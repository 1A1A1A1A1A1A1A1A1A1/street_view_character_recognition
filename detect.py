"""
街景字符检测与识别脚本

本文件是街景字符识别系统的主要执行脚本，用于检测和识别图像中的字符（如门牌号、路标等）。

主要功能：
1. 加载预训练的YOLOv5模型
2. 处理输入图像/视频/摄像头数据
3. 检测并识别图像中的字符
4. 将检测到的字符按位置排序并组合
5. 输出识别结果到CSV和JSON文件

使用方法：
    python detect.py --source [图像/视频路径] --weights [模型权重路径]

参数说明：
    --weights: 模型权重文件路径，默认为weights/yolov5s.pt
    --source: 输入源，可以是图像、视频文件夹路径或0（网络摄像头）
    --output: 输出文件夹路径，默认为inference/output
    --img-size: 推理图像大小，默认为320像素
    --conf-thres: 对象置信度阈值，默认为0.4
    --iou-thres: NMS的IOU阈值，默认为0.5
    --device: 计算设备，可指定CUDA设备或CPU
    --view-img: 是否显示结果，默认为False
    --save-txt: 是否保存结果到文本文件，默认为False
    --classes: 按类别筛选，默认为None（所有类别）
    --agnostic-nms: 是否使用类别无关的NMS，默认为False
    --augment: 是否使用增强推理，默认为False

输出文件：
    submission-yolov5x.csv: 包含文件名和识别字符的CSV文件
    yolov5x.json: 包含详细检测结果的JSON文件
"""

import argparse
import os
import torch.backends.cudnn as cudnn
import cv2
import time
import torch
import shutil
import random
from utils import google_utils
from utils.datasets import *
from utils.utils import *
import pandas as pd
import json


def detect(save_img=False):
    """
    执行字符检测和识别的主函数
    
    参数:
        save_img: 是否保存图像结果，默认为False
        
    功能流程:
        1. 初始化设备和模型
        2. 设置数据加载器
        3. 执行推理检测
        4. 处理检测结果
        5. 保存识别结果到CSV和JSON文件
    """
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    file_name = []  # 存储处理的文件名
    file_code = []  # 存储识别的字符代码
    result = dict()  # 存储详细检测结果
    
    # 初始化设备
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # 删除已存在的输出文件夹
    os.makedirs(out)  # 创建新的输出文件夹
    half = device.type != 'cpu'  # 半精度仅在CUDA上支持

    # 加载模型
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # 加载为FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # 如果出现SourceChangeWarning则更新模型
    # model.fuse()
    model.to(device).eval()
    if half:
        model.half()  # 转换为FP16

    # 第二阶段分类器（可选）
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # 初始化
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # 加载权重
        modelc.to(device).eval()

    # 设置数据加载器
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # 设置为True以加速恒定图像大小推理
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # 获取类别名称和颜色
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # 运行推理
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # 初始化图像
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # 先运行一次
    
    # 逐图像处理
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8转为fp16/32
        img /= 255.0  # 0-255缩放到0.0-1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 执行推理
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # 应用非极大值抑制
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # 应用分类器（如果启用）
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # 记录当前处理的文件名
        file_name.append(path.split('/')[-1])
        result[path.split('/')[-1]] = []
        res = ''
        
        # 在这里初始化x_value，确保它始终被定义
        x_value = dict()  # 保存x坐标-类别对应关系，用于按位置排序
        
        # 处理检测结果
        for i, det in enumerate(pred):  # 每张图像的检测结果
            if webcam:  # 如果是网络摄像头（批量大小>=1）
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # 打印图像尺寸
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化增益wh
            
            # 如果有检测结果
            if det is not None and len(det):
                # 将检测框尺寸从img_size缩放到im0尺寸
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # 打印检测到的各类别数量
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # 每个类别的检测数
                    s += '%g %ss, ' % (n, names[int(c)])  # 添加到输出字符串

                # 处理并写入结果
                for *xyxy, conf, cls in det:
                    # 提取边界框信息
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # 规范化xywh
                    cls = torch.tensor(cls).tolist()
                    conf = torch.tensor(conf).tolist()
                    x_value[xywh[0]] = int(cls)  # 使用x坐标作为排序键
                    
                    if save_txt:  # 写入文件
                        # 计算完整边界框坐标
                        r = [xywh[0] - xywh[2] / 2, xywh[1] - xywh[3] / 2, xywh[0] + xywh[2] / 2, xywh[1] + xywh[3] / 2]
                        result[path.split('/')[-1]].append(r + [conf, int(cls)])
                    
                    if save_img or view_img:  # 在图像上添加边界框
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # 打印处理时间（推理+NMS）
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # 显示结果
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # 按q退出
                    raise StopIteration

            # 根据需要保存图像/视频结果（此部分已注释）
            
        # 按x坐标顺序合并识别的字符
        for key in sorted(x_value):
            res += str(x_value[key])
        file_code.append(res)
    
    # 保存结果到CSV和JSON文件
    sub = pd.DataFrame({"file_name": file_name, 'file_code': file_code})
    sub.to_csv('submission-yolov5x.csv', index=False)
    with open("yolov5x.json", "w") as f:
        json.dump(result, f)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='模型权重文件路径')
    parser.add_argument('--source', type=str, default='inference/images', help='源文件路径，文件/文件夹，0表示网络摄像头')
    parser.add_argument('--output', type=str, default='inference/output', help='输出文件夹')
    parser.add_argument('--img-size', type=int, default=320, help='推理尺寸（像素）')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='对象置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS的IOU阈值')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='输出视频编解码器（需验证ffmpeg支持）')
    parser.add_argument('--device', default='', help='cuda设备，例如0或0,1,2,3或cpu')
    parser.add_argument('--view-img', action='store_true', help='显示结果')
    parser.add_argument('--save-txt', action='store_true', help='保存结果到*.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='按类别过滤')
    parser.add_argument('--agnostic-nms', action='store_true', help='类别无关的NMS')
    parser.add_argument('--augment', action='store_true', help='增强推理')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)

    with torch.no_grad():
        detect()

        # 更新所有模型
        # for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
        #    detect()
        #    create_pretrained(opt.weights, opt.weights)
