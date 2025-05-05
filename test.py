"""
模型测试与评估脚本

本脚本用于测试和评估YOLOv5模型在验证集或测试集上的性能表现，计算各种评估指标。

主要功能：
1. 加载训练好的模型
2. 在验证集或测试集上执行目标检测
3. 计算精确率(P)、召回率(R)、mAP@0.5和mAP@0.5:0.95等评估指标
4. 支持保存结果为COCO格式JSON文件，兼容COCO评估API
5. 可视化测试结果（标注真实框和预测框）
6. 提供"研究"模式，在不同图像尺寸下评估模型性能

使用方法：
    python test.py --weights [权重文件路径] --data [数据配置文件路径] --task [val/test/study]

参数说明：
    --weights: 模型权重文件路径，默认为weights/yolov5s.pt
    --data: 数据集配置文件路径，默认为data/coco128.yaml
    --batch-size: 批处理大小，默认为32
    --img-size: 推理图像尺寸，默认为640像素
    --conf-thres: 对象置信度阈值，默认为0.001
    --iou-thres: NMS的IOU阈值，默认为0.65
    --save-json: 保存结果为COCO格式JSON文件
    --task: 任务类型（val - 验证，test - 测试，study - 研究），默认为val
    --device: 计算设备，可指定CUDA设备或CPU
    --single-cls: 将数据集视为单类别数据集
    --augment: 使用增强推理（测试时增强）
    --merge: 使用Merge NMS
    --verbose: 按类别报告mAP
"""

import argparse
import json

from utils import google_utils
from utils.datasets import *
from utils.utils import *


def test(data,
         weights=None,
         batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         merge=False):
    """
    在验证集或测试集上测试模型性能
    
    参数:
        data: 数据集配置文件路径
        weights: 模型权重文件路径，默认为None
        batch_size: 批处理大小，默认为16
        imgsz: 输入图像尺寸，默认为640
        conf_thres: 置信度阈值，默认为0.001
        iou_thres: NMS的IOU阈值，默认为0.6
        save_json: 是否保存结果为COCO格式JSON文件，默认为False
        single_cls: 是否将数据集视为单类别，默认为False
        augment: 是否使用增强推理，默认为False
        verbose: 是否打印详细信息，默认为False
        model: 预加载的模型，默认为None（用于train.py调用）
        dataloader: 预加载的数据加载器，默认为None（用于train.py调用）
        merge: 是否使用Merge NMS，默认为False
        
    返回:
        (mp, mr, map50, map, ...): 各种评估指标（精确率均值，召回率均值，mAP@0.5，mAP@0.5:0.95等）
        maps: 每个类别的AP值
        t: 推理和NMS的时间
    """
    # 初始化/加载模型并设置设备
    if model is None:
        training = False
        device = torch_utils.select_device(opt.device, batch_size=batch_size)

        # 删除之前的测试结果
        for f in glob.glob('test_batch*.jpg'):
            os.remove(f)

        # 加载模型
        google_utils.attempt_download(weights)
        model = torch.load(weights, map_location=device)['model'].float()  # 加载为FP32
        torch_utils.model_info(model)
        model.fuse()
        model.to(device)

        # 多GPU暂不支持，与.half()不兼容 https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    else:  # 由train.py调用
        training = True
        device = next(model.parameters()).device  # 获取模型所在设备

    # 半精度
    half = device.type != 'cpu' and torch.cuda.device_count() == 1  # 半精度仅支持单GPU
    if half:
        model.half()  # 转换为FP16

    # 配置
    model.eval()
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # 模型字典
    nc = 1 if single_cls else int(data['nc'])  # 类别数量
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # mAP@0.5:0.95的IOU向量
    niou = iouv.numel()

    # 数据加载器
    if dataloader is None:  # 非训练模式
        merge = opt.merge  # 使用Merge NMS
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # 初始化图像
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # 预热一次
        path = data['test'] if opt.task == 'test' else data['val']  # 验证/测试图像路径
        dataloader = create_dataloader(path, imgsz, batch_size, int(max(model.stride)), opt,
                                       hyp=None, augment=False, cache=False, pad=0.5, rect=True)[0]

    seen = 0
    names = model.names if hasattr(model, 'names') else model.module.names
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('类别', '图像数', '目标数', '精确率', '召回率', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    
    # 迭代处理每个批次
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device)
        img = img.half() if half else img.float()  # uint8转为fp16/32
        img /= 255.0  # 0-255缩放到0.0-1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # 批次大小、通道数、高度、宽度
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # 禁用梯度计算
        with torch.no_grad():
            # 运行模型
            t = torch_utils.time_synchronized()
            inf_out, train_out = model(img, augment=augment)  # 推理和训练输出
            t0 += torch_utils.time_synchronized() - t

            # 计算损失
            if training:  # 如果模型有损失超参数
                loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # GIoU, obj, cls

            # 运行NMS
            t = torch_utils.time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, merge=merge)
            t1 += torch_utils.time_synchronized() - t

        # 统计每张图像的信息
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # 目标类别
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # 将检测框裁剪到图像边界
            clip_coords(pred, (height, width))

            # 添加到pycocotools JSON字典
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])  # 还原到原始图像尺寸
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy中心点转为左上角坐标
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])],
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # 初始化所有预测为不正确
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # 已检测到的目标索引
                tcls_tensor = labels[:, 0]

                # 目标框
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # 针对每个目标类别
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # 该类别的目标索引
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # 该类别的预测索引

                    # 搜索检测结果
                    if pi.shape[0]:
                        # 预测框与目标框的IOU
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # 最佳IOU和索引

                        # 添加检测结果
                        for j in (ious > iouv[0]).nonzero():
                            d = ti[i[j]]  # 检测到的目标
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres是1xn
                                if len(detected) == nl:  # 图像中的所有目标都已定位
                                    break

            # 添加统计信息（正确性、置信度、预测类别、目标类别）
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # 绘制图像
        if batch_i < 1:
            f = 'test_batch%g_gt.jpg' % batch_i  # 文件名
            plot_images(img, targets, paths, f, names)  # 真实框
            f = 'test_batch%g_pred.jpg' % batch_i
            plot_images(img, output_to_target(output, width, height), paths, f, names)  # 预测框

    # 计算统计信息
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # 转为numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # 每个类别的目标数量
    else:
        nt = torch.zeros(1)

    # 打印结果
    pf = '%20s' + '%12.3g' * 6  # 打印格式
    print(pf % ('所有类别', seen, nt.sum(), mp, mr, map50, map))

    # 按类别打印结果
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # 打印速度
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # 元组
    if not training:
        print('速度: %.1f/%.1f/%.1f 毫秒 推理/NMS/总计 每张%gx%g图像，批次大小为%g' % t)

    # 保存JSON结果
    if save_json and map50 and len(jdict):
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataloader.dataset.img_files]
        f = 'detections_val2017_%s_results.json' % \
            (weights.split(os.sep)[-1].replace('.pt', '') if weights else '')  # 文件名
        print('\n使用pycocotools计算COCO mAP... 保存 %s...' % f)
        with open(f, 'w') as file:
            json.dump(jdict, file)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            cocoGt = COCO(glob.glob('../coco/annotations/instances_val*.json')[0])  # 初始化COCO真实值API
            cocoDt = cocoGt.loadRes(f)  # 初始化COCO预测API

            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # 要评估的图像ID
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            map, map50 = cocoEval.stats[:2]  # 更新结果(mAP@0.5:0.95, mAP@0.5)
        except:
            print('警告: 必须使用numpy==1.17安装pycocotools才能正确运行。'
                  '参见 https://github.com/cocodataset/cocoapi/issues/356')

    # 返回结果
    model.float()  # 用于训练
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='模型权重文件路径')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='数据配置文件路径')
    parser.add_argument('--batch-size', type=int, default=32, help='批处理大小')
    parser.add_argument('--img-size', type=int, default=640, help='推理图像尺寸(像素)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='对象置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='NMS的IOU阈值')
    parser.add_argument('--save-json', action='store_true', help='保存结果为COCO格式JSON文件')
    parser.add_argument('--task', default='val', help="'val'(验证), 'test'(测试), 'study'(研究)")
    parser.add_argument('--device', default='', help='cuda设备，如0或0,1,2,3或cpu')
    parser.add_argument('--single-cls', action='store_true', help='将数据集视为单类别数据集')
    parser.add_argument('--augment', action='store_true', help='增强推理')
    parser.add_argument('--merge', action='store_true', help='使用Merge NMS')
    parser.add_argument('--verbose', action='store_true', help='按类别报告mAP')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    opt.save_json = opt.save_json or opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # 检查文件
    print(opt)

    # 任务 = 'val', 'test', 'study'
    if opt.task in ['val', 'test']:  # （默认）正常运行
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose)

    elif opt.task == 'study':  # 在一系列设置下运行并保存/绘制结果
        for weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
            f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem)  # 保存的文件名
            x = list(range(352, 832, 64))  # x轴
            y = []  # y轴
            for i in x:  # 图像尺寸
                print('\n运行 %s 点 %s...' % (f, i))
                r, _, t = test(opt.data, weights, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json)
                y.append(r + t)  # 结果和时间
            np.savetxt(f, y, fmt='%10.4g')  # 保存
        os.system('zip -r study.zip study_*.txt')
        # plot_study_txt(f, x)  # 绘图
