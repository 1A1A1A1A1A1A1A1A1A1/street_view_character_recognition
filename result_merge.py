"""
模型结果合并工具

本脚本用于合并多个YOLOv5模型的检测结果，通过非极大值抑制(NMS)算法消除重叠的检测框，
生成最终的街景字符识别结果。这种模型集成方法可以有效提高识别准确率。

主要功能：
1. 读取多个模型生成的JSON结果文件
2. 对每个测试图像，合并所有模型的检测结果
3. 使用NMS算法过滤重叠的检测框，保留置信度最高的结果
4. 按照字符的水平位置(x坐标)排序，生成最终的字符序列
5. 将合并后的结果保存为比赛提交格式的CSV文件

使用方法：
    修改result_list变量为你的模型结果文件路径，然后直接运行脚本
    python result_merge.py

参数设置：
    result_list: 存放各个模型结果JSON文件的路径列表
    thresh: NMS阈值，默认为0.3，用于控制检测框重叠度的容忍程度
    
输出文件：
    submit.csv: 最终的提交文件，包含file_name和file_code两列
"""

import json
import numpy as np
import pandas as pd

# 要合并的模型结果文件列表
result_list = ['./json/yolov5l_0.922.json', './json/yolov5s_0.902.json']
result = []


def py_cpu_nms(dets, thresh):
    """
    非极大值抑制(NMS)算法的CPU实现
    
    该函数用于过滤重叠的检测框，保留置信度最高的结果，并按x坐标排序生成字符序列
    
    参数:
        dets: 检测框数组，每行格式为[x1, y1, x2, y2, score, class]
              其中(x1,y1)是左上角坐标，(x2,y2)是右下角坐标，score是置信度，class是类别
        thresh: IoU阈值，用于判断两个框是否重叠，默认0.3
        
    返回:
        out: 字典，键为x坐标，值为对应的类别，用于后续按位置排序
    """
    x1 = dets[:, 0]  # 左上角x坐标
    y1 = dets[:, 1]  # 左上角y坐标
    x2 = dets[:, 2]  # 右下角x坐标
    y2 = dets[:, 3]  # 右下角y坐标
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)  # 计算每个框的面积
    scores = dets[:, 4]  # 置信度分数
    keep = []  # 保留的检测框索引
    
    # 按置信度降序排序
    index = scores.argsort()[::-1]
    
    while index.size > 0:
        i = index[0]  # 每次取置信度最高的框
        keep.append(i)

        # 计算当前最高置信度框与其他框的重叠区域
        x11 = np.maximum(x1[i], x1[index[1:]])  # 重叠区域的左上角x
        y11 = np.maximum(y1[i], y1[index[1:]])  # 重叠区域的左上角y
        x22 = np.minimum(x2[i], x2[index[1:]])  # 重叠区域的右下角x
        y22 = np.minimum(y2[i], y2[index[1:]])  # 重叠区域的右下角y

        # 计算重叠区域的宽和高
        w = np.maximum(0, x22 - x11 + 1)  # 重叠区域的宽
        h = np.maximum(0, y22 - y11 + 1)  # 重叠区域的高

        # 计算重叠面积和IoU
        overlaps = w * h  # 重叠区域的面积
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)  # 计算IoU

        # 保留IoU小于阈值的框（即与当前框重叠度较小的框）
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # 更新索引（+1是因为idx是从index[1:]计算的）
    
    # 将保留的检测框信息转换为字典，键为x坐标中心，值为类别
    out = dict()
    for _ in keep:
        out[dets[_][0]] = int(dets[_][5])  # 使用x1坐标作为键（用于后续排序）
    return out


# 加载所有模型的结果
for i in range(len(result_list)):
    result.append(json.load(open(result_list[i])))

file_name = []  # 文件名列表
file_code = []  # 识别结果列表

# 处理每个测试图像
for key in result[0]:
    print(key)  # 打印当前处理的图像文件名
    file_name.append(key)
    
    # 收集所有模型对当前图像的检测结果
    t = []
    for i in range(len(result_list)):
        for _ in result[i][key]:
            t.append(_)
    t = np.array(t)
    
    # 应用NMS并生成最终字符序列
    res = ''
    if len(t) == 0:
        res = '1'  # 如果没有检测到任何字符，默认输出'1'
    else:
        # 应用NMS算法，并按x坐标排序生成字符序列
        x_value = py_cpu_nms(t, 0.3)
        for x in sorted(x_value.keys()):  # 按x坐标（从左到右）排序
            res += str(x_value[x])
    
    file_code.append(res)

# 生成最终的提交文件
sub = pd.DataFrame({'file_name': file_name, 'file_code': file_code})
sub.to_csv('./submit.csv', index=False)
