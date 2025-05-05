"""
标签转换工具

本脚本用于将原始JSON格式的标注数据转换为YOLO模型训练所需的TXT格式标签文件。

主要功能：
1. 读取训练集和验证集的JSON标注文件
2. 读取对应图像以获取尺寸信息
3. 提取每个字符的标注信息（标签、位置、大小）
4. 计算YOLO格式所需的归一化坐标（中心点x,y及宽高）
5. 将转换后的标注数据写入TXT文件（每行格式：类别 中心x 中心y 宽 高）
6. 验证集标签文件添加"val_"前缀，与图像文件命名保持一致

使用方法：
    python make_label.py [--train_image_path TRAIN_IMG_PATH] [--val_image_path VAL_IMG_PATH] 
                         [--train_annotation_path TRAIN_ANN_PATH] [--val_annotation_path VAL_ANN_PATH]
                         [--label_path LABEL_PATH]

参数说明：
    --train_image_path: 训练集图像路径
    --val_image_path: 验证集图像路径
    --train_annotation_path: 训练集JSON标注文件路径
    --val_annotation_path: 验证集JSON标注文件路径
    --label_path: 输出标签文件的目标路径

输出格式：
    每个图像对应一个TXT文件，文件中每行表示一个字符标注，格式为：
    <类别ID> <归一化中心x> <归一化中心y> <归一化宽度> <归一化高度>
"""

import cv2
import json
import argparse
import os
import numpy as np


def clip_to_range(value, min_val=0.0, max_val=1.0):
    """
    将值限制在指定范围内
    
    参数:
        value: 需要限制的值
        min_val: 最小值，默认为0.0
        max_val: 最大值，默认为1.0
        
    返回:
        限制后的值
    """
    return max(min_val, min(value, max_val))


def main():
    """
    主函数，负责解析命令行参数并执行标签转换操作
    
    流程：
    1. 解析命令行参数获取路径信息
    2. 加载训练集和验证集的JSON标注数据
    3. 处理训练集标注，转换为YOLO格式并写入TXT文件
    4. 处理验证集标注，转换为YOLO格式并写入TXT文件（添加"val_"前缀）
    """
    parser = argparse.ArgumentParser(description="为数据集生成YOLO格式标签文件")
    parser.add_argument('-ti', '--train_image_path', default='./coco/images/train/mchar_train/',
                        help="训练集图像路径")
    parser.add_argument('-vi', '--val_image_path', default='./coco/images/val/mchar_val',
                        help="验证集图像路径")
    parser.add_argument('-ta', '--train_annotation_path',
                        default='./coco/labels/train/mchar_train.json',
                        help="训练集标注文件路径")
    parser.add_argument('-va', '--val_annotation_path',
                        default='./coco/labels/val/mchar_val.json',
                        help="验证集标注文件路径")
    parser.add_argument('-l', '--label_path', default='./coco/labels/', help='标签文件输出路径')
    args = parser.parse_args()

    # 确保标签目录存在
    os.makedirs(args.label_path, exist_ok=True)
    
    # 检查输入文件是否存在
    input_files = [args.train_annotation_path, args.val_annotation_path]
    missing_files = [f for f in input_files if not os.path.exists(f)]
    if missing_files:
        print("错误：以下文件不存在：")
        for f in missing_files:
            print(f"  - {f}")
        print("\n请确保标注文件位于正确的位置，或者使用参数指定正确的路径。")
        return

    # 加载JSON格式的标注数据
    try:
        train_data = json.load(open(args.train_annotation_path))
        val_data = json.load(open(args.val_annotation_path))
    except json.JSONDecodeError:
        print("错误：JSON标注文件格式不正确，请检查文件内容。")
        return

    # 处理训练集标注数据
    train_processed = 0
    train_errors = 0
    train_out_of_bounds = 0
    
    for key in train_data:
        try:
            if not os.path.exists(args.train_image_path + key):
                print(f"警告：找不到训练图像 {key}，已跳过")
                train_errors += 1
                continue
                
            f = open(args.label_path + key.replace('.png', '.txt'), 'w')
            img = cv2.imread(args.train_image_path + key)
            if img is None:
                print(f"警告：无法读取训练图像 {key}，已跳过")
                train_errors += 1
                continue
                
            shape = img.shape  # 获取图像尺寸，用于坐标归一化
            label = train_data[key]['label']  # 字符类别标签
            left = train_data[key]['left']    # 左上角x坐标
            top = train_data[key]['top']      # 左上角y坐标
            height = train_data[key]['height']  # 高度
            width = train_data[key]['width']    # 宽度
            
            # 处理每个字符的标注
            has_out_of_bounds = False
            for i in range(len(label)):
                # 计算归一化的中心点坐标和宽高
                # 确保坐标在图像范围内
                left_val = max(0, min(left[i], shape[1]-1))
                top_val = max(0, min(top[i], shape[0]-1))
                width_val = max(1, min(width[i], shape[1] - left_val))
                height_val = max(1, min(height[i], shape[0] - top_val))
                
                # 计算归一化坐标
                x_center = 1.0 * (left_val + width_val / 2) / shape[1]  # 中心点x坐标归一化
                y_center = 1.0 * (top_val + height_val / 2) / shape[0]  # 中心点y坐标归一化
                w = 1.0 * width_val / shape[1]   # 宽度归一化
                h = 1.0 * height_val / shape[0]  # 高度归一化
                
                # 检查是否有坐标超出范围
                if (x_center < 0 or x_center > 1 or 
                    y_center < 0 or y_center > 1 or 
                    w <= 0 or w > 1 or 
                    h <= 0 or h > 1):
                    has_out_of_bounds = True
                    # 截断到合法范围
                    x_center = clip_to_range(x_center)
                    y_center = clip_to_range(y_center)
                    w = clip_to_range(w, 0.001, 1.0)  # 宽度最小为0.001
                    h = clip_to_range(h, 0.001, 1.0)  # 高度最小为0.001
                
                # 写入YOLO格式标签：类别 中心x 中心y 宽度 高度
                f.write(str(label[i]) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(w) + ' ' + str(h) + '\n')
            
            f.close()
            
            if has_out_of_bounds:
                train_out_of_bounds += 1
                if train_out_of_bounds <= 5:  # 只显示前5个异常
                    print(f"警告：训练图像 {key} 的坐标存在越界，已截断到[0,1]范围")
            
            train_processed += 1
            if train_processed % 100 == 0:
                print(f"已处理 {train_processed}/{len(train_data)} 个训练集图像")
        except Exception as e:
            print(f"处理训练图像 {key} 时出错: {str(e)}")
            train_errors += 1

    # 处理验证集标注数据
    val_processed = 0
    val_errors = 0
    val_out_of_bounds = 0
    
    for key in val_data:
        try:
            if not os.path.exists(args.val_image_path + key):
                print(f"警告：找不到验证图像 {key}，已跳过")
                val_errors += 1
                continue
                
            # 标签文件名添加"val_"前缀，与图像文件命名保持一致
            f = open(args.label_path + 'val_' + key.replace('.png', '.txt'), 'w')
            img = cv2.imread(args.val_image_path + key)
            if img is None:
                print(f"警告：无法读取验证图像 {key}，已跳过")
                val_errors += 1
                continue
                
            shape = img.shape
            label = val_data[key]['label']
            left = val_data[key]['left']
            top = val_data[key]['top']
            height = val_data[key]['height']
            width = val_data[key]['width']
            
            has_out_of_bounds = False
            for i in range(len(label)):
                # 确保坐标在图像范围内
                left_val = max(0, min(left[i], shape[1]-1))
                top_val = max(0, min(top[i], shape[0]-1))
                width_val = max(1, min(width[i], shape[1] - left_val))
                height_val = max(1, min(height[i], shape[0] - top_val))
                
                # 计算归一化的中心点坐标和宽高
                x_center = 1.0 * (left_val + width_val / 2) / shape[1]
                y_center = 1.0 * (top_val + height_val / 2) / shape[0]
                w = 1.0 * width_val / shape[1]
                h = 1.0 * height_val / shape[0]
                
                # 检查是否有坐标超出范围
                if (x_center < 0 or x_center > 1 or 
                    y_center < 0 or y_center > 1 or 
                    w <= 0 or w > 1 or 
                    h <= 0 or h > 1):
                    has_out_of_bounds = True
                    # 截断到合法范围
                    x_center = clip_to_range(x_center)
                    y_center = clip_to_range(y_center)
                    w = clip_to_range(w, 0.001, 1.0)  # 宽度最小为0.001
                    h = clip_to_range(h, 0.001, 1.0)  # 高度最小为0.001
                
                # 写入YOLO格式标签
                f.write(str(label[i]) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(w) + ' ' + str(h) + '\n')
            
            f.close()
            
            if has_out_of_bounds:
                val_out_of_bounds += 1
                if val_out_of_bounds <= 5:  # 只显示前5个异常
                    print(f"警告：验证图像 {key} 的坐标存在越界，已截断到[0,1]范围")
            
            val_processed += 1
            if val_processed % 100 == 0:
                print(f"已处理 {val_processed}/{len(val_data)} 个验证集图像")
        except Exception as e:
            print(f"处理验证图像 {key} 时出错: {str(e)}")
            val_errors += 1

    # 删除之前可能存在的缓存文件
    if os.path.exists('./coco/labels.npy'):
        os.remove('./coco/labels.npy')
    
    # 生成numpy缓存文件
    print("正在生成缓存文件...")
    if train_processed > 0 or val_processed > 0:
        # 创建空的labels.npy和images.shapes文件
        np.save('./coco/labels.npy', np.array([]))
        with open('./coco/images.shapes', 'w') as f:
            f.write("")
        print("已创建缓存文件占位符，训练时会自动填充")
    
    # 输出统计信息
    print(f"\n处理完成:")
    print(f"训练集: 已处理 {train_processed}/{len(train_data)} 个图像，错误 {train_errors} 个，坐标越界 {train_out_of_bounds} 个")
    print(f"验证集: 已处理 {val_processed}/{len(val_data)} 个图像，错误 {val_errors} 个，坐标越界 {val_out_of_bounds} 个")
    print(f"标签文件输出到: {args.label_path}")
    
    if train_processed == 0 and val_processed == 0:
        print("\n警告: 未能生成任何标签文件，请检查图像和标注路径是否正确。")


if __name__ == '__main__':
    main()
