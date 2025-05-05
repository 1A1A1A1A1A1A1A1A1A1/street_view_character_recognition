"""
数据集合并工具

本脚本用于将训练集和验证集的图像合并到同一个目录中，便于统一管理和处理。

主要功能：
1. 读取指定路径下的训练集和验证集图像
2. 按照数字顺序排序图像（文件名为数字.jpg格式）
3. 将训练集图像直接复制到目标路径
4. 将验证集图像复制到目标路径，并添加"val_"前缀以防止文件名冲突
5. 显示复制进度信息

使用方法：
    python image_merge.py [--train_image_path TRAIN_PATH] [--val_image_path VAL_PATH] [--dst_image_path DST_PATH]

参数说明：
    --train_image_path: 训练集图像路径，默认为'coco/images/train/'
    --val_image_path: 验证集图像路径，默认为'coco/images/val/'
    --dst_image_path: 目标路径（合并后的图像存储位置），默认为'./coco/images/'

注意事项：
    - 确保目标路径已存在，否则会出错
    - 验证集图像会添加"val_"前缀，以防止与训练集图像文件名冲突
"""

import os
import shutil
import argparse


def main():
    """
    主函数，负责解析命令行参数并执行图像合并操作
    
    流程：
    1. 解析命令行参数获取路径信息
    2. 读取训练集和验证集图像列表并排序
    3. 复制训练集图像到目标路径
    4. 复制验证集图像到目标路径（添加"val_"前缀）
    5. 显示进度信息
    """
    parser = argparse.ArgumentParser(description="合并训练数据集和验证数据集")
    parser.add_argument('-t', '--train_image_path', default='coco/images/train/',
                        help="训练数据集路径")
    parser.add_argument('-v', '--val_image_path', default='coco/images/val/',
                        help="验证数据集路径")
    parser.add_argument('-d', '--dst_image_path', default='./coco/images/', help="合并后数据集的目标路径")
    args = parser.parse_args()

    # 获取并排序训练集和验证集图像列表
    train_image_list = os.listdir(args.train_image_path)
    train_image_list.sort(key=lambda x: int(x[:-4]))  # 按文件名数字排序（去掉.jpg后缀）
    val_image_list = os.listdir(args.val_image_path)
    val_image_list.sort(key=lambda x: int(x[:-4]))  # 按文件名数字排序

    # 复制训练集图像
    for img in train_image_list:
        shutil.copy(args.train_image_path + img, args.dst_image_path + img)
        print("训练集图像 {0}/{1} 已复制".format(img[:-4], len(train_image_list)))
    print("训练数据集复制完成")
    
    # 复制验证集图像，添加"val_"前缀以防止重名
    for img in val_image_list:
        shutil.copy(args.val_image_path + img, args.dst_image_path + 'val_' + img)  # 防止重名
        print("验证集图像 {0}/{1} 已复制".format(img[:-4], len(val_image_list)))
    print("验证数据集复制完成")


if __name__ == '__main__':
    main()
