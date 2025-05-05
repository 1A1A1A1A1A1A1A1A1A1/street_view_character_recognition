#!/bin/bash

# 街景字符识别项目目录结构创建脚本

echo -e "\n===== 创建目录结构 ====="
mkdir -p coco/images/train
mkdir -p coco/images/val
mkdir -p coco/images/test
mkdir -p coco/labels/train
mkdir -p coco/labels/val
mkdir -p weights
mkdir -p data
mkdir -p results

echo "目录结构创建完成:"
echo "- coco/images/{train,val,test}"
echo "- coco/labels/{train,val}"
echo "- weights"
echo "- data"
echo "- results"
echo "===== 目录结构创建完成 =====" 