# 街景字符识别系统

## 项目介绍

本项目是一个基于计算机视觉的街景字符识别系统，主要针对街景图像中的门牌号、路标等文字进行检测和识别。该系统基于Google的SVNH(Street View Number House)数据集开发，旨在解决实际场景中文字检测与识别的挑战性问题。

项目最初源于阿里巴巴天池比赛，通过结合多种模型、数据增强和全局非极大值抑制等技术，实现了高准确率的街景字符识别。实验结果表明，我们的方法在多种测试场景中都取得了优异的表现。

## 技术方案

本项目采用了以下核心技术：

1. **目标检测框架**: 使用YOLOv5作为基础检测框架，实现对图像中字符的精确定位
2. **多模型融合**: 结合不同规模的YOLOv5模型(s/m/l/x)，提高识别的准确性和鲁棒性
3. **数据增强**: 应用多种图像增强技术，如旋转、缩放、色彩调整等，提高模型泛化能力
4. **全局非极大值抑制(NMS)**: 优化检测结果，消除重复检测和错误识别
5. **字符排序与组合**: 根据检测到的字符位置，自动进行排序和组合，形成完整的识别结果

系统处理流程：
- 输入图像预处理
- 字符检测与定位
- 字符识别分类
- 结果后处理与排序
- 输出最终识别结果

## 目录结构

```
├── coco/                 # 数据集目录
│   ├── images/           # 所有图像文件
│   │   ├── train/        # 训练集图像
│   │   ├── val/          # 验证集图像
│   │   └── test/         # 测试集图像
│   └── labels/           # 标签文件
│       ├── train/        # 训练集标签
│       └── val/          # 验证集标签
│
├── data/                 # 数据配置目录
│   └── coco.yaml         # 数据集配置文件
│
├── models/               # 模型定义目录
│   ├── yolov5s.yaml      # 小型YOLOv5模型配置
│   ├── yolov5m.yaml      # 中型YOLOv5模型配置
│   ├── yolov5l.yaml      # 大型YOLOv5模型配置
│   └── yolov5x.yaml      # 超大型YOLOv5模型配置
│
├── utils/                # 工具函数目录
│   ├── activations.py    # 激活函数
│   ├── datasets.py       # 数据集加载工具
│   ├── google_utils.py   # Google云存储工具
│   ├── torch_utils.py    # PyTorch工具函数
│   └── utils.py          # 通用工具函数
│
├── weights/              # 模型权重目录
│   └── yolov5*.pt        # 预训练模型权重
│
├── detect.py             # 字符检测与识别脚本
├── image_merge.py        # 训练集和验证集图像合并工具
├── make_label.py         # JSON标注转YOLO格式标签工具
├── train.py              # 模型训练脚本
├── test.py               # 模型测试与评估脚本
├── result_merge.py       # 多模型结果合并工具
├── create_dirs.sh        # 创建目录结构脚本
├── install_conda.sh      # Conda环境安装脚本
├── run_pipeline.sh       # 项目流程自动化脚本
```

## 创建文件夹

项目提供了便捷的目录结构创建脚本`create_dirs.sh`，执行该脚本将自动创建所需的所有目录：

```bash
# 赋予执行权限
chmod +x create_dirs.sh

# 执行脚本
./create_dirs.sh
```

脚本执行后将创建以下目录：
- coco/images/{train,val,test}
- coco/labels/{train,val}
- weights
- data
- results

## 环境搭建

项目提供了自动化的Conda环境安装脚本`install_conda.sh`：

```bash
# 赋予执行权限
chmod +x install_conda.sh

# 执行脚本
./install_conda.sh
```

此脚本会：
1. 检查Conda是否已安装
2. 配置国内镜像源以加速下载
3. 创建名为"yolov5-env"的Python 3.7环境
4. 安装基础依赖和PyTorch
5. 验证安装和CUDA支持
6. 下载YOLOv5预训练权重

主要安装的依赖包括：
- PyTorch 1.12.0 和 torchvision
- CUDA 工具包 11.3
- numpy 1.17（COCO API所需版本）
- pandas
- matplotlib
- OpenCV
- tensorboard

## 模型训练与预测

### 1. 数据预处理

#### 合并图像数据集

使用`image_merge.py`脚本合并训练集和验证集图像，便于统一管理：

```bash
python image_merge.py --train_image_path coco/images/train/mchar_train/ \
                      --val_image_path coco/images/val/mchar_val/ \
                      --dst_image_path ./coco/images/
```

#### 标签转换

使用`make_label.py`脚本将JSON格式的标注转换为模型需要的格式：

```bash
python make_label.py --train_image_path coco/images/train/mchar_train/ \
                     --val_image_path coco/images/val/mchar_val/ \
                     --train_annotation_path coco/labels/train/mchar_train.json \
                     --val_annotation_path coco/labels/val/mchar_val.json \
                     --label_path ./coco/labels/
```

#### 创建数据配置文件

在`data/coco.yaml`中配置数据集信息：
```yaml
train: ./coco/images/
val: ./coco/images/

# 类别数量
nc: 10

# 类别名称
names: ['0','1','2','3','4','5','6','7','8','9']
```

### 2. 模型训练

使用`train.py`脚本训练YOLOv5模型：

```bash
python train.py --data ./data/coco.yaml \
                --cfg ./models/yolov5m.yaml \
                --weights weights/yolov5m.pt \
                --batch-size 32 \
                --img-size 640 \
                --epochs 50 \
                --device 0 \
                --noautoanchor \
                --cache-images
```

参数说明：
- `--data`: 数据集配置文件路径
- `--cfg`: 模型配置文件路径，可选择s/m/l/x不同大小的模型
- `--weights`: 预训练权重路径
- `--batch-size`: 批处理大小，根据GPU显存调整
- `--img-size`: 训练图像尺寸
- `--epochs`: 训练轮数
- `--device`: 使用的GPU设备，0表示第一个GPU
- `--noautoanchor`: 禁用自动锚框检查
- `--cache-images`: 缓存图像以加速训练

### 3. 模型评估

使用`test.py`脚本评估模型性能：

```bash
python test.py --weights weights/best.pt \
               --data data/coco.yaml \
               --img-size 640 \
               --batch-size 32 \
               --device 0
```

此脚本会计算精确率、召回率、mAP@0.5和mAP@0.5:0.95等评估指标。

### 4. 预测识别

使用`detect.py`脚本进行字符检测与识别：

```bash
python detect.py --source coco/images/test/mchar_test_a/ \
                 --weights weights/best.pt \
                 --img-size 640 \
                 --conf-thres 0.4 \
                 --iou-thres 0.5 \
                 --device 0
```

参数说明：
- `--source`: 输入图像或文件夹路径
- `--weights`: 模型权重文件路径
- `--img-size`: 输入图像尺寸
- `--conf-thres`: 置信度阈值
- `--iou-thres`: NMS的IOU阈值
- `--device`: 计算设备

预测结果将保存为CSV文件（submission-yolov5x.csv）和JSON文件（yolov5x.json）。

### 5. 多模型结果合并

如果使用了多个不同的模型进行预测，可以使用`result_merge.py`脚本合并结果以提高识别准确率：

```bash
python result_merge.py
```

此脚本会读取多个模型的JSON结果文件，通过NMS算法合并检测框，并按字符位置排序生成最终识别结果。

### 6. 自动化流程

项目提供了完整的自动化流程脚本`run_pipeline.sh`，可一键执行从数据处理到训练再到预测的全流程：

```bash
# 赋予执行权限
chmod +x run_pipeline.sh

# 执行脚本
./run_pipeline.sh
```