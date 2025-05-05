#!/bin/bash

# 街景字符识别项目依赖安装脚本（Conda版本）
# 本脚本会创建conda虚拟环境并安装所有依赖

echo "===== 开始创建conda环境并安装项目依赖 ====="

# 检查conda是否已安装
if ! command -v conda &> /dev/null; then
    echo "错误: Conda 未安装，请先安装Miniconda或Anaconda"
    echo "您可以从以下地址下载安装: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "Conda版本:"
conda --version

# 配置国内镜像源
echo "配置conda国内镜像源..."
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes

# 环境名称
ENV_NAME="yolov5-env"

# 检查是否已存在同名环境
if conda env list | grep -q "$ENV_NAME"; then
    echo "警告: 环境 '$ENV_NAME' 已存在"
    read -p "是否移除现有环境并重新创建? [y/n] (默认: y): " remove_env
    if [[ $remove_env != "n" && $remove_env != "N" ]]; then
        echo "正在移除现有环境..."
        conda env remove -n "$ENV_NAME" -y
    else
        echo "将使用现有环境"
    fi
fi

# 直接创建环境（不依赖environment.yml）
echo "正在创建conda环境..."
conda create -n "$ENV_NAME" python=3.7 -y

# 激活环境
echo "正在激活环境..."
if [[ "$SHELL" == *"zsh"* ]]; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate "$ENV_NAME"
else
    # 默认使用bash初始化
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
fi

# 安装基础依赖
echo "正在安装基础依赖..."
conda install -n "$ENV_NAME" numpy=1.17 pandas cython matplotlib pillow "pyyaml>=5.3" scipy tensorboard -y
# 安装pip依赖
echo "配置pip国内镜像..."
conda run -n "$ENV_NAME" pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

echo "正在安装pip依赖..."
conda run -n "$ENV_NAME" pip install opencv-python tqdm tensorboardX

# 安装PyTorch (CUDA版本)
echo "正在安装PyTorch (CUDA版本)..."
conda install -n "$ENV_NAME" pytorch=1.12.0 torchvision cudatoolkit=11.3 -c pytorch -y

# 验证numpy版本
echo "验证numpy版本..."
conda run -n "$ENV_NAME" python -c "import numpy; print('✓ Numpy版本:', numpy.__version__)"

# 检查PyTorch是否有CUDA支持
echo "检查PyTorch CUDA支持..."
if conda run -n "$ENV_NAME" python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" &> /dev/null; then
    echo "✓ PyTorch已启用CUDA支持"
    conda run -n "$ENV_NAME" python -c "import torch; print('CUDA版本:', torch.version.cuda if torch.cuda.is_available() else '无')"
else
    echo "警告: PyTorch未启用CUDA支持，这可能会影响训练性能"
    echo "系统CUDA信息:"
    nvidia-smi
    echo ""
    echo "尝试使用CPU版本进行训练，添加 --device cpu 参数"
fi

# 下载YOLOv5模型权重
download_weights() {
    if [ ! -d "weights" ] || [ -z "$(ls -A weights)" ]; then
        echo "YOLOv5模型权重未下载或weights目录为空"
        read -p "是否下载YOLOv5预训练权重? [y/n] (默认: y): " download_weights
        if [[ $download_weights != "n" && $download_weights != "N" ]]; then
            mkdir -p weights
            
            # 下载YOLOv5预训练权重（指定国内镜像）
            echo "正在下载YOLOv5预训练权重..."
            wget -P weights https://scoop.201704.xyz/https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5s.pt
            wget -P weights https://scoop.201704.xyz/https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5m.pt
            
            echo "是否下载更大的模型权重? (需要更多存储空间和内存)"
            read -p "下载YOLOv5l和YOLOv5x权重? [y/n] (默认: n): " download_large
            if [[ $download_large == "y" || $download_large == "Y" ]]; then
                wget -P weights https://scoop.201704.xyz/https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5l.pt
                wget -P weights https://scoop.201704.xyz/https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5x.pt
            fi
            echo "预训练权重下载完成"
        fi
    else
        echo "✓ weights目录已存在，跳过下载"
    fi
}

# 下载权重
download_weights

echo "===== 环境创建完成 ====="
echo "要使用此环境，请运行: conda activate $ENV_NAME"
echo "使用CPU训练命令示例: python train.py --batch-size 16 --cfg ./models/yolov5s.yaml --data ./data/coco.yaml --img-size 640 --weights weights/yolov5s.pt --noautoanchor --cache-images --device cpu --epochs 50"
echo "如果CUDA可用，GPU训练命令示例: python train.py --batch-size 32 --cfg ./models/yolov5s.yaml --data ./data/coco.yaml --img-size 640 --weights weights/yolov5s.pt --noautoanchor --cache-images --device 0 --epochs 50"

# 提示用户如何激活环境
cat << EOF

要在新终端中使用此环境，请运行:
-----------------------------------------
conda activate $ENV_NAME
-----------------------------------------

EOF 