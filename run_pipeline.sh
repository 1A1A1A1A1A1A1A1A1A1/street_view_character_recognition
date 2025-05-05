#!/bin/bash

# 街景字符识别项目自动化流程脚本
# 适用于RTX 3060 12GB显存、28核CPU、32GB内存的系统配置

# 全局变量
train_img_path="coco/images/train/mchar_train/"
val_img_path="coco/images/val/mchar_val/"
train_anno_path="coco/labels/train/mchar_train.json"
val_anno_path="coco/labels/val/mchar_val.json"
test_path="coco/images/test/mchar_test_a/"
model_cfg="yolov5m"
batch_size=32
img_size=640
epochs=50
workers=12
cache_option="--cache-images"
conf_thres=0.4
iou_thres=0.5

# 函数: 步骤2 - 合并训练集和验证集图像
merge_images() {
    echo -e "\n===== 步骤2: 合并训练集和验证集图像 ====="
    read -p "请确认训练集图像路径 (默认: $train_img_path): " input_train_img_path
    train_img_path=${input_train_img_path:-$train_img_path}

    read -p "请确认验证集图像路径 (默认: $val_img_path): " input_val_img_path
    val_img_path=${input_val_img_path:-$val_img_path}

    echo "正在合并图像..."
    python image_merge.py --train_image_path $train_img_path \
                          --val_image_path $val_img_path \
                          --dst_image_path ./coco/images/

    if [ $? -ne 0 ]; then
        echo "图像合并失败，请检查路径和脚本"
        return 1
    fi
    
    echo "图像合并完成"
    echo "===== 步骤2完成: 合并训练集和验证集图像 ====="
}

# 函数: 步骤3 - 转换标注格式
convert_annotations() {
    echo -e "\n===== 步骤3: 转换标注格式 ====="
    read -p "请确认训练集标注路径 (默认: $train_anno_path): " input_train_anno_path
    train_anno_path=${input_train_anno_path:-$train_anno_path}

    read -p "请确认验证集标注路径 (默认: $val_anno_path): " input_val_anno_path
    val_anno_path=${input_val_anno_path:-$val_anno_path}

    echo "正在转换标注..."
    python make_label.py --train_image_path $train_img_path \
                         --val_image_path $val_img_path \
                         --train_annotation_path $train_anno_path \
                         --val_annotation_path $val_anno_path \
                         --label_path ./coco/labels/

    if [ $? -ne 0 ]; then
        echo "标注转换失败，请检查路径和脚本"
        return 1
    fi
    
    echo "标注转换完成"
    echo "===== 步骤3完成: 转换标注格式 ====="
}

# 函数: 步骤4 - 创建数据配置文件
create_data_config() {
    echo -e "\n===== 步骤4: 创建数据配置文件 ====="
    echo "正在创建 data/coco.yaml..."

    cat > data/coco.yaml << EOF
train: ./coco/images/
val: ./coco/images/

# 类别数量
nc: 10

# 类别名称
names: ['0','1','2','3','4','5','6','7','8','9']
EOF

    echo "数据配置文件创建完成"
    echo "===== 步骤4完成: 创建数据配置文件 ====="
}

# 函数: 步骤5 - 训练模型
train_model() {
    echo -e "\n===== 步骤5: 训练模型 ====="
    read -p "选择模型大小 [s/m/l/x] (默认: m): " input_model_size
    local model_size=${input_model_size:-"m"}

    case $model_size in
        s) model_cfg="yolov5s" ;;
        m) model_cfg="yolov5m" ;;
        l) model_cfg="yolov5l" ;;
        x) model_cfg="yolov5x" ;;
        *) 
            echo "无效的模型选择，使用默认值 'yolov5m'"
            model_cfg="yolov5m" 
            ;;
    esac

    read -p "设置批处理大小 (默认: $batch_size): " input_batch_size
    batch_size=${input_batch_size:-$batch_size}

    read -p "设置图像尺寸 (默认: $img_size): " input_img_size
    img_size=${input_img_size:-$img_size}

    read -p "设置训练轮数 (默认: $epochs): " input_epochs
    epochs=${input_epochs:-$epochs}

    read -p "设置工作进程数 (默认: $workers): " input_workers
    workers=${input_workers:-$workers}

    read -p "是否使用缓存图像加速训练? [y/n] (默认: y): " use_cache
    if [[ $use_cache == "n" || $use_cache == "N" ]]; then
        cache_option=""
    else
        cache_option="--cache-images"
    fi

    echo "正在开始训练模型..."
    echo "模型: $model_cfg.yaml"
    echo "批处理大小: $batch_size"
    echo "图像尺寸: $img_size"
    echo "训练轮数: $epochs"
    echo "工作进程数: $workers"
    echo "缓存图像: $cache_option"

    python train.py --data ./data/coco.yaml \
                    --cfg ./models/${model_cfg}.yaml \
                    --weights weights/${model_cfg}.pt \
                    --batch-size $batch_size \
                    --img-size $img_size \
                    --epochs $epochs \
                    --device 0 \
                    --noautoanchor \
                    $cache_option

    if [ $? -ne 0 ]; then
        echo "模型训练失败，请检查配置和脚本"
        return 1
    fi
    
    echo "模型训练完成"
    echo "===== 步骤5完成: 训练模型 ====="
}

# 函数: 步骤7 - 测试集推理
run_inference() {
    echo -e "\n===== 步骤6: 测试集推理 ====="
    read -p "请确认测试集路径 (默认: $test_path): " input_test_path
    test_path=${input_test_path:-$test_path}

    read -p "设置置信度阈值 (默认: $conf_thres): " input_conf_thres
    conf_thres=${input_conf_thres:-$conf_thres}

    read -p "设置IoU阈值 (默认: $iou_thres): " input_iou_thres
    iou_thres=${input_iou_thres:-$iou_thres}

    read -p "设置图像尺寸 (默认: $img_size): " input_img_size
    img_size=${input_img_size:-$img_size}

    echo "正在进行测试集推理..."
    python detect.py --source $test_path \
                     --weights weights/best.pt \
                     --img-size $img_size \
                     --conf-thres $conf_thres \
                     --iou-thres $iou_thres \
                     --device 0

    if [ $? -ne 0 ]; then
        echo "测试集推理失败，请检查配置和脚本"
        return 1
    fi
    
    echo "测试集推理完成"
    echo "===== 步骤6完成: 测试集推理 ====="
}

# 函数: 步骤8 - 多模型结果合并
merge_results() {
    echo -e "\n===== 步骤7: 多模型结果合并 ====="
    echo "注意: 确保已使用不同模型进行了推理并生成了各自的结果文件"
        
    echo "正在合并模型结果..."
    python result_merge.py
    
    if [ $? -ne 0 ]; then
        echo "结果合并失败，请检查result_merge.py脚本"
        return 1
    fi
    
    echo "结果合并完成"
    echo "===== 步骤7完成: 多模型结果合并 ====="
}

# 函数: 执行所有步骤
run_all_steps() {
    merge_images
    convert_annotations
    create_data_config
    train_model
    run_inference
    merge_results
    
    echo -e "\n===== 全部步骤完成 ====="
    echo "整个街景字符识别项目流程已成功执行！"
}

# 显示菜单并处理用户选择
show_menu() {
    echo "===== 街景字符识别项目自动化流程 ====="
    echo "请选择要执行的步骤:"
    echo "0. 执行所有步骤"
    echo "1. 合并训练集和验证集图像"
    echo "2. 将JSON标注转换为YOLO格式"
    echo "3. 创建数据配置文件"
    echo "4. 训练模型"
    echo "5. 在测试集上进行推理"
    echo "6. 多模型结果合并"
    echo "7. 退出"
    
    read -p "请输入选项 (0-7): " choice
    
    case $choice in
        0) run_all_steps ;;
        1) merge_images ;;
        2) convert_annotations ;;
        3) create_data_config ;;
        4) train_model ;;
        5) run_inference ;;
        6) merge_results ;;
        7) 
            echo "退出脚本"
            exit 0
            ;;
        *)
            echo "无效选项，请重新选择"
            show_menu
            ;;
    esac
    
    # 询问是否继续执行其他步骤
    read -p "是否继续执行其他步骤? (y/n): " continue_choice
    if [[ $continue_choice == "y" || $continue_choice == "Y" ]]; then
        show_menu
    else
        echo "脚本执行结束"
        exit 0
    fi
}

# 脚本入口
show_menu 