"""
YOLOv5模型训练脚本

本脚本用于训练YOLOv5目标检测模型，支持多种高级训练功能。

主要功能：
1. 加载和初始化YOLOv5模型
2. 支持从预训练权重继续训练
3. 支持混合精度训练(Automatic Mixed Precision)提高训练速度
4. 支持分布式训练和多GPU训练
5. 使用余弦学习率调度器动态调整学习率
6. 使用模型指数移动平均(EMA)提高模型稳定性
7. 支持超参数进化(Hyperparameter Evolution)自动优化训练参数
8. 定期在验证集上评估模型性能并保存最佳模型
9. 支持TensorBoard可视化训练过程

使用方法：
    python train.py --data [数据配置文件] --cfg [模型配置文件] --weights [权重文件]

主要参数：
    --epochs: 训练轮数，默认300
    --batch-size: 批处理大小，默认16
    --cfg: 模型配置文件路径(*.yaml)
    --data: 数据集配置文件路径，默认./data/coco128.yaml
    --img-size: 训练和测试图像尺寸，默认[320, 320]
    --weights: 初始权重文件路径，空字符串表示从头训练
    --device: 计算设备，如0(表示第一块GPU)或0,1,2,3(多GPU)或cpu
    --adam: 使用Adam优化器，默认使用SGD
    --multi-scale: 使用多尺度训练(图像大小在±50%范围内变化)
    --single-cls: 将数据集视为单类别数据集
    --resume: 从上次中断处继续训练
    --evolve: 使用遗传算法自动优化超参数
    --cache-images: 缓存图像以加速训练
    --rect: 使用矩形训练，提高小批量训练效率
    --noautoanchor: 禁用自动锚点检查

输出：
    - weights/best.pt: 验证集上性能最佳的模型权重
    - weights/last.pt: 最后一个周期的模型权重
    - results.txt/results.png: 训练结果记录和可视化
"""

import argparse
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from tensorboardX import SummaryWriter

import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils import google_utils
from utils.datasets import *
from utils.utils import *

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    # print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

wdir = 'weights' + os.sep  # weights dir
os.makedirs(wdir, exist_ok=True)
last = wdir + 'last.pt'
best = wdir + 'best.pt'
results_file = 'results.txt'

# 超参数，控制训练过程的各种参数设置
hyp = {'lr0': 0.01,  # 初始学习率 (SGD=1E-2, Adam=1E-3)
       'momentum': 0.937,  # SGD动量
       'weight_decay': 5e-4,  # 优化器权重衰减
       'giou': 0.05,  # GIoU损失增益
       'cls': 0.58,  # 分类损失增益
       'cls_pw': 1.0,  # 分类BCELoss正样本权重
       'obj': 1.0,  # 目标损失增益 (*=img_size/320 如果img_size!=320)
       'obj_pw': 1.0,  # 目标BCELoss正样本权重
       'iou_t': 0.20,  # IoU训练阈值
       'anchor_t': 4.0,  # 锚点-多重阈值
       'fl_gamma': 0.0,  # 聚焦损失gamma (efficientDet默认gamma=1.5)
       'hsv_h': 0.014,  # 图像HSV-色调增强
       'hsv_s': 0.68,  # 图像HSV-饱和度增强
       'hsv_v': 0.36,  # 图像HSV-亮度增强
       'degrees': 0.0,  # 图像旋转角度 (+/- 度数)
       'translate': 0.0,  # 图像平移 (+/- 分数)
       'scale': 0.5,  # 图像缩放 (+/- 增益)
       'shear': 0.0}  # 图像剪切 (+/- 度数)
# print(hyp)

# 从hyp*.txt覆盖超参数（可选）
f = glob.glob('hyp*.txt')
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v

# 如果gamma>0则使用聚焦损失
if hyp['fl_gamma']:
    print('Using FocalLoss(gamma=%g)' % hyp['fl_gamma'])


def train(hyp):
    """
    训练主函数
    
    参数:
        hyp: 训练超参数字典
        
    主要流程:
        1. 配置训练环境和参数
        2. 创建模型和优化器
        3. 加载数据集
        4. 执行训练循环
           - 前向传播计算损失
           - 反向传播更新权重
           - 定期在验证集上评估
           - 保存检查点
        5. 保存最终模型和结果
    """
    epochs = opt.epochs  # 训练总轮数，默认300
    batch_size = opt.batch_size  # 批处理大小，默认16
    weights = opt.weights  # 初始训练权重路径

    # 配置环境
    init_seeds(1)  # 初始化随机种子，确保结果可复现
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # 加载数据集配置
    train_path = data_dict['train']  # 训练集路径
    test_path = data_dict['val']  # 验证集路径
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # 类别数量（单类或多类）

    # 删除先前的结果文件
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
        os.remove(f)

    # 创建模型
    model = Model(opt.cfg).to(device)  # 创建模型并移至指定设备
    assert model.md['nc'] == nc, '%s nc=%g classes but %s nc=%g classes' % (opt.data, nc, opt.cfg, model.md['nc'])
    model.names = data_dict['names']  # 设置类别名称

    # 图像尺寸
    gs = int(max(model.stride))  # 网格大小（最大步长）
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # 验证图像尺寸是否为gs的倍数

    # 优化器设置
    nbs = 64  # 标称批处理大小
    accumulate = max(round(nbs / batch_size), 1)  # 梯度累积步数
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # 缩放权重衰减
    pg0, pg1, pg2 = [], [], []  # 优化器参数分组
    for k, v in model.named_parameters():
        if v.requires_grad:
            if '.bias' in k:
                pg2.append(v)  # 偏置项
            elif '.weight' in k and '.bn' not in k:
                pg1.append(v)  # 应用权重衰减的权重
            else:
                pg0.append(v)  # 其他参数

    # 设置优化器（Adam或SGD）
    optimizer = optim.Adam(pg0, lr=hyp['lr0']) if opt.adam else \
        optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # 添加带权重衰减的参数组
    optimizer.add_param_group({'params': pg2})  # 添加偏置参数组
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # 加载预训练模型（如果指定）
    google_utils.attempt_download(weights)  # 尝试下载权重文件
    start_epoch, best_fitness = 0, 0.0
    if weights.endswith('.pt'):  # pytorch格式
        try:
            # 添加安全全局变量，以支持较新版本的PyTorch
            import numpy as np
            import torch.serialization
            try:
                torch.serialization.add_safe_globals(['numpy.core.multiarray._reconstruct'])
            except (AttributeError, TypeError):
                pass  # 在旧版本PyTorch中忽略此错误
                
            # 尝试加载检查点
            try:
                ckpt = torch.load(weights, map_location=device)  # 首先尝试直接加载
            except Exception as e:
                # 如果直接加载失败，尝试关闭权重仅模式
                try:
                    ckpt = torch.load(weights, map_location=device, weights_only=False)
                except TypeError:  # 如果是旧版本PyTorch (weights_only参数不存在)
                    ckpt = torch.load(weights, map_location=device)

            # 加载模型权重
            try:
                ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
                                 if k in model.state_dict() and model.state_dict()[k].shape == v.shape}  # 过滤与当前模型匹配的权重
                model.load_state_dict(ckpt['model'], strict=False)
            except KeyError as e:
                s = "%s is not compatible with %s. This may be due to model differences or %s may be out of date. " \
                    "Please delete or update %s and try again, or use --weights '' to train from scatch." \
                    % (opt.weights, opt.cfg, opt.weights, opt.weights)
                raise KeyError(s) from e

            # 加载优化器状态
            if ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
                best_fitness = ckpt['best_fitness']

            # 加载之前的训练结果
            if ckpt.get('training_results') is not None:
                with open(results_file, 'w') as file:
                    file.write(ckpt['training_results'])  # 写入results.txt

            start_epoch = ckpt['epoch'] + 1  # 从上次训练的下一个周期开始
            del ckpt

        except Exception as e:
            print('WARNING: Error loading %s: %s' % (weights, e))

    # 混合精度训练
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # 学习率调度器 - 余弦退火策略
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.9 + 0.1  # 余弦退火公式
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1  # 从正确的周期恢复
    # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # 初始化分布式训练（如果可用）
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend='nccl',  # 分布式后端
                                init_method='tcp://127.0.0.1:9999',  # 初始化方法
                                world_size=1,  # 节点数量
                                rank=0)  # 节点排名
        model = torch.nn.parallel.DistributedDataParallel(model)
        # pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html

    # 创建训练数据加载器
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect)
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # 最大标签类别
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Correct your labels or your model.' % (mlc, nc, opt.cfg)

    # 创建验证数据加载器
    testloader = create_dataloader(test_path, imgsz_test, batch_size, gs, opt,
                                   hyp=hyp, augment=False, cache=opt.cache_images, rect=True)[0]

    # 设置模型参数
    hyp['cls'] *= nc / 80.  # 根据当前数据集类别数调整分类损失增益
    model.nc = nc  # 设置模型的类别数
    model.hyp = hyp  # 附加超参数
    model.gr = 1.0  # GIoU损失比率(obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # 附加类别权重

    # 统计类别频率
    labels = np.concatenate(dataset.labels, 0)
    c = torch.tensor(labels[:, 0])  # 类别
    # cf = torch.bincount(c.long(), minlength=nc) + 1.
    # model._initialize_biases(cf.to(device))
    if tb_writer:
        plot_labels(labels)  # 绘制标签分布
        tb_writer.add_histogram('classes', c, 0)  # 添加类别直方图到TensorBoard

    # 检查锚点配置是否合适
    if not opt.noautoanchor:
        check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    # 创建模型指数移动平均(EMA)
    ema = torch_utils.ModelEMA(model)

    # 开始训练
    t0 = time.time()
    nb = len(dataloader)  # 批次数量
    n_burn = max(3 * nb, 1e3)  # 预热迭代次数，最少3个周期或1000次迭代
    maps = np.zeros(nc)  # 每个类别的mAP
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    print('Image sizes %g train, %g test' % (imgsz, imgsz_test))
    print('Using %g dataloader workers' % dataloader.num_workers)
    print('Starting training for %g epochs...' % epochs)
    # torch.autograd.set_detect_anomaly(True)
    
    # 训练循环 - 迭代每个周期
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()  # 设置为训练模式

        # 更新图像权重（可选）
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # 类别权重
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # 加权随机采样

        # 更新马赛克增强边界
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # 高度、宽度边界

        mloss = torch.zeros(4, device=device)  # 平均损失
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(dataloader), total=nb)  # 进度条
        
        # 批次处理循环
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # 积累的批次数（自训练开始）
            imgs = imgs.to(device).float() / 255.0  # uint8转为float32，值域从0-255变为0.0-1.0

            # 预热阶段
            if ni <= n_burn:
                xi = [0, n_burn]  # x插值
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())  # 梯度累积步数
                for j, x in enumerate(optimizer.param_groups):
                    # 偏置学习率从0.1降至lr0，其他从0.0升至lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

            # 多尺度训练
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # 随机选择尺寸
                sf = sz / max(imgs.shape[2:])  # 缩放因子
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # 新尺寸（伸展至gs倍数）
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)  # 图像缩放

            # 前向传播
            pred = model(imgs)

            # 计算损失
            loss, loss_items = compute_loss(pred, targets.to(device), model)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # 反向传播
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # 优化
            if ni % accumulate == 0:
                optimizer.step()  # 更新参数
                optimizer.zero_grad()  # 清零梯度
                ema.update(model)  # 更新EMA参数

            # 输出训练信息
            mloss = (mloss * i + loss_items) / (i + 1)  # 更新平均损失
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # 内存占用(GB)
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)  # 设置进度条描述

            # 绘制训练批次图像
            if ni < 3:
                f = 'train_batch%g.jpg' % ni  # 文件名
                result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                if tb_writer and result is not None:
                    tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    # tb_writer.add_graph(model, imgs)  # 添加模型到TensorBoard

            # 批次处理结束 ------------------------------------------------------------------------------------------------

        # 学习率调度更新
        scheduler.step()

        # 模型评估 - 计算mAP
        ema.update_attr(model)  # 更新EMA属性
        final_epoch = epoch + 1 == epochs
        if not opt.notest or final_epoch:  # 计算mAP
            results, maps, times = test.test(opt.data,
                                             batch_size=batch_size,
                                             imgsz=imgsz_test,
                                             save_json=final_epoch and opt.data.endswith(os.sep + 'coco.yaml'),
                                             model=ema.ema,
                                             single_cls=opt.single_cls,
                                             dataloader=testloader)

        # 写入结果
        with open(results_file, 'a') as f:
            f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        if len(opt.name) and opt.bucket:
            os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (opt.bucket, opt.name))

        # 写入TensorBoard
        if tb_writer:
            tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/F1',
                    'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
            for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                tb_writer.add_scalar(tag, x, epoch)

        # 更新最佳mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = [P, R, mAP, F1]的加权组合
        if fi > best_fitness:
            best_fitness = fi

        # 保存模型
        save = (not opt.nosave) or (final_epoch and not opt.evolve)
        if save:
            with open(results_file, 'r') as f:  # 创建检查点
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': f.read(),
                        'model': ema.ema.module if hasattr(model, 'module') else ema.ema,
                        'optimizer': None if final_epoch else optimizer.state_dict()}

            # 保存最后一个、最好的和删除
            torch.save(ckpt, last)
            if (best_fitness == fi) and not final_epoch:
                torch.save(ckpt, best)
            del ckpt

        # 周期结束 ----------------------------------------------------------------------------------------------------
    # 训练结束

    # 重命名结果文件（如果指定了名称）
    n = opt.name
    if len(n):
        n = '_' + n if not n.isnumeric() else n
        fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
        for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)  # 重命名
                ispt = f2.endswith('.pt')  # 是否为*.pt文件
                strip_optimizer(f2) if ispt else None  # 剥离优化器
                os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # 上传

    # 绘制结果
    if not opt.evolve:
        plot_results()  # 保存为results.png
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    
    # 释放资源
    dist.destroy_process_group() if device.type != 'cpu' and torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    #check_git_status()  # 检查git状态，建议git pull更新
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='总批处理大小')
    parser.add_argument('--cfg', type=str, default='', help='模型配置文件路径*.yaml')
    parser.add_argument('--data', type=str, default='./data/coco128.yaml', help='数据集配置文件路径*.yaml')
    parser.add_argument('--img-size', nargs='+', type=int, default=[320, 320], help='训练和测试图像大小')
    parser.add_argument('--rect', action='store_true', help='矩形训练')
    parser.add_argument('--resume', action='store_true', help='恢复最近训练')
    parser.add_argument('--nosave', action='store_true', help='只保存最终检查点')
    parser.add_argument('--notest', action='store_true', help='只测试最后一轮')
    parser.add_argument('--noautoanchor', action='store_true', help='禁用自动锚点检查')
    parser.add_argument('--evolve', action='store_true', help='进化超参数')
    parser.add_argument('--bucket', type=str, default='', help='gsutil桶')
    parser.add_argument('--cache-images', action='store_true', help='缓存图像以加快训练')
    parser.add_argument('--weights', type=str, default='', help='初始权重路径')
    parser.add_argument('--name', default='', help='重命名results.txt为results_name.txt')
    parser.add_argument('--device', default='', help='cuda设备，例如0或0,1,2,3或cpu')
    parser.add_argument('--adam', action='store_true', help='使用adam优化器')
    parser.add_argument('--multi-scale', action='store_true', help='图像大小变化±50%')
    parser.add_argument('--single-cls', action='store_true', help='将数据集作为单类别训练')
    opt = parser.parse_args()
    opt.weights = last if opt.resume else opt.weights
    opt.cfg = check_file(opt.cfg)  # 检查文件
    opt.data = check_file(opt.data)  # 检查文件
    print(opt)
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # 扩展到2个尺寸(训练，测试)
    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    if device.type == 'cpu':
        mixed_precision = False

    # 训练
    if not opt.evolve:  # 正常训练
        tb_writer = SummaryWriter(comment=opt.name)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        train(hyp)

    # 超参数进化（可选）
    else:
        tb_writer = None
        opt.notest, opt.nosave = True, True  # 只测试/保存最终周期
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # 如果存在则下载evolve.txt

        for _ in range(10):  # 进化代数
            if os.path.exists('evolve.txt'):  # 如果evolve.txt存在：选择最佳超参数并突变
                # 选择父代
                parent = 'single'  # 父代选择方法: 'single'或'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # 考虑前n个结果
                x = x[np.argsort(-fitness(x))][:n]  # 前n个突变
                w = fitness(x) - fitness(x).min()  # 权重
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # 随机选择
                    x = x[random.choices(range(n), weights=w)[0]]  # 加权选择
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # 加权组合

                # 突变
                mp, s = 0.9, 0.2  # 突变概率，sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 0, 1, 1, 1, 1, 1, 1, 1])  # 增益
                ng = len(g)
                v = np.ones(ng)
                while all(v == 1):  # 突变直到发生变化（防止重复）
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = x[i + 7] * v[i]  # 突变

            # 裁剪到限制范围
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # 训练突变
            results = train(hyp.copy())

            # 写入突变结果
            print_mutation(hyp, results, opt.bucket)

            # 绘制结果
            # plot_evolution_results(hyp)
