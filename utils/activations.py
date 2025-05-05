"""
激活函数库

本文件实现了多种高级激活函数，用于深度学习模型，主要包括：
1. Swish激活函数 - x * sigmoid(x)，谷歌提出的激活函数，性能优于ReLU
2. Mish激活函数 - x * tanh(softplus(x))，比Swish更平滑的激活函数

每种激活函数提供三种实现：
- 内存效率版本：利用自定义autograd函数优化内存使用
- 标准版本：直接使用PyTorch基础操作实现
- 硬件优化版本(仅Swish)：使用分段函数近似，提高计算效率

使用方法：
- 在神经网络模型中直接导入所需的激活函数类
- 作为nn.Module层使用，例如：`nn.Sequential(nn.Conv2d(...), Mish())`

这些激活函数主要用于改善模型性能和收敛特性，特别适用于计算机视觉任务。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn


# Swish 激活函数 ----------------------------------------------------------------
class SwishImplementation(torch.autograd.Function):
    """
    Swish激活函数的自定义autograd实现
    
    实现了自定义的前向和反向传播以优化内存使用
    公式: f(x) = x * sigmoid(x)
    """
    @staticmethod
    def forward(ctx, x):
        """
        前向传播计算
        
        参数:
            ctx: 上下文对象，用于保存张量以在反向传播中使用
            x: 输入张量
            
        返回:
            x * sigmoid(x)
        """
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播计算
        
        参数:
            ctx: 上下文对象，含有前向传播保存的张量
            grad_output: 输出梯度
            
        返回:
            计算的输入梯度
        """
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        return grad_output * (sx * (1 + x * (1 - sx)))


class MemoryEfficientSwish(nn.Module):
    """
    内存优化的Swish激活函数模块
    
    使用自定义的autograd函数实现，减少内存使用
    """
    @staticmethod
    def forward(x):
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            应用Swish激活后的结果
        """
        return SwishImplementation.apply(x)


class HardSwish(nn.Module):
    """
    硬件优化的Swish激活函数
    
    使用分段函数近似Swish，减少计算开销
    来源: https://arxiv.org/pdf/1905.02244.pdf
    """
    @staticmethod
    def forward(x):
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            应用HardSwish激活后的结果: x * hardtanh(x+3, 0, 6)/6
        """
        return x * F.hardtanh(x + 3, 0., 6., True) / 6.


class Swish(nn.Module):
    """
    标准Swish激活函数
    
    直接使用PyTorch操作实现，简单但可能内存消耗较大
    """
    @staticmethod
    def forward(x):
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            应用Swish激活后的结果: x * sigmoid(x)
        """
        return x * torch.sigmoid(x)


# Mish 激活函数 -----------------------------------------------------------------
class MishImplementation(torch.autograd.Function):
    """
    Mish激活函数的自定义autograd实现
    
    实现了自定义的前向和反向传播以优化内存使用
    公式: f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    @staticmethod
    def forward(ctx, x):
        """
        前向传播计算
        
        参数:
            ctx: 上下文对象，用于保存张量以在反向传播中使用
            x: 输入张量
            
        返回:
            x * tanh(ln(1 + exp(x)))
        """
        ctx.save_for_backward(x)
        return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播计算
        
        参数:
            ctx: 上下文对象，含有前向传播保存的张量
            grad_output: 输出梯度
            
        返回:
            计算的输入梯度
        """
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        fx = F.softplus(x).tanh()
        return grad_output * (fx + x * sx * (1 - fx * fx))


class MemoryEfficientMish(nn.Module):
    """
    内存优化的Mish激活函数模块
    
    使用自定义的autograd函数实现，减少内存使用
    """
    @staticmethod
    def forward(x):
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            应用Mish激活后的结果
        """
        return MishImplementation.apply(x)


class Mish(nn.Module):
    """
    标准Mish激活函数
    
    直接使用PyTorch操作实现，来源: https://github.com/digantamisra98/Mish
    """
    @staticmethod
    def forward(x):
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            应用Mish激活后的结果: x * tanh(softplus(x))
        """
        return x * F.softplus(x).tanh()
