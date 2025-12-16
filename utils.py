"""
工具函数模块
"""

import torch
import numpy as np
from typing import Tuple, Dict

def compute_gradient_variance(
    field: torch.Tensor,
    edge_index: torch.Tensor
) -> float:
    """
    计算场的梯度方差（衡量空间连续性）
    
    梯度方差越小，场越平滑
    
    Args:
        field: [N] 场值
        edge_index: [2, E] 边索引
        
    Returns:
        gradient_variance: 梯度方差
    """
    src, dst = edge_index[0], edge_index[1]
    gradients = field[dst] - field[src]
    return torch.var(gradients).item()

def compute_mesh_metrics(
    field_original: torch.Tensor,
    field_fused: torch.Tensor,
    sensor_indices: torch.Tensor,
    sensor_values: torch.Tensor,
    edge_index: torch.Tensor
) -> Dict:
    """
    计算融合前后的质量指标
    
    Returns:
        metrics: 指标字典
    """
    metrics = {}
    
    # 1. 传感器拟合误差
    sensor_error_original = torch.abs(field_original[sensor_indices] - sensor_values)
    sensor_error_fused = torch.abs(field_fused[sensor_indices] - sensor_values)
    
    metrics['sensor_fit'] = {
        'original_max_error': sensor_error_original.max().item(),
        'original_mean_error': sensor_error_original.mean().item(),
        'fused_max_error': sensor_error_fused.max().item(),
        'fused_mean_error': sensor_error_fused.mean().item(),
        'improvement': (sensor_error_original.mean() - sensor_error_fused.mean()).item()
    }
    
    # 2. 梯度连续性
    grad_var_original = compute_gradient_variance(field_original, edge_index)
    grad_var_fused = compute_gradient_variance(field_fused, edge_index)
    
    metrics['continuity'] = {
        'original_gradient_var': grad_var_original,
        'fused_gradient_var': grad_var_fused,
        'smoothness_improvement': (grad_var_original - grad_var_fused) / grad_var_original * 100
    }
    
    # 3. 场值统计
    metrics['field_stats'] = {
        'original': {
            'min': field_original.min().item(),
            'max': field_original.max().item(),
            'mean': field_original.mean().item(),
            'std': field_original.std().item()
        },
        'fused': {
            'min': field_fused.min().item(),
            'max': field_fused.max().item(),
            'mean': field_fused.mean().item(),
            'std': field_fused.std().item()
        }
    }
    
    return metrics

def print_metrics(metrics: Dict):
    """打印格式化的指标"""
    print(f"\n{'='*70}")
    print(f"📊 质量指标对比")
    print(f"{'='*70}")
    
    print(f"\n1️⃣  传感器拟合:")
    sf = metrics['sensor_fit']
    print(f"   原始场最大误差: {sf['original_max_error']:.4e}")
    print(f"   融合场最大误差: {sf['fused_max_error']:.4e}")
    print(f"   原始场平均误差: {sf['original_mean_error']:.4e}")
    print(f"   融合场平均误差: {sf['fused_mean_error']:.4e}")
    print(f"   改善: {sf['improvement']:.4e}")
    
    print(f"\n2️⃣  空间连续性:")
    cont = metrics['continuity']
    print(f"   原始场梯度方差: {cont['original_gradient_var']:.4e}")
    print(f"   融合场梯度方差: {cont['fused_gradient_var']:.4e}")
    print(f"   平滑度提升: {cont['smoothness_improvement']:.2f}%")
    
    print(f"\n3️⃣  场值范围:")
    orig = metrics['field_stats']['original']
    fused = metrics['field_stats']['fused']
    print(f"   原始场: [{orig['min']:.4e}, {orig['max']:.4e}], "
          f"均值={orig['mean']:.4e}, 标准差={orig['std']:.4e}")
    print(f"   融合场: [{fused['min']:.4e}, {fused['max']:.4e}], "
          f"均值={fused['mean']:.4e}, 标准差={fused['std']:.4e}")
    
    print(f"\n{'='*70}\n")
