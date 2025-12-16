"""
梯度保持的场融合方法
GPU 加速实现
"""

from .gradient_fusion import GradientPreservingFusion
from .utils import compute_gradient_variance, compute_mesh_metrics

__all__ = [
    'GradientPreservingFusion',
    'compute_gradient_variance',
    'compute_mesh_metrics'
]

__version__ = '1.0.0'
