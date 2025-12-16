"""
梯度保持的场融合算法 - GPU 高性能实现
预期性能：19万节点 < 5秒（GPU）/ < 30秒（CPU）
日期：2025-12-08
"""

import torch
import torch.sparse as sparse
from typing import Tuple, Optional, Dict
import time
import numpy as np

class GradientPreservingFusion:
    """
    梯度保持的数据融合类
    
    特点：
    1. GPU 加速的稀疏矩阵运算
    2. 预条件共轭梯度法快速求解
    3. 预计算和缓存优化
    4. 自适应收敛控制
    
    算法原理：
    最小化能量泛函：
        E(x) = ||x_sensors - y_measured||²     # 数据拟合项
             + λ₁ ||L·x||²                     # 拉普拉斯平滑项
             + λ₂ ||∇x - ∇x_CAE||²             # 梯度保持项
    """
    
    def __init__(
        self, 
        edge_index: torch.Tensor, 
        num_nodes: int,
        node_coords: torch.Tensor = None,
        device: str = 'cuda',
        use_double_precision: bool = False
    ):
        """
        初始化融合器（预计算图结构相关矩阵）
        
        Args:
            edge_index: [2, E] 边索引（无向图）
            num_nodes: 节点数 N
            node_coords: [N, 3] 节点坐标（用于距离归一化梯度）
            device: 'cuda' 或 'cpu'
            use_double_precision: 是否使用双精度（提高精度但降低速度）
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_nodes = num_nodes
        self.edge_index = edge_index.to(self.device)
        self.num_edges = edge_index.shape[1]
        self.dtype = torch.float64 if use_double_precision else torch.float32
        
        # 节点坐标（用于梯度距离归一化）
        if node_coords is not None:
            self.node_coords = node_coords.to(self.device).to(self.dtype)
        else:
            self.node_coords = None
        
        print(f"\n{'='*70}")
        print(f"🚀 初始化梯度保持融合器")
        print(f"{'='*70}")
        print(f"📊 图规模:")
        print(f"   - 节点数: {num_nodes:,}")
        print(f"   - 边数: {self.num_edges:,}")
        print(f"   - 平均度: {2*self.num_edges/num_nodes:.1f}")
        print(f"⚙️  计算设置:")
        print(f"   - 设备: {self.device}")
        print(f"   - 精度: {'双精度 (float64)' if use_double_precision else '单精度 (float32)'}")
        
        init_start = time.time()
        
        # 预计算图拉普拉斯矩阵（只需一次）
        self._build_laplacian()
        
        # 预计算梯度算子（只需一次）
        self._build_gradient_operator()
        
        init_time = time.time() - init_start
        
        print(f"\n✅ 初始化完成！耗时: {init_time:.2f}秒")
        print(f"{'='*70}\n")
    
    def _build_laplacian(self):
        """构建加权图拉普拉斯矩阵 L [N×N]（稀疏），权重为 w_ij = 1/d_ij²"""
        start_time = time.time()
        print(f"\n🔧 构建图拉普拉斯矩阵...")
        
        src, dst = self.edge_index[0], self.edge_index[1]
        
        # 计算边权重（如果提供了坐标）
        if self.node_coords is not None:
            edge_lengths = torch.norm(
                self.node_coords[src] - self.node_coords[dst],
                dim=1
            )  # [E]
            edge_lengths = torch.clamp(edge_lengths, min=1e-10)
            edge_weights = 1.0 / (edge_lengths ** 2)  # w_ij = 1/d_ij²
            print(f"   📐 使用距离权重 w_ij = 1/d_ij²:")
            print(f"      - 权重范围: [{edge_weights.min().item():.4e}, {edge_weights.max().item():.4e}]")
        else:
            # 未提供坐标，使用单位权重
            edge_weights = torch.ones(self.num_edges, device=self.device, dtype=self.dtype)
            print(f"   ⚠️  未提供节点坐标，使用单位权重")
        
        # 计算加权度数: D_ii = Σ w_ij
        degree = torch.zeros(self.num_nodes, device=self.device, dtype=self.dtype)
        degree.index_add_(0, src, edge_weights)
        degree.index_add_(0, dst, edge_weights)
        
        # 构建稀疏拉普拉斯矩阵: L = D - W
        
        # 对角线：加权度数
        indices_diag = torch.arange(self.num_nodes, device=self.device).unsqueeze(0).repeat(2, 1)
        values_diag = degree
        
        # 非对角线：-w_ij（加权邻接）
        indices_offdiag = torch.cat([
            self.edge_index,
            torch.stack([dst, src], dim=0)  # 对称边
        ], dim=1)
        values_offdiag = -torch.cat([edge_weights, edge_weights])  # 对称权重
        
        # 合并
        indices = torch.cat([indices_diag, indices_offdiag], dim=1)
        values = torch.cat([values_diag, values_offdiag])
        
        # 创建稀疏张量
        self.L = torch.sparse_coo_tensor(
            indices, values, 
            (self.num_nodes, self.num_nodes),
            device=self.device,
            dtype=self.dtype
        ).coalesce()
        
        nnz = self.L._nnz()
        sparsity = 100 * (1 - nnz / (self.num_nodes ** 2))
        
        print(f"   ✓ 拉普拉斯矩阵: {self.num_nodes}×{self.num_nodes}, "
              f"非零元: {nnz:,}, 稀疏度: {sparsity:.2f}%")
        print(f"   ✓ 耗时: {time.time()-start_time:.3f}秒")
    
    def _build_gradient_operator(self):
        """构建梯度算子 G [E×N]（稀疏），带距离归一化"""
        start_time = time.time()
        print(f"\n🔧 构建梯度算子矩阵...")
        
        src, dst = self.edge_index[0], self.edge_index[1]
        edge_ids = torch.arange(self.num_edges, device=self.device)
        
        # 计算边的欧氏距离（如果提供了坐标）
        if self.node_coords is not None:
            edge_lengths = torch.norm(
                self.node_coords[src] - self.node_coords[dst],
                dim=1
            )  # [E]
            # 避免除零
            edge_lengths = torch.clamp(edge_lengths, min=1e-10)
            weights = 1.0 / edge_lengths
            print(f"   📐 使用距离归一化:")
            print(f"      - 平均边长: {edge_lengths.mean().item():.4e}")
            print(f"      - 边长范围: [{edge_lengths.min().item():.4e}, {edge_lengths.max().item():.4e}]")
        else:
            # 未提供坐标，使用单位权重（简化版本）
            weights = torch.ones(self.num_edges, device=self.device, dtype=self.dtype)
            print(f"   ⚠️  未提供节点坐标，使用简化版本（无距离归一化）")
        
        # G[e, src[e]] = -weights[e], G[e, dst[e]] = +weights[e]
        indices_src = torch.stack([edge_ids, src], dim=0)
        indices_dst = torch.stack([edge_ids, dst], dim=0)
        
        indices = torch.cat([indices_src, indices_dst], dim=1)
        values = torch.cat([
            -weights,  # 起点：负权重
            weights    # 终点：正权重
        ])
        
        self.G = torch.sparse_coo_tensor(
            indices, values,
            (self.num_edges, self.num_nodes),
            device=self.device,
            dtype=self.dtype
        ).coalesce()
        
        nnz = self.G._nnz()
        sparsity = 100 * (1 - nnz / (self.num_edges * self.num_nodes))
        
        print(f"   ✓ 梯度算子: {self.num_edges}×{self.num_nodes}, "
              f"非零元: {nnz:,}, 稀疏度: {sparsity:.2f}%")
        print(f"   ✓ 耗时: {time.time()-start_time:.3f}秒")
    
    def fuse(
        self,
        x_cae: torch.Tensor,           # [N] CAE 场
        sensor_indices: torch.Tensor,  # [M] 传感器索引
        sensor_values: torch.Tensor,   # [M] 实测值
        lambda_smooth: float = 0.1,    # 平滑强度
        lambda_grad: float = 1.0,      # 梯度保持强度
        max_iter: int = 1000,          # 最大迭代次数
        tol: float = 1e-7,             # 收敛容差
        verbose: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        执行梯度保持的场融合
        
        Args:
            x_cae: [N] CAE 场值
            sensor_indices: [M] 传感器节点索引
            sensor_values: [M] 实测值
            lambda_smooth: 拉普拉斯平滑权重（越大越平滑）
            lambda_grad: 梯度保持权重（越大越接近 CAE 梯度）
            max_iter: PCG 最大迭代次数
            tol: 收敛容差（残差范数）
            verbose: 是否打印详细信息
            
        Returns:
            x_fused: [N] 融合后的场
            info: 求解信息字典
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"🔄 开始梯度保持融合")
            print(f"{'='*70}")
            print(f"📌 传感器数量: {len(sensor_indices)}")
            print(f"⚙️  参数设置:")
            print(f"   - λ_smooth (平滑): {lambda_smooth}")
            print(f"   - λ_grad (梯度保持): {lambda_grad}")
            print(f"   - 最大迭代: {max_iter}")
            print(f"   - 收敛容差: {tol:.1e}")
        
        total_start = time.time()
        
        # 转移到 GPU 并统一数据类型
        x_cae = x_cae.to(self.device).to(self.dtype)
        if x_cae.dim() > 1:
            x_cae = x_cae.squeeze()
        
        sensor_indices = sensor_indices.to(self.device).long()
        sensor_values = sensor_values.to(self.device).to(self.dtype)
        if sensor_values.dim() > 1:
            sensor_values = sensor_values.squeeze()
        
        # 1. 计算 CAE 梯度场
        if verbose:
            print(f"\n📐 计算 CAE 梯度场...")
        grad_start = time.time()
        grad_cae = torch.sparse.mm(self.G, x_cae.unsqueeze(1)).squeeze()  # [E]
        if verbose:
            print(f"   ✓ 梯度计算完成，耗时: {time.time()-grad_start:.3f}秒")
            print(f"   ✓ CAE 梯度统计: min={grad_cae.min():.4e}, "
                  f"max={grad_cae.max():.4e}, mean={grad_cae.mean():.4e}")
        
        # 2. 使用预条件共轭梯度法求解
        if verbose:
            print(f"\n⚙️  求解线性系统（预条件共轭梯度法）...")
        
        x_fused, info = self._solve_pcg(
            x_cae=x_cae,
            sensor_indices=sensor_indices,
            sensor_values=sensor_values,
            grad_cae=grad_cae,
            lambda_smooth=lambda_smooth,
            lambda_grad=lambda_grad,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose
        )
        
        # 3. 计算融合后的质量指标
        # 传感器拟合误差（无论verbose与否都需要计算，因为要返回给info）
        sensor_error = torch.abs(x_fused[sensor_indices] - sensor_values)
        
        if verbose:
            print(f"\n📊 融合质量评估:")
            print(f"   ✓ 传感器拟合:")
            print(f"      - 最大误差: {sensor_error.max():.4e}")
            print(f"      - 平均误差: {sensor_error.mean():.4e}")
            print(f"      - RMS 误差: {torch.sqrt((sensor_error**2).mean()):.4e}")
            
            # 梯度保持度
            grad_fused = torch.sparse.mm(self.G, x_fused.unsqueeze(1)).squeeze()
            grad_diff = grad_fused - grad_cae
            print(f"   ✓ 梯度保持:")
            print(f"      - 梯度差异 RMS: {torch.sqrt((grad_diff**2).mean()):.4e}")
            print(f"      - 相对梯度差: {(grad_diff.abs().mean() / grad_cae.abs().mean() * 100):.2f}%")
            
            # 场值范围
            print(f"   ✓ 融合场统计:")
            print(f"      - 最小值: {x_fused.min():.4e}")
            print(f"      - 最大值: {x_fused.max():.4e}")
            print(f"      - 平均值: {x_fused.mean():.4e}")
        
        total_time = time.time() - total_start
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"✅ 融合完成！总耗时: {total_time:.2f}秒")
            print(f"{'='*70}\n")
        
        info['total_time'] = total_time
        info['sensor_error'] = sensor_error.cpu().numpy()
        info['lambda_smooth'] = lambda_smooth
        info['lambda_grad'] = lambda_grad
        
        return x_fused, info
    
    def _apply_A(self, x, sensor_indices, lambda_smooth, lambda_grad):
        """
        高效计算 A·x（无需显式构建 A）
        
        A = S^T·S + λ₁·L^T·L + λ₂·G^T·G
        
        其中 S 是传感器选择矩阵
        """
        # S^T·S·x: 传感器约束项
        result = torch.zeros_like(x)
        result[sensor_indices] = x[sensor_indices]
        
        # λ₁·L^T·L·x = λ₁·L·(L·x)
        if lambda_smooth > 0:
            Lx = torch.sparse.mm(self.L, x.unsqueeze(1)).squeeze()
            LtLx = torch.sparse.mm(self.L.t(), Lx.unsqueeze(1)).squeeze()
            result = result + lambda_smooth * LtLx
        
        # λ₂·G^T·G·x = λ₂·G^T·(G·x)
        if lambda_grad > 0:
            Gx = torch.sparse.mm(self.G, x.unsqueeze(1)).squeeze()
            GtGx = torch.sparse.mm(self.G.t(), Gx.unsqueeze(1)).squeeze()
            result = result + lambda_grad * GtGx
        
        return result
    
    def _solve_pcg(
        self, x_cae, sensor_indices, sensor_values, grad_cae,
        lambda_smooth, lambda_grad, max_iter, tol, verbose
    ):
        """
        预条件共轭梯度法（Preconditioned Conjugate Gradient）
        
        求解: A·x = b
        其中:
            A = S^T·S + λ₁·L^T·L + λ₂·G^T·G
            b = S^T·y + λ₂·G^T·(G·x_cae)
        """
        # 构建右端项 b
        b = torch.zeros_like(x_cae)
        b[sensor_indices] = sensor_values
        
        if lambda_grad > 0:
            # b += λ₂·G^T·(G·x_cae) = λ₂·G^T·grad_cae
            Gt_grad_cae = torch.sparse.mm(self.G.t(), grad_cae.unsqueeze(1)).squeeze()
            b = b + lambda_grad * Gt_grad_cae
        
        # 初始化（用 CAE 作为初值，加速收敛）
        x = x_cae.clone()
        
        # 计算初始残差 r = b - A·x
        Ax = self._apply_A(x, sensor_indices, lambda_smooth, lambda_grad)
        r = b - Ax
        
        # 对角预条件子 M = diag(A) 的逆
        # 近似计算对角线元素
        diag_A = torch.ones_like(x_cae)
        diag_A[sensor_indices] = diag_A[sensor_indices] + 1.0  # S^T·S 贡献
        
        # 如果需要平滑或梯度约束，计算节点度数
        if lambda_smooth > 0 or lambda_grad > 0:
            degree = torch.zeros_like(x_cae)
            src, dst = self.edge_index[0], self.edge_index[1]
            degree.index_add_(0, src, torch.ones(self.num_edges, device=self.device, dtype=self.dtype))
            degree.index_add_(0, dst, torch.ones(self.num_edges, device=self.device, dtype=self.dtype))
            
            if lambda_smooth > 0:
                # L^T·L 的对角线近似为 2*degree
                diag_A = diag_A + lambda_smooth * 2.0 * degree
            
            if lambda_grad > 0:
                # G^T·G 的对角线近似为 2*degree
                diag_A = diag_A + lambda_grad * 2.0 * degree
        
        M_inv = 1.0 / (diag_A + 1e-10)
        
        # z = M^{-1}·r
        z = M_inv * r
        p = z.clone()
        
        rz = torch.dot(r, z)
        
        # 记录收敛历史
        residuals = []
        
        # 初始化残差（防止 max_iter=0 或提前 break 时未定义）
        residual_norm = torch.norm(r).item()
        
        # PCG 迭代
        solve_start = time.time()
        i = -1  # 初始化迭代计数（如果 max_iter=0，i 将保持为 -1）
        for i in range(max_iter):
            # α = (r^T·z) / (p^T·A·p)
            Ap = self._apply_A(p, sensor_indices, lambda_smooth, lambda_grad)
            pAp = torch.dot(p, Ap)
            
            if pAp < 1e-20:
                if verbose:
                    print(f"   ⚠️  警告: pAp 过小 ({pAp:.2e})，提前终止")
                break
            
            alpha = rz / pAp
            
            # x = x + α·p
            x = x + alpha * p
            
            # r_new = r - α·A·p
            r = r - alpha * Ap
            
            # 检查收敛
            residual_norm = torch.norm(r).item()
            residuals.append(residual_norm)
            
            if residual_norm < tol:
                if verbose:
                    print(f"   ✓ 收敛于第 {i+1} 次迭代")
                    print(f"   ✓ 最终残差: {residual_norm:.4e}")
                break
            
            # z = M^{-1}·r
            z = M_inv * r
            
            # β = (r_new^T·z_new) / (r^T·z)
            rz_new = torch.dot(r, z)
            beta = rz_new / (rz + 1e-20)
            
            # p = z + β·p
            p = z + beta * p
            
            rz = rz_new
            
            # 打印进度
            if verbose and ((i+1) % 50 == 0 or i < 10):
                print(f"   迭代 {i+1:4d}/{max_iter}: 残差={residual_norm:.4e}, "
                      f"收敛率={residuals[-2]/residual_norm if len(residuals)>1 else 0:.3f}")
        
        if i == max_iter - 1 and residual_norm >= tol:
            if verbose:
                print(f"   ⚠️  达到最大迭代次数，未完全收敛")
                print(f"   ⚠️  最终残差: {residual_norm:.4e} (目标: {tol:.4e})")
        
        solve_time = time.time() - solve_start
        num_iterations = i + 1
        
        if verbose:
            print(f"   ✓ 求解耗时: {solve_time:.2f}秒")
            if num_iterations > 0:
                print(f"   ✓ 平均每次迭代: {solve_time/num_iterations*1000:.1f}ms")
        
        info = {
            'iterations': num_iterations,
            'residual': residual_norm,
            'residuals': residuals,
            'solve_time': solve_time,
            'converged': residual_norm < tol,
            'avg_iter_time': solve_time / num_iterations if num_iterations > 0 else 0.0
        }
        
        return x, info
