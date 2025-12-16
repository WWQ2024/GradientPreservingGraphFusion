"""
梯度保持融合版本的数据合并脚本
对标 4ALLProcess/MergeCAEandMeasurePoint.py

主要改进：
1. 使用梯度保持融合替代 DAG 传播
2. 无片状割裂
3. 保持 CAE 梯度场连续性
4. 更快的计算速度
5. 集成可视化
"""

import torch
import os
import sys

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from fusion_methods import GradientPreservingFusion
from fusion_methods.utils import compute_mesh_metrics, print_metrics
from visualization import FieldVisualizer, create_comparison_plot
import csv
from scipy.io import loadmat
import numpy as np
import time

def MergeCAEandMeasure_GradientFusion(parameters):
    """
    使用梯度保持融合的数据合并
    
    参数结构（与原版相同）:
    {
        "GraphDataPath": 图数据路径,
        "MeasureDataFile": 实测数据文件,
        "SensorInformationFile": 传感器信息文件,
        "Result_base_save_path": 结果保存路径,
        "rms_file_path": CAE RMS 文件路径,
        "direction": 方向 (1/2/3),
        "Config_ID": 配置 ID,
        
        # 新增参数
        "lambda_smooth": 平滑强度（默认 0.1）,
        "lambda_grad": 梯度保持强度（默认 1.0）,
        "enable_visualization": 是否生成可视化（默认 True）
    }
    
    返回:
    包含融合结果的字典
    """
    try:
        print(f"\n{'='*70}")
        print(f"🚀 梯度保持融合 - 数据合并")
        print(f"{'='*70}\n")
        
        total_start = time.time()
        
        # ========== 1. 参数解析 ==========
        GraphDataPath = os.path.normpath(parameters.get("GraphDataPath"))
        MeasureDataFilePath = os.path.normpath(parameters.get("MeasureDataFile"))
        SensorInformationFileCsvPath = os.path.normpath(parameters.get("SensorInformationFile"))
        direction = int(parameters.get("direction", 1))
        Config_ID = parameters.get("Config_ID", "default")
        
        # 新增参数
        lambda_smooth = float(parameters.get("lambda_smooth", 0.1))
        lambda_grad = float(parameters.get("lambda_grad", 1.0))
        enable_viz = parameters.get("enable_visualization", True)
        
        print(f"📂 输入文件:")
        print(f"   - 图数据: {GraphDataPath}")
        print(f"   - 实测数据: {MeasureDataFilePath}")
        print(f"   - 传感器信息: {SensorInformationFileCsvPath}")
        print(f"\n⚙️  参数:")
        print(f"   - 配置 ID: {Config_ID}")
        print(f"   - 方向: {direction}")
        print(f"   - λ_smooth: {lambda_smooth}")
        print(f"   - λ_grad: {lambda_grad}")
        
        # 检查文件
        for path in [GraphDataPath, MeasureDataFilePath, SensorInformationFileCsvPath]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"文件不存在: {path}")
        
        # ========== 2. 加载图数据 ==========
        print(f"\n📊 加载图数据...")
        loaded_data = torch.load(GraphDataPath)
        graph_data = loaded_data['graph_data']
        node_id_mapping = loaded_data['node_id_mapping']
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        graph_data = graph_data.to(device)
        
        print(f"   ✓ 节点数: {graph_data.num_nodes:,}")
        print(f"   ✓ 边数: {graph_data.num_edges:,}")
        print(f"   ✓ 设备: {device}")
        
        # ========== 3. 读取实测数据 ==========
        print(f"\n📡 读取实测数据...")
        measure_data = loadmat(MeasureDataFilePath)
        dataArray_est = measure_data['dataArray_est'].T  # [DOFs, FreqPoints]
        dofs = dataArray_est.shape[0]
        frequency_resolution = measure_data['FrequencyResolution']
        
        # 计算 RMS
        RMS_Array = np.zeros(dofs)
        for dof in range(dofs):
            rms_value = np.sqrt(np.sum(dataArray_est[dof, :] * frequency_resolution))
            RMS_Array[dof] = rms_value
        
        print(f"   ✓ DOF 数: {dofs}")
        print(f"   ✓ RMS 范围: [{RMS_Array.min():.4e}, {RMS_Array.max():.4e}]")
        
        # ========== 4. 读取传感器信息 ==========
        print(f"\n📌 读取传感器信息...")
        sensor_ids = []
        sensor_node_ids = []
        with open(SensorInformationFileCsvPath, 'r') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                sensor_ids.append(row['ID'])
                sensor_node_ids.append(int(row['NodeID']))
        
        print(f"   ✓ 传感器数量: {len(sensor_ids)}")
        print(f"   ✓ 传感器 ID: {sensor_ids[:5]}..." if len(sensor_ids) > 5 else f"   ✓ 传感器 ID: {sensor_ids}")
        
        # ========== 5. 提取使用的传感器 ==========
        modified_nodes_ids = parameters.get("ModifiedNodesSelectedIds", [])
        if isinstance(modified_nodes_ids, str):
            modified_nodes_ids = modified_nodes_ids.strip('[]').split(',')
            modified_nodes_ids = [id.strip() for id in modified_nodes_ids]
        
        print(f"\n🎯 本次使用的传感器: {modified_nodes_ids}")
        
        # 获取对应的节点 ID 和 RMS 值
        sensor_values = []
        modified_nodes = []
        for modified_id in modified_nodes_ids:
            try:
                index = sensor_ids.index(modified_id)
                node_id = sensor_node_ids[index]
                rms_index = int(index) * 3 + direction
                rms_value = RMS_Array[rms_index]
                
                modified_nodes.append(node_id - 1)  # 节点 ID 转索引
                sensor_values.append(rms_value)
                
                print(f"   ✓ {modified_id}: NodeID={node_id}, RMS={rms_value:.4e}")
            except ValueError:
                print(f"   ⚠️  未找到传感器 {modified_id}")
        
        modified_nodes = torch.tensor(modified_nodes, dtype=torch.long, device=device)
        sensor_values = torch.tensor(sensor_values, dtype=torch.float, device=device)
        
        # ========== 6. 保存原始 CAE 场（用于对比）==========
        x_cae_original = graph_data.y.clone().squeeze()
        
        # ========== 7. 初始化梯度融合器 ==========
        print(f"\n{'='*70}")
        print(f"🔧 初始化梯度保持融合器")
        print(f"{'='*70}")
        
        fusion = GradientPreservingFusion(
            edge_index=graph_data.edge_index,
            num_nodes=graph_data.num_nodes,
            device=device
        )
        
        # ========== 8. 执行融合 ==========
        x_fused, info = fusion.fuse(
            x_cae=graph_data.y.squeeze(),
            sensor_indices=modified_nodes,
            sensor_values=sensor_values,
            lambda_smooth=lambda_smooth,
            lambda_grad=lambda_grad,
            max_iter=1000,
            tol=1e-7,
            verbose=True
        )
        
        # 更新图数据
        graph_data.y = x_fused.unsqueeze(1)
        
        # ========== 9. 计算质量指标 ==========
        metrics = compute_mesh_metrics(
            field_original=x_cae_original,
            field_fused=x_fused,
            sensor_indices=modified_nodes,
            sensor_values=sensor_values,
            edge_index=graph_data.edge_index
        )
        print_metrics(metrics)
        
        # ========== 10. 保存结果 ==========
        base_save_path = parameters["Result_base_save_path"]
        save_dir = os.path.join(base_save_path, Config_ID)
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n💾 保存结果到: {save_dir}")
        
        # 保存融合场
        y_fused = x_fused.cpu().numpy()
        npy_path = os.path.join(save_dir, "fused_field.npy")
        np.save(npy_path, y_fused)
        print(f"   ✓ 融合场: {npy_path}")
        
        # 保存传感器数据对比
        sensor_y_values = [y_fused[node_id-1] for node_id in sensor_node_ids]
        csv_path = os.path.join(save_dir, "sensor_data.csv")
        
        # 读取 CAE RMS 用于对比
        db_value = os.path.basename(os.path.dirname(MeasureDataFilePath)).lower().replace("db", "")
        rms_file_path = parameters["rms_file_path"] + f"\\RMS{db_value}dB.csv"
        cae_rms_values = []
        if os.path.exists(rms_file_path):
            with open(rms_file_path, 'r') as f:
                csv_reader = csv.DictReader(f)
                cae_rms_data = {int(row['nodes']): float(row['RMS']) for row in csv_reader}
            
            # 使用 node_id_mapping 将原始节点 ID 转换为图索引
            # CAE RMS 文件中的 nodes 列是 1-based 的图索引
            for node_id in sensor_node_ids:
                if node_id in node_id_mapping:
                    graph_idx = node_id_mapping[node_id]
                    cae_idx = graph_idx + 1  # CAE 文件使用 1-based 索引
                    cae_rms_values.append(cae_rms_data.get(cae_idx, 0.0))
                else:
                    cae_rms_values.append(0.0)
        
        # 写入 CSV
        with open(csv_path, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['SensorID', 'NodeID', 'Measured_RMS', 'CAE_RMS', 'Fused_RMS', 'Error'])
            
            for i, (sid, nid) in enumerate(zip(sensor_ids, sensor_node_ids)):
                measured = RMS_Array[3*i+direction]
                cae = cae_rms_values[i] if cae_rms_values else 0.0
                fused = sensor_y_values[i]
                error = abs(fused - measured)
                csv_writer.writerow([sid, nid, f"{measured:.6e}", f"{cae:.6e}", 
                                   f"{fused:.6e}", f"{error:.6e}"])
        
        print(f"   ✓ 传感器数据: {csv_path}")
        
        # 保存融合信息
        info_path = os.path.join(save_dir, "fusion_info.txt")
        with open(info_path, 'w') as f:
            f.write(f"梯度保持融合信息\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"配置 ID: {Config_ID}\n")
            f.write(f"传感器: {modified_nodes_ids}\n\n")
            f.write(f"参数:\n")
            f.write(f"  λ_smooth: {lambda_smooth}\n")
            f.write(f"  λ_grad: {lambda_grad}\n\n")
            f.write(f"求解信息:\n")
            f.write(f"  迭代次数: {info['iterations']}\n")
            f.write(f"  最终残差: {info['residual']:.4e}\n")
            f.write(f"  收敛: {'是' if info['converged'] else '否'}\n")
            f.write(f"  求解时间: {info['solve_time']:.2f}秒\n")
            f.write(f"  总时间: {info['total_time']:.2f}秒\n\n")
            f.write(f"质量指标:\n")
            f.write(f"  传感器最大误差: {metrics['sensor_fit']['fused_max_error']:.4e}\n")
            f.write(f"  传感器平均误差: {metrics['sensor_fit']['fused_mean_error']:.4e}\n")
            f.write(f"  梯度方差: {metrics['continuity']['fused_gradient_var']:.4e}\n")
        
        print(f"   ✓ 融合信息: {info_path}")
        
        # ========== 11. Visualization ==========
        if enable_viz:
            print(f"\n{'='*70}")
            print(f"Generating Visualization")
            print(f"{'='*70}")
            
            # Read node coordinates
            if 'pos' in graph_data:
                node_coords = graph_data.pos.cpu().numpy()
            elif hasattr(graph_data, 'x') and graph_data.x.shape[1] >= 3:
                node_coords = graph_data.x[:, :3].cpu().numpy()
            else:
                print(f"   Warning: Node coordinates not found, skipping visualization")
                node_coords = None
            
            if node_coords is not None:
                viz_dir = os.path.join(save_dir, "visualizations")
                os.makedirs(viz_dir, exist_ok=True)
                
                visualizer = FieldVisualizer(node_coords)
                
                create_comparison_plot(
                    visualizer=visualizer,
                    field_original=x_cae_original.cpu().numpy(),
                    field_fused=y_fused,
                    sensor_indices=modified_nodes.cpu().numpy(),
                    output_dir=viz_dir,
                    config_id=Config_ID
                )
        
        # ========== 12. 总结 ==========
        total_time = time.time() - total_start
        
        print(f"\n{'='*70}")
        print(f"✅ 融合完成！")
        print(f"{'='*70}")
        print(f"⏱️  总耗时: {total_time:.2f}秒")
        print(f"📁 结果路径: {save_dir}")
        print(f"{'='*70}\n")
        
        return {
            'success': True,
            'config_id': Config_ID,
            'fused_field': y_fused,
            'metrics': metrics,
            'info': info,
            'save_dir': save_dir,
            'total_time': total_time
        }
    
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    # 测试示例
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    parameters = {
        "GraphDataPath": os.path.join(project_root, '1ExperiStep', '1AddCAEData2ConstructGraphData', 'GraphData', 'GraphData.pt'),
        "MeasureDataFile": os.path.join(project_root, '0Data', '1ExperiData', '0dB', 'dataArray_est.mat'),
        "SensorInformationFile": os.path.join(project_root, '0Data', '3MeasurePointInformation', 'measurement_points_AllInfo.csv'),
        "Result_base_save_path": os.path.join(os.path.dirname(__file__), 'FusionResults'),
        "rms_file_path": os.path.join(project_root, '0Data', '0CAEData'),
        "direction": 1,
        "Config_ID": "Test_A1_A2",
        "ModifiedNodesSelectedIds": ["A1", "A2"],
        "lambda_smooth": 0.1,
        "lambda_grad": 1.0,
        "enable_visualization": True
    }
    
    result = MergeCAEandMeasure_GradientFusion(parameters)
    
    if result['success']:
        print(f"\n🎉 测试成功！")
        print(f"查看结果: {result['save_dir']}")
    else:
        print(f"\n❌ 测试失败: {result.get('error', '未知错误')}")
