import torch
import numpy as np
from torch_geometric.data import HeteroData


def build_heterogeneous_graph(usv_features, task_features, usv_task_distances):
    """
    构建包含USV和任务节点的异构图，包含USV-任务边、任务-任务边和USV-USV边
    """
    # 优化：将输入转换为numpy数组后再创建张量
    usv_features = np.array(usv_features)
    task_features = np.array(task_features)
    usv_task_distances = np.array(usv_task_distances)

    data = HeteroData()

    # 添加USV节点 [num_usvs, 4]
    data['usv'].x = torch.tensor(usv_features, dtype=torch.float32)

    # 添加任务节点 [num_tasks, 6]
    data['task'].x = torch.tensor(task_features, dtype=torch.float32)

    # ================== 添加USV-任务边 ==================
    usv_num, task_num = usv_features.shape[0], task_features.shape[0]
    usv_indices, task_indices = np.meshgrid(
        np.arange(usv_num), np.arange(task_num), indexing='ij'
    )
    edge_index = torch.tensor(
        [usv_indices.flatten(), task_indices.flatten()], dtype=torch.long
    )
    edge_attr = torch.tensor(usv_task_distances.flatten(), dtype=torch.float32)
    data['usv', 'to', 'task'].edge_index = edge_index
    data['usv', 'to', 'task'].edge_attr = edge_attr

    # ================== 添加任务-任务边 ==================
    task_coords = task_features[:, :2]  # 提取任务坐标 [x, y]
    task_distances = np.zeros((task_num, task_num))

    # 计算任务间距离矩阵
    for i in range(task_num):
        for j in range(task_num):
            if i != j:
                task_distances[i, j] = np.linalg.norm(task_coords[i] - task_coords[j])

    task_edges = []
    task_edge_attr = []
    for i in range(task_num):
        for j in range(task_num):
            if i != j:
                task_edges.append([i, j])
                task_edge_attr.append(task_distances[i, j])

    if task_edges:
        edge_index = torch.tensor(np.array(task_edges).T, dtype=torch.long)
        edge_attr = torch.tensor(task_edge_attr, dtype=torch.float32)
        data['task', 'to', 'task'].edge_index = edge_index
        data['task', 'to', 'task'].edge_attr = edge_attr

    # ================== 添加USV-USV边 ==================
    usv_coords = usv_features[:, :2]  # 提取USV坐标 [x, y]
    usv_distances = np.zeros((usv_num, usv_num))

    # 计算USV间距离矩阵
    for i in range(usv_num):
        for j in range(usv_num):
            if i != j:
                usv_distances[i, j] = np.linalg.norm(usv_coords[i] - usv_coords[j])

    usv_edges = []
    usv_edge_attr = []
    for i in range(usv_num):
        for j in range(usv_num):
            if i != j:
                usv_edges.append([i, j])
                usv_edge_attr.append(usv_distances[i, j])

    if usv_edges:
        edge_index = torch.tensor(np.array(usv_edges).T, dtype=torch.long)
        edge_attr = torch.tensor(usv_edge_attr, dtype=torch.float32)
        data['usv', 'to', 'usv'].edge_index = edge_index
        data['usv', 'to', 'usv'].edge_attr = edge_attr

    # 打印图结构信息（调试用）
    print(f"[GRAPH DEBUG] USV节点数: {usv_num}, 任务节点数: {task_num}")
    print(f"[GRAPH DEBUG] USV-任务边数: {len(edge_attr)}")
    print(f"[GRAPH DEBUG] 任务-任务边数: {len(task_edge_attr) if task_edges else 0}")
    print(f"[GRAPH DEBUG] USV-USV边数: {len(usv_edge_attr) if usv_edges else 0}")

    return data


def calculate_usv_task_distances(usv_positions, task_positions):
    """计算USV和任务之间的距离矩阵"""
    return np.array([
        [np.linalg.norm(usv - task) for task in task_positions]
        for usv in usv_positions
    ])