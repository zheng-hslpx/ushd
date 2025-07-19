import numpy as np
import torch
import dgl


def build_heterogeneous_graph(usv_features, task_features, usv_task_distances, eta=2):
    """
    构建异构图，任务节点仅与η个最近邻任务聚合
    :param eta: 最近邻任务数量
    """
    # 计算任务间距离矩阵
    task_positions = task_features[:, :2]
    task_distances = np.sqrt(((task_positions[:, np.newaxis] - task_positions) ** 2).sum(axis=2))

    # 对每个任务，选择η个最近邻任务（排除自身）
    task_edges = []
    for i in range(len(task_features)):
        neighbors = np.argsort(task_distances[i])[1:eta + 1]  # 排除自身，取前η个
        for j in neighbors:
            task_edges.append((i, j))

    # 构建USV-任务边
    usv_task_edges = []
    for usv_idx in range(len(usv_features)):
        for task_idx in range(len(task_features)):
            if usv_task_distances[usv_idx, task_idx] < np.inf:
                usv_task_edges.append((usv_idx, task_idx))

    # 构建异构图
    graph = dgl.heterograph({
        ('usv', 'to', 'task'): usv_task_edges,
        ('task', 'to', 'task'): task_edges
    })

    # 设置节点特征
    graph.nodes['usv'].data['feat'] = torch.tensor(usv_features, dtype=torch.float32)
    graph.nodes['task'].data['feat'] = torch.tensor(task_features, dtype=torch.float32)

    return graph


def calculate_usv_task_distances(usv_positions, task_positions):
    """
    计算USV与任务之间的距离矩阵
    """
    usv_task_distances = np.sqrt(((usv_positions[:, np.newaxis] - task_positions) ** 2).sum(axis=2))
    return usv_task_distances
