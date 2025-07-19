import numpy as np
from scipy.spatial.distance import cdist


def generate_task_instance(num_tasks, num_usvs, area_size=100):
    """生成任务和USV初始状态"""
    # 生成任务坐标和处理时间（三角模糊时间）
    tasks = {
        'coords': np.random.rand(num_tasks, 2) * area_size,
        'processing_time': np.random.randint(5, 15, size=(num_tasks, 3))  # [t1, t2, t3]
    }

    # 生成USV初始位置（原点）和电量
    usvs = {
        'coords': np.zeros((num_usvs, 2)),
        'battery': np.full(num_usvs, 100),  # 初始电量
        'speed': np.random.uniform(5, 10, num_usvs),  # 速度
        'charge_time': 5  # 充电时间
    }

    return tasks, usvs


def generate_batch_instances(num_instances, max_tasks=20, max_usvs=5, area_size=100):
    """生成批量调度实例"""
    instances = []
    for i in range(num_instances):
        num_t = np.random.randint(5, max_tasks + 1)
        num_u = np.random.randint(1, min(max_usvs, num_t) + 1)
        tasks, usvs = generate_task_instance(num_t, num_u, area_size)
        instances.append((tasks, usvs))
    return instances
