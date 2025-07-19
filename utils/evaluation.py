import numpy as np


def calculate_completion_time(usv_tasks, usv_coords, usv_speed, charge_time):
    """计算USV完成所有任务的时间"""
    completion_times = []

    for usv_idx, tasks in enumerate(usv_tasks):
        if not tasks:
            completion_times.append(0)
            continue

        # 任务执行顺序和路径
        task_coords = [usv_coords[usv_idx]] + [usv_tasks[task]['coords'] for task in tasks]
        total_distance = 0
        processing_time = 0

        # 计算总距离和处理时间
        for i in range(len(task_coords) - 1):
            total_distance += np.linalg.norm(task_coords[i] - task_coords[i + 1])
            processing_time += np.mean(usv_tasks[tasks[i]]['processing_time'])

        # 计算航行时间和充电时间
        travel_time = total_distance / usv_speed[usv_idx]
        charge_times = max(0, len(tasks) // 5 - 1)  # 每5个任务充一次电
        total_time = travel_time + processing_time + charge_times * charge_time
        completion_times.append(total_time)

    return max(completion_times)  # 返回最大完成时间


def evaluate_scheduling(usvs, tasks):
    """评估调度方案的完成时间和路径效率"""
    # 假设已分配任务到USV（实际需根据调度结果计算）
    # 这里简化为随机分配，实际应根据模型输出结果计算
    usv_tasks = {i: [] for i in range(usvs['coords'].shape[0])}
    for task_idx in range(tasks['coords'].shape[0]):
        usv_idx = task_idx % usvs['coords'].shape[0]
        usv_tasks[usv_idx].append(task_idx)

    # 计算完成时间
    completion_time = calculate_completion_time(
        usv_tasks, usvs['coords'], usvs['speed'], usvs['charge_time']
    )

    # 计算路径效率（总距离/最短可能距离）
    total_distance = 0
    for usv_idx, tasks_idx in usv_tasks.items():
        if not tasks_idx:
            continue
        coords = [usvs['coords'][usv_idx]] + [tasks['coords'][i] for i in tasks_idx]
        for i in range(len(coords) - 1):
            total_distance += np.linalg.norm(coords[i] - coords[i + 1])

    # 假设最短路径为任务点的最小生成树距离（简化计算）
    min_distance = 0  # 实际应计算最小生成树或TSP最短路径
    efficiency = min_distance / max(1, total_distance) if min_distance else 0

    return completion_time, efficiency
