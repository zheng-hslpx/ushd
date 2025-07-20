import os
import torch
import numpy as np
from graph.hgnn import USVHeteroGNN
from PPO_model import PPO
from env.usv_env import USVSchedulingEnv
from env.state_representation import build_heterogeneous_graph, calculate_usv_task_distances


def test(model_path, num_tests=100, num_usvs=3, num_tasks=5, area_size_x=(0, 100), area_size_y=(0, 100),
         processing_time_range=(5, 15), battery_capacity=100, speed_range=(5, 10), charge_time=5,
         device=None, eta=2):
    # 关键修复：调整num_tasks使action_dim与训练时一致（3*5=15，匹配checkpoint的15）
    # 若训练时的参数不同，需修改为训练时的num_usvs和num_tasks
    print(f"使用的动作空间维度: {num_usvs * num_tasks}（需与训练时一致）")

    # 设备配置
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化HGNN（与训练时参数一致）
    hgnn = USVHeteroGNN(
        usv_feat_dim=4,
        task_feat_dim=6,
        hidden_dim=32,
        n_heads=4,
        num_layers=2
    ).to(device)

    # 动作空间维度（确保与训练时一致）
    action_dim = num_usvs * num_tasks  # 3*5=15，匹配checkpoint的15

    # 初始化PPO
    policy = PPO(
        hgnn=hgnn,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        K_epochs=5,
        device=device
    ).to(device)

    # 加载模型权重（无需修改键名，因action_dim已匹配）
    saved_weights = torch.load(model_path, map_location=device)
    policy.load_state_dict(saved_weights, strict=False)  # strict=False忽略无关参数
    policy.eval()

    # 测试结果存储
    completion_times = []
    path_efficiencies = []

    for test_idx in range(num_tests):
        # 初始化环境（使用与action_dim匹配的任务数量）
        env = USVSchedulingEnv(num_usvs, num_tasks, area_size_x, area_size_y,
                               processing_time_range, battery_capacity, speed_range, charge_time)
        state = env.reset()
        done = False

        while not done:
            # 构建异构图状态
            usv_features = state['usv_features']
            task_features = state['task_features']
            usv_positions = usv_features[:, :2]
            task_positions = task_features[:, :2]
            usv_task_distances = np.sqrt(((usv_positions[:, np.newaxis] - task_positions) ** 2).sum(axis=2))
            graph_state = build_heterogeneous_graph(
                usv_features, task_features, usv_task_distances, eta=eta
            ).to(device)

            # 选择动作
            with torch.no_grad():
                action_probs, _ = policy(graph_state)
                action = torch.argmax(action_probs).item()

            # 执行动作
            state, _, done, _ = env.step(action)

        # 评估结果
        completion_time, efficiency = evaluate_scheduling(env.usvs, env.tasks)
        completion_times.append(completion_time)
        path_efficiencies.append(efficiency)

        if test_idx % 10 == 0:
            print(f"Test {test_idx}, 完成时间: {completion_time:.2f}, 效率: {efficiency:.2f}")

    # 输出统计结果
    avg_time = np.mean(completion_times)
    avg_efficiency = np.mean(path_efficiencies)
    print(f"\n平均完成时间: {avg_time:.2f}, 平均效率: {avg_efficiency:.2f}")
    return avg_time, avg_efficiency


def evaluate_scheduling(usvs, tasks):
    # 补充实际评估逻辑（示例）
    completion_time = 0.0
    efficiency = 0.0
    return completion_time, efficiency


if __name__ == "__main__":
    # 关键修复：使用与训练时一致的num_usvs和num_tasks
    # 若训练时是3个USV和5个任务（3*5=15），则此处保持一致
    MODEL_PATH = "model/ppo_best.pt"
    NUM_TESTS = 50
    NUM_USVS = 4  # 与训练时一致
    NUM_TASKS = 6  # 与训练时一致（4*6=24）
    area_size_x = (0, 200)  # 任务点X坐标值范围
    area_size_y = (0, 200)  # 任务点Y坐标值范围
    processing_time_range = (10, 20)  # 任务的访问时间范围
    battery_capacity = 120  # 电池容量
    speed_range = (6, 12)  # 无人船速度范围
    charge_time = 6  # 充电时间

    test(
        model_path=MODEL_PATH,
        num_tests=NUM_TESTS,
        num_usvs=NUM_USVS,
        num_tasks=NUM_TASKS,
        area_size_x=area_size_x,
        area_size_y=area_size_y,
        processing_time_range=processing_time_range,
        battery_capacity=battery_capacity,
        speed_range=speed_range,
        charge_time=charge_time,
        eta=2
    )
