from PPO_model import ActorCritic
from graph.hgnn import USVHeteroGNN
from env.usv_env import USVSchedulingEnv
from env.state_representation import build_heterogeneous_graph
import torch
import numpy as np
from utils.evaluation import evaluate_scheduling


def test(model_path, num_tests=100, num_usvs=3, num_tasks=10):
    # 加载模型
    hgnn = USVHeteroGNN(usv_feat_dim=3, task_feat_dim=5, hidden_dim=32)
    policy = ActorCritic(hgnn, action_dim=num_usvs * num_tasks)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    # 测试结果存储
    completion_times = []
    path_efficiencies = []

    for test_idx in range(num_tests):
        env = USVSchedulingEnv(num_usvs, num_tasks)
        state = env.reset()
        done = False

        while not done:
            # 构建异构图状态
            graph_state = build_heterogeneous_graph(
                state['usv_features'], state['task_features'], state['edge_features']
            )

            # 选择动作（贪心策略）
            with torch.no_grad():
                action_probs, _ = policy(graph_state)
                action = torch.argmax(action_probs).item()

            # 执行动作
            state, _, done, _ = env.step(action)

        # 评估调度结果
        completion_time, efficiency = evaluate_scheduling(env.usvs, env.tasks)
        completion_times.append(completion_time)
        path_efficiencies.append(efficiency)

        if test_idx % 10 == 0:
            print(f"Test {test_idx}, Completion Time: {completion_time:.2f}, Efficiency: {efficiency:.2f}")

    # 输出统计结果
    avg_time = np.mean(completion_times)
    avg_efficiency = np.mean(path_efficiencies)
    print(f"平均完成时间: {avg_time:.2f}, 平均路径效率: {avg_efficiency:.2f}")
    return avg_time, avg_efficiency