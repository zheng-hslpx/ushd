import os
import time
import random
import torch
import numpy as np
from tqdm import tqdm  # 用于显示训练进度条
from torch.utils.tensorboard import SummaryWriter
from env.usv_env import USVSchedulingEnv  # 假设这是您的环境类
from env.state_representation import build_heterogeneous_graph, calculate_usv_task_distances
from graph.hgnn import USVHeteroGNN
from PPO_model import PPO, Memory


def setup_seed(seed):
    """设置随机种子确保结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # 设置随机种子
    setup_seed(42)

    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | {'GPU加速已启用' if device.type == 'cuda' else '使用CPU'}")

    # 超参数配置
    num_usvs = 3  # USV数量
    num_tasks = 5  # 任务数量
    hidden_dim = 32  # 隐藏层维度
    n_heads = 4  # GAT注意力头数
    num_layers = 2  # HGNN层数
    max_episodes = 100  # 最大训练回合数
    max_steps = num_tasks * 2  # 每回合最大步数
    lr = 3e-4  # 学习率
    gamma = 0.99  # 折扣因子
    eps_clip = 0.2  # PPO裁剪参数
    K_epochs = 5  # 每次更新的训练轮数
    early_stop_patience = 50  # 早停耐心值
    eta = 2  # 构建图时考虑的最近邻数量

    # 创建模型保存目录
    os.makedirs("model", exist_ok=True)

    # 初始化环境
    env = USVSchedulingEnv(num_usvs=num_usvs, num_tasks=num_tasks)

    # 初始化HGNN模型
    state = env.reset()  # 获取初始状态
    usv_feats = state['usv_features']  # USV特征 [num_usvs, usv_feat_dim]
    task_feats = state['task_features']  # 任务特征 [num_tasks, task_feat_dim]

    # 计算USV与任务之间的距离（用于构建图）
    distances = calculate_usv_task_distances(usv_feats[:, :2], task_feats[:, :2])

    # 构建异构图（关键：验证边类型）
    graph = build_heterogeneous_graph(usv_feats, task_feats, distances, eta=eta)
    graph = graph.to(device)

    # 打印图的结构信息（关键：使用完整三元组）
    print(f"图中节点类型: {graph.ntypes}")
    print(f"图中完整边类型三元组: {graph.canonical_etypes}")

    # 打印边的详细信息（帮助调试）
    for etype in graph.canonical_etypes:
        src_type, edge_name, dst_type = etype
        print(f"边类型 '{edge_name}': {src_type} → {dst_type}, 边数量: {graph.num_edges(etype)}")

    # 初始化HGNN模型（输入维度由环境特征决定）
    hgnn = USVHeteroGNN(
        usv_feat_dim=usv_feats.shape[1],  # USV特征维度
        task_feat_dim=task_feats.shape[1],  # 任务特征维度
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        num_layers=num_layers
    ).to(device)

    # 初始化PPO代理
    action_dim = num_usvs * num_tasks  # 动作空间维度：每个USV可分配到任意任务
    ppo = PPO(
        hgnn=hgnn,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        eps_clip=eps_clip,
        K_epochs=K_epochs,
        device=device
    ).to(device)

    # 初始化TensorBoard日志记录器
    writer = SummaryWriter("runs/usv_scheduling")

    # 训练统计
    best_makespan = float('inf')  # 最佳完成时间（越小越好）
    no_improve_count = 0  # 早停计数器
    start_time = time.time()  # 记录训练开始时间

    # 主训练循环（使用tqdm显示进度条）
    pbar = tqdm(range(max_episodes), desc="训练进度", unit="episode")
    for episode in pbar:
        # 重置环境和内存缓冲区
        state = env.reset()
        memory = Memory()
        done = False
        total_reward = 0
        steps = 0

        # 一个回合的交互
        while not done and steps < max_steps:
            # 构建异构图状态
            usv_feats = state['usv_features']
            task_feats = state['task_features']
            distances = calculate_usv_task_distances(usv_feats[:, :2], task_feats[:, :2])
            graph = build_heterogeneous_graph(usv_feats, task_feats, distances, eta=eta)
            graph = graph.to(device)

            # 选择动作
            action, log_prob, state_value = ppo.select_action(graph)

            # 执行动作
            next_state, reward, done, _ = env.step(action)

            # 存储经验到内存缓冲区
            memory.states.append(graph)
            memory.actions.append(action)
            memory.logprobs.append(log_prob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            memory.state_values.append(state_value)

            # 更新状态和统计
            state = next_state
            total_reward += reward
            steps += 1

        # 使用收集的经验更新PPO策略
        ppo.update(memory)

        # 计算性能指标
        makespan = env.makespan_batch[env.scheduled_tasks].mean() if env.scheduled_tasks else float('inf')

        # 早停检查：如果当前回合的完成时间优于最佳记录，则保存模型
        if makespan < best_makespan:
            best_makespan = makespan
            no_improve_count = 0
            torch.save(ppo.policy.state_dict(), "model/ppo_best.pt")
            pbar.write(f"Episode {episode}: 新的最佳完成时间 {best_makespan:.4f}，已保存模型")
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                pbar.write(f"\n早停触发：连续{early_stop_patience}轮未改进最佳完成时间")
                break

        # 记录训练日志
        writer.add_scalar("Reward/Episode", total_reward, episode)
        writer.add_scalar("Makespan/Episode", makespan, episode)
        writer.add_scalar("Steps/Episode", steps, episode)

        # 更新进度条显示
        pbar.set_postfix({
            "奖励": f"{total_reward:.2f}",
            "完成时间": f"{makespan:.2f}",
            "最佳时间": f"{best_makespan:.2f}",
            "耗时": f"{time.time() - start_time:.1f}s"
        })

    # 训练结束
    writer.close()
    print(f"训练完成！最佳完成时间: {best_makespan:.4f}")
    print(f"模型已保存至: model/ppo_best.pt")


if __name__ == "__main__":
    main()
