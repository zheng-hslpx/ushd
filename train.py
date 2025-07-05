import copy
import json
import os
import random
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from env.usv_env import USVSchedulingEnv
from env.state_representation import build_heterogeneous_graph, calculate_usv_task_distances
from PPO_model import PPO, Memory
from graph.hgnn import USVHeteroGNN


def setup_seed(seed):
    """设置随机种子确保实验可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    """主训练函数"""
    # 设置随机种子
    setup_seed(42)

    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 超参数配置
    num_usvs = 3
    num_tasks = 10
    hidden_dim = 32
    n_heads = 4
    max_episodes = 200
    learning_rate = 3e-4
    gamma = 0.99
    eps_clip = 0.2
    K_epochs = 5

    # 创建保存模型的目录
    os.makedirs("model", exist_ok=True)

    # 初始化环境
    env = USVSchedulingEnv(num_usvs=num_usvs, num_tasks=num_tasks)

    # 初始化HGNN模型
    hgnn = USVHeteroGNN(
        usv_feat_dim=4,
        task_feat_dim=6,
        hidden_dim=hidden_dim,
        n_heads=n_heads
    ).to(device)

    # 初始化PPO算法
    action_dim = num_usvs * num_tasks
    ppo = PPO(
        hgnn=hgnn,
        action_dim=action_dim,
        lr=learning_rate,
        gamma=gamma,
        eps_clip=eps_clip,
        K_epochs=K_epochs,
        device=device
    )

    # 初始化内存
    memory = Memory()

    # 初始化TensorBoard日志
    writer = SummaryWriter(log_dir="runs/usv_scheduling")

    # 训练循环
    total_rewards = []
    total_losses = []
    best_makespan = float('inf')

    start_time = time.time()

    for episode in range(max_episodes):
        # 重置环境
        state = env.reset()
        done = False
        episode_reward = 0
        transitions = []

        while not done:
            # 构建异构图
            usv_features = state['usv_features']
            task_features = state['task_features']
            usv_positions = usv_features[:, :2]
            task_positions = task_features[:, :2]
            usv_task_distances = calculate_usv_task_distances(usv_positions, task_positions)
            graph_state = build_heterogeneous_graph(usv_features, task_features, usv_task_distances)
            graph_state = graph_state.to(device)  # 移至设备

            # 选择动作
            action, log_prob = ppo.select_action(graph_state)

            # 执行动作
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            # 保存过渡数据（注意：只保存图数据的副本）
            transitions.append((
                copy.deepcopy(graph_state),  # 保存图数据副本
                action,
                log_prob,
                reward,
                next_state,
                done
            ))

            state = next_state

        # 更新策略
        loss = ppo.update(transitions, device)
        total_losses.append(loss)
        total_rewards.append(episode_reward)

        # 计算完成时间（仅考虑已调度的任务）
        if len(env.scheduled_tasks) > 0:
            makespan = env.makespan_batch[env.scheduled_tasks].mean()
        else:
            makespan = 0

        # 记录训练日志
        writer.add_scalar("Reward/Episode", episode_reward, episode)
        writer.add_scalar("Loss/Episode", loss, episode)
        writer.add_scalar("Makespan/Episode", makespan, episode)

        # 每50集打印训练信息
        if episode % 50 == 0:
            end_time = time.time()
            avg_reward = np.mean(total_rewards[-50:])
            avg_loss = np.mean(total_losses[-50:])

            print(
                f"Episode {episode} | Reward: {episode_reward:.2f} | Avg Reward: {avg_reward:.2f} | Loss: {loss:.4f} | Makespan: {makespan:.2f} | Time: {end_time - start_time:.2f}s")

            # 保存最佳模型
            if makespan < best_makespan and makespan > 0:
                best_makespan = makespan
                torch.save(ppo.policy.state_dict(), f"model/ppo_best_{episode}.pt")
                print(f"保存最佳模型，完成时间: {makespan:.2f}")

    # 训练完成
    writer.close()
    end_time = time.time()
    print(f"训练完成！总耗时: {end_time - start_time:.2f}秒")
    print(f"最佳完成时间: {best_makespan:.2f}")


if __name__ == "__main__":
    main()