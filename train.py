import os
import time
import random
import torch
import numpy as np
from tqdm import tqdm  # 用于显示训练进度条
from torch.utils.tensorboard import SummaryWriter
from env.usv_env import USVSchedulingEnv  # 导入环境类
from env.state_representation import build_heterogeneous_graph, calculate_usv_task_distances
from graph.hgnn import USVHeteroGNN
from PPO_model import PPO, Memory
import visdom
import utils.data_generator as data_generator
import matplotlib.pyplot as plt

# 超参数：固定任务和USV数量（核心：与环境保持一致）
num_usvs = 3  # USV数量
num_tasks = 30  # 任务数量（与环境、算例生成器保持一致）

# 生成固定任务数量的算例文件
num_instances = 10
# 明确指定算例的任务和USV数量（与环境匹配）
instances = data_generator.generate_batch_instances(
    num_instances=num_instances,
    fixed_tasks=num_tasks,
    fixed_usvs=num_usvs
)
file_path = "data/fixed_instances.pkl"
data_generator.save_instances_to_file(instances, file_path)

# 初始化环境（使用固定数量的任务和USV）
env = USVSchedulingEnv(num_usvs=num_usvs, num_tasks=num_tasks)

# 在每个周期读取算例
instance_index = 0
for episode in tqdm(range(100), desc="读取算例进度", unit="episode"):
    # 读取算例（任务数量已固定为num_tasks）
    tasks, usvs = instances[instance_index % num_instances]
    instance_index += 1
    # 重置环境（此时任务数量与环境匹配）
    env.reset_with_instances(tasks, usvs)


def setup_seed(seed):
    """设置随机种子确保结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_next_model_number(model_dir):
    """获取下一个可用的模型编号"""
    existing_models = [f for f in os.listdir(model_dir) if f.startswith('ppo_best_') and f.endswith('.pt')]
    numbers = []
    for model in existing_models:
        try:
            number = int(model.split('_')[-1].split('.')[0])
            numbers.append(number)
        except ValueError:
            continue
    if numbers:
        return max(numbers) + 1
    return 0


def generate_gantt_chart(env):
    fig, ax = plt.subplots()
    usv_tasks = {i: [] for i in range(env.num_usvs)}
    for task_idx in env.scheduled_tasks:
        usv_idx = task_idx % env.num_usvs
        usv_tasks[usv_idx].append(task_idx)
    for usv_idx, tasks in usv_tasks.items():
        for task_idx in tasks:
            start_time = env.makespan_batch[task_idx] - np.mean(env.tasks['processing_time'][task_idx])
            end_time = env.makespan_batch[task_idx]
            ax.barh(usv_idx, end_time - start_time, left=start_time)
    ax.set_xlabel('Time')
    ax.set_ylabel('USV')
    ax.set_title('Gantt Chart')
    plt.show()


def main():
    # 设置随机种子
    setup_seed(42)

    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | {'GPU加速已启用' if device.type == 'cuda' else '使用CPU'}")

    # 超参数配置（与全局参数保持一致）
    hidden_dim = 32  # 隐藏层维度
    n_heads = 4  # GAT注意力头数
    num_layers = 2  # HGNN层数
    max_episodes = 100  # 最大训练回合数
    max_steps = num_tasks * 2  # 每回合最大步数
    lr = 3e-4  # 学习率
    gamma = 0.99  # 折扣因子
    eps_clip = 0.2  # PPO裁剪参数
    K_epochs = 5  # 每次更新的训练轮数
    early_stop_patience = 100  # 早停耐心值
    eta = 2  # 构建图时考虑的最近邻数量

    # 创建模型保存目录
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)

    # 获取下一个可用的模型编号
    next_model_number = get_next_model_number(model_dir)

    # 初始化HGNN模型
    state = env.reset()  # 获取初始状态
    usv_feats = state['usv_features']  # USV特征 [num_usvs, usv_feat_dim]
    task_feats = state['task_features']  # 任务特征 [num_tasks, task_feat_dim]

    # 计算USV与任务之间的距离（用于构建图）
    distances = calculate_usv_task_distances(usv_feats[:, :2], task_feats[:, :2])

    # 构建异构图
    graph = build_heterogeneous_graph(usv_feats, task_feats, distances, eta=eta)
    graph = graph.to(device)

    # 打印图的结构信息
    print(f"图中节点类型: {graph.ntypes}")
    print(f"图中完整边类型三元组: {graph.canonical_etypes}")
    for etype in graph.canonical_etypes:
        src_type, edge_name, dst_type = etype
        print(f"边类型 '{edge_name}': {src_type} → {dst_type}, 边数量: {graph.num_edges(etype)}")

    # 初始化HGNN模型
    hgnn = USVHeteroGNN(
        usv_feat_dim=usv_feats.shape[1],
        task_feat_dim=task_feats.shape[1],
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        num_layers=num_layers
    ).to(device)

    # 初始化PPO代理
    action_dim = num_usvs * num_tasks  # 动作空间维度与任务/USV数量匹配
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

    # 初始化visdom
    vis = visdom.Visdom()
    reward_window = vis.line(
        Y=torch.zeros((1)).cpu(),
        X=torch.zeros((1)).cpu(),
        opts=dict(xlabel='Episode', ylabel='Reward', title='Training Reward')
    )
    makespan_window = vis.line(
        Y=torch.zeros((1)).cpu(),
        X=torch.zeros((1)).cpu(),
        opts=dict(xlabel='Episode', ylabel='Makespan', title='Training Makespan')
    )

    # 训练统计
    best_makespan = float('inf')
    no_improve_count = 0
    start_time = time.time()

    # 主训练循环
    pbar = tqdm(range(max_episodes), desc="训练进度", unit="episode")
    for episode in pbar:
        state = env.reset()
        memory = Memory()
        done = False
        total_reward = 0
        steps = 0

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

            # 存储经验
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

        # 更新PPO策略
        ppo.update(memory)

        # 计算性能指标
        makespan = env.makespan_batch[env.scheduled_tasks].mean() if env.scheduled_tasks else float('inf')

        # 早停检查与模型保存
        if makespan < best_makespan:
            best_makespan = makespan
            no_improve_count = 0
            model_name = f"ppo_best_{next_model_number}.pt"
            model_path = os.path.join(model_dir, model_name)
            torch.save(ppo.policy.state_dict(), model_path)
            pbar.write(f"Episode {episode}: 新的最佳完成时间 {best_makespan:.4f}，已保存模型至 {model_path}")
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                pbar.write(f"\n早停触发：连续{early_stop_patience}轮未改进最佳完成时间")
                break

        # 记录日志
        writer.add_scalar("Reward/Episode", total_reward, episode)
        writer.add_scalar("Makespan/Episode", makespan, episode)
        writer.add_scalar("Steps/Episode", steps, episode)

        # 更新可视化
        vis.line(
            Y=torch.tensor([total_reward]).cpu(),
            X=torch.tensor([episode]).cpu(),
            win=reward_window,
            update='append'
        )
        vis.line(
            Y=torch.tensor([makespan]).cpu(),
            X=torch.tensor([episode]).cpu(),
            win=makespan_window,
            update='append'
        )

        # 更新进度条
        pbar.set_postfix({
            "奖励": f"{total_reward:.2f}",
            "完成时间": f"{makespan:.2f}",
            "最佳时间": f"{best_makespan:.2f}",
            "耗时": f"{time.time() - start_time:.1f}s"
        })

    # 训练结束
    writer.close()
    print(f"训练完成！最佳完成时间: {best_makespan:.4f}")
    print(f"模型已保存至: {model_path}")
    generate_gantt_chart(env)


if __name__ == "__main__":
    main()
