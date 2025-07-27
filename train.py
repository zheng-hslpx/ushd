import os
import time
import random
import torch
import numpy as np
from tqdm import tqdm  # 用于显示训练进度条
from torch.utils.tensorboard import SummaryWriter
from env.usv_env import USVSchedulingEnv  # 导入修改后的环境类
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
# 确保 data 目录存在
os.makedirs(os.path.dirname(file_path), exist_ok=True)
data_generator.save_instances_to_file(instances, file_path)
# 初始化环境（使用固定数量的任务和USV）
env = USVSchedulingEnv(num_usvs=num_usvs, num_tasks=num_tasks)

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
    """
    生成并显示甘特图，为每个调度的任务添加任务编号标识。
    修复：处理 processing_time 可能是数组的情况。
    """
    if not env.scheduled_tasks:
        print("警告：没有已调度的任务，无法生成甘特图。")
        return
    fig, ax = plt.subplots(figsize=(12, 6)) # 增加图形尺寸以获得更好可读性
    usv_tasks = {i: [] for i in range(env.num_usvs)}
    # 将任务按USV分组
    for task_idx in env.scheduled_tasks:
        # --- 修改：优先使用 env.task_assignment ---
        if hasattr(env, 'task_assignment') and env.task_assignment is not None:
             if task_idx < len(env.task_assignment) and env.task_assignment[task_idx] is not None and env.task_assignment[task_idx] != -1:
                 usv_idx = env.task_assignment[task_idx]
             else:
                 print(f"Warning: Task {task_idx} assignment not found or invalid, skipping in Gantt chart.")
                 continue
        else:
            print("Warning: Direct task-to-USV mapping not found in env. Using modulo method (might be inaccurate).")
            usv_idx = task_idx % env.num_usvs
        if 0 <= usv_idx < env.num_usvs:
            usv_tasks[usv_idx].append(task_idx)
        else:
            print(f"Warning: Task {task_idx} assigned to invalid USV index {usv_idx}, skipping.")
    # 对每个 USV 上的任务按开始时间排序
    for usv_idx in usv_tasks:
        sorted_tasks = []
        for tid in usv_tasks[usv_idx]:
            if tid < len(env.makespan_batch) and tid < len(env.tasks['processing_time']):
                end_time = env.makespan_batch[tid]
                # --- 修复关键点 ---
                # 安全地获取处理时间（使用平均值作为近似）
                proc_time_data = env.tasks['processing_time'][tid]
                if isinstance(proc_time_data, (list, tuple, np.ndarray)):
                    if len(proc_time_data) >= 2:
                        processing_time = proc_time_data[1] # 取 t2 (按时完成时间)
                    else:
                        processing_time = np.mean(proc_time_data) # Fallback
                else:
                    processing_time = proc_time_data
                start_time = end_time - processing_time
                sorted_tasks.append((tid, start_time))
            else:
                 print(f"Warning: Task {tid} data missing, skipping in sort.")
        # 根据计算出的开始时间排序
        sorted_tasks.sort(key=lambda x: x[1])
        # 只保留任务ID
        usv_tasks[usv_idx] = [tid for tid, _ in sorted_tasks]
    # 绘制每个任务的条形图并添加任务编号
    y_labels = [] # 存储Y轴标签
    for usv_idx, tasks in usv_tasks.items():
        y_labels.append(f'USV {usv_idx}')
        for i, task_idx in enumerate(tasks):
            if task_idx < len(env.makespan_batch) and task_idx < len(env.tasks['processing_time']):
                # --- 再次安全地获取处理时间用于绘图 ---
                proc_time_data = env.tasks['processing_time'][task_idx]
                if isinstance(proc_time_data, (list, tuple, np.ndarray)):
                    if len(proc_time_data) >= 2:
                        processing_time = proc_time_data[1] # 与排序时保持一致
                    else:
                        processing_time = np.mean(proc_time_data)
                else:
                    processing_time = proc_time_data
                end_time = env.makespan_batch[task_idx]
                start_time = end_time - processing_time
                # 绘制条形图
                bar = ax.barh(usv_idx, processing_time, left=start_time, height=0.5, label=f'Task {task_idx}' if i == 0 else "")
                # 在条形图中心添加任务编号文本
                ax.text(start_time + processing_time/2, usv_idx, f'{task_idx}',
                        ha='center', va='center', fontsize=8, color='white')
            else:
                 print(f"Warning: Task {task_idx} data missing in env.makespan_batch or env.tasks['processing_time'].")
    # 设置图表属性
    ax.set_yticks(range(env.num_usvs))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Time')
    ax.set_ylabel('USV')
    ax.set_title('Gantt Chart of USV Task Scheduling')
    ax.grid(True, axis='x', linestyle='--', alpha=0.6) # 添加网格线
    fig.tight_layout() # 调整布局
    # --- 修改：显示并保存甘特图 ---
    plt.show()
    # 可选：保存图片到文件
    # plt.savefig("gantt_chart_final.png")
    print("甘特图已生成并显示。")
    # -----------------------------

def main():
    # 设置随机种子
    setup_seed(42)
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | {'GPU加速已启用' if device.type == 'cuda' else '使用CPU'}")
    # 超参数配置（与全局参数保持一致）
    # --- 修改：调整超参数以减少波动 ---
    hidden_dim = 32  # 隐藏层维度
    n_heads = 4  # GAT注意力头数
    num_layers = 2  # HGNN层数
    max_episodes = 2000  # 最大训练回合数
    max_steps = num_tasks * 2  # 每回合最大步数
    lr = 5e-5  # 学习率 (从 3e-4 降低到 1e-4)
    gamma = 0.98  # 折扣因子
    eps_clip = 0.2  # PPO裁剪参数 (从 0.2 降低到 0.1)
    K_epochs = 10  # 每次更新的训练轮数
    early_stop_patience = 2000  # 早停耐心值
    eta = 2  # 构建图时考虑的最近邻数量
    # ----------------------------------
    # 创建模型保存目录
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    # 获取下一个可用的模型编号
    next_model_number = get_next_model_number(model_dir)
    # --- 修改：实例索引 ---
    instance_index = 0
    # 初始化HGNN模型 (在循环外初始化一次即可)
    state = env.reset() # 先 reset 获取初始状态以确定维度
    usv_feats = state['usv_features']  # USV特征 [num_usvs, usv_feat_dim]
    task_feats = state['task_features']  # 任务特征 [num_tasks, task_feat_dim]
    distances = calculate_usv_task_distances(usv_feats[:, :2], task_feats[:, :2])
    graph = build_heterogeneous_graph(usv_feats, task_feats, distances, eta=eta)
    graph = graph.to(device)
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
    if not vis.check_connection():
        print("Warning: Visdom server not connected. Plots will not be displayed.")
        vis = None # 设置为 None 以便后续检查
    reward_window = None
    makespan_window = None
    # --- 修改：为 Policy Loss 和 Value Loss 创建 Visdom 窗口 ---
    policy_loss_window = None
    value_loss_window = None
    # ---------------------------------------------------------------
    if vis:
        try:
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
            # --- 新增：创建 Policy Loss 和 Value Loss 的 Visdom 窗口 ---
            policy_loss_window = vis.line(
                Y=torch.zeros((1)).cpu(),
                X=torch.zeros((1)).cpu(),
                opts=dict(xlabel='Episode', ylabel='Loss', title='Policy Loss')
            )
            value_loss_window = vis.line(
                Y=torch.zeros((1)).cpu(),
                X=torch.zeros((1)).cpu(),
                opts=dict(xlabel='Episode', ylabel='Loss', title='Value Loss')
            )
            # ---------------------------------------------------------------
        except Exception as e:
            print(f"Warning: Failed to create Visdom windows: {e}")
            vis = None # 如果创建窗口失败，也禁用 visdom
    # 训练统计
    best_makespan = float('inf')
    no_improve_count = 0
    start_time = time.time()
    # 主训练循环
    pbar = tqdm(range(max_episodes), desc="训练进度", unit="episode")
    for episode in pbar:
        # --- 修改：在每个 episode 开始时重置环境 ---
        tasks, usvs = instances[instance_index % num_instances]
        instance_index += 1
        state = env.reset_with_instances(tasks, usvs)
        # --- 修改结束 ---
        memory = Memory()
        done = False
        total_reward = 0
        steps = 0
        episode_makespan = 0 # 用于记录最终的 makespan
        while not done and steps < max_steps:
            # 构建异构图状态
            usv_feats = state['usv_features']
            task_feats = state['task_features']
            distances = calculate_usv_task_distances(usv_feats[:, :2], task_feats[:, :2])
            graph = build_heterogeneous_graph(usv_feats, task_feats, distances, eta=eta)
            graph = graph.to(device)
            # 选择动作
            action, log_prob, state_value = ppo.select_action(graph)
            # --- 修改：执行动作并接收 info ---
            next_state, reward, done, info = env.step(action)
            # --- 修改：使用 info 中的最终 makespan ---
            if done and 'final_makespan' in info:
                episode_makespan = info['final_makespan']
                # --- 修改：在 episode 结束时给予稀疏奖励 ---
                # 奖励与 makespan 成反比，系数可调 (这里是示例)
                # --- 修改：调整稀疏奖励权重 ---
                sparse_reward = -episode_makespan * 0.1 # 系数可以根据训练效果调整 (例如从 0.2 调整回 0.1 或更低)
                reward += sparse_reward
                # print(f"Episode {episode} finished. Sparse reward: {sparse_reward}, Total step reward: {reward - sparse_reward}")
            elif not done:
                 # 如果未完成，使用当前最大已调度任务完成时间作为近似 (用于早停和日志)
                 episode_makespan = info.get("makespan", episode_makespan)
            # --- 修改结束 ---
            # 存储经验
            memory.states.append(graph)
            memory.actions.append(action)
            memory.logprobs.append(log_prob)
            memory.rewards.append(reward) # 使用可能被修改后的 reward
            memory.is_terminals.append(done)
            memory.state_values.append(state_value)
            # 更新状态和统计
            state = next_state
            total_reward += reward
            steps += 1
        # --- 修改：更新PPO策略并获取损失 ---
        # ppo.update(memory) # 原来的调用
        policy_loss_avg, value_loss_avg = ppo.update(memory) # 修改后的调用，接收损失
        # --- 修改结束 ---
        # --- 修改：使用 episode_makespan 作为性能指标 ---
        makespan = episode_makespan if episode_makespan > 0 else float('inf')
        # --- 修改结束 ---
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
                pbar.write(f"早停触发：连续{early_stop_patience}轮未改进最佳完成时间")
                break
        # 记录日志
        writer.add_scalar("Reward/Episode", total_reward, episode)
        writer.add_scalar("Makespan/Episode", makespan, episode)
        writer.add_scalar("Steps/Episode", steps, episode)
        # --- 新增：记录损失到 TensorBoard ---
        writer.add_scalar("Policy_Loss/Episode", policy_loss_avg, episode)
        writer.add_scalar("Value_Loss/Episode", value_loss_avg, episode)
        # -------------------------------------
        # 更新可视化
        if vis and reward_window is not None and makespan_window is not None and policy_loss_window is not None and value_loss_window is not None:
            try:
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
                # --- 新增：更新 Policy Loss 和 Value Loss 的 Visdom 曲线 ---
                vis.line(
                    Y=torch.tensor([policy_loss_avg]).cpu(), # 使用返回的平均损失
                    X=torch.tensor([episode]).cpu(),
                    win=policy_loss_window,
                    update='append',
                    name='Policy Loss' # 可选：为曲线命名
                )
                vis.line(
                    Y=torch.tensor([value_loss_avg]).cpu(), # 使用返回的平均损失
                    X=torch.tensor([episode]).cpu(),
                    win=value_loss_window,
                    update='append',
                    name='Value Loss' # 可选：为曲线命名
                )
                # ----------------------------------------------------------------
            except Exception as e:
                print(f"Warning: Failed to update Visdom plots: {e}")
                # 可以选择禁用 visdom 或继续尝试
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
    # --- 修改：在训练结束后生成甘特图 ---
    try:
        generate_gantt_chart(env) # 生成最终调度结果的甘特图
    except Exception as e:
        print(f"生成甘特图时出错: {e}")
    # ------------------------------------

if __name__ == "__main__":
    main()
