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
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

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

# --- 修改：generate_gantt_chart 函数 ---
def generate_gantt_chart(env):
    """
    生成并显示甘特图，为每个调度的任务添加任务编号标识。
    修改：
    1. 为每个任务分配唯一颜色（至少300种）。
    2. 细化 X 轴刻度间隔。
    3. 添加航行时间条（灰色）并标注 "Navigation"。
    4. 确保每个 USV 的任务是独立绘制的。
    5. 增强容错机制。
    """
    if not env.scheduled_tasks:
        print("警告：没有已调度的任务，无法生成甘特图。")
        return

    fig, ax = plt.subplots(figsize=(15, 8)) # 增加图形尺寸

    # --- 修改 1：生成唯一颜色 ---
    # 生成至少 num_tasks 个唯一颜色，上限为 300
    num_colors_needed = max(env.num_tasks, 300)
    # 使用 HSV 色彩空间生成区分度高的颜色
    hues = np.linspace(0, 1, num_colors_needed, endpoint=False)
    colors_list = [mcolors.hsv_to_rgb((h, 0.9, 0.8)) for h in hues]
    # --- 修改结束 ---

    # 用于存储每个 USV 的任务及其时间信息
    usv_task_data = {i: [] for i in range(env.num_usvs)}

    # --- 修改 4 & 5：收集任务数据，确保独立绘制，并增强容错 ---
    # 使用 env.task_schedule_details 来获取精确的时间信息
    if not hasattr(env, 'task_schedule_details') or not env.task_schedule_details:
        print("警告：环境未存储任务调度详情 (task_schedule_details) 或其为空，甘特图将不包含航行时间。")
        return # 如果没有存储或为空，则不生成甘特图

    # 填充 usv_task_data
    for task_idx in env.scheduled_tasks:
        if task_idx in env.task_schedule_details:
            details = env.task_schedule_details[task_idx]
            try:
                # 确保 task_idx 存在
                task_idx = details['task_idx']
                usv_idx = details['usv_idx']
                usv_task_data[usv_idx].append(details)
            except KeyError as e:
                print(f"Warning: Task {task_idx} details missing key {e}, skipping in Gantt chart.")
        else:
             print(f"Warning: Task {task_idx} details not found in env.task_schedule_details, skipping in Gantt chart.")

    # 对每个 USV 上的任务按处理开始时间排序
    for usv_idx in usv_task_data:
        if usv_task_data[usv_idx]: # 增强容错：检查列表是否为空
            usv_task_data[usv_idx].sort(key=lambda x: x['processing_start_time'])
        else:
            print(f"Warning: No tasks found for USV {usv_idx}, skipping sorting.")

    # 绘制每个任务的条形图
    y_labels = []
    y_positions = []
    y_spacing = 1.5 # USV 之间的垂直间距
    bar_height = 0.4 # 条形图高度

    for usv_idx, tasks_data in usv_task_data.items():
        y_pos = usv_idx * y_spacing
        y_labels.append(f'USV {usv_idx}')
        y_positions.append(y_pos)

        # 增强容错：检查 tasks_data 是否为空
        if not tasks_data:
            print(f"Warning: No task data to plot for USV {usv_idx}.")
            continue

        for task_data in tasks_data:
            try:
                task_idx = task_data['task_idx']  # 确保 task_idx 存在
                processing_start_time = task_data['processing_start_time']
                processing_time = task_data['processing_time']
                travel_start_time = task_data['travel_start_time']
                travel_time = task_data['travel_time']

                # --- 修改 3：绘制航行时间条 ---
                if travel_time > 0:
                    ax.barh(y_pos, travel_time, left=travel_start_time, height=bar_height, color='gray', label='Navigation' if usv_idx == 0 and task_idx == usv_task_data[usv_idx][0]['task_idx'] else "")
                # --- 绘制处理时间条 ---
                ax.barh(y_pos, processing_time, left=processing_start_time, height=bar_height, color=colors_list[task_idx])
                # --- 在处理时间条中心添加任务编号 ---
                ax.text(processing_start_time + processing_time/2, y_pos, f'{task_idx}',
                        ha='center', va='center', fontsize=8, color='white')
            except KeyError as e:
                print(f"Error plotting task for USV {usv_idx}: Missing key {e} in task_data {task_data}")
            except Exception as e:
                print(f"Unexpected error plotting task for USV {usv_idx}: {e}")
    # --- 修改结束 ---

    # --- 修改 2：细化 X 轴刻度 ---
    # 设置主刻度定位器
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    if x_range > 0:
        major_tick_interval = max(1, int(x_range / 20)) # 大约20个主刻度
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=major_tick_interval))
        # 设置主刻度标签格式和旋转
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # 添加 minor ticks
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True, which='major', axis='x', linestyle='-', alpha=0.5)
        ax.grid(True, which='minor', axis='x', linestyle='--', alpha=0.3)
    # --- 修改结束 ---

    # 设置 Y 轴
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Time')
    ax.set_ylabel('USV')
    ax.set_title('Gantt Chart of USV Task Scheduling')

    # --- 修改 3：添加 "Navigation" 图例 ---
    handles, labels = ax.get_legend_handles_labels()
    if 'Navigation' in labels:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='gray', label='Navigation')]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    # --- 修改结束 ---

    fig.tight_layout()
    plt.show()
    print("甘特图已生成并显示。")
# --- 修改结束 ---

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
            # --- 修改：细化 Y 轴范围和刻度 ---
            reward_window = vis.line(
                Y=torch.zeros((1)).cpu(),
                X=torch.zeros((1)).cpu(),
                opts=dict(
                    xlabel='Episode',
                    ylabel='Reward',
                    title='Training Reward',
                    ylim=[-3200, 0],
                    ytickmarks=list(range(-3200, 0, 50))
                )
            )
            makespan_window = vis.line(
                Y=torch.zeros((1)).cpu(),
                X=torch.zeros((1)).cpu(),
                opts=dict(
                    xlabel='Episode',
                    ylabel='Makespan',
                    title='Training Makespan'
                )
            )
            # --- 新增：创建 Policy Loss 和 Value Loss 的 Visdom 窗口，细化 Y 轴范围和刻度 ---
            policy_loss_window = vis.line(
                Y=torch.zeros((1)).cpu(),
                X=torch.zeros((1)).cpu(),
                opts=dict(
                    xlabel='Episode',
                    ylabel='Loss',
                    title='Policy Loss'
                )
            )
            value_loss_window = vis.line(
                Y=torch.zeros((1)).cpu(),
                X=torch.zeros((1)).cpu(),
                opts=dict(
                    xlabel='Episode',
                    ylabel='Loss',
                    title='Value Loss'
                )
            )
            # ----------------------------------------------------------------
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
                # --- 修改：细化 Y 轴范围和刻度 ---
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
                # --- 新增：更新 Policy Loss 和 Value Loss 的 Visdom 曲线，细化 Y 轴范围和刻度 ---
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
