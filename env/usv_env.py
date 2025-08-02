import gym
from gym import spaces
import numpy as np
from utils.data_generator import generate_task_instance  # 导入任务生成函数

class USVSchedulingEnv(gym.Env):
    def __init__(self, num_usvs=3, num_tasks=30, area_size_x=(0, 100), area_size_y=(0, 100),
                 processing_time_range=(5, 15), battery_capacity=100, speed_range=(5, 10), charge_time=5):
        super().__init__()
        # 定义观测空间和动作空间（形状与任务/USV数量匹配）
        self.observation_space = spaces.Dict({
            'usv_features': spaces.Box(low=0, high=1, shape=(num_usvs, 4)),  # [x, y, battery, speed]
            'task_features': spaces.Box(low=0, high=100, shape=(num_tasks, 6)),  # [x, y, t1, t2, t3, is_pending]
            'edge_features': spaces.Box(low=0, high=100, shape=(num_usvs, num_tasks))  # 距离矩阵
        })
        self.action_space = spaces.Discrete(num_usvs * num_tasks)  # 动作空间维度匹配
        # 初始化环境参数
        self.num_usvs = num_usvs
        self.num_tasks = num_tasks
        self.area_size_x = area_size_x
        self.area_size_y = area_size_y
        self.processing_time_range = processing_time_range
        self.battery_capacity = battery_capacity
        self.speed_range = speed_range
        self.charge_time = charge_time
        # 记录任务完成时间（长度与任务数量匹配）
        self.makespan_batch = np.zeros(num_tasks)
        # --- 新增：记录任务到USV的分配 ---
        self.task_assignment = np.full(self.num_tasks, -1, dtype=int)  # -1 表示未分配
        # --- 新增：记录任务调度详情用于甘特图 ---
        self.task_schedule_details = {}  # {task_idx: {'usv_idx': ..., 'travel_start_time': ..., 'travel_time': ..., 'processing_start_time': ..., 'processing_time': ...}}
        # -------------------------------------
        # 初始化状态（避免在__init__中调用reset，改为显式调用）

    def reset(self):
        """重置环境为初始状态，返回有效的观测值（关键修复）"""
        # 生成默认的初始任务和USV数据（与环境参数匹配）
        default_tasks, default_usvs = generate_task_instance(
            num_tasks=self.num_tasks,
            num_usvs=self.num_usvs,
            area_size_x=self.area_size_x,
            area_size_y=self.area_size_y,
            processing_time_range=self.processing_time_range,
            battery_capacity=self.battery_capacity,
            speed_range=self.speed_range,
            charge_time=self.charge_time
        )
        # 复用reset_with_instances逻辑，确保返回格式一致
        return self.reset_with_instances(default_tasks, default_usvs)

    def reset_with_instances(self, tasks, usvs):
        """使用给定的任务和USV状态重置环境，确保返回有效观测"""
        self.tasks = tasks
        self.usvs = usvs
        self.scheduled_tasks = []  # 清空已调度任务
        self.current_time = 0  # 重置当前时间
        self.usv_positions = np.zeros((self.num_usvs, 2))  # USV初始位置（形状：[num_usvs, 2]）
        self.usv_batteries = np.full(self.num_usvs, self.battery_capacity)  # 初始电量（满电）
        # --- 重置任务分配记录 ---
        self.task_assignment = np.full(self.num_tasks, -1, dtype=int)
        # --- 重置任务调度详情记录 ---
        self.task_schedule_details = {}
        # ----------------------------
        # 处理USV速度数组（确保长度与USV数量一致）
        speed_data = np.array(usvs['speed'])
        if len(speed_data) > self.num_usvs:
            self.usv_speeds = speed_data[:self.num_usvs]  # 截断过多数据
        elif len(speed_data) < self.num_usvs:
            default_speed = np.mean(self.speed_range)  # 用速度范围均值填充
            padding_length = self.num_usvs - len(speed_data)
            self.usv_speeds = np.pad(
                speed_data,
                pad_width=(0, padding_length),
                constant_values=default_speed
            )
        else:
            self.usv_speeds = speed_data  # 长度匹配时直接使用
        self.usv_next_available_time = np.zeros(self.num_usvs)  # 重置可用时间
        self.makespan_batch = np.zeros(self.num_tasks)  # 重置完成时间记录
        # 生成并返回观测值（确保非None）
        observation = self._get_observation()
        return observation

    def step(self, action):
        """执行动作并更新环境状态"""
        usv_idx = action // self.num_tasks  # 解析USV索引
        task_idx = action % self.num_tasks  # 解析任务索引
        # 使用 while 循环确保动作有效性
        while True:
            # 检查动作有效性
            if (
                    task_idx in self.scheduled_tasks  # 任务已调度
                    or self.current_time < self.usv_next_available_time[usv_idx]  # USV忙碌
                    or self.usv_batteries[usv_idx] <= 10  # 电量不足
            ):
                # 如果动作无效，尝试下一个动作
                action += 1
                usv_idx = action // self.num_tasks  # 更新 USV 索引
                task_idx = action % self.num_tasks  # 更新任务索引
                if action >= self.action_space.n:  # 如果所有动作都无效，返回惩罚
                    return self._get_observation(), -50, False, {"invalid_action": True}
            else:
                break  # 动作有效，退出循环
        # 获取USV和任务信息
        usv_pos = self.usv_positions[usv_idx]
        task_pos = self.tasks['coords'][task_idx]
        # --- 修改：处理可能的列表形式的 processing_time ---
        task_processing_time = self.tasks['processing_time'][task_idx]
        if isinstance(task_processing_time, (list, tuple, np.ndarray)):
            processing_time = np.mean(task_processing_time)
        else:
            processing_time = task_processing_time
        # 计算距离和时间
        distance = np.linalg.norm(usv_pos - task_pos)
        # --- 修改：避免除零错误 ---
        if self.usv_speeds[usv_idx] > 0:
            travel_time = distance / self.usv_speeds[usv_idx]
        else:
            travel_time = float('inf')  # 或者给予极大惩罚
            return self._get_observation(), -100, False, {"invalid_action": True, "reason": "zero_speed"}

        # --- 在计算 travel_time 后，记录调度详情 ---
        travel_start_time = self.usv_next_available_time[usv_idx]  # USV开始移动的时间
        # --- 修改：记录分配关系 ---
        self.task_assignment[task_idx] = usv_idx
        # --- 修改：更新USV状态 ---
        # 更新USV位置和电量 (在移动和处理 *之后*)
        self.usv_positions[usv_idx] = task_pos
        # --- 修改：调整电量消耗系数 ---
        self.usv_batteries[usv_idx] -= distance * 0.1  # 电量消耗系数从 0.5 调整为 0.1
        if self.usv_batteries[usv_idx] < 0:
            self.usv_batteries[usv_idx] = 0  # 确保电量不为负
        # --- 更新 current_time 为所有 USV 中最早可用的时间 (允许并行) ---
        # self.current_time = self.makespan_batch[task_idx] # 旧逻辑：等待单个任务完成
        self.current_time = np.min(self.usv_next_available_time)  # 新逻辑：允许并行，时间推进到最早空闲的 USV 时间点

        processing_start_time = self.current_time + travel_time  # USV到达并开始处理的时间
        processing_end_time = self.current_time + travel_time + processing_time  # USV完成处理的时间

        # 存储任务调度详情 (确保包含 task_idx)
        self.task_schedule_details[task_idx] = {
            'task_idx': task_idx, # 显式添加 task_idx
            'usv_idx': usv_idx,
            'travel_start_time': travel_start_time,
            'travel_time': travel_time,
            'processing_start_time': processing_start_time,
            'processing_time': processing_time
        }
        # --- 记录结束 ---

        # 更新USV下次可用时间
        self.usv_next_available_time[usv_idx] = processing_end_time
        # 标记任务为已调度并记录完成时间
        self.scheduled_tasks.append(task_idx)
        self.makespan_batch[task_idx] = processing_end_time  # 使用处理结束时间

        # --- 修改：计算奖励 ---
        reward = self._calculate_reward(task_idx, usv_idx, travel_time, processing_time, distance)
        done = len(self.scheduled_tasks) == self.num_tasks  # 所有任务完成则结束
        # --- 修改：在 info 中返回更多信息用于调试和最终奖励 ---
        info = {
            "makespan": np.max(self.makespan_batch[self.scheduled_tasks]) if self.scheduled_tasks else 0,
            "avg_completion_time": np.mean(self.makespan_batch[self.scheduled_tasks]) if self.scheduled_tasks else 0,
            "total_time": self.current_time,  # 这现在是全局时间
            "usv_battery": self.usv_batteries[usv_idx],
            "distance": distance,
            "travel_time": travel_time,
            "processing_time": processing_time,
            "done": done,
            "action_taken": action,  # 记录实际执行的动作
            "original_action": action  # 记录原始选择的动作
        }
        if done:
            info["final_makespan"] = np.max(self.makespan_batch)  # 最终的 makespan
        return self._get_observation(), reward, done, info

    def _calculate_reward(self, task_idx, usv_idx, travel_time, processing_time, distance):
        """
        基于任务完成情况、时间效率计算奖励。
        移除了电量和时间惩罚项。
        """
        # 1. 基础奖励：完成任务总是好的
        base_reward = -70.0  # 完成一个任务的基础奖励（负数）
        # 2. 效率奖励：鼓励快速完成任务 (与处理时间窗口 t1,t2,t3 相比)
        # --- 修改：安全地获取 t1, t2, t3 ---
        proc_time_entry = self.tasks['processing_time'][task_idx]
        if isinstance(proc_time_entry, (list, tuple, np.ndarray)) and len(proc_time_entry) >= 3:
            t1, t2, t3 = proc_time_entry[0], proc_time_entry[1], proc_time_entry[2]
        else:
            # Fallback if not 3-element array
            t1, t2, t3 = processing_time, processing_time, processing_time
        efficiency_bonus = 0.0
        if processing_time <= t1:
            efficiency_bonus -= 30.0  # 提前完成大奖励（负数）
        elif processing_time <= t2:
            efficiency_bonus -= 15.0  # 按时完成中等奖励（负数）
        elif processing_time <= t3:
            efficiency_bonus -= 5.0  # 轻微延迟小奖励（负数）
        # else: 严重延迟无效率奖励，甚至可以考虑小惩罚
        # 3. 总奖励
        total_reward = base_reward + efficiency_bonus
        return total_reward

    def _get_observation(self):
        """生成环境观测值（确保数组维度匹配）"""
        # USV特征：[x, y, battery, speed]（形状：[num_usvs, 4]）
        usv_features = np.column_stack([
            self.usv_positions / 200.0,  # 归一化位置
            self.usv_batteries / self.battery_capacity,  # 归一化电量
            self.usv_speeds / 12.0  # 归一化速度
        ])
        # 任务特征：[x, y, t1, t2, t3, is_pending]（形状：[num_tasks, 6]）
        task_features = np.zeros((self.num_tasks, 6))
        task_features[:, :2] = self.tasks['coords'] / 200.0  # 归一化位置
        # --- 修改：处理可能的列表形式的 processing_time ---
        proc_times = self.tasks['processing_time']
        if isinstance(proc_times, (list, np.ndarray)) and len(proc_times) > 0 and isinstance(proc_times[0],
                                                                                             (list, np.ndarray)):
            # 如果是二维数组，取平均
            task_features[:, 2:5] = np.mean(proc_times, axis=1, keepdims=True).repeat(3, axis=1) / 20.0
        else:
            # 如果是一维数组，复制到 t1,t2,t3
            task_features[:, 2:5] = np.tile(proc_times, (3, 1)).T / 20.0
        # 标记未调度任务
        for task_idx in range(self.num_tasks):
            if task_idx not in self.scheduled_tasks:
                task_features[task_idx, 5] = 1.0
        # 距离矩阵（形状：[num_usvs, num_tasks]）
        distances = np.zeros((self.num_usvs, self.num_tasks))
        for i in range(self.num_usvs):
            for j in range(self.num_tasks):
                if j not in self.scheduled_tasks:
                    distances[i, j] = np.linalg.norm(self.usv_positions[i] - self.tasks['coords'][j])
        return {
            'usv_features': usv_features.astype(np.float32),
            'task_features': task_features.astype(np.float32),
            'edge_features': distances.astype(np.float32)
        }

    def render(self, mode='human'):
        """可视化环境状态（调试用）"""
        if mode == 'human':
            print(f"当前时间: {self.current_time:.2f}")
            print(f"USV位置: {self.usv_positions}")
            print(f"USV电量: {self.usv_batteries}")
            print(f"已调度任务: {self.scheduled_tasks}")
            print(f"待调度任务: {[i for i in range(self.num_tasks) if i not in self.scheduled_tasks]}")
            print(f"平均完成时间: {self.makespan_batch[self.scheduled_tasks].mean():.2f}")
            # --- 修改：打印任务分配 ---
            print(f"任务分配: {self.task_assignment}")
            # --- 修改：打印 USV 下次可用时间 ---
            print(f"USV下次可用时间: {self.usv_next_available_time}")
