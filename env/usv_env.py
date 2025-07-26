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
        self.task_assignment = np.full(self.num_tasks, -1, dtype=int) # -1 表示未分配
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

    def _find_next_feasible_time_and_action(self, original_usv_idx, original_task_idx):
        """
        计算使原始动作 (original_usv_idx, original_task_idx) 可行的最早时间点。
        如果任务已被调度，则返回 None, None, None。
        """
        if original_task_idx in self.scheduled_tasks:
            # 任务已被调度，无法执行
            return None, None, None

        # 计算 USV 到达任务位置所需的时间
        usv_pos = self.usv_positions[original_usv_idx]
        task_pos = self.tasks['coords'][original_task_idx]
        distance = np.linalg.norm(usv_pos - task_pos)
        if self.usv_speeds[original_usv_idx] > 0:
            travel_time = distance / self.usv_speeds[original_usv_idx]
        else:
            # 速度为0，永远无法到达
            return None, None, None

        # 计算任务处理时间
        task_processing_time = self.tasks['processing_time'][original_task_idx]
        if isinstance(task_processing_time, (list, tuple, np.ndarray)):
            processing_time = np.mean(task_processing_time)
        else:
            processing_time = task_processing_time

        # 计算使动作可行的最早时间点
        # 条件1: USV 空闲且电量足够
        earliest_time_for_usv = self.usv_next_available_time[original_usv_idx]
        if self.usv_batteries[original_usv_idx] <= distance * 0.1: # 简化电量检查
             # 电量不足，无法执行此动作
             return None, None, None

        # 确定最终的可行时间点
        feasible_time = max(self.current_time, earliest_time_for_usv)

        # 计算在可行时间点执行此动作的完整完成时间
        completion_time = feasible_time + travel_time + processing_time

        return feasible_time, completion_time, (original_usv_idx, original_task_idx)


    def _execute_action(self, usv_idx, task_idx, execution_time):
        """
        在指定时间点执行调度动作。
        """
        # 获取任务信息
        task_pos = self.tasks['coords'][task_idx]
        task_processing_time = self.tasks['processing_time'][task_idx]
        if isinstance(task_processing_time, (list, tuple, np.ndarray)):
            processing_time = np.mean(task_processing_time)
        else:
            processing_time = task_processing_time

        # 计算距离和时间
        usv_pos = self.usv_positions[usv_idx]
        distance = np.linalg.norm(usv_pos - task_pos)
        if self.usv_speeds[usv_idx] > 0:
            travel_time = distance / self.usv_speeds[usv_idx]
        else:
            # 这种情况在_find_next_feasible_time_and_action中已被过滤
            travel_time = float('inf')

        # --- 更新USV状态 ---
        # 更新USV位置和电量 (在移动和处理 *之后*)
        self.usv_positions[usv_idx] = task_pos
        self.usv_batteries[usv_idx] -= distance * 0.1 # 电量消耗系数
        if self.usv_batteries[usv_idx] < 0:
            self.usv_batteries[usv_idx] = 0 # 确保电量不为负

        # 更新USV下次可用时间
        self.usv_next_available_time[usv_idx] = execution_time + travel_time + processing_time

        # 标记任务为已调度并记录完成时间
        self.scheduled_tasks.append(task_idx)
        self.makespan_batch[task_idx] = execution_time + travel_time + processing_time
        self.task_assignment[task_idx] = usv_idx # 记录分配

        # 更新全局时间
        self.current_time = self.makespan_batch[task_idx]

        # 计算奖励
        reward = self._calculate_reward(task_idx, usv_idx, travel_time, processing_time, distance)

        done = len(self.scheduled_tasks) == self.num_tasks  # 所有任务完成则结束

        # --- 在 info 中返回更多信息 ---
        info = {
            "makespan": np.max(self.makespan_batch[self.scheduled_tasks]) if self.scheduled_tasks else 0,
            "avg_completion_time": np.mean(self.makespan_batch[self.scheduled_tasks]) if self.scheduled_tasks else 0,
            "total_time": self.current_time,
            "usv_battery": self.usv_batteries[usv_idx],
            "distance": distance,
            "travel_time": travel_time,
            "processing_time": processing_time,
            "done": done,
            # "action_taken": action, # action_taken 不再适用，因为我们可能执行了不同的动作
            "original_action": usv_idx * self.num_tasks + task_idx # 保留原始动作索引
        }
        if done:
            info["final_makespan"] = np.max(self.makespan_batch) # 最终的 makespan

        return reward, done, info


    def step(self, action):
        """执行动作并更新环境状态，实现时钟移动逻辑"""
        original_usv_idx = action // self.num_tasks
        original_task_idx = action % self.num_tasks

        # 1. 检查原始动作是否立即可行
        task_processing_time = self.tasks['processing_time'][original_task_idx]
        if isinstance(task_processing_time, (list, tuple, np.ndarray)):
            processing_time = np.mean(task_processing_time)
        else:
            processing_time = task_processing_time

        usv_pos = self.usv_positions[original_usv_idx]
        task_pos = self.tasks['coords'][original_task_idx]
        distance = np.linalg.norm(usv_pos - task_pos)
        if self.usv_speeds[original_usv_idx] > 0:
            travel_time = distance / self.usv_speeds[original_usv_idx]
        else:
            travel_time = float('inf')

        if (
            original_task_idx not in self.scheduled_tasks and
            self.current_time >= self.usv_next_available_time[original_usv_idx] and
            self.usv_batteries[original_usv_idx] > distance * 0.1 # 检查电量
        ):
            # 原始动作立即可行，直接执行
            # print(f"Action ({original_usv_idx}, {original_task_idx}) is immediately feasible at time {self.current_time}. Executing.")
            reward, done, info = self._execute_action(original_usv_idx, original_task_idx, self.current_time)
            return self._get_observation(), reward, done, info

        else:
            # 2. 原始动作不可行，寻找下一个可行时间点
            # print(f"Action ({original_usv_idx}, {original_task_idx}) not immediately feasible. Finding next feasible time...")
            feasible_time, completion_time, feasible_action_tuple = self._find_next_feasible_time_and_action(original_usv_idx, original_task_idx)

            if feasible_time is not None and feasible_action_tuple is not None:
                # 3. 找到可行时间点，执行动作
                # print(f"Found feasible time {feasible_time} for action {feasible_action_tuple}. Advancing time and executing.")
                # 将时钟推进到可行时间点
                self.current_time = feasible_time
                usv_idx_to_execute, task_idx_to_execute = feasible_action_tuple
                reward, done, info = self._execute_action(usv_idx_to_execute, task_idx_to_execute, feasible_time)
                return self._get_observation(), reward, done, info
            else:
                # 4. 无法使原始动作可行（例如任务已调度或USV永远无法到达）
                # print(f"Action ({original_usv_idx}, {original_task_idx}) cannot be made feasible. Returning penalty.")
                return self._get_observation(), -50, False, {
                    "invalid_action": True,
                    "reason": "Action cannot be made feasible (task scheduled or USV unreachable)",
                    "original_action": action
                }


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
        if isinstance(proc_times, (list, np.ndarray)) and len(proc_times) > 0 and isinstance(proc_times[0], (list, np.ndarray)):
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

    # --- 修改：重写奖励函数 ---
    def _calculate_reward(self, task_idx, usv_idx, travel_time, processing_time, distance):
        """
        基于任务完成情况、时间效率计算奖励。
        移除了电量和时间惩罚项。
        """
        # 1. 基础奖励：完成任务总是好的
        base_reward = 70.0 # 完成一个任务的基础奖励

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
            efficiency_bonus += 30.0  # 提前完成大奖励
        elif processing_time <= t2:
            efficiency_bonus += 15.0   # 按时完成中等奖励
        elif processing_time <= t3:
            efficiency_bonus += 5.0   # 轻微延迟小奖励
        # else: 严重延迟无效率奖励，甚至可以考虑小惩罚

        # 3. 总奖励 (移除了惩罚项)
        total_reward = base_reward + efficiency_bonus

        return total_reward

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
