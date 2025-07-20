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
        task_idx = action % self.num_tasks   # 解析任务索引

        # 检查动作有效性
        if (
            task_idx in self.scheduled_tasks  # 任务已调度
            or self.current_time < self.usv_next_available_time[usv_idx]  # USV忙碌
            or self.usv_batteries[usv_idx] <= 10  # 电量不足
        ):
            return self._get_observation(), -10, False, {}  # 无效动作惩罚

        # 获取USV和任务信息
        usv_pos = self.usv_positions[usv_idx]
        task_pos = self.tasks['coords'][task_idx]
        task_processing_time = self.tasks['processing_time'][task_idx]

        # 计算距离和时间
        distance = np.linalg.norm(usv_pos - task_pos)
        travel_time = distance / self.usv_speeds[usv_idx]
        processing_time = np.mean(task_processing_time)  # 平均处理时间

        # 更新USV状态
        self.usv_positions[usv_idx] = task_pos  # 更新位置
        self.usv_batteries[usv_idx] -= distance * 0.5  # 电量消耗
        self.usv_next_available_time[usv_idx] = self.current_time + travel_time + processing_time

        # 标记任务为已调度并记录完成时间
        self.scheduled_tasks.append(task_idx)
        self.makespan_batch[task_idx] = self.current_time + travel_time + processing_time

        # 更新当前时间
        self.current_time = max(self.current_time, self.usv_next_available_time[usv_idx])

        # 计算奖励
        reward = self._calculate_reward(task_idx, travel_time, processing_time)
        done = len(self.scheduled_tasks) == self.num_tasks  # 所有任务完成则结束

        return self._get_observation(), reward, done, {}

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
        task_features[:, 2:5] = self.tasks['processing_time'] / 20.0  # 归一化处理时间
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

    def _calculate_reward(self, task_idx, travel_time, processing_time):
        """基于任务完成情况和资源消耗计算奖励"""
        t1, t2, t3 = self.tasks['processing_time'][task_idx]
        # 基于处理时间的奖励
        if processing_time <= t1:
            reward = 10.0  # 提前完成
        elif processing_time <= t2:
            reward = 5.0  # 按时完成
        elif processing_time <= t3:
            reward = 1.0  # 轻微延迟
        else:
            reward = -5.0  # 严重延迟

        # 电量惩罚（消耗越多惩罚越大）
        usv_idx = task_idx // self.num_tasks
        energy_penalty = (self.battery_capacity - self.usv_batteries[usv_idx]) * 0.01
        # 时间惩罚（完成时间越晚惩罚越大）
        time_penalty = self.current_time * 0.001

        return reward - energy_penalty - time_penalty

    def render(self, mode='human'):
        """可视化环境状态（调试用）"""
        if mode == 'human':
            print(f"当前时间: {self.current_time:.2f}")
            print(f"USV位置: {self.usv_positions}")
            print(f"USV电量: {self.usv_batteries}")
            print(f"已调度任务: {self.scheduled_tasks}")
            print(f"待调度任务: {[i for i in range(self.num_tasks) if i not in self.scheduled_tasks]}")
            print(f"平均完成时间: {self.makespan_batch[self.scheduled_tasks].mean():.2f}\n")
