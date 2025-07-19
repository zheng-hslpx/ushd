import gym
from gym import spaces
import numpy as np
from utils.data_generator import generate_task_instance


class USVSchedulingEnv(gym.Env):
    def __init__(self, num_usvs=3, num_tasks=10):
        super().__init__()
        # 定义观测空间和动作空间
        self.observation_space = spaces.Dict({
            'usv_features': spaces.Box(low=0, high=1, shape=(num_usvs, 4)),  # [x, y, battery, speed]
            'task_features': spaces.Box(low=0, high=100, shape=(num_tasks, 6)),  # [x, y, t1, t2, t3, is_pending]
            'edge_features': spaces.Box(low=0, high=100, shape=(num_usvs, num_tasks))  # 距离矩阵
        })
        self.action_space = spaces.Discrete(num_usvs * num_tasks)

        # 初始化环境参数
        self.num_usvs = num_usvs
        self.num_tasks = num_tasks
        self.reset()

        # 新增：记录任务完成时间
        self.makespan_batch = np.zeros(num_tasks)

    def reset(self):
        """重置环境状态"""
        # 生成初始任务和USV状态
        self.tasks, self.usvs = generate_task_instance(self.num_tasks, self.num_usvs)
        self.scheduled_tasks = []  # 已调度的任务列表
        self.current_time = 0  # 当前时间步
        self.usv_positions = np.zeros((self.num_usvs, 2))  # USV位置
        self.usv_batteries = np.full(self.num_usvs, 100.0)  # USV电量
        self.usv_speeds = np.random.uniform(5, 10, self.num_usvs)  # USV速度
        self.usv_next_available_time = np.zeros(self.num_usvs)  # USV下一次可用时间

        # 重置完成时间记录
        self.makespan_batch = np.zeros(self.num_tasks)

        return self._get_observation()

    def step(self, action):
        """执行动作并更新环境状态"""
        usv_idx = action // self.num_tasks
        task_idx = action % self.num_tasks

        # 检查动作有效性
        if (
                task_idx in self.scheduled_tasks  # 任务已调度
                or self.current_time < self.usv_next_available_time[usv_idx]  # USV不可用
                or self.usv_batteries[usv_idx] <= 10  # 电量不足
        ):
            return self._get_observation(), -10, False, {}

        # 获取USV和任务信息
        usv_pos = self.usv_positions[usv_idx]
        task_pos = self.tasks['coords'][task_idx]
        task_processing_time = self.tasks['processing_time'][task_idx]

        # 计算距离和所需时间
        distance = np.linalg.norm(usv_pos - task_pos)
        travel_time = distance / self.usv_speeds[usv_idx]
        processing_time = np.mean(task_processing_time)  # 使用平均处理时间

        # 更新USV状态
        self.usv_positions[usv_idx] = task_pos
        self.usv_batteries[usv_idx] -= distance * 0.5  # 电量消耗
        self.usv_next_available_time[usv_idx] = self.current_time + travel_time + processing_time

        # 标记任务为已调度
        self.scheduled_tasks.append(task_idx)

        # 记录任务完成时间
        completion_time = self.current_time + travel_time + processing_time
        self.makespan_batch[task_idx] = completion_time

        # 更新当前时间
        self.current_time = max(self.current_time, self.usv_next_available_time[usv_idx])

        # 计算奖励（基于任务完成时间和电量消耗）
        reward = self._calculate_reward(task_idx, travel_time, processing_time)

        # 检查是否所有任务都已完成
        done = len(self.scheduled_tasks) == self.num_tasks

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        """返回当前环境状态的观测值"""
        # USV特征: [x, y, battery, speed]
        usv_features = np.column_stack([
            self.usv_positions / 100.0,  # 归一化位置
            self.usv_batteries / 100.0,  # 归一化电量
            self.usv_speeds / 10.0  # 归一化速度
        ])

        # 任务特征: [x, y, t1, t2, t3, is_pending]
        task_features = np.zeros((self.num_tasks, 6))
        task_features[:, :2] = self.tasks['coords'] / 100.0  # 归一化位置
        task_features[:, 2:5] = self.tasks['processing_time'] / 100.0  # 归一化处理时间

        # 标记未调度的任务
        for task_idx in range(self.num_tasks):
            if task_idx not in self.scheduled_tasks:
                task_features[task_idx, 5] = 1.0

        # 计算USV和任务之间的距离矩阵
        distances = np.zeros((self.num_usvs, self.num_tasks))
        for i in range(self.num_usvs):
            for j in range(self.num_tasks):
                if j not in self.scheduled_tasks:
                    distances[i, j] = np.linalg.norm(self.usv_positions[i] - self.tasks['coords'][j])

        # 返回观测
        return {
            'usv_features': usv_features.astype(np.float32),
            'task_features': task_features.astype(np.float32),
            'edge_features': distances.astype(np.float32)
        }

    def _calculate_reward(self, task_idx, travel_time, processing_time):
        """计算奖励函数"""
        # 获取任务的处理时间参数
        t1, t2, t3 = self.tasks['processing_time'][task_idx]

        # 基于三角模糊时间的奖励
        if processing_time <= t1:
            reward = 10.0  # 提前完成
        elif processing_time <= t2:
            reward = 5.0  # 按时完成
        elif processing_time <= t3:
            reward = 1.0  # 延迟完成
        else:
            reward = -5.0  # 严重延迟

        # 电量惩罚
        energy_penalty = (100 - self.usv_batteries[task_idx // self.num_tasks]) * 0.01

        # 时间惩罚（完成时间越晚，奖励越低）
        time_penalty = self.current_time * 0.001

        # 综合奖励
        return reward - energy_penalty - time_penalty

    def render(self, mode='human'):
        """可视化环境（简化实现）"""
        if mode == 'human':
            print(f"Time: {self.current_time:.2f}")
            print(f"USV Positions: {self.usv_positions}")
            print(f"USV Batteries: {self.usv_batteries}")
            print(f"Scheduled Tasks: {self.scheduled_tasks}")
            print(f"Pending Tasks: {[i for i in range(self.num_tasks) if i not in self.scheduled_tasks]}")
            print(f"Makespan: {self.makespan_batch[self.scheduled_tasks].mean():.2f}")
