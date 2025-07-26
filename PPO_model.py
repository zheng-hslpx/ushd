import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class Memory:
    """存储智能体与环境交互的经验数据"""
    def __init__(self):
        self.states = []  # 存储异构图状态
        self.actions = []  # 存储选择的动作
        self.logprobs = []  # 存储动作的对数概率
        self.rewards = []  # 存储获得的奖励
        self.is_terminals = []  # 存储是否为终止状态
        self.state_values = []  # 存储状态价值估计

    def clear_memory(self):
        """清空内存中的所有经验数据"""
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.state_values[:]


class PPO(nn.Module):
    """近端策略优化算法的实现"""
    def __init__(self, hgnn, action_dim, lr, gamma, eps_clip, K_epochs, device):
        """
        初始化PPO模型
        参数:
            hgnn: 异构图神经网络，用于特征提取
            action_dim: 动作空间维度
            lr: 学习率
            gamma: 折扣因子
            eps_clip: PPO裁剪参数
            K_epochs: 每次更新的训练轮数
            device: 计算设备（CPU或GPU）
        """
        super(PPO, self).__init__()
        self.hgnn = hgnn
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        # 策略网络：将HGNN提取的特征映射到动作概率分布
        self.policy = nn.Sequential(
            nn.Linear(self.hgnn.global_pool.out_features, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)  # 输出动作概率分布
        ).to(device)

        # 价值网络：估计当前状态的价值
        self.value_net = nn.Sequential(
            nn.Linear(self.hgnn.global_pool.out_features, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)  # 输出状态价值
        ).to(device)

        # 旧策略网络：用于计算策略更新的重要性采样比率
        self.policy_old = nn.Sequential(
            nn.Linear(self.hgnn.global_pool.out_features, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())  # 初始化为与当前策略相同

        # 优化器：同时优化策略网络、价值网络和HGNN
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) +
            list(self.value_net.parameters()) +
            list(self.hgnn.parameters()),
            lr=lr
        )
        self.MseLoss = nn.MSELoss()  # 均方误差损失函数

    def forward(self, graph):
        """
        前向传播计算动作概率和状态价值
        参数:
            graph: 异构图状态
        返回:
            action_probs: 动作概率分布
            state_value: 状态价值估计
        """
        # 通过HGNN提取全局状态表示
        _, _, global_h = self.hgnn(graph)
        # 计算动作概率分布
        action_probs = self.policy(global_h)
        # 计算状态价值
        state_value = self.value_net(global_h)
        return action_probs, state_value

    def select_action(self, graph):
        """
        根据当前状态选择动作，并记录相关信息
        参数:
            graph: 异构图状态
        返回:
            action.item(): 选择的动作索引
            dist.log_prob(action): 动作的对数概率
            state_value.item(): 状态价值估计
        """
        # 计算动作概率和状态价值
        action_probs, state_value = self.forward(graph)
        # 创建分类分布
        dist = Categorical(action_probs)
        # 从分布中采样动作
        action = dist.sample()
        # 记录动作、对数概率和状态价值
        return action.item(), dist.log_prob(action), state_value.item()

    def update(self, memory):
        """
        使用存储的经验数据更新策略
        参数:
            memory: 存储经验数据的Memory对象
        """
        # 修复：将zip对象转换为列表后再反转（解决'zip' object is not reversible错误）
        rewards = []
        discounted_reward = 0
        # 先将zip结果转为列表，再反转
        for reward, is_terminal in reversed(list(zip(memory.rewards, memory.is_terminals))):
            if is_terminal:
                discounted_reward = 0  # 如果是终止状态，重置累积奖励
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)  # 插入到列表开头，保持时间顺序

        # 标准化奖励以稳定训练
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 提取经验数据
        old_states = memory.states
        old_actions = torch.tensor(memory.actions, dtype=torch.long).to(self.device)
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32).to(self.device)
        old_state_values = torch.tensor(memory.state_values, dtype=torch.float32).to(self.device)

        # 计算优势函数：实际奖励与估计价值的差异
        advantages = rewards - old_state_values.detach()

        # 多次迭代更新策略，提高样本效率
        for _ in range(self.K_epochs):
            logprobs = []
            state_values = []
            # 对每个存储的状态重新计算动作概率和状态价值
            for state in old_states:
                action_probs, state_val = self.forward(state)
                dist = Categorical(action_probs)
                logprobs.append(dist.log_prob(old_actions))
                state_values.append(state_val)

            # 整理张量形状
            logprobs = torch.stack(logprobs).to(self.device)
            state_values = torch.stack(state_values).squeeze().to(self.device)

            # 计算重要性采样比率：新策略与旧策略的概率比值
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 计算PPO的裁剪目标函数
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()  # 取负号因为要最大化

            # 计算价值损失
            value_loss = 0.5 * self.MseLoss(state_values, rewards)

            # 总损失
            loss = policy_loss + value_loss

            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 更新旧策略网络，使其与当前策略一致
        self.policy_old.load_state_dict(self.policy.state_dict())
