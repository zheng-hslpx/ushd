import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, GATv2Conv


class Memory:
    def __init__(self):
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        # 其他属性...

    def clear_memory(self):
        # 清空内存...
        pass


class ActorCritic(nn.Module):
    def __init__(self, hgnn, action_dim, hidden_dim=128, device=None):
        super().__init__()
        self.hgnn = hgnn.to(device) if device else hgnn
        self.fc = nn.Linear(hgnn.output_dim, hidden_dim).to(device)

        # 策略网络（Actor）
        self.actor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        ).to(device)

        # 价值网络（Critic）
        self.critic = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)

        self.device = device if device else torch.device("cpu")

    def forward(self, data):
        data = data.to(self.device)
        _, global_state = self.hgnn(data)
        x = torch.relu(self.fc(global_state))
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value


class PPO:
    def __init__(self, hgnn, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=10, device=None):
        self.device = device if device else torch.device("cpu")
        self.policy = ActorCritic(hgnn, action_dim, device=device)
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()

        # 超参数
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

    def select_action(self, data):
        with torch.no_grad():
            action_probs, _ = self.policy(data)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def _compute_returns(self, rewards, dones):
        """计算折扣回报（避免原地操作）"""
        returns = []
        n = len(rewards)

        for i in range(n):
            R = 0
            for j in range(i, n):
                R = R * self.gamma + rewards[j]
                if dones[j]:
                    break
            returns.append(R)

        # 归一化回报（避免原地操作）
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(-1)
        returns_mean = returns_tensor.mean()
        returns_std = returns_tensor.std() + 1e-5
        normalized_returns = (returns_tensor - returns_mean) / returns_std
        return normalized_returns

    def update(self, transitions, device):
        """更新策略网络（修复计算图和原地操作问题）"""
        # 解析过渡数据
        states, actions, log_probs, rewards, next_states, dones = zip(*transitions)

        # 计算折扣回报
        returns = self._compute_returns(rewards, dones)

        # 转换为张量并移至正确设备
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        old_log_probs = torch.stack(log_probs).to(device)

        # PPO更新
        for _ in range(self.K_epochs):
            # 每次迭代都重新计算前向传播，避免使用旧的计算图
            action_probs_list = []
            state_values_list = []
            for state in states:
                state = state.to(device)
                ap, sv = self.policy(state)
                action_probs_list.append(ap)
                state_values_list.append(sv)

            action_probs = torch.stack(action_probs_list)
            state_values = torch.stack(state_values_list)

            # 计算概率比
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # 计算优势函数
            advantages = returns - state_values.detach()

            # PPO损失函数
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(state_values, returns)

            # 合并损失
            loss = actor_loss + 0.5 * critic_loss

            # 反向传播（不需要retain_graph=True，因为每次迭代都重新计算前向传播）
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 复制新策略到旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())

        return loss.item()