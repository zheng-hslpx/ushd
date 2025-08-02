import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.state_values = [] # 新增：存储旧的状态价值

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.state_values[:] # 新增：清空旧的状态价值

class ActorCritic(nn.Module):
    def __init__(self, hgnn, action_dim, has_continuous_action_space=False, action_std_init=0.6):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.hgnn = hgnn  # 注入的HGNN模型，用于特征提取
        self.action_dim = action_dim # Store action_dim

        # Actor: 输出动作概率 (分类分布)
        # 假设 HGNN 输出的全局特征维度为 hidden_dim (例如 32)
        # 我们需要知道这个维度来定义 Actor 和 Critic 网络
        # 一个常见的做法是在第一次调用 forward 时动态获取或要求传入
        # 这里我们假设在 PPO 初始化时已经知道 hidden_dim 并传递给了 HGNN
        # 因此，HGNN 的 forward 应该返回一个固定大小的向量
        # 例如，如果 HGNN(hidden_dim=32)，则输出是 [32]
        # self.actor = None # 将在首次 act/evaluate 时定义
        # self.critic = None # 将在首次 act/evaluate 时定义

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            # ... (连续动作空间逻辑不变)
            print("--------------------------------------------------------------------------------------------")
            print("WARNING: set_action_std has no effect. ActorCritic does not use continuous action space.")
            print("--------------------------------------------------------------------------------------------")
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING: set_action_std has no effect. ActorCritic does not use continuous action space.")
            print("--------------------------------------------------------------------------------------------")


    def forward(self):
        raise NotImplementedError

    # def act(self, state, device): # 旧签名
    def act(self, state, device, action_mask=None):  # <-- 新签名，添加 action_mask 参数
        # 通过HGNN提取特征 (假设 state 是一个 DGL 图)
        # 根据修改后的 hgnn.py，self.hgnn(state) 现在返回一个 [hidden_dim] 的张量
        graph_embedding = self.hgnn(state)  # Shape: [hidden_dim]
        # 确保 graph_embedding 在正确的设备上
        graph_embedding = graph_embedding.to(device)
        # 定义 Actor 和 Critic 网络（在首次调用时）
        # 检查是否已定义网络头
        if not hasattr(self, 'actor_head'):
            embedding_dim = graph_embedding.shape[0]  # 获取隐藏层维度
            self.actor_head = nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, self.action_dim)  # 输出与动作空间匹配
            ).to(device)
            self.critic_head = nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            ).to(device)
        # 计算动作概率和状态价值
        action_logits = self.actor_head(graph_embedding)  # Shape: [action_dim]

        # --- 新增：应用 Action Mask ---
        if action_mask is not None:
            # 将 numpy array 转换为 tensor 并移到对应设备
            action_mask_tensor = torch.from_numpy(action_mask).to(device)
            # 应用 mask: 将无效动作的 logits 设为一个极大的负数 (-1e8)
            # 这样 softmax 后，无效动作的概率接近 0
            action_logits = action_logits.masked_fill(~action_mask_tensor, -1e8)
        # --- 新增结束 ---

        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic_head(graph_embedding)  # Shape: [1]
        return action.detach().cpu().item(), action_logprob.detach(), state_val.detach()
        # 返回标量 action (转为 CPU item)，logprob，state_val


    def evaluate(self, states, actions):
        # states 是一个列表，包含多个 DGL 图 (Memory 中存储的)
        # actions 是一个包含动作索引的 tensor
        if not isinstance(states, list):
            raise ValueError("states should be a list of DGL graphs for batch evaluation.")

        # 批量通过 HGNN 处理图
        graph_embeddings = []
        for s in states:
            emb = self.hgnn(s) # 对每个图调用 HGNN，得到 [hidden_dim]
            graph_embeddings.append(emb)
        graph_embeddings = torch.stack(graph_embeddings) # Shape: [batch_size, hidden_dim]
        batch_size = graph_embeddings.shape[0]

        # 确保网络头已定义 (通常在 act 之后已定义)
        device = graph_embeddings.device
        if not hasattr(self, 'actor_head'):
             # Fallback: 如果 evaluate 在 act 之前被调用 (不太可能在PPO流程中)
             # 但这保证了安全性
             embedding_dim = graph_embeddings.shape[1]
             self.actor_head = nn.Sequential(
                 nn.Linear(embedding_dim, 64),
                 nn.Tanh(),
                 nn.Linear(64, 64),
                 nn.Tanh(),
                 nn.Linear(64, self.action_dim)
             ).to(device)
             self.critic_head = nn.Sequential(
                 nn.Linear(embedding_dim, 64),
                 nn.Tanh(),
                 nn.Linear(64, 64),
                 nn.Tanh(),
                 nn.Linear(64, 1)
             ).to(device)

        # 计算动作概率和状态价值
        action_logits = self.actor_head(graph_embeddings) # Shape: [batch_size, action_dim]
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic_head(graph_embeddings) # Shape: [batch_size, 1]

        return action_logprobs, torch.squeeze(state_values, -1), dist_entropy
        # 返回 [batch_size,], [batch_size,], [batch_size,]


class PPO:
    def __init__(self, hgnn, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=10, device='cpu'):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.policy = ActorCritic(hgnn, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        # --- 可选：添加学习率调度器 ---
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)
        # -----------------------------

        self.policy_old = ActorCritic(hgnn, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # --- 损失函数 ---
        self.loss_fn = nn.SmoothL1Loss() # 使用更稳定的损失函数

    def select_action(self, state):

        graph = state  # DGL 图
        action_mask = getattr(graph, 'action_mask', None)

        with torch.no_grad():
            # 传递 action_mask 给 act 方法
            action, action_logprob, state_value = self.policy_old.act(graph, self.device, action_mask)
        # action 已经是 cpu().item()
        return action, action_logprob.cpu(), state_value.cpu()  # 确保返回的 tensor 在 CPU 上


    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        # --- 逆序计算折扣回报 ---
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = reward
            else:
                discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        # -------------------------------

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        # --- 添加数值稳定性 ---
        reward_std = rewards.std()
        if reward_std > 1e-6: # 避免除以零或极小的数
            rewards = (rewards - rewards.mean()) / (reward_std + 1e-8)
        else:
             rewards = rewards - rewards.mean() # 仅中心化
        # -----------------------------

        # Convert list to tensor
        old_states = memory.states # List of DGL graphs
        old_actions = torch.tensor(memory.actions, dtype=torch.long).to(self.device)
        old_logprobs = torch.stack(memory.logprobs).to(self.device)
        # --- 新增：获取旧的状态价值 ---
        old_state_values = torch.stack(memory.state_values).to(self.device).detach() # detach is crucial
        # -------------------------------

        # Optimize policy for K epochs:
        policy_loss_accum = 0.0
        value_loss_accum = 0.0
        for _ in range(self.K_epochs):
            # Evaluating old actions and values:
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            # --- 使用 detach 的旧状态价值计算优势 ---
            advantages = rewards - old_state_values.squeeze(-1) # 确保维度匹配 [batch_size]
            # --------------------------------------------
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # --- 分别计算策略损失和价值损失 ---
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.loss_fn(state_values, rewards) # 使用更稳定的损失函数
            entropy_loss = -0.01 * dist_entropy.mean() # 熵正则化项 (系数可调)

            # Total loss
            loss = policy_loss + 0.5 * value_loss + entropy_loss
            # ------------------------------------

            # Take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            # --- 可选：添加梯度裁剪 ---
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            # -------------------------
            self.optimizer.step()

            # Accumulate losses for averaging
            policy_loss_accum += policy_loss.item()
            value_loss_accum += value_loss.item()

        # --- 可选：更新学习率调度器 ---
        # if hasattr(self, 'scheduler'):
        #     self.scheduler.step()
        # -----------------------------

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear memory
        memory.clear_memory()

        # Return average losses
        avg_policy_loss = policy_loss_accum / self.K_epochs
        avg_value_loss = value_loss_accum / self.K_epochs
        return avg_policy_loss, avg_value_loss # 返回平均损失
