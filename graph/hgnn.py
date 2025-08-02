import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn

class USVHeteroGNN(nn.Module):
    def __init__(self, usv_feat_dim, task_feat_dim, hidden_dim, n_heads, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.feat_per_head = hidden_dim // n_heads if hidden_dim % n_heads == 0 else hidden_dim

        # 输入特征映射
        self.usv_encoder = nn.Linear(usv_feat_dim, hidden_dim)
        self.task_encoder = nn.Linear(task_feat_dim, hidden_dim)

        # 图中实际边类型三元组 (根据您的描述)
        self.actual_etypes = [('task', 'to', 'task'), ('usv', 'to', 'task')]

        # 多层GAT
        self.gat_layers = nn.ModuleList()
        # 用于在GAT后将特征维度恢复/投影到 hidden_dim
        self.post_conv_layers = nn.ModuleList()

        for _ in range(num_layers):
            # GAT卷积层 (注意 in_feats 和 out_feats 的设置)
            conv_dict = {
                etype: dglnn.GATConv(
                    in_feats=(hidden_dim, hidden_dim), # 输入特征维度
                    out_feats=self.feat_per_head,      # *每个注意力头*的输出特征维度
                    num_heads=n_heads,
                    allow_zero_in_degree=True
                ) for etype in self.actual_etypes
            }
            self.gat_layers.append(dglnn.HeteroGraphConv(conv_dict, aggregate='sum'))

            # 线性层：将 GAT 多头输出 (N, n_heads, feat_per_head) 映射回 (N, hidden_dim)
            # 通常在 GAT 后会做类似操作，这里显式添加一层线性变换确保维度
            # self.post_conv_layers.append(nn.Linear(n_heads * self.feat_per_head, hidden_dim))
            # 或者，如果 GATConv 的 out_feats 是 feat_per_head，且 num_heads=n_heads,
            # 它的输出是 (N, n_heads, feat_per_head)。我们先 reshape，再线性变换。
            # 更简洁的方式是直接让 post_conv_layer 处理平均后的结果 (N, feat_per_head)
            # 并将其映射回 (N, hidden_dim)。
            # 为了与原始逻辑匹配并简化，我们假设 GAT 输出 (N, n_heads, feat_per_head)
            # 我们在 post_conv_layer 中处理。
            # 修正：GATConv out_feats=feat_per_head, num_heads=n_heads -> 输出 (N, n_heads, feat_per_head)
            # 平均多头: (N, n_heads, feat_per_head) -> (N, feat_per_head)
            # 线性变换: (N, feat_per_head) -> (N, hidden_dim)
            # 但是 feat_per_head = hidden_dim // n_heads, 所以维度会变小。
            # 更合理的做法是指定 GATConv 输出 (N, n_heads, out_feat_per_head)
            # 使得 n_heads * out_feat_per_head = hidden_dim
            # 让我们重新定义 feat_per_head
            self.post_conv_layers.append(nn.Linear(self.feat_per_head, hidden_dim))


        # 输出层 (用于生成最终的节点嵌入)
        self.task_decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.usv_decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        # 全局池化层，用于生成图的全局表示
        # 输入是 USV 和 Task 全局特征的拼接
        self.global_pool = nn.Linear(hidden_dim * 2, hidden_dim)


    def forward(self, graph):
        """
        前向传播。
        Args:
            graph (dgl.DGLGraph): 输入的异构图。
        Returns:
            torch.Tensor: 图的全局嵌入向量 (1D tensor of shape [hidden_dim]).
        """
        # 1. 编码原始特征
        usv_h = self.usv_encoder(graph.nodes['usv'].data['feat'])  # [num_usvs, hidden_dim]
        task_h = self.task_encoder(graph.nodes['task'].data['feat'])  # [num_tasks, hidden_dim]
        h = {'usv': usv_h, 'task': task_h}

        # 2. 多层GAT传播
        for layer_idx in range(self.num_layers):
            # GAT卷积
            h_gat = self.gat_layers[layer_idx](graph, h)
            # 处理GAT输出 (通常需要合并多头注意力)
            h_merged = {}
            if 'task' in h_gat:
                # h_gat['task'] shape: [num_tasks, n_heads, feat_per_head]
                # 平均多头输出
                h_merged['task'] = h_gat['task'].mean(dim=1) # [num_tasks, feat_per_head]
            if 'usv' in h_gat:
                # h_gat['usv'] shape: [num_usvs, n_heads, feat_per_head]
                h_merged['usv'] = h_gat['usv'].mean(dim=1) # [num_usvs, feat_per_head]

            # 通过线性层将特征维度映射回 hidden_dim
            h = {}
            if 'task' in h_merged:
                h['task'] = self.post_conv_layers[layer_idx](h_merged['task']) # [num_tasks, hidden_dim]
            if 'usv' in h_merged:
                h['usv'] = self.post_conv_layers[layer_idx](h_merged['usv']) # [num_usvs, hidden_dim]

        # 3. 解码节点特征 (可选，但保持结构完整性)
        usv_emb = self.usv_decoder(h['usv']) if 'usv' in h else usv_h
        task_emb = self.task_decoder(h['task']) if 'task' in h else task_h

        # 4. 全局池化，生成图的单一特征向量
        # 计算 USV 节点的全局表示 (例如，平均)
        global_usv = torch.mean(usv_emb, dim=0, keepdim=True) # [1, hidden_dim]
        # 计算 Task 节点的全局表示
        global_task = torch.mean(task_emb, dim=0, keepdim=True) # [1, hidden_dim]
        # 拼接并再次线性变换得到最终的全局图嵌入
        global_emb = self.global_pool(torch.cat([global_usv, global_task], dim=1)) # [1, hidden_dim]
        # 移除批次维度，返回一维向量
        global_emb = global_emb.squeeze(0) # [hidden_dim]

        # 返回全局嵌入向量，供 PPO 使用
        return global_emb
