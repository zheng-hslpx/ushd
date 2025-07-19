import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn

class USVHeteroGNN(nn.Module):
    def __init__(self, usv_feat_dim, task_feat_dim, hidden_dim, n_heads, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim  # 目标维度：32
        self.n_heads = n_heads        # 注意力头数：4
        self.feat_per_head = hidden_dim // n_heads  # 每个头的特征维度：8

        # 输入特征映射：将原始特征映射到hidden_dim（32）
        self.usv_encoder = nn.Linear(usv_feat_dim, hidden_dim)  # 4→32
        self.task_encoder = nn.Linear(task_feat_dim, hidden_dim)  # 6→32

        # 图中实际边类型三元组
        self.actual_etypes = [('task', 'to', 'task'), ('usv', 'to', 'task')]

        # 多层GAT：每层后添加线性层，确保输出维度保持32
        self.gat_layers = nn.ModuleList()
        self.post_conv_layers = nn.ModuleList()  # 用于恢复维度的线性层

        for _ in range(num_layers):
            # 1. GAT卷积层（输出维度：[num_nodes, n_heads, feat_per_head] → [N,4,8]）
            conv_dict = {
                etype: dglnn.GATConv(
                    in_feats=(hidden_dim, hidden_dim),  # 输入32维
                    out_feats=self.feat_per_head,       # 每个头输出8维
                    num_heads=n_heads,
                    allow_zero_in_degree=True
                ) for etype in self.actual_etypes
            }
            self.gat_layers.append(dglnn.HeteroGraphConv(conv_dict, aggregate='sum'))

            # 2. 线性层：将多头合并后的8维恢复为32维（关键修复）
            self.post_conv_layers.append(nn.Linear(self.feat_per_head, hidden_dim))

        # 输出层
        self.task_decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.usv_decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.global_pool = nn.Linear(hidden_dim * 2, hidden_dim * 2)

    def forward(self, graph):
        # 1. 编码原始特征（4→32，6→32）
        usv_h = self.usv_encoder(graph.nodes['usv'].data['feat'])  # [3,32]
        task_h = self.task_encoder(graph.nodes['task'].data['feat'])  # [10,32]
        h = {'usv': usv_h, 'task': task_h}

        # 2. 多层传播：确保每层输出维度保持32
        for layer_idx in range(self.num_layers):
            # 2.1 GAT卷积（输出：[N,4,8]）
            h_gat = self.gat_layers[layer_idx](graph, h)

            # 2.2 合并多头注意力（[N,4,8] → [N,8]）
            h_merged = {}
            if 'task' in h_gat:
                h_merged['task'] = h_gat['task'].mean(dim=1)  # [10,8]
            if 'usv' in h_gat:
                h_merged['usv'] = h_gat['usv'].mean(dim=1)  # [3,8]

            # 2.3 线性层恢复维度（8→32，关键修复）
            h = {}
            if 'task' in h_merged:
                h['task'] = self.post_conv_layers[layer_idx](h_merged['task'])  # [10,32]
                print(f"第{layer_idx}层后任务特征维度: {h['task'].shape}")
            if 'usv' in h_merged:
                h['usv'] = self.post_conv_layers[layer_idx](h_merged['usv'])  # [3,32]
                print(f"第{layer_idx}层后USV特征维度: {h['usv'].shape}")

        # 3. 解码与全局池化
        usv_emb = self.usv_decoder(h['usv']) if 'usv' in h else usv_h
        task_emb = self.task_decoder(h['task']) if 'task' in h else task_h

        global_usv = torch.mean(usv_emb, dim=0)
        global_task = torch.mean(task_emb, dim=0)
        global_emb = self.global_pool(torch.cat([global_usv, global_task], dim=0))

        return usv_emb, task_emb, global_emb
