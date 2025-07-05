import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn import HeteroConv, GATConv, global_mean_pool


class USVHeteroGNN(nn.Module):
    def __init__(self, usv_feat_dim=4, task_feat_dim=6, hidden_dim=32, n_heads=4):
        super().__init__()

        # 节点特征转换层
        self.usv_linear = nn.Linear(usv_feat_dim, hidden_dim)
        self.task_linear = nn.Linear(task_feat_dim, hidden_dim)

        # 第一层异构图卷积（显式定义所有边类型）
        self.conv1 = HeteroConv({
            ('usv', 'to', 'task'): GATConv(
                in_channels=(hidden_dim, hidden_dim),
                out_channels=hidden_dim,
                heads=n_heads,
                dropout=0.2,
                add_self_loops=False
            ),
            ('task', 'to', 'task'): GATConv(
                in_channels=(hidden_dim, hidden_dim),
                out_channels=hidden_dim,
                heads=n_heads,
                dropout=0.2,
                add_self_loops=False
            ),
            ('usv', 'to', 'usv'): GATConv(
                in_channels=(hidden_dim, hidden_dim),
                out_channels=hidden_dim,
                heads=n_heads,
                dropout=0.2,
                add_self_loops=True  # 允许USV自环
            )
        })

        # 第二层异构图卷积
        self.conv2 = HeteroConv({
            ('usv', 'to', 'task'): GATConv(
                in_channels=(n_heads * hidden_dim, n_heads * hidden_dim),
                out_channels=hidden_dim,
                heads=1,
                dropout=0.2,
                add_self_loops=False
            ),
            ('task', 'to', 'task'): GATConv(
                in_channels=(n_heads * hidden_dim, n_heads * hidden_dim),
                out_channels=hidden_dim,
                heads=1,
                dropout=0.2,
                add_self_loops=False
            ),
            ('usv', 'to', 'usv'): GATConv(
                in_channels=(n_heads * hidden_dim, n_heads * hidden_dim),
                out_channels=hidden_dim,
                heads=1,
                dropout=0.2,
                add_self_loops=True  # 允许USV自环
            )
        })

        # 最终特征融合层
        self.usv_global = nn.Linear(hidden_dim, hidden_dim)
        self.task_global = nn.Linear(hidden_dim, hidden_dim)

        # 输出维度
        self.output_dim = hidden_dim * 2

    def forward(self, data):
        # 节点特征转换
        x = {}
        x['usv'] = torch.relu(self.usv_linear(data['usv'].x))
        x['task'] = torch.relu(self.task_linear(data['task'].x))
        print(f"[DEBUG] After linear: usv={x['usv'].shape}, task={x['task'].shape}")

        # 第一层异构图卷积（确保所有节点类型都有特征）
        x = self.conv1(x, data.edge_index_dict)
        # 处理可能为None的USV特征
        if 'usv' not in x or x['usv'] is None:
            x['usv'] = x['usv'].new_zeros((data['usv'].x.size(0),
                                           self.conv1['usv', 'to', 'usv'].out_channels * self.conv1[
                                               'usv', 'to', 'usv'].heads))
        x = {k: torch.relu(v) for k, v in x.items()}
        print(f"[DEBUG] After conv1: usv={x['usv'].shape}, task={x['task'].shape}")

        # 第二层异构图卷积
        x = self.conv2(x, data.edge_index_dict)
        x = {k: torch.relu(v) for k, v in x.items()}
        print(f"[DEBUG] After conv2: usv={x['usv'].shape}, task={x['task'].shape}")

        # 全局池化
        usv_emb = global_mean_pool(x['usv'], torch.zeros(x['usv'].size(0), dtype=torch.long, device=x['usv'].device))
        task_emb = global_mean_pool(x['task'],
                                    torch.zeros(x['task'].size(0), dtype=torch.long, device=x['task'].device))
        print(f"[DEBUG] After pooling: usv={usv_emb.shape}, task={task_emb.shape}")

        # 特征融合
        usv_global = torch.relu(self.usv_global(usv_emb))
        task_global = torch.relu(self.task_global(task_emb))

        # 拼接全局表示
        global_state = torch.cat([usv_global, task_global], dim=1).squeeze()
        print(f"[DEBUG] Global state: {global_state.shape}")

        return x, global_state