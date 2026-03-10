import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


# 1. 动态门控融合模块 (用于 Ours 模式)
class GatedResidualFusion(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(GatedResidualFusion, self).__init__()
        # 门控生成网络：输入拼接后的特征，输出融合权重 z (0~1)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        # 两个特征的变换层
        self.ltef_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()  # 使用 Tanh 保持数值稳定性
        )
        self.stef_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

    def forward(self, ltef, stef):
        # 计算动态权重 ：越接近 1，模型越信任 LTEF；越接近 0，模型越信任 STEF
        combined = torch.cat([ltef, stef], dim=1)
        z = self.gate_net(combined)

        # 对特征进行非线性变换（防止直接加和导致特征空间冲突）
        ltef_h = self.ltef_transform(ltef)
        stef_h = self.stef_transform(stef)

        # 加权融合
        final = z * ltef_h + (1 - z) * stef_h

        return final


class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs, text_masks):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        weights = torch.bmm(inputs,
                            self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1))
        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)
        masked = attentions * text_masks
        _sums = masked.sum(-1, keepdim=True) + 1e-9
        attentions = masked / _sums
        weighted = inputs * attentions.unsqueeze(-1)
        representations = weighted.sum(1)
        return representations, attentions


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, class_num, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, class_num))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        result = self.mlp(x)
        return result.squeeze(1)


class PostLevel_GRU_Model(nn.Module):
    '''
    支持动态门控残差融合的消融实验模型
    ablation 参数:
    - 'none': Ours (LTEF 主干 + Gate * STEF)
    - 'no_stef': 只保留 LTEF (去掉 GRU 最后状态)
    - 'no_ltef': 只保留 STEF (去掉共性计算)
    - 'mlp_fusion': 传统的拼接后接 MLP 融合
    '''

    def __init__(self, args, device):
        super(PostLevel_GRU_Model, self).__init__()
        self.args = args
        self.device = device
        self.ablation = args.ablation

        self.input_dim = args.embedding_dim
        self.gru_size = args.gru_size
        self.class_num = args.class_num
        self.dropout = args.dropout
        self.hidden_dim = 2 * self.gru_size

        # 提取 STEF (短期/序列末尾) 相关的模块
        if self.ablation in ['none', 'no_ltef', 'mlp_fusion']:
            self.sentence_gru = nn.GRU(
                input_size=self.input_dim,
                hidden_size=self.gru_size,
                num_layers=1,
                bidirectional=True,
                batch_first=True
            )

        # 提取 LTEF (长期/共性) 相关的模块
        if self.ablation in ['none', 'no_stef', 'mlp_fusion']:
            self.ltef_project = nn.Linear(self.input_dim, self.hidden_dim)

        # 融合层
        if self.ablation == 'none':
            self.gated_fusion = GatedResidualFusion(self.hidden_dim, self.dropout)
        elif self.ablation == 'mlp_fusion':
            self.fusion_mlp = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU()
            )

        # 分类器
        # 无论哪种模式，最终都统一到 hidden_dim 维度进行分类
        self.class_fc = MultiLayerPerceptron(
            input_dim=self.hidden_dim,
            embed_dims=[self.gru_size],
            dropout=self.dropout,
            class_num=self.class_num
        )

    def compute_commonality(self, post_vectors, post_masks):
        mask = post_masks.unsqueeze(-1)
        masked_vectors = post_vectors * mask
        lengths = torch.sum(post_masks, dim=1).unsqueeze(-1) + 1e-9
        sum_vec = torch.sum(masked_vectors, dim=1)
        sq_sum_vec = torch.sum(masked_vectors ** 2, dim=1)
        s_pair = (sum_vec ** 2 - sq_sum_vec) / (2 * lengths)
        return torch.nn.LeakyReLU(negative_slope=0.01)(s_pair)

    def forward(self, x, post_masks):
        # 提取 LTEF (长期/共性)
        ltef = None
        if self.ablation in ['none', 'no_stef', 'mlp_fusion']:
            ltef_raw = self.compute_commonality(x, post_masks)
            ltef = self.ltef_project(ltef_raw)

        # 提取 STEF (短期/状态)
        stef = None
        if self.ablation in ['none', 'no_ltef', 'mlp_fusion']:
            post_lengths = post_masks.sum(dim=1).cpu()
            if post_lengths.sum() == 0:
                return torch.zeros(x.size(0), self.class_num).to(self.device)

            packed_input = nn.utils.rnn.pack_padded_sequence(x, post_lengths, batch_first=True, enforce_sorted=False)
            _, hidden = self.sentence_gru(packed_input)
            stef = torch.cat([hidden[-2], hidden[-1]], dim=1)

        # 特征融合
        if self.ablation == 'none':
            # Ours: 门控残差融合 (LTEF 为基准)
            final_features = self.gated_fusion(ltef, stef)

        elif self.ablation == 'no_stef':
            # 消融: 只有长期
            final_features = ltef

        elif self.ablation == 'no_ltef':
            # 消融: 只有短期
            final_features = stef

        elif self.ablation == 'mlp_fusion':
            # 消融: 拼接后 MLP
            combined = torch.cat([ltef, stef], dim=1)
            final_features = self.fusion_mlp(combined)

        # 分类
        logits = self.class_fc(final_features)
        return logits