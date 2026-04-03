#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class LOBEncoder(nn.Module):

    def __init__(
        self,
        lob_feature_dim: int,
        lob_history_len: int,
        lob_depth: int,
        lstm_hidden_size: int = 128,
        lstm_input_dropout: float = 0.2,
        lstm_output_dropout: float = 0.0,
        cnn_pair_filters: int = 16,
        cnn_leaky_relu_neg_slope: float = 0.01,
        temporal_kernel: int = 2,
        temporal_repeats: int = 1,
        inception_enabled: bool = True,
        inception_out_channels: int = 32,
        inception_pool_time_kernel: int = 3,
    ):
        super().__init__()
        self.K = lob_history_len
        self.D = int(lob_depth)
        self.F = int(lob_feature_dim)
        self.neg_slope = float(cnn_leaky_relu_neg_slope)

        C = int(cnn_pair_filters)
        self.conv12_a = nn.Conv2d(1, C, kernel_size=(1, 2), stride=(1, 2), padding=(0, 0), bias=True)
        self.conv12_b = nn.Conv2d(C, C, kernel_size=(1, 2), stride=(1, 2), padding=(0, 0), bias=True)
        pad_t = max(0, (temporal_kernel - 1) // 2)
        self.temporal_repeats = 2 if int(temporal_repeats) >= 2 else 1
        def make_temporal_group():
            return nn.ModuleList([
                nn.Conv2d(C, C, kernel_size=(temporal_kernel, 1), stride=(1, 1), padding=(pad_t, 0), bias=True)
                for _ in range(self.temporal_repeats)
            ])
        self.convs_t_after_a = make_temporal_group()
        self.convs_t_after_b = make_temporal_group()
        self.conv_levels = nn.Conv2d(C, C, kernel_size=(1, self.D), stride=(1, 1), padding=(0, 0), bias=True)
        self.convs_t_after_levels = make_temporal_group()

        self.inception_enabled = bool(inception_enabled)
        if self.inception_enabled:
            out_each = max(1, inception_out_channels // 4)
            self.incp_1x1_a = nn.Conv2d(C, out_each, kernel_size=(1, 1), padding=(0, 0))
            self.incp_1x1_b3 = nn.Conv2d(C, out_each, kernel_size=(1, 1), padding=(0, 0))
            self.incp_1x1_b5 = nn.Conv2d(C, out_each, kernel_size=(1, 1), padding=(0, 0))
            self.incp_3x1 = nn.Conv2d(out_each, out_each, kernel_size=(3, 1), padding=(1, 0))
            self.incp_5x1 = nn.Conv2d(out_each, out_each, kernel_size=(5, 1), padding=(2, 0))
            self.incp_pool = nn.MaxPool2d(kernel_size=(inception_pool_time_kernel, 1), stride=(1, 1), padding=(inception_pool_time_kernel // 2, 0))
            self.incp_pool_1x1 = nn.Conv2d(C, out_each, kernel_size=(1, 1), padding=(0, 0))
            self.incp_act = nn.LeakyReLU(self.neg_slope)
            self.incp_out_channels = out_each * 4
            C_lstm_in = self.incp_out_channels
        else:
            C_lstm_in = C

        self.lstm = nn.LSTM(input_size=C_lstm_in, hidden_size=lstm_hidden_size, batch_first=True)
        self.input_dropout = nn.Dropout(lstm_input_dropout)
        self.output_dropout = nn.Dropout(lstm_output_dropout)
        self.act = nn.LeakyReLU(self.neg_slope)

    def forward(self, lob_seq: torch.Tensor) -> torch.Tensor:
        # lob_seq: (B, K, F=4D)
        x = lob_seq.unsqueeze(1)  # (B,1,K,F)
        x = self.act(self.conv12_a(x))      # (B,C,K,2D)
        for conv in self.convs_t_after_a:
            x = self.act(conv(x))           # (B,C,K,2D)
        x = self.act(self.conv12_b(x))      # (B,C,K,D)
        for conv in self.convs_t_after_b:
            x = self.act(conv(x))           # (B,C,K,D)
        x = self.act(self.conv_levels(x))   # (B,C,K,1)
        for conv in self.convs_t_after_levels:
            x = self.act(conv(x))           # (B,C,K,1)

        x = x.squeeze(-1)                   # (B,C,K)
        x = x.transpose(1, 2).contiguous()  # (B,K,C)

        if self.inception_enabled:

            xi = x.transpose(1, 2).unsqueeze(-1)  # (B,C,K,1)
            b1 = self.incp_act(self.incp_1x1_a(xi))
            b2 = self.incp_act(self.incp_3x1(self.incp_1x1_b3(xi)))
            b3 = self.incp_act(self.incp_5x1(self.incp_1x1_b5(xi)))
            b4 = self.incp_act(self.incp_pool_1x1(self.incp_pool(xi)))
            xi = torch.cat([b1, b2, b3, b4], dim=1)  # (B, C', K, 1)
            x = xi.squeeze(-1).transpose(1, 2).contiguous()  # (B, K, C')

        x = self.input_dropout(x)
        out, _ = self.lstm(x)               # (B,K,H)
        feat = out[:, -1, :]                # (B,H)
        return self.output_dropout(feat)


class TradingFeatureEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, trade_vec: torch.Tensor) -> torch.Tensor:
        return self.net(trade_vec)


class SharedStateFusion(nn.Module):
    def __init__(self, lob_hidden: int, trade_hidden: int, fused_dim: int = 256, mlp_dropout: float = 0.1, trade_gate: float = 1.0):
        super().__init__()
        try:
            tg = float(trade_gate)
        except Exception:
            tg = 1.0
        self.trade_gate = max(0.0, min(tg, 1.0))
        
        self.fc = nn.Sequential(
            nn.Linear(lob_hidden + trade_hidden, fused_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
        )

    def forward(self, lob_feat: torch.Tensor, trade_feat: torch.Tensor) -> torch.Tensor:
        if self.trade_gate != 1.0:
            trade_feat = trade_feat * float(self.trade_gate)
        fused = torch.cat([lob_feat, trade_feat], dim=-1)
        return self.fc(fused)


class BDQCore(nn.Module):
    def __init__(
        self,
        lob_feature_dim: int,
        lob_history_len: int,
        lob_depth: int,
        price_branch_dim: int,
        ratio_branch_dim: int,
        lstm_hidden_size: int = 128,
        trade_hidden: int = 128,
        fused_dim: int = 256,
        mlp_dropout: float = 0.1,
        trade_gate: float = 1.0,
        cnn_pair_filters: int = 16,
        cnn_leaky_relu_neg_slope: float = 0.01,
        temporal_kernel: int = 2,
        temporal_repeats: int = 1,
        inception_enabled: bool = True,
        inception_out_channels: int = 32,
        inception_pool_time_kernel: int = 3,
        lstm_input_dropout: float = 0.2,
        lstm_output_dropout: float = 0.0,
    ):
        super().__init__()
        self.price_branch_dim = price_branch_dim
        self.ratio_branch_dim = ratio_branch_dim
        
        try:
            tg = float(trade_gate)
        except Exception:
            tg = 1.0
        self.trade_gate = max(0.0, min(tg, 1.0))

        self.lob_encoder = LOBEncoder(
            lob_feature_dim=lob_feature_dim,
            lob_history_len=lob_history_len,
            lob_depth=lob_depth,
            lstm_hidden_size=lstm_hidden_size,
            lstm_input_dropout=lstm_input_dropout,
            lstm_output_dropout=lstm_output_dropout,
            cnn_pair_filters=cnn_pair_filters,
            cnn_leaky_relu_neg_slope=cnn_leaky_relu_neg_slope,
            temporal_kernel=temporal_kernel,
            temporal_repeats=temporal_repeats,
            inception_enabled=inception_enabled,
            inception_out_channels=inception_out_channels,
            inception_pool_time_kernel=inception_pool_time_kernel,
        )
        self.trade_encoder = TradingFeatureEncoder(trade_hidden)
        self.fusion = SharedStateFusion(lstm_hidden_size, trade_hidden, fused_dim=fused_dim, mlp_dropout=mlp_dropout, trade_gate=self.trade_gate)

        # Dueling: V(s)
        self.value_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Advantage for two branches
        self.adv_price = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Linear(128, price_branch_dim),
        )
        self.adv_ratio = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Linear(128, ratio_branch_dim),
        )

    def forward(self, lob_seq: torch.Tensor, trade_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lob_feat = self.lob_encoder(lob_seq)
        trade_feat = self.trade_encoder(trade_vec)
        shared = self.fusion(lob_feat, trade_feat)

        V = self.value_head(shared)  # (B,1)
        A_p = self.adv_price(shared)  # (B, P)
        A_r = self.adv_ratio(shared)  # (B, R)

        A_p_centered = A_p - A_p.mean(dim=-1, keepdim=True)
        A_r_centered = A_r - A_r.mean(dim=-1, keepdim=True)
        Q_p = V + A_p_centered
        Q_r = V + A_r_centered
        return V, Q_p, Q_r


class BDQPolicyHead(nn.Module):
    def __init__(self, core: BDQCore):
        super().__init__()
        self.core = core

    def forward(self, lob_seq: torch.Tensor, trade_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, q_p, q_r = self.core(lob_seq, trade_vec)
        a_p = torch.argmax(q_p, dim=-1)
        a_r = torch.argmax(q_r, dim=-1)
        return a_p, a_r


