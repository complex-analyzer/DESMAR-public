#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class MarketCNNLSTMEncoder(nn.Module):

    def __init__(self,
                 num_assets: int,
                 seq_len: int,
                 feature_dim: int,
                 num_attention_heads: int = 8,
                 conv_out_channels: int = 64,
                 conv_kernel_size: int = 1,
                 pool_kernel_size: int = 1,
                 lstm_hidden_size: int = 128,
                 cnn_dropout: float = 0.2):
        super(MarketCNNLSTMEncoder, self).__init__()

        self.num_assets = num_assets
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.lstm_hidden_size = lstm_hidden_size

        # CNN层：核大小1
        self.conv = nn.Conv1d(in_channels=feature_dim, out_channels=conv_out_channels, kernel_size=conv_kernel_size)
        self.conv_act = nn.ReLU()
        self.cnn_dropout = nn.Dropout(p=cnn_dropout)
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size)

        self.lstm = nn.LSTM(input_size=conv_out_channels, hidden_size=lstm_hidden_size, num_layers=1,
                             batch_first=True, bidirectional=False)

        self.cross_asset_attention = MultiHeadAttention(
            feature_dim=lstm_hidden_size,
            num_heads=num_attention_heads,
            dropout=0.1
        )
        self.layer_norm = nn.LayerNorm(lstm_hidden_size)

    def forward(self, market_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            market_features: (batch, assets, seq_len, feature_dim)
        Returns:
            per_asset_features: (batch, assets, 128)
        """
        batch_size, assets, seq_len, feat = market_features.shape
        if not torch.jit.is_scripting():
            assert assets == self.num_assets, f"number of assets mismatch: {assets} vs {self.num_assets}"
            assert seq_len == self.seq_len, f"sequence length mismatch: {seq_len} vs {self.seq_len}"
            assert feat == self.feature_dim, f"feature dimension mismatch: {feat} vs {self.feature_dim}"

        x = market_features.reshape(batch_size * assets, seq_len, feat)  # (B*A, L, F)
        x = x.permute(0, 2, 1)  # (B*A, F, L)
        x = self.conv_act(self.conv(x))  # (B*A, C, L)
        x = self.cnn_dropout(x)
        x = self.pool(x)  # (B*A, 64, L)

        # LSTM requires (B*A, L, C)
        x = x.permute(0, 2, 1)  # (B*A, L, 64)
        lstm_out, _ = self.lstm(x)  # (B*A, L, 128)
        per_asset = lstm_out[:, -1, :]  # (B*A, H)
        per_asset = per_asset.view(batch_size, assets, self.lstm_hidden_size)  # (B, A, H)

        # cross-asset attention + residual
        attn_out = self.cross_asset_attention(per_asset)  # (B, A, 128)
        per_asset_features = self.layer_norm(attn_out + per_asset)

        return per_asset_features

class MultiHeadAttention(nn.Module):
    
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = float(self.head_dim) ** -0.5
        
        assert self.head_dim * num_heads == feature_dim, "feature_dim must be divisible by num_heads"
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: shape (batch_size, assets, feature_dim)
        Returns:
            shape (batch_size, assets, feature_dim)
        """
        batch_size, assets, feature_dim = x.shape
        
        Q = self.query(x).view(batch_size, assets, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, assets, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, assets, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, assets, feature_dim
        )
        
        output = self.out_proj(attention_output)
        return output

class StateFusion(nn.Module):

    def __init__(self, num_assets: int, per_asset_dim: int,
                 market_reduce_dim: int = 256, weight_fc_dim: int = 256):
        super(StateFusion, self).__init__()
        self.num_assets = num_assets
        self.per_asset_dim = per_asset_dim
        self.market_reduce_dim = market_reduce_dim
        self.weight_fc_dim = weight_fc_dim

        self.market_reducer = nn.Sequential(
            nn.Linear(num_assets * per_asset_dim, market_reduce_dim),
            nn.ReLU()
        )
        self.weight_encoder = nn.Sequential(
            nn.Linear(num_assets, weight_fc_dim),
            nn.ReLU()
        )

    @property
    def fused_dim(self) -> int:
        return self.market_reduce_dim + self.weight_fc_dim

    def forward(self, per_asset_features: torch.Tensor, portfolio_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            per_asset_features: (batch, assets, 128)
            portfolio_weights: (batch, assets)
        Returns:
            fused_state: (batch, 512)
        """
        batch_size = per_asset_features.shape[0]
        market_flat = per_asset_features.reshape(batch_size, self.num_assets * self.per_asset_dim)
        market_256 = self.market_reducer(market_flat)
        weights_256 = self.weight_encoder(portfolio_weights)
        fused = torch.cat([market_256, weights_256], dim=-1)
        return fused

class PolicyNetwork(nn.Module):

    def __init__(self,
                 low_freq_feature_dim: int,
                 low_freq_seq_len: int,
                 num_assets: int,
                 fusion_dim: int = 128,
                 hidden_dim: int = 256,
                 min_concentration: float = 0.02,
                 weights_gate: float = 1.0,
                 conv_out_channels: int = 64,
                 conv_kernel_size: int = 1,
                 pool_kernel_size: int = 1,
                 lstm_hidden_size: int = 128,
                 attention_num_heads: int = 8,
                 market_reduce_dim: int = 256,
                 weight_fc_dim: int = 256,
                 mlp_dropout: float = 0.1,
                 cnn_dropout: float = 0.2):
        super(PolicyNetwork, self).__init__()

        self.num_assets = num_assets
        self.min_concentration = float(min_concentration)
        
        try:
            wg = float(weights_gate)
        except Exception:
            wg = 1.0
        self.weights_gate = max(0.0, min(wg, 1.0))

        self.market_encoder = MarketCNNLSTMEncoder(
            num_assets=num_assets,
            seq_len=low_freq_seq_len,
            feature_dim=low_freq_feature_dim,
            num_attention_heads=attention_num_heads,
            conv_out_channels=conv_out_channels,
            conv_kernel_size=conv_kernel_size,
            pool_kernel_size=pool_kernel_size,
            lstm_hidden_size=lstm_hidden_size,
            cnn_dropout=cnn_dropout
        )

        self.state_fusion = StateFusion(
            num_assets=num_assets,
            per_asset_dim=lstm_hidden_size,
            market_reduce_dim=market_reduce_dim,
            weight_fc_dim=weight_fc_dim
        )

        self.policy_head = nn.Sequential(
            nn.Linear(self.state_fusion.fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout)
        )

        self.alpha_layer = nn.Linear(hidden_dim, num_assets)

    def forward(self, state: Dict[str, torch.Tensor], deterministic: bool = False):
        """
        Args:
            state: {
              'market_features': (batch, assets, seq_len, features),
              'portfolio_weights': (batch, assets)
            }
        Returns:
            action, log_prob, mean
        """
        market = state['market_features']
        weights = state['portfolio_weights']

        per_asset_features = self.market_encoder(market)  # (B, A, 128)
        batch_size = per_asset_features.shape[0]
        market_flat = per_asset_features.reshape(batch_size, self.state_fusion.num_assets * self.state_fusion.per_asset_dim)
        market_256 = self.state_fusion.market_reducer(market_flat)
        weights_256 = self.state_fusion.weight_encoder(weights)
        
        if self.weights_gate != 1.0:
            weights_256 = weights_256 * float(self.weights_gate)
            
        fused_state = torch.cat([market_256, weights_256], dim=-1)

        policy_features = self.policy_head(fused_state)
        alpha_unconstrained = self.alpha_layer(policy_features)
        concentration = F.softplus(alpha_unconstrained) + self.min_concentration

        if deterministic:
            action = concentration / concentration.sum(dim=-1, keepdim=True)
            log_prob = None
        else:
            dist = torch.distributions.Dirichlet(concentration)
            action = dist.rsample()
            log_prob = dist.log_prob(action).unsqueeze(-1)  # (B,1)

        eps = getattr(self, 'sampling_eps', 1e-6)
        action = (1.0 - eps) * action + eps * (1.0 / self.num_assets)

        return action, log_prob, concentration

    @torch.jit.export
    def forward_infer(self, market_features: torch.Tensor, portfolio_weights: torch.Tensor) -> torch.Tensor:

        per_asset_features = self.market_encoder(market_features)
        batch_size = per_asset_features.shape[0]
        market_flat = per_asset_features.reshape(batch_size, self.state_fusion.num_assets * self.state_fusion.per_asset_dim)
        market_256 = self.state_fusion.market_reducer(market_flat)
        weights_256 = self.state_fusion.weight_encoder(portfolio_weights)
        
        if self.weights_gate != 1.0:
            weights_256 = weights_256 * float(self.weights_gate)
            
        fused_state = torch.cat([market_256, weights_256], dim=-1)
        policy_features = self.policy_head(fused_state)
        alpha_unconstrained = self.alpha_layer(policy_features)
        concentration = F.softplus(alpha_unconstrained) + self.min_concentration
        action = concentration / concentration.sum(dim=-1, keepdim=True)
        eps = getattr(self, 'sampling_eps', 1e-6)
        action = (1.0 - eps) * action + eps * (1.0 / self.num_assets)
        return action

class QNetwork(nn.Module):

    def __init__(self,
                 low_freq_feature_dim: int,
                 low_freq_seq_len: int,
                 num_assets: int,
                 fusion_dim: int = 128,
                 hidden_dim: int = 256,
                 weights_gate: float = 1.0,
                 conv_out_channels: int = 64,
                 conv_kernel_size: int = 1,
                 pool_kernel_size: int = 1,
                 lstm_hidden_size: int = 128,
                 attention_num_heads: int = 8,
                 market_reduce_dim: int = 256,
                 weight_fc_dim: int = 256,
                 mlp_dropout: float = 0.1,
                 cnn_dropout: float = 0.2):
        super(QNetwork, self).__init__()

        self.num_assets = num_assets
        
        try:
            wg = float(weights_gate)
        except Exception:
            wg = 1.0
        self.weights_gate = max(0.0, min(wg, 1.0))

        self.market_encoder = MarketCNNLSTMEncoder(
            num_assets=num_assets,
            seq_len=low_freq_seq_len,
            feature_dim=low_freq_feature_dim,
            num_attention_heads=attention_num_heads,
            conv_out_channels=conv_out_channels,
            conv_kernel_size=conv_kernel_size,
            pool_kernel_size=pool_kernel_size,
            lstm_hidden_size=lstm_hidden_size,
            cnn_dropout=cnn_dropout
        )

        self.state_fusion = StateFusion(
            num_assets=num_assets,
            per_asset_dim=lstm_hidden_size,
            market_reduce_dim=market_reduce_dim,
            weight_fc_dim=weight_fc_dim
        )

        self.q_network = nn.Sequential(
            nn.Linear(self.state_fusion.fused_dim + num_assets, hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: Dict[str, torch.Tensor], action: torch.Tensor):
        market = state['market_features']
        weights = state['portfolio_weights']
        per_asset_features = self.market_encoder(market)
        
        batch_size = per_asset_features.shape[0]
        market_flat = per_asset_features.reshape(batch_size, self.state_fusion.num_assets * self.state_fusion.per_asset_dim)
        market_256 = self.state_fusion.market_reducer(market_flat)
        weights_256 = self.state_fusion.weight_encoder(weights)
        
        if self.weights_gate != 1.0:
            weights_256 = weights_256 * float(self.weights_gate)
            
        fused_state = torch.cat([market_256, weights_256], dim=-1)
        q_input = torch.cat([fused_state, action], dim=-1)
        q_value = self.q_network(q_input)
        return q_value
