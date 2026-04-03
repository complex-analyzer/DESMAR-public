#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
import logging
import yaml
import random
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import glob

from rl_modules.policy_networks import PolicyNetwork, QNetwork
from rl_modules.experience_replay import ExperienceReplayBuffer

def setup_sac_logger():
    log_dir = os.path.join(os.path.dirname(__file__), 'SAC_Portfolio_Allocationl', 'run_log')
    os.makedirs(log_dir, exist_ok=True)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    log_file = os.path.join(log_dir, 'sac_agent.log')
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_sac_logger()

class SACAgent:
    def __init__(self, 
                 low_freq_feature_dim: int, 
                 low_freq_seq_len: int,
                 num_assets: int,
                 fusion_dim_heterogeneous: int = 131,
                 fusion_dim_unified: int = 128,
                 hidden_dim: int = 256,
                 alpha: float = 0.2,
                 alpha_min: float = 0.05,
                 gamma: float = 0.98,
                 tau: float = 0.005,
                 auto_alpha: bool = False,
                 target_entropy: Optional[float] = None,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4,
                 lr_alpha: float = 3e-4,
                 replay_buffer_size: int = 100000,
                 batch_size: int = 32,
                 device: str = 'auto',
                 seed: Optional[int] = None,
                 training_enabled: bool = True,
                 deterministic_policy: bool = False,
                 train_frequency: int = 8,
                 train_steps_per_update: int = 2,
                 min_experiences_before_training: int = 64,
                 entropy_logging_enabled: bool = True,
                 prioritized_replay_enabled: bool = False,
                 prioritized_replay_half_life_days: float = 3.0,
                 prioritized_replay_uniform_mix: float = 0.1,
                 gradient_clipping_enabled: bool = True,
                 early_clip_threshold: int = 2000,
                 middle_clip_threshold: int = 5000,
                 early_clip_norm: float = 5.0,
                 middle_clip_norm: float = 2.0,
                 late_clip_norm: float = 1.5,
                 q_value_clamp_min: float = -1000.0,
                 q_value_clamp_max: float = 1000.0,
                 reward_scaling_enabled: bool = True,
                 reward_scaler_warmup_experiences: int = 1000,
                 reward_scale_method: str = 'q95',  # 'std' | 'q95' | 'iqr'
                 reward_clip_value: float = 20.0,
                 reward_min_c: float = 1e-6,
                 **network_kwargs):
        
        self.fusion_dim = fusion_dim_unified
        self.auto_alpha = auto_alpha
        self.alpha = alpha
        self.alpha_min = float(alpha_min)
        self.gamma = gamma
        
        self.low_freq_feature_dim = low_freq_feature_dim
        self.high_freq_feature_dim = 0
        self.high_freq_seq_len = 0
        self.low_freq_seq_len = low_freq_seq_len
        self.num_assets = num_assets
        self.hidden_dim = hidden_dim
        self.tau = tau
        
        if target_entropy is None:
            mode = network_kwargs.get('target_entropy_mode', None)
            if isinstance(mode, str):
                m = mode.strip().lower()
                if m == 'sqrt_k':
                    self.target_entropy = -float(np.sqrt(self.num_assets))
                elif m == 'k_minus_1':
                    self.target_entropy = -(self.num_assets - 1)
                else:
                    self.target_entropy = -(self.num_assets - 1)
            else:
                self.target_entropy = -(self.num_assets - 1)
        else:
            self.target_entropy = target_entropy
        
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_alpha = lr_alpha
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.training_enabled = training_enabled
        self.deterministic_policy = deterministic_policy
        self.train_frequency = train_frequency
        self.train_steps_per_update = train_steps_per_update
        self.min_experiences_before_training = min_experiences_before_training
        self.entropy_logging_enabled = entropy_logging_enabled
        self.prioritized_replay_enabled = prioritized_replay_enabled
        self.prioritized_replay_half_life_days = prioritized_replay_half_life_days
        self.prioritized_replay_uniform_mix = prioritized_replay_uniform_mix
        
        self.gradient_clipping_enabled = gradient_clipping_enabled
        self.early_clip_threshold = early_clip_threshold
        self.middle_clip_threshold = middle_clip_threshold
        self.early_clip_norm = early_clip_norm
        self.middle_clip_norm = middle_clip_norm
        self.late_clip_norm = late_clip_norm
        self.q_value_clamp_min = q_value_clamp_min
        self.q_value_clamp_max = q_value_clamp_max

        self.reward_scaling_enabled = bool(reward_scaling_enabled)
        self.reward_scaler_warmup_experiences = int(reward_scaler_warmup_experiences)
        self.reward_scale_method = str(reward_scale_method).lower().strip()
        if self.reward_scale_method not in {'std', 'q95', 'iqr'}:
            self.reward_scale_method = 'std'
        self.reward_clip_value = float(reward_clip_value)
        self.reward_min_c = float(reward_min_c)
        self._reward_c = 1.0
        self._reward_scaler_ready = False

        self.network_cfg = {
            'conv_out_channels': network_kwargs.get('conv_out_channels', 64),
            'conv_kernel_size': network_kwargs.get('conv_kernel_size', 1),
            'pool_kernel_size': network_kwargs.get('pool_kernel_size', 1),
            'lstm_hidden_size': network_kwargs.get('lstm_hidden_size', 128),
            'attention_num_heads': network_kwargs.get('attention_num_heads', 8),
            'market_reduce_dim': network_kwargs.get('market_reduce_dim', 256),
            'weight_fc_dim': network_kwargs.get('weight_fc_dim', 256),
            'mlp_dropout': network_kwargs.get('mlp_dropout', 0.1),
            'cnn_dropout': network_kwargs.get('cnn_dropout', 0.2),
            'actor_weights_gate': float(network_kwargs.get('actor_weights_gate', 1.0)),
            'dirichlet_min_concentration': network_kwargs.get('dirichlet_min_concentration', 0.02),
            'sampling_eps': network_kwargs.get('sampling_eps', 1e-6),
        }
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.seed = seed
        if seed is not None:
            try:
                random.seed(int(seed))
            except Exception:
                pass
            try:
                np.random.seed(int(seed))
            except Exception:
                pass
            try:
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))
            except Exception:
                pass
            
        self._initialize_networks()
        
        self.training_step = 0
        self.total_training_steps = 0
        self.experience_count = 0
        self.last_training_count = 0
        
        import time
        self.session_timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        self.history_replay_buffer_days = 10
        
        self.performance_stats = {
            'action_generation_times': [],
            'experience_processing_times': []
        }
        
        self.entropy_records = []
        
        self.heterogeneous_params = None
        
        logger.info(f"SAC agent initialized - SAC - random seed: {self.seed}, fusion dimension: {self.fusion_dim}, auto_alpha: {self.auto_alpha}, device: {self.device}")
        logger.info(f"entropy recording switch: {self.entropy_logging_enabled}")
        logger.info(f"prioritized replay(recent first): enabled={self.prioritized_replay_enabled}, half_life_days={self.prioritized_replay_half_life_days}, uniform_mix={self.prioritized_replay_uniform_mix}")
        if self.reward_scaling_enabled:
            logger.info(
                f"reward scaling: enabled | method={self.reward_scale_method} | initialized once (no half-life) | warmup experiences={self.reward_scaler_warmup_experiences} | clipping=±{self.reward_clip_value}"
            )
        else:
            logger.info("reward scaling: disabled")
        self._scripted_actor_cache = None
        
    @classmethod
    def from_config(cls, config_path: str = None, **kwargs):
        import yaml
        
        if config_path is None:
            config_path = "Simulator_configs/config_templates/sac_config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        params = {
            'fusion_dim_unified': config.get('fusion_dim', 128),
            'alpha': config.get('alpha', 0.2),
            'alpha_min': float(config.get('alpha_min', 0.05)),
            'gamma': config.get('gamma', 0.98),
            'auto_alpha': config.get('auto_alpha', True),
            'target_entropy': config.get('target_entropy', None),
            'target_entropy_mode': config.get('target_entropy_mode', None),
            'conv_out_channels': config.get('conv_out_channels', 64),
            'conv_kernel_size': config.get('conv_kernel_size', 1),
            'pool_kernel_size': config.get('pool_kernel_size', 1),
            'lstm_hidden_size': config.get('lstm_hidden_size', 128),
            'attention_num_heads': config.get('attention_num_heads', 8),
            'market_reduce_dim': config.get('market_reduce_dim', 256),
            'weight_fc_dim': config.get('weight_fc_dim', 256),
            'mlp_dropout': config.get('mlp_dropout', 0.1),
            'cnn_dropout': config.get('cnn_dropout', 0.2),
            'actor_weights_gate': float(config.get('actor_weights_gate', 1.0)),
            'dirichlet_min_concentration': float(config.get('dirichlet_min_concentration', 0.02)),
            'sampling_eps': float(config.get('sampling_eps', 1e-6)),
        }
        
        gradient_clipping = config.get('gradient_clipping', {})
        
        base_params = {
            'low_freq_feature_dim': config.get('low_freq_feature_dim', 5),
            'low_freq_seq_len': config.get('low_freq_seq_len', 20),
            'num_assets': config.get('num_assets', 4),
            'hidden_dim': config.get('hidden_dim', 256),
            'tau': config.get('tau', 0.005),
            'lr_actor': config.get('lr_actor', 3e-4),
            'lr_critic': config.get('lr_critic', 3e-4),
            'lr_alpha': config.get('lr_alpha', 3e-4),
            'replay_buffer_size': config.get('replay_buffer_size', 100000),
            'batch_size': config.get('batch_size', 32),
            'device': config.get('device', 'auto'),
            'seed': config.get('seed'),
            'training_enabled': config.get('training_enabled', True),
            'deterministic_policy': config.get('deterministic_policy', False),
            'train_frequency': config.get('train_frequency', 8),
            'train_steps_per_update': config.get('train_steps_per_update', 2),
            'min_experiences_before_training': config.get('min_experiences_before_training', 64),
            'entropy_logging_enabled': config.get('entropy_logging_enabled', True),
            'prioritized_replay_enabled': config.get('prioritized_replay_enabled', False),
            'prioritized_replay_half_life_days': float(config.get('prioritized_replay_half_life_days', 3.0)),
            'prioritized_replay_uniform_mix': float(config.get('prioritized_replay_uniform_mix', 0.1)),
            'gradient_clipping_enabled': gradient_clipping.get('enabled', True),
            'early_clip_threshold': gradient_clipping.get('early_threshold', 2000),
            'middle_clip_threshold': gradient_clipping.get('middle_threshold', 5000),
            'early_clip_norm': gradient_clipping.get('early_clip_norm', 5.0),
            'middle_clip_norm': gradient_clipping.get('middle_clip_norm', 2.0),
            'late_clip_norm': gradient_clipping.get('late_clip_norm', 1.5),
            'q_value_clamp_min': gradient_clipping.get('q_value_clamp_min', -1000.0),
            'q_value_clamp_max': gradient_clipping.get('q_value_clamp_max', 1000.0),
            'reward_scaling_enabled': config.get('reward_scaling_enabled', True),
            'reward_scaler_warmup_experiences': int(config.get('reward_scaler_warmup_experiences', 1000)),
            'reward_scale_method': str(config.get('reward_scale_method', 'q95')),
            'reward_clip_value': float(config.get('reward_clip_value', 20.0)),
            'reward_min_c': float(config.get('reward_min_c', 1e-6)),
        }
        
        final_params = {**base_params, **params, **kwargs}
        
        logger.info(f"create SAC agent from config file: {config_path}")
        logger.info("mode: classic SAC")
        
        agent = cls(**final_params)
        
        agent.history_replay_buffer_days = config.get('history_replay_buffer_days', 10)
        
        logger.info(f"history replay buffer days: {agent.history_replay_buffer_days}")
        
        agent._auto_load_latest_model()
        if agent.training_enabled:
            agent._auto_load_recent_replay_buffers()
        else:
            logger.info("inference mode: skip loading history replay buffer")
        
        return agent
    
    def _initialize_networks(self):
        self.actor = PolicyNetwork(
            low_freq_feature_dim=self.low_freq_feature_dim,
            low_freq_seq_len=self.low_freq_seq_len,
            num_assets=self.num_assets,
            fusion_dim=self.fusion_dim,
            hidden_dim=self.hidden_dim,
            min_concentration=self.network_cfg['dirichlet_min_concentration'],
            weights_gate=self.network_cfg['actor_weights_gate'],
            conv_out_channels=self.network_cfg['conv_out_channels'],
            conv_kernel_size=self.network_cfg['conv_kernel_size'],
            pool_kernel_size=self.network_cfg['pool_kernel_size'],
            lstm_hidden_size=self.network_cfg['lstm_hidden_size'],
            attention_num_heads=self.network_cfg['attention_num_heads'],
            market_reduce_dim=self.network_cfg['market_reduce_dim'],
            weight_fc_dim=self.network_cfg['weight_fc_dim'],
            mlp_dropout=self.network_cfg['mlp_dropout'],
            cnn_dropout=self.network_cfg['cnn_dropout']
        ).to(self.device)
        try:
            # set attribute for runtime use
            setattr(self.actor, 'sampling_eps', float(self.network_cfg.get('sampling_eps', 1e-6)))
        except Exception:
            pass
        try:
            weights_gate = float(getattr(self.actor, 'weights_gate', 1.0))
            logger.info(f"Actor portfolio feature weight gate: {weights_gate:.3f} (weight ratio of portfolio feature to market feature, 1.0=equal weight)")
        except Exception:
            pass
        
        self.critic1 = QNetwork(
            low_freq_feature_dim=self.low_freq_feature_dim,
            low_freq_seq_len=self.low_freq_seq_len,
            num_assets=self.num_assets,
            fusion_dim=self.fusion_dim,
            hidden_dim=self.hidden_dim,
            weights_gate=self.network_cfg['actor_weights_gate'],
            conv_out_channels=self.network_cfg['conv_out_channels'],
            conv_kernel_size=self.network_cfg['conv_kernel_size'],
            pool_kernel_size=self.network_cfg['pool_kernel_size'],
            lstm_hidden_size=self.network_cfg['lstm_hidden_size'],
            attention_num_heads=self.network_cfg['attention_num_heads'],
            market_reduce_dim=self.network_cfg['market_reduce_dim'],
            weight_fc_dim=self.network_cfg['weight_fc_dim'],
            mlp_dropout=self.network_cfg['mlp_dropout'],
            cnn_dropout=self.network_cfg['cnn_dropout']
        ).to(self.device)
        
        self.critic2 = QNetwork(
            low_freq_feature_dim=self.low_freq_feature_dim,
            low_freq_seq_len=self.low_freq_seq_len,
            num_assets=self.num_assets,
            fusion_dim=self.fusion_dim,
            hidden_dim=self.hidden_dim,
            weights_gate=self.network_cfg['actor_weights_gate'],
            conv_out_channels=self.network_cfg['conv_out_channels'],
            conv_kernel_size=self.network_cfg['conv_kernel_size'],
            pool_kernel_size=self.network_cfg['pool_kernel_size'],
            lstm_hidden_size=self.network_cfg['lstm_hidden_size'],
            attention_num_heads=self.network_cfg['attention_num_heads'],
            market_reduce_dim=self.network_cfg['market_reduce_dim'],
            weight_fc_dim=self.network_cfg['weight_fc_dim'],
            mlp_dropout=self.network_cfg['mlp_dropout'],
            cnn_dropout=self.network_cfg['cnn_dropout']
        ).to(self.device)
        
        self.target_critic1 = QNetwork(
            low_freq_feature_dim=self.low_freq_feature_dim,
            low_freq_seq_len=self.low_freq_seq_len,
            num_assets=self.num_assets,
            fusion_dim=self.fusion_dim,
            hidden_dim=self.hidden_dim,
            weights_gate=self.network_cfg['actor_weights_gate'],
            conv_out_channels=self.network_cfg['conv_out_channels'],
            conv_kernel_size=self.network_cfg['conv_kernel_size'],
            pool_kernel_size=self.network_cfg['pool_kernel_size'],
            lstm_hidden_size=self.network_cfg['lstm_hidden_size'],
            attention_num_heads=self.network_cfg['attention_num_heads'],
            market_reduce_dim=self.network_cfg['market_reduce_dim'],
            weight_fc_dim=self.network_cfg['weight_fc_dim'],
            mlp_dropout=self.network_cfg['mlp_dropout'],
            cnn_dropout=self.network_cfg['cnn_dropout']
        ).to(self.device)
        
        self.target_critic2 = QNetwork(
            low_freq_feature_dim=self.low_freq_feature_dim,
            low_freq_seq_len=self.low_freq_seq_len,
            num_assets=self.num_assets,
            fusion_dim=self.fusion_dim,
            hidden_dim=self.hidden_dim,
            weights_gate=self.network_cfg['actor_weights_gate'],
            conv_out_channels=self.network_cfg['conv_out_channels'],
            conv_kernel_size=self.network_cfg['conv_kernel_size'],
            pool_kernel_size=self.network_cfg['pool_kernel_size'],
            lstm_hidden_size=self.network_cfg['lstm_hidden_size'],
            attention_num_heads=self.network_cfg['attention_num_heads'],
            market_reduce_dim=self.network_cfg['market_reduce_dim'],
            weight_fc_dim=self.network_cfg['weight_fc_dim'],
            mlp_dropout=self.network_cfg['mlp_dropout'],
            cnn_dropout=self.network_cfg['cnn_dropout']
        ).to(self.device)
        
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        try:
            c1g = float(getattr(self.critic1, 'weights_gate', 1.0))
            c2g = float(getattr(self.critic2, 'weights_gate', 1.0))
            tc1g = float(getattr(self.target_critic1, 'weights_gate', 1.0))
            tc2g = float(getattr(self.target_critic2, 'weights_gate', 1.0))
            logger.info(f"Critic portfolio feature weight gate: c1={c1g:.3f}, c2={c2g:.3f}, target_c1={tc1g:.3f}, target_c2={tc2g:.3f}")
        except Exception:
            pass
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr_critic)
        
        if self.auto_alpha:
            initial_log_alpha = np.log(self.alpha)
            self.log_alpha = torch.tensor([initial_log_alpha], requires_grad=True, device=self.device, dtype=torch.float32)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)
            logger.info(f"auto alpha adjustment: initial alpha={self.alpha:.4f}, target entropy={self.target_entropy}")
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
            logger.info(f"fixed alpha value: {self.alpha}")
        
        self.replay_buffer = ExperienceReplayBuffer(
            capacity=self.replay_buffer_size,
            device=self.device,
            seed=self.seed
        )
        
        logger.info("SAC networks built successfully")
    
    def get_actions_batch(self, agent_observations: Dict[str, Dict[str, np.ndarray]], 
                         deterministic: Optional[bool] = None) -> Dict[str, np.ndarray]:
        if not agent_observations:
            return {}
        
        if deterministic is None:
            deterministic = self.deterministic_policy
        
        try:
            batch_market = []
            batch_weights = []
            
            agent_names_ordered = []
            
            for agent_name, state in agent_observations.items():
                self._validate_state_shapes(state)

                market_np = state['market_features']  # (A, L, F)
                weights_np = state['portfolio_weights']  # (A,)

                market_tensor = torch.FloatTensor(market_np).unsqueeze(0).to(self.device)  # (1, A, L, F)
                weights_tensor = torch.FloatTensor(weights_np).unsqueeze(0).to(self.device)  # (1, A)

                batch_market.append(market_tensor)
                batch_weights.append(weights_tensor)
                agent_names_ordered.append(agent_name)
            
            batch_market_tensor = torch.cat(batch_market, dim=0)
            batch_weights_tensor = torch.cat(batch_weights, dim=0)
            logger.debug(f"batch inference feature shape - market: {batch_market_tensor.shape}, weights: {batch_weights_tensor.shape}")
            
            with torch.no_grad():
                start_time = time.perf_counter()
                self.actor.eval()
                
                actions_tensor, log_probs, _ = self.actor(
                    {
                        'market_features': batch_market_tensor,
                        'portfolio_weights': batch_weights_tensor
                    },
                    deterministic=deterministic
                )
                
                if self.entropy_logging_enabled and (log_probs is not None):
                    entropy_batch = -log_probs.sum(dim=-1)  # (batch_size,)
                    for i, agent_name in enumerate(agent_names_ordered):
                        entropy_value = entropy_batch[i].item()
                        self.entropy_records.append({
                            'timestamp': time.time(),
                            'agent_name': agent_name,
                            'entropy': entropy_value
                        })
                
                inference_time = time.perf_counter() - start_time
                self.performance_stats['action_generation_times'].append(inference_time)
                
                if len(self.performance_stats['action_generation_times']) > 100:
                    self.performance_stats['action_generation_times'] = \
                        self.performance_stats['action_generation_times'][-100:]
            
            actions_batch = actions_tensor.cpu().numpy()
            result_dict = {}
            for i, agent_name in enumerate(agent_names_ordered):
                weights = actions_batch[i]
                weights = self._ensure_valid_weights(weights)
                result_dict[agent_name] = weights
            
            avg_time = inference_time / len(agent_observations)
            logger.debug(f"batch generate {len(agent_observations)} agent actions completed, "
                        f"total time: {inference_time:.4f}s, average time per agent: {avg_time:.4f}s")
            
            return result_dict
            
        except Exception as e:
            logger.error(f"batch get actions failed: {e}")
            logger.error(f"input observation count: {len(agent_observations)}")
            for agent_name, obs in list(agent_observations.items())[:3]:
                try:
                    logger.error(f"Agent {agent_name} - state keys: {list(obs.keys())}")
                except Exception:
                    pass
            raise
    
    def add_experience(self, experience_tuple) -> bool:
        if not self.training_enabled:
            logger.debug("inference mode: skip adding experience")
            return True
            
        try:
            self.replay_buffer.push_from_rl_controller(experience_tuple)
            self.experience_count += 1
            try:
                timestamp_ns = int(getattr(experience_tuple, 'timestamp', 0))
                if timestamp_ns > 0:
                    self._latest_sim_timestamp_ns = timestamp_ns
            except Exception:
                pass
            return True
        except Exception as e:
            logger.error(f"add experience failed: {e}")
            return False
    
    def add_experiences_batch(self, experience_tuples: List) -> int:
        if not self.training_enabled:
            logger.debug(f"inference mode: skip batch adding {len(experience_tuples)} experiences")
            return len(experience_tuples)
            
        start_time = time.perf_counter()
        
        try:
            added_count = self.replay_buffer.push_batch_from_rl_controller(
                experience_tuples)
            self.experience_count += added_count
            try:
                max_ts = 0
                for exp_tuple in experience_tuples:
                    timestamp_ns = int(getattr(exp_tuple, 'timestamp', 0))
                    if timestamp_ns > max_ts:
                        max_ts = timestamp_ns
                if max_ts > 0:
                    self._latest_sim_timestamp_ns = max_ts
            except Exception:
                pass
            
            processing_time = time.perf_counter() - start_time
            self.performance_stats['experience_processing_times'].append(processing_time)
            
            logger.info(f"batch add experience: {added_count}/{len(experience_tuples)}, "
                       f"total experience count: {self.experience_count}, time: {processing_time:.4f}s")
            
            if self._should_train():
                self.train()
            
            return added_count
            
        except Exception as e:
            logger.error(f"batch add experience failed: {e}")
            return 0
    
    def _should_train(self) -> bool:
        if not self.training_enabled:
            return False
        
        if self.experience_count < self.min_experiences_before_training:
            return False
        
        experiences_since_last_training = self.experience_count - self.last_training_count
        
        return experiences_since_last_training >= self.train_frequency
    
    def can_train(self) -> bool:
        return self.replay_buffer.can_sample(self.batch_size)
    
    def _get_gradient_clip_norm(self) -> float:
        """
        get gradient clip norm based on current experience count
        """
        if not self.gradient_clipping_enabled:
            return float('inf')
        
        if self.experience_count < self.early_clip_threshold:
            return self.early_clip_norm
        elif self.experience_count < self.middle_clip_threshold:
            return self.middle_clip_norm
        else:
            return self.late_clip_norm

    def _gather_rewards_from_buffer(self, max_samples: int = 200000) -> np.ndarray:

        try:
            if len(self.replay_buffer) == 0:
                return np.asarray([], dtype=np.float32)
            
            buf = getattr(self.replay_buffer, 'buffer', [])
            if not buf:
                return np.asarray([], dtype=np.float32)
            
            rewards = []
            recent_experiences = buf[-max_samples:] if len(buf) > max_samples else buf
            
            for exp in recent_experiences:
                try:
                    reward_val = float(getattr(exp, 'reward', 0.0))
                    rewards.append(reward_val)
                except Exception:
                    pass
            
            logger.debug(f"extract rewards from replay buffer: total experience count={len(buf)}, successfully extracted rewards count={len(rewards)}")
            return np.asarray(rewards, dtype=np.float32)
        except Exception as e:
            logger.warning(f"extract rewards from replay buffer failed: {e}")
            return np.asarray([], dtype=np.float32)

    def _robust_scale_of(self, tensor: torch.Tensor) -> float:

        try:
            x = tensor.detach().abs().flatten().float()
            if x.numel() == 0:
                return float(self._reward_c)
            if self.reward_scale_method == 'std':
                return float(x.std(unbiased=False).clamp(min=self.reward_min_c).item())
            elif self.reward_scale_method == 'q95':
                q = torch.quantile(x, 0.95)
                return float(max(self.reward_min_c, q.item()))
            else:  # 'iqr'
                q75 = torch.quantile(x, 0.75)
                q25 = torch.quantile(x, 0.25)
                iqr = (q75 - q25).abs()
                robust_sigma = 0.7413 * iqr
                return float(robust_sigma.clamp(min=self.reward_min_c).item())
        except Exception:
            return float(self._reward_c)

    def _try_initialize_reward_scale(self, rewards_tensor: Optional[torch.Tensor] = None) -> None:
        if self._reward_scaler_ready:
            return
        if not self.reward_scaling_enabled:
            return
        if self.experience_count < self.reward_scaler_warmup_experiences:
            return
        arr = self._gather_rewards_from_buffer()
        if arr.size == 0:
            logger.warning(f"reward scale initialization failed: extracted 0 rewards from {len(self.replay_buffer)} experiences, check experience object structure")
            try:
                if len(self.replay_buffer) > 0:
                    first_exp = list(self.replay_buffer.buffer)[0]
                    logger.debug(f"first experience type: {type(first_exp)}, has reward attribute: {hasattr(first_exp, 'reward')}")
                    if hasattr(first_exp, 'reward'):
                        logger.debug(f"first experience reward value: {first_exp.reward}")
            except Exception as e:
                logger.debug(f"check experience structure failed: {e}")
            return
        abs_arr = np.abs(arr)
        try:
            if self.reward_scale_method == 'std':
                c0 = float(abs_arr.std(dtype=np.float64))
            elif self.reward_scale_method == 'q95':
                c0 = float(np.quantile(abs_arr, 0.95))
            else:  # 'iqr'
                q75 = float(np.quantile(abs_arr, 0.75))
                q25 = float(np.quantile(abs_arr, 0.25))
                c0 = 0.7413 * max(0.0, q75 - q25)
        except Exception:
            return
        if not np.isfinite(c0) or c0 <= 0:
            return
        self._reward_c = max(self.reward_min_c, c0)
        self._reward_scaler_ready = True
        try:
            logger.info(f"reward scale initializer completed: c={self._reward_c:.6f} | method={self.reward_scale_method} | sample count={arr.size} | ready_from=buffer")
        except Exception:
            pass

    def _update_reward_scale(self, rewards_tensor: torch.Tensor) -> None:
        if not self.reward_scaling_enabled:
            return
        self._try_initialize_reward_scale(rewards_tensor)
        return

    def _apply_reward_scaling(self, rewards_tensor: torch.Tensor) -> torch.Tensor:
        c = max(self.reward_min_c, float(self._reward_c))
        r = rewards_tensor / c
        if self.reward_clip_value is not None and self.reward_clip_value > 0:
            r = torch.clamp(r, -self.reward_clip_value, self.reward_clip_value)
        return r
    
    def train(self, num_steps: int = None) -> Dict[str, Any]:
        if num_steps is None:
            num_steps = self.train_steps_per_update
            
        if not self.can_train():
            logger.debug("experience insufficient, skip training")
            return {}
        
        clip_norm = self._get_gradient_clip_norm()
        if self.gradient_clipping_enabled:
            clip_display = f"enabled ({clip_norm})"
        else:
            clip_display = "disabled"
        
        logger.info(f"training parameters: experience count: {self.experience_count} | training frequency: every {self.train_frequency} experiences | gradient clipping: {clip_display}")
        
        start_time = time.perf_counter()
        all_stats = []
        
        for step in range(num_steps):
            try:
                stats = self._train_step()
                all_stats.append(stats)
            except Exception as e:
                logger.error(f"training step {step + 1} failed: {e}")
                break
        
        train_time = time.perf_counter() - start_time
        
        if all_stats:
            agg = {}
            for s in all_stats:
                for k, v in s.items():
                    try:
                        agg.setdefault(k, []).append(float(v))
                    except Exception:
                        pass
            avg_stats = { f'avg_{k}': float(np.mean(vs)) for k, vs in agg.items() if len(vs) > 0 }
            
            avg_stats['num_steps'] = len(all_stats)
            avg_stats['train_time'] = train_time
            
            self.total_training_steps += len(all_stats)
            self.last_training_count = self.experience_count
            
            self._log_training_progress(avg_stats)
            
            logger.info(f"training completed - steps: {len(all_stats)}, actor loss: {avg_stats['avg_actor_loss']:.4f}, critic1 loss: {avg_stats['avg_critic1_loss']:.4f}/{avg_stats['avg_critic2_loss']:.4f}")
            
            return avg_stats
        
        return {}
    
    def _train_step(self) -> Dict[str, float]:
        """perform one training step"""
        if self.prioritized_replay_enabled:
            batch = self.replay_buffer.prioritized_sample(
                self.batch_size,
                half_life_days=self.prioritized_replay_half_life_days,
                uniform_mix=self.prioritized_replay_uniform_mix
            )
            try:
                dbg = getattr(self.replay_buffer, 'last_sampling_debug', None)
                if dbg is not None:
                    dates = sorted(dbg.get('theoretical_ratio_by_date', {}).keys(), reverse=True)
                    theo = dbg.get('theoretical_ratio_by_date', {})
                    real = dbg.get('sample_ratio_by_date', {})
                    msg_parts = []
                    for d in dates:
                        msg_parts.append(f"{d}: theo={theo.get(d, 0.0):.4f}, real={real.get(d, 0.0):.4f}")
                    logger.info(
                        "time-based sampling distribution | latest_date=%s half_life=%.2f uniform_mix=%.2f | %s",
                        dbg.get('latest_date'),
                        dbg.get('half_life_days', 0.0),
                        dbg.get('uniform_mix', 0.0),
                        ' | '.join(msg_parts)
                    )
            except Exception:
                pass
        else:
            batch = self.replay_buffer.sample(self.batch_size)
        state, actions, rewards, next_state, dones = batch

        avg_abs_reward = float('nan')
        avg_abs_scaled_reward = float('nan')
        reward_c_now = float('nan')
        try:
            with torch.no_grad():
                avg_abs_reward = rewards.abs().mean().item()
            if self.reward_scaling_enabled:
                self._update_reward_scale(rewards)
                reward_c_now = float(self._reward_c)
                if self._reward_scaler_ready:
                    rewards = self._apply_reward_scaling(rewards)
                    with torch.no_grad():
                        avg_abs_scaled_reward = rewards.abs().mean().item()
        except Exception:
            pass
        
        current_alpha = self.get_alpha()
        alpha_values = torch.full((self.batch_size, 1), current_alpha, device=self.device)
        gamma_values = torch.full((self.batch_size, 1), self.gamma, device=self.device)
        network_heterogeneous_params = None
        
        c1, c2, td_target_abs, minq_abs = self._update_critics(
            state, actions, rewards,
            next_state, dones,
            alpha_values, gamma_values
        )
        critic_loss = (c1, c2)
        
        actor_loss = self._update_actor(state, alpha_values)
        
        if self.auto_alpha:
            alpha_loss = self._update_alpha(state)
        else:
            alpha_loss = 0.0

        try:
            with torch.no_grad():
                actions_dbg, log_probs_step, concentration_dbg = self.actor(state)
                if log_probs_step is not None:
                    policy_entropy = (-log_probs_step).mean().item()
                    mean_logp = (log_probs_step).mean().item()
                    mean_neglogp = (-log_probs_step).mean().item()
                    alpha_grad_term = (log_probs_step + self.target_entropy).mean().item()
                    try:
                        K = concentration_dbg.size(-1)
                        dirichlet_A_mean_chk = concentration_dbg.sum(dim=-1).mean().item()
                        dirichlet_alpha_mean_chk = concentration_dbg.mean(dim=-1).mean().item()
                        denom = max(1.0, abs(dirichlet_A_mean_chk))
                        if abs(dirichlet_A_mean_chk - K * dirichlet_alpha_mean_chk) / denom > 1e-3:
                            logger.warning(f"[SAC][Stats] A_mean!=K*alpha_mean: A={dirichlet_A_mean_chk:.6f} K*alpha_mean={(K*dirichlet_alpha_mean_chk):.6f}")
                    except Exception:
                        pass
                else:
                    policy_entropy = float('nan')
                    mean_logp = float('nan')
                    mean_neglogp = float('nan')
                    alpha_grad_term = float('nan')
                try:
                    dirichlet_A_mean = concentration_dbg.sum(dim=-1).mean().item()
                    dirichlet_alpha_mean = concentration_dbg.mean().item()
                except Exception:
                    dirichlet_A_mean = float('nan')
                    dirichlet_alpha_mean = float('nan')
        except Exception:
            policy_entropy = float('nan')
            dirichlet_A_mean = float('nan')
            dirichlet_alpha_mean = float('nan')

        try:
            alpha_value = float(self.get_alpha())
        except Exception:
            alpha_value = float(self.alpha)
        
        self._soft_update_targets()
        
        try:
            denom = max(1e-6, float(alpha_value) * float(mean_neglogp))
            q_entropy_ratio = float(minq_abs) / denom
        except Exception:
            q_entropy_ratio = float('nan')

        return {
            'critic1_loss': critic_loss[0],
            'critic2_loss': critic_loss[1],
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'policy_entropy': policy_entropy,
            'mean_logp': mean_logp,
            'mean_neglogp': mean_neglogp,
            'alpha_grad_term': alpha_grad_term,
            'alpha_value': alpha_value,
            'dirichlet_A_mean': dirichlet_A_mean,
            'dirichlet_alpha_mean': dirichlet_alpha_mean,
            'target_entropy_value': float(self.target_entropy),
            'td_target_abs': td_target_abs,
            'minQ_abs': minq_abs,
            'q_entropy_ratio': q_entropy_ratio,
            'reward_c': reward_c_now,
            'abs_reward': avg_abs_reward,
            'abs_scaled_reward': avg_abs_scaled_reward,
        }
    
    def _update_critics(self, state, actions, rewards,
                       next_state, dones, 
                       alpha_values, gamma_values):
        with torch.no_grad():
            if torch.is_autocast_enabled():
                with torch.cuda.amp.autocast(enabled=False):
                    next_actions, next_log_probs, _ = self.actor({
                        'market_features': next_state['market_features'].float(),
                        'portfolio_weights': next_state['portfolio_weights'].float()
                    })
            else:
                next_actions, next_log_probs, _ = self.actor(next_state)
            target_q1 = self.target_critic1(next_state, next_actions)
            target_q2 = self.target_critic2(next_state, next_actions)
            
            target_q = torch.min(target_q1, target_q2) - alpha_values * next_log_probs
            
            target_q_values = rewards + gamma_values * (1 - dones.float()) * target_q
            try:
                td_target_abs = target_q_values.abs().mean().item()
            except Exception:
                td_target_abs = float('nan')
            
            target_q_values = torch.clamp(target_q_values, self.q_value_clamp_min, self.q_value_clamp_max)
        
        current_q1 = self.critic1(state, actions)
        current_q2 = self.critic2(state, actions)
        try:
            minq_abs = torch.min(current_q1, current_q2).abs().mean().item()
        except Exception:
            minq_abs = float('nan')
        
        current_q1 = torch.clamp(current_q1, self.q_value_clamp_min, self.q_value_clamp_max)
        current_q2 = torch.clamp(current_q2, self.q_value_clamp_min, self.q_value_clamp_max)
        
        critic1_loss = F.mse_loss(current_q1, target_q_values)
        critic2_loss = F.mse_loss(current_q2, target_q_values)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        clip_norm = self._get_gradient_clip_norm()
        if clip_norm != float('inf'):
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=clip_norm)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_norm = self._get_gradient_clip_norm()
        if clip_norm != float('inf'):
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=clip_norm)
        self.critic2_optimizer.step()
        
        return critic1_loss.item(), critic2_loss.item(), td_target_abs, minq_abs
    
    def _update_actor(self, state, alpha_values):
        if torch.is_autocast_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                actions, log_probs, _ = self.actor({
                    'market_features': state['market_features'].float(),
                    'portfolio_weights': state['portfolio_weights'].float()
                })
        else:
            actions, log_probs, _ = self.actor(state)
        q1_pi = self.critic1(state, actions)
        q2_pi = self.critic2(state, actions)
        
        q1_pi = torch.clamp(q1_pi, -1000, 1000)
        q2_pi = torch.clamp(q2_pi, -1000, 1000)
        
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = (alpha_values * log_probs - min_q_pi).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_norm = self._get_gradient_clip_norm()
        if clip_norm != float('inf'):
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=clip_norm)
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _update_alpha(self, state):
        if not self.auto_alpha or self.log_alpha is None:
            return 0.0
        
        if torch.is_autocast_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                actions, log_probs, _ = self.actor({
                    'market_features': state['market_features'].float(),
                    'portfolio_weights': state['portfolio_weights'].float()
                })
        else:
            actions, log_probs, _ = self.actor(state)
        
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        clip_norm = self._get_gradient_clip_norm()
        if clip_norm != float('inf'):
            torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=clip_norm)
        self.alpha_optimizer.step()
        
        return alpha_loss.item()
    
    def _soft_update_targets(self):
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def get_alpha(self) -> float:
        if self.auto_alpha:
            return max(self.alpha_min, float(torch.exp(self.log_alpha).item()))
        else:
            return max(self.alpha_min, float(self.alpha))
    
    def _validate_state_shapes(self, state: Dict[str, np.ndarray]) -> None:
        if not isinstance(state, dict):
            raise ValueError("state must be a dictionary, containing 'market_features' and 'portfolio_weights'")
        if 'market_features' not in state or 'portfolio_weights' not in state:
            raise ValueError("state missing required keys: 'market_features' or 'portfolio_weights'")

        market = state['market_features']
        weights = state['portfolio_weights']

        if market.ndim != 3:
            raise ValueError(f"market_features should be 3D (assets, seq_len, features), actual: {market.ndim}D")
        if weights.ndim != 1:
            raise ValueError(f"portfolio_weights should be 1D (assets,), actual: {weights.ndim}D")

        expected_market_shape = (self.num_assets, self.low_freq_seq_len, self.low_freq_feature_dim)
        expected_weights_shape = (self.num_assets,)
        if tuple(market.shape) != expected_market_shape:
            raise ValueError(f"market_features shape should be {expected_market_shape}, actual: {market.shape}")
        if tuple(weights.shape) != expected_weights_shape:
            raise ValueError(f"portfolio_weights shape should be {expected_weights_shape}, actual: {weights.shape}")
    
    def _ensure_valid_weights(self, weights: np.ndarray) -> np.ndarray:
        weights = np.maximum(weights, 0)
        
        weight_sum = weights.sum()
        if weight_sum > 1e-8:
            weights = weights / weight_sum
        else:
            weights = np.ones_like(weights) / len(weights)
        
        return weights
    
    def save_models(self, filepath_prefix: str) -> bool:
        try:
            torch.save({
                'actor': self.actor.state_dict(),
                'critic1': self.critic1.state_dict(),
                'critic2': self.critic2.state_dict(),
                'target_critic1': self.target_critic1.state_dict(),
                'target_critic2': self.target_critic2.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic1_optimizer': self.critic1_optimizer.state_dict(),
                'critic2_optimizer': self.critic2_optimizer.state_dict(),
                'log_alpha': self.log_alpha.data if self.log_alpha is not None else None,
                'alpha_optimizer': self.alpha_optimizer.state_dict() if self.alpha_optimizer is not None else None,
                'total_training_steps': self.total_training_steps,
                'experience_count': self.experience_count,
                'meta_num_assets': self.num_assets,
                'meta_low_freq_seq_len': self.low_freq_seq_len,
                'meta_low_freq_feature_dim': self.low_freq_feature_dim,
                'meta_network_cfg': self.network_cfg
            }, f"{filepath_prefix}.pth")
            
            logger.info(f"model saved: {filepath_prefix}.pth")

            # 额外导出：C++可加载（LibTorch）用的TorchScript Actor
            try:
                self._export_actor_torchscript(filepath_prefix)
            except Exception as e:
                logger.warning(f"export TorchScript Actor failed (not affect .pth saving): {e}")
            return True
        except Exception as e:
            logger.error(f"save model failed: {e}")
            return False
    
    def load_models(self, filepath: str) -> bool:
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic1.load_state_dict(checkpoint['critic1'])
            self.critic2.load_state_dict(checkpoint['critic2'])
            self.target_critic1.load_state_dict(checkpoint['target_critic1'])
            self.target_critic2.load_state_dict(checkpoint['target_critic2'])
            
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
            self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
            
            if checkpoint['log_alpha'] is not None and self.log_alpha is not None:
                self.log_alpha.data = checkpoint['log_alpha']
            if checkpoint['alpha_optimizer'] is not None and self.alpha_optimizer is not None:
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            
            self.total_training_steps = checkpoint.get('total_training_steps', 0)
            self.experience_count = checkpoint.get('experience_count', 0)

            try:
                na = int(checkpoint.get('meta_num_assets', self.num_assets))
                wl = int(checkpoint.get('meta_low_freq_seq_len', self.low_freq_seq_len))
                ft = int(checkpoint.get('meta_low_freq_feature_dim', self.low_freq_feature_dim))
                if (na != self.num_assets) or (wl != self.low_freq_seq_len) or (ft != self.low_freq_feature_dim):
                    logger.warning(f"load model shape mismatch: ckpt=({na},{wl},{ft}) current=({self.num_assets},{self.low_freq_seq_len},{self.low_freq_feature_dim})")
            except Exception:
                pass
            
            logger.info(f"model loaded successfully: {filepath}")
            return True
        except Exception as e:
            logger.error(f"load model failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {
            'experience_count': self.experience_count,
            'total_training_steps': self.total_training_steps,
            'replay_buffer_size': len(self.replay_buffer),
            'current_alpha': self.get_alpha(),
            'training_enabled': self.training_enabled
        }
        
        for key, values in self.performance_stats.items():
            if values:
                stats[f'avg_{key}'] = np.mean(values)
                stats[f'latest_{key}'] = values[-1]
        
        return stats
    
    def enable_training(self):
        self.training_enabled = True
        logger.info("training enabled")
    
    def disable_training(self):
        self.training_enabled = False
        logger.info("training disabled")
    
    def cleanup(self):
        if self.total_training_steps > 0:
            try:
                save_dir = "./rl_modules/SAC_Portfolio_Allocationl/saved_models"
                os.makedirs(save_dir, exist_ok=True)
                date_str = self._get_simulation_date_for_naming()
                filepath_prefix = os.path.join(save_dir, f"sac_agent_final_{date_str}")
                self.save_models(filepath_prefix)
                logger.info("final model saved")
            except Exception as e:
                logger.error(f"save final model failed: {e}")
        
        try:
            if len(self.replay_buffer) > 0:
                success = self.replay_buffer.save_by_date()
                if success:
                    logger.info("replay buffer saved by date")
                else:
                    logger.warning("save replay buffer failed")
            else:
                logger.info("replay buffer is empty, skip saving")
        except Exception as e:
            logger.error(f"save replay buffer failed: {e}")
        
        self._save_entropy_records()
        
        logger.info("SAC agent cleaned up")

    def _export_actor_torchscript(self, filepath_prefix: str) -> None:
        original_device = next(self.actor.parameters()).device
        was_training = self.actor.training
        try:
            self.actor.eval()
            out_path = f"{filepath_prefix}_actor.pt"
            if self._scripted_actor_cache is not None:
                try:
                    src_sd = self.actor.state_dict()
                    dst_sd = self._scripted_actor_cache.state_dict()
                    mapped = {}
                    for k_dst in dst_sd.keys():
                        k_src = k_dst
                        if k_src.startswith('core.'):
                            k_src = k_src[len('core.'):] 
                        if k_src in src_sd and src_sd[k_src].shape == dst_sd[k_dst].shape:
                            mapped[k_dst] = src_sd[k_src]
                    self._scripted_actor_cache.load_state_dict(mapped, strict=False)
                    try:
                        g = float(getattr(self.actor, 'weights_gate', 1.0))
                        self._scripted_actor_cache.setattr('weights_gate', torch.tensor(float(g), dtype=torch.float32))
                    except Exception:
                        pass
                    try:
                        eps_val = float(self.network_cfg.get('sampling_eps', 1e-6))
                        self._scripted_actor_cache.setattr('sampling_eps', torch.tensor(float(eps_val), dtype=torch.float32))
                    except Exception:
                        pass
                    self._scripted_actor_cache.save(out_path)
                    try:
                        g_logged = float(self._scripted_actor_cache.attr('weights_gate').toTensor().item())
                    except Exception:
                        g_logged = float(getattr(self.actor, 'weights_gate', 1.0))
                    try:
                        eps_logged = float(self._scripted_actor_cache.attr('sampling_eps').toTensor().item())
                    except Exception:
                        eps_logged = float(self.network_cfg.get('sampling_eps', 1e-6))
                    logger.info(f"TorchScript Actor(cached) exported: weights_gate={g_logged:.3f}, sampling_eps={eps_logged:.6f}")
                    logger.info(f"TorchScript Actor exported (reuse cache): {out_path}")
                    return
                except Exception as e:
                    logger.warning(f"update cached script module failed, try recompile: {e}")
                    self._scripted_actor_cache = None

            try:
                class ActorInferModule(torch.nn.Module):
                    def __init__(self, core: torch.nn.Module, seed: int = 0, eps: float = 1e-6):
                        super().__init__()
                        self.market_encoder = core.market_encoder
                        self.state_fusion = core.state_fusion
                        self.policy_head = core.policy_head
                        self.alpha_layer = core.alpha_layer
                        self.min_concentration = core.min_concentration
                        self.register_buffer('weights_gate', torch.tensor(float(getattr(core, 'weights_gate', 1.0)), dtype=torch.float32))
                        self.register_buffer('sampling_eps', torch.tensor(float(eps), dtype=torch.float32))
                        self.register_buffer('sampling_seed', torch.tensor([int(seed)], dtype=torch.int64))
                    def forward(self, market_features: torch.Tensor, portfolio_weights: torch.Tensor):
                        per_asset_features = self.market_encoder(market_features)
                        batch_size = per_asset_features.shape[0]
                        market_flat = per_asset_features.reshape(batch_size, self.state_fusion.num_assets * self.state_fusion.per_asset_dim)
                        market_256 = self.state_fusion.market_reducer(market_flat)
                        weights_256 = self.state_fusion.weight_encoder(portfolio_weights)
                        weights_256 = weights_256 * self.weights_gate
                        fused_state = torch.cat([market_256, weights_256], dim=-1)
                        policy_features = self.policy_head(fused_state)
                        alpha_unconstrained = self.alpha_layer(policy_features)
                        concentration = torch.nn.functional.softplus(alpha_unconstrained) + float(self.min_concentration)
                        mean_action = concentration / concentration.sum(dim=-1, keepdim=True)
                        return mean_action, concentration
                seed_val = int(self.seed) if (self.seed is not None) else 0
                eps_val = float(self.network_cfg.get('sampling_eps', 1e-6))
                wrapper = ActorInferModule(self.actor, seed=seed_val, eps=eps_val)
                self._scripted_actor_cache = torch.jit.script(wrapper)
                self._scripted_actor_cache.save(out_path)
                try:
                    g_logged = float(self._scripted_actor_cache.attr('weights_gate').toTensor().item())
                except Exception:
                    g_logged = float(getattr(self.actor, 'weights_gate', 1.0))
                try:
                    eps_logged = float(self._scripted_actor_cache.attr('sampling_eps').toTensor().item())
                except Exception:
                    eps_logged = float(self.network_cfg.get('sampling_eps', 1e-6))
                logger.info(f"TorchScript Actor(script) exported: weights_gate={g_logged:.3f}, sampling_eps={eps_logged:.6f}")
                logger.info(f"TorchScript Actor exported (script): {out_path}")
                return
            except Exception as e_script:
                logger.warning(f"script export failed, try trace fallback: {e_script}")

            try:
                class TraceWrapper(torch.nn.Module):
                    def __init__(self, core: torch.nn.Module, seed: int = 0, eps: float = 1e-6):
                        super().__init__()
                        self.core = core
                        self.register_buffer('weights_gate', torch.tensor(float(getattr(core, 'weights_gate', 1.0)), dtype=torch.float32))
                        self.register_buffer('sampling_eps', torch.tensor(float(eps), dtype=torch.float32))
                        self.register_buffer('sampling_seed', torch.tensor([int(seed)], dtype=torch.int64))
                    def forward(self, market_features: torch.Tensor, portfolio_weights: torch.Tensor):
                        per_asset_features = self.core.market_encoder(market_features)
                        batch_size = per_asset_features.shape[0]
                        market_flat = per_asset_features.reshape(batch_size, self.core.state_fusion.num_assets * self.core.state_fusion.per_asset_dim)
                        market_256 = self.core.state_fusion.market_reducer(market_flat)
                        weights_256 = self.core.state_fusion.weight_encoder(portfolio_weights)
                        weights_256 = weights_256 * self.weights_gate
                        fused_state = torch.cat([market_256, weights_256], dim=-1)
                        policy_features = self.core.policy_head(fused_state)
                        alpha_unconstrained = self.core.alpha_layer(policy_features)
                        concentration = torch.nn.functional.softplus(alpha_unconstrained) + float(self.core.min_concentration)
                        mean_action = concentration / concentration.sum(dim=-1, keepdim=True)
                        return mean_action, concentration
                seed_val = int(self.seed) if (self.seed is not None) else 0
                eps_val = float(self.network_cfg.get('sampling_eps', 1e-6))
                t_wrapper = TraceWrapper(self.actor, seed=seed_val, eps=eps_val).to(original_device)
                example_market = torch.zeros(1, self.num_assets, self.low_freq_seq_len, self.low_freq_feature_dim, device=original_device, dtype=torch.float32)
                example_weights = torch.ones(1, self.num_assets, device=original_device, dtype=torch.float32) / max(1, self.num_assets)
                traced = torch.jit.trace(t_wrapper, (example_market, example_weights))
                traced.save(out_path)
                try:
                    g_logged = float(getattr(self.actor, 'weights_gate', 1.0))
                except Exception:
                    g_logged = 1.0
                try:
                    eps_logged = float(self.network_cfg.get('sampling_eps', 1e-6))
                except Exception:
                    eps_logged = 1e-6
                logger.info(f"TorchScript Actor(trace) exported: weights_gate={g_logged:.3f}, sampling_eps={eps_logged:.6f}")
                logger.info(f"TorchScript Actor exported (trace): {out_path}")
                return
            except Exception as e_trace:
                raise RuntimeError(f"trace export failed: {e_trace}")
        finally:
            self.actor.to(original_device)
            if was_training:
                self.actor.train()

    def _auto_load_latest_model(self):
        try:
            save_dir = "./rl_modules/SAC_Portfolio_Allocationl/saved_models"
            model_pattern = os.path.join(save_dir, "sac_agent_final_*.pth")
            model_files = glob.glob(model_pattern)
            if model_files:
                latest_model = max(model_files, key=os.path.getmtime)
                logger.info(f"found history model file: {latest_model}")
                
                logger.info(f"before loading state: experience_count={self.experience_count}, "
                           f"total_training_steps={self.total_training_steps}")
                
                if self.load_models(latest_model):
                    logger.info(f"success load history model: {latest_model}")
                    logger.info(f"restore state: experience_count={self.experience_count}, "
                               f"total_training_steps={self.total_training_steps}")
                    logger.info("continue training mode enabled")
                else:
                    logger.warning(f"load model failed: {latest_model}")
                    logger.info("start training from scratch")
            else:
                logger.info("no history model file found, start training from scratch")
        except Exception as e:
            logger.error(f"auto load model failed: {e}")
            logger.info("start training from scratch")

    def _auto_load_recent_replay_buffers(self):
        try:
            available_dates = self.replay_buffer.get_available_dates()
            
            if not available_dates:
                logger.info("no history replay buffer found, start training from scratch")
                return
            
            days_to_load = min(self.history_replay_buffer_days, len(available_dates))
            recent_dates = available_dates[-days_to_load:]
            
            logger.info(f"prepare to load recent {days_to_load} days of replay buffer: {recent_dates}")
            
            self.replay_buffer.clear()
            
            total_loaded = 0
            successful_dates = []
            
            for date in recent_dates:
                try:
                    temp_buffer = ExperienceReplayBuffer(
                        capacity=self.replay_buffer.capacity,
                        device=self.device
                    )
                    
                    if temp_buffer.load_by_date(date):
                        for exp in temp_buffer.buffer:
                            self.replay_buffer.push(exp)
                        
                        total_loaded += len(temp_buffer.buffer)
                        successful_dates.append(date)
                        logger.info(f"success load {date} experience: {len(temp_buffer.buffer)}")
                    else:
                        logger.warning(f"load {date} experience replay buffer failed")
                        
                except Exception as e:
                    logger.error(f"load {date} experience replay buffer failed: {e}")
            
            if successful_dates:
                self.experience_count = total_loaded
                self.last_training_count = total_loaded
                
                logger.info(f"cumulative load experience replay buffer completed"
                           f"successful dates: {successful_dates}, "
                           f"total experience: {total_loaded}")
                logger.info(f"continue training mode: experience_count={self.experience_count}, "
                           f"last_training_count={self.last_training_count}")
                
                logger.info(f"train frequency: every {self.train_frequency} new experience")
            else:
                logger.warning("no history experience replay buffer loaded")
                logger.info("start training from scratch")
                
        except Exception as e:
            logger.error(f"auto load experience replay buffer failed: {e}")
            logger.info("use empty experience replay buffer")

    def _log_training_progress(self, stats: Dict[str, float]):
        try:
            save_dir = "./rl_modules/SAC_Portfolio_Allocationl/saved_models"
            os.makedirs(save_dir, exist_ok=True)
            date_str = self._get_simulation_date_for_naming()
            filepath = os.path.join(save_dir, f"training_progress_{date_str}.csv")
            
            desired_headers = ['timestamp', 'total_steps',
                               'avg_critic1_loss','avg_critic2_loss','avg_actor_loss','avg_alpha_loss',
                               'avg_policy_entropy','avg_alpha_value',
                               'avg_dirichlet_A_mean','avg_dirichlet_alpha_mean','avg_target_entropy_value',
                               'avg_mean_logp','avg_mean_neglogp','avg_alpha_grad_term',
                               'avg_td_target_abs','avg_minQ_abs',
                               'avg_reward_c','avg_abs_reward','avg_abs_scaled_reward']
            file_exists = os.path.exists(filepath)

            if file_exists:
                try:
                    with open(filepath, 'r', encoding='utf-8') as rf:
                        first_line = rf.readline().rstrip('\n')
                    existing_headers = first_line.split(',') if first_line else []
                except Exception:
                    existing_headers = []

                if existing_headers != desired_headers:
                    try:
                        with open(filepath, 'r', encoding='utf-8') as rf:
                            lines = rf.read().splitlines()
                        upgraded_lines = []
                        upgraded_lines.append(','.join(desired_headers))
                        if len(lines) > 1:
                            for line in lines[1:]:
                                if not line.strip():
                                    continue
                                cols = line.split(',')
                                if len(cols) < len(desired_headers):
                                    cols.extend([''] * (len(desired_headers) - len(cols)))
                                elif len(cols) > len(desired_headers):
                                    cols = cols[:len(desired_headers)]
                                upgraded_lines.append(','.join(cols))
                        with open(filepath, 'w', encoding='utf-8') as wf:
                            for l in upgraded_lines:
                                wf.write(l + '\n')
                    except Exception as e:
                        logger.warning(f"upgrade training progress CSV header failed, continue append: {e}")

            with open(filepath, 'a', encoding='utf-8') as f:
                if not file_exists:
                    f.write(','.join(desired_headers) + '\n')
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                row_values = []
                missing_keys = []
                for h in desired_headers[2:]:
                    v = stats.get(h, None)
                    if v is None:
                        base = h[4:] if h.startswith('avg_') else h
                        v = stats.get(base, None)
                        if v is None:
                            if h == 'avg_target_entropy_value':
                                v = float(self.target_entropy)
                            else:
                                v = ''
                                missing_keys.append(h)
                    row_values.append(str(v))
                if missing_keys:
                    try:
                        logger.warning(f"missing columns: {missing_keys}; actual available keys: {list(stats.keys())}")
                    except Exception:
                        pass
                values = [timestamp, str(self.total_training_steps)] + row_values
                if len(values) < len(desired_headers):
                    values.extend([''] * (len(desired_headers) - len(values)))
                elif len(values) > len(desired_headers):
                    values = values[:len(desired_headers)]
                f.write(','.join(values) + '\n')
        except Exception as e:
            logger.error(f"record training progress failed: {e}")
    
    def _save_entropy_records(self):
        if not self.entropy_records:
            logger.info("no entropy records to save")
            return
        
        try:
            save_dir = "./rl_modules/SAC_Portfolio_Allocationl/saved_models"
            os.makedirs(save_dir, exist_ok=True)
            date_str = self._get_simulation_date_for_naming()
            filepath = os.path.join(save_dir, f"entropy_records_{date_str}.csv")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('timestamp,agent_name,entropy\n')
                
                for record in self.entropy_records:
                    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record['timestamp']))
                    f.write(f"{formatted_time},{record['agent_name']},{record['entropy']}\n")
            
            logger.info(f"entropy records saved to: {filepath}, total {len(self.entropy_records)} records")
        except Exception as e:
            logger.error(f"save entropy records failed: {e}") 

    def _get_simulation_date_for_naming(self) -> str:
        try:
            ts_latest = int(getattr(self, '_latest_sim_timestamp_ns', 0) or 0)
            if ts_latest > 0:
                import datetime
                dt = datetime.datetime.fromtimestamp(ts_latest / 1_000_000_000)
                return dt.strftime("%Y%m%d")
        except Exception:
            pass
        try:
            if len(self.replay_buffer) > 0:
                recent = self.replay_buffer.get_recent_experiences(min(64, len(self.replay_buffer)))
                latest_ts = 0
                for exp in recent:
                    try:
                        ts = int(getattr(exp, 'timestamp', 0))
                    except Exception:
                        ts = 0
                    if ts > latest_ts:
                        latest_ts = ts
                if latest_ts > 0:
                    import datetime
                    dt = datetime.datetime.fromtimestamp(latest_ts / 1_000_000_000)
                    return dt.strftime("%Y%m%d")
        except Exception:
            pass
        try:
            dates = self.replay_buffer.get_available_dates()
            if dates:
                return dates[-1]
        except Exception:
            pass
        try:
            return time.strftime("%Y%m%d")
        except Exception:
            return "19700101"

    