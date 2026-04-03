#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import random
import datetime
from collections import deque, namedtuple
from typing import List, Tuple, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

# 定义经验元组结构（新版：state字典）
Experience = namedtuple('Experience', [
    'state',               # {'market_features': (A, L, F), 'portfolio_weights': (A,)}
    'action',              # action
    'reward',              # reward
    'next_state',          # next_state same structure
    'done',                # done
    'agent_name',          # agent name
    'timestamp'            # timestamp
])

class ExperienceReplayBuffer:
    
    def __init__(self, 
                 capacity: int = 100000,
                 device: str = 'cpu',
                 seed: Optional[int] = None):

        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        # 采样调试信息（供上层日志使用）
        self.last_sampling_debug: Optional[Dict[str, Any]] = None
        
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        logger.info(f"initialize experience replay buffer, capacity: {capacity}, device: {device}")
    
    def push(self, experience: Experience) -> None:
        self.buffer.append(experience)
        
    def push_from_rl_controller(self, experience_tuple) -> None:
        if not experience_tuple.is_complete:
            logger.warning(f"experience tuple incomplete, skip adding: {experience_tuple.agent_name}")
            return
        
        if not hasattr(experience_tuple, 'state') or not hasattr(experience_tuple, 'next_state'):
            raise ValueError("experience tuple missing state/next_state field (please provide dictionary containing 'market_features' and 'portfolio_weights')")

        experience = Experience(
            state=experience_tuple.state,
            action=experience_tuple.action,
            reward=experience_tuple.reward,
            next_state=experience_tuple.next_state,
            done=False,
            agent_name=experience_tuple.agent_name,
            timestamp=experience_tuple.timestamp
        )
        
        self.push(experience)
        logger.debug(f"add experience from rl controller: {experience_tuple.agent_name}, "
                    f"reward: {experience_tuple.reward:.4f}")
    
    def push_batch_from_rl_controller(self, experience_tuples: List) -> int:
        added_count = 0
        for exp_tuple in experience_tuples:
            if exp_tuple.is_complete:
                self.push_from_rl_controller(exp_tuple)
                added_count += 1
        
        logger.info(f"add batch of experiences: {added_count}/{len(experience_tuples)}")
        return added_count
    
    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        if len(self.buffer) < batch_size:
            raise ValueError(f"buffer size ({len(self.buffer)}) is less than batch size ({batch_size})")
        
        experiences = random.sample(self.buffer, batch_size)
        
        state_market = []
        state_weights = []
        actions = []
        rewards = []
        next_state_market = []
        next_state_weights = []
        dones = []
        
        
        for exp in experiences:
            state_market.append(exp.state['market_features'])
            state_weights.append(exp.state['portfolio_weights'])
            actions.append(exp.action)
            rewards.append(exp.reward)
            next_state_market.append(exp.next_state['market_features'])
            next_state_weights.append(exp.next_state['portfolio_weights'])
            dones.append(exp.done)
            
        try:
            state_market_tensor = torch.FloatTensor(np.array(state_market)).to(self.device)  # (B, A, L, F)
            state_weights_tensor = torch.FloatTensor(np.array(state_weights)).to(self.device)  # (B, A)
            action_tensor = torch.FloatTensor(np.array(actions)).to(self.device)  # (B, A)
            reward_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_state_market_tensor = torch.FloatTensor(np.array(next_state_market)).to(self.device)
            next_state_weights_tensor = torch.FloatTensor(np.array(next_state_weights)).to(self.device)
            done_tensor = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

            state = {
                'market_features': state_market_tensor,
                'portfolio_weights': state_weights_tensor
            }
            next_state = {
                'market_features': next_state_market_tensor,
                'portfolio_weights': next_state_weights_tensor
            }

            logger.debug(f"sampling batch - market: {state_market_tensor.shape}, weights: {state_weights_tensor.shape}, action: {action_tensor.shape}")

            return (state, action_tensor, reward_tensor, next_state, done_tensor)
                   
        except Exception as e:
            logger.error(f"error converting experiences to tensor: {e}")
            # 输出调试信息
            logger.error(f"market_features sample shape: {[s.shape for s in state_market[:3]]}")
            logger.error(f"action sample shape: {[a.shape for a in actions[:3]]}")
            raise
    
    def can_sample(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def get_stats(self) -> Dict[str, Any]:
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0.0,
                'avg_reward': 0.0,
                'min_reward': 0.0,
                'max_reward': 0.0
            }
        
        rewards = [exp.reward for exp in self.buffer]
        
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'avg_reward': np.mean(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'unique_agents': len(set(exp.agent_name for exp in self.buffer))
        }
    
    def clear(self) -> int:
        size = len(self.buffer)
        self.buffer.clear()
        logger.info(f"clear experience replay buffer, delete {size} experiences")
        return size
    
    def get_recent_experiences(self, n: int) -> List[Experience]:
        if n <= 0:
            return []
        
        if n >= len(self.buffer):
            return list(self.buffer)
        
        return list(self.buffer)[-n:]
    
    def get_experiences_by_agent(self, agent_name: str) -> List[Experience]:
        return [exp for exp in self.buffer if exp.agent_name == agent_name]
    
    def get_experiences_in_timerange(self, start_time: int, end_time: int) -> List[Experience]:
        return [exp for exp in self.buffer 
                if start_time <= exp.timestamp <= end_time]
    
    def prioritized_sample(self, batch_size: int, half_life_days: float = 3.0, uniform_mix: float = 0.1) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        buf_len = len(self.buffer)
        if buf_len < batch_size:
            raise ValueError(f"buffer size ({buf_len}) is less than batch size ({batch_size})")

        half_life_days = float(max(1e-6, half_life_days))
        uniform_mix = float(min(max(uniform_mix, 0.0), 1.0))

        ts_ns_array = np.array([int(getattr(exp, 'timestamp', 0)) for exp in self.buffer], dtype=np.int64)
        if not np.any(ts_ns_array > 0):
            now_ns = int(datetime.datetime.now().timestamp() * 1_000_000_000)
            ts_ns_array[:] = now_ns
        latest_ns = int(np.max(ts_ns_array))
        seconds_per_day = 86400.0
        age_days = (latest_ns - ts_ns_array) / (seconds_per_day * 1_000_000_000.0)
        age_days = np.maximum(age_days, 0.0)

        ln2 = np.log(2.0)
        weights = np.exp(-ln2 * (age_days / half_life_days))
        weights = np.maximum(weights, 1e-12)
        weights_sum = float(np.sum(weights))
        if weights_sum <= 0.0 or not np.isfinite(weights_sum):
            probabilities = np.ones(buf_len, dtype=np.float64) / buf_len
        else:
            probs_recency = weights / weights_sum
            probabilities = (1.0 - uniform_mix) * probs_recency + uniform_mix * (1.0 / buf_len)
            probabilities = probabilities / probabilities.sum()

        indices = np.random.choice(buf_len, size=batch_size, p=probabilities)
        experiences = [self.buffer[i] for i in indices]

        try:
            dates_all = [self._extract_date_from_nanosecond_timestamp(ts) for ts in ts_ns_array]
            dates_unique = sorted(set(dates_all))
            # 理论日期概率 = 按同日期下样本概率求和
            prob_by_date = {}
            for d, p in zip(dates_all, probabilities):
                prob_by_date[d] = prob_by_date.get(d, 0.0) + float(p)
            # 实际样本日期计数
            dates_sampled = [dates_all[i] for i in indices]
            from collections import Counter
            cnt = Counter(dates_sampled)
            # 归一化为比例
            sample_ratio_by_date = {d: cnt[d] / float(batch_size) for d in cnt}
            # 记录调试信息
            self.last_sampling_debug = {
                'latest_date': max(dates_unique) if dates_unique else None,
                'unique_dates': dates_unique,
                'theoretical_ratio_by_date': prob_by_date,
                'sample_ratio_by_date': sample_ratio_by_date,
                'batch_size': int(batch_size),
                'half_life_days': float(half_life_days),
                'uniform_mix': float(uniform_mix)
            }
        except Exception:
            # 忽略调试统计失败
            self.last_sampling_debug = None

        # 转换为tensor
        state_market = []
        state_weights = []
        actions = []
        rewards = []
        next_state_market = []
        next_state_weights = []
        dones = []

        for exp in experiences:
            state_market.append(exp.state['market_features'])
            state_weights.append(exp.state['portfolio_weights'])
            actions.append(exp.action)
            rewards.append(exp.reward)
            next_state_market.append(exp.next_state['market_features'])
            next_state_weights.append(exp.next_state['portfolio_weights'])
            dones.append(exp.done)

        state_market_tensor = torch.FloatTensor(np.array(state_market)).to(self.device)
        state_weights_tensor = torch.FloatTensor(np.array(state_weights)).to(self.device)
        action_tensor = torch.FloatTensor(np.array(actions)).to(self.device)
        reward_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state_market_tensor = torch.FloatTensor(np.array(next_state_market)).to(self.device)
        next_state_weights_tensor = torch.FloatTensor(np.array(next_state_weights)).to(self.device)
        done_tensor = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        try:
            # 统计信息（仅调试）
            mean_age = float(np.mean(age_days))
            logger.debug(f"priority sampling batch - half_life_days: {half_life_days}, uniform_mix: {uniform_mix}, mean age: {mean_age:.2f}")
        except Exception:
            pass

        state = {
            'market_features': state_market_tensor,
            'portfolio_weights': state_weights_tensor
        }
        next_state = {
            'market_features': next_state_market_tensor,
            'portfolio_weights': next_state_weights_tensor
        }

        return (state, action_tensor, reward_tensor, next_state, done_tensor)
    
    def save_to_file(self, filepath: str) -> bool:
        try:
            import pickle
            
            # 准备保存的数据
            save_data = {
                'experiences': list(self.buffer),
                'capacity': self.capacity,
                'device': self.device
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info(f"experience replay buffer saved to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"save experience replay buffer failed: {e}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            self.buffer = deque(save_data['experiences'], maxlen=self.capacity)
            
            logger.info(f"load experience replay buffer from: {filepath}, "
                       f"load {len(self.buffer)} experiences")
            return True
            
        except Exception as e:
            logger.error(f"load experience replay buffer failed: {e}")
            return False

    def _extract_date_from_nanosecond_timestamp(self, ns_timestamp: int) -> str:
        try:
            seconds = ns_timestamp / 1_000_000_000
            dt = datetime.datetime.fromtimestamp(seconds)
            return dt.strftime("%Y%m%d")
        except Exception as e:
            logger.error(f"error converting timestamp to date: {ns_timestamp}, error: {e}")
            return "19700101"
    
    def save_by_date(self, base_dir: str = "./rl_modules/SAC_Portfolio_Allocationl/replay_buffer") -> bool:
        if len(self.buffer) == 0:
            logger.warning("experience replay buffer is empty, skip saving")
            return False
        
        try:
            import os
            import pickle
            import datetime
            
            os.makedirs(base_dir, exist_ok=True)
            
            experiences_by_date = {}
            for exp in self.buffer:
                date_str = self._extract_date_from_nanosecond_timestamp(exp.timestamp)
                if date_str not in experiences_by_date:
                    experiences_by_date[date_str] = []
                experiences_by_date[date_str].append(exp)
            
            latest_date = max(experiences_by_date.keys())
            latest_experiences = experiences_by_date[latest_date]
            
            # 构建文件路径（SAC前缀）
            filename = f"sac_replay_buffer_{latest_date}.pkl"
            filepath = os.path.join(base_dir, filename)
            
            # 准备保存的数据
            save_data = {
                'experiences': latest_experiences,
                'capacity': self.capacity,
                'device': self.device,
                'date': latest_date,
                'total_experiences': len(latest_experiences)
            }
            
            # 保存到文件
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info(f"save experience replay buffer by date: {filepath}, "
                       f"date: {latest_date}, experiences: {len(latest_experiences)} "
                       f"(buffer size: {len(self.buffer)}, groups: {list(experiences_by_date.keys())})")
            return True
            
        except Exception as e:
            logger.error(f"save experience replay buffer by date failed: {e}")
            return False
    
    def load_by_date(self, date_str: str, base_dir: str = "./rl_modules/SAC_Portfolio_Allocationl/replay_buffer") -> bool:
        try:
            import os
            import pickle
            
            filename = f"sac_replay_buffer_{date_str}.pkl"
            filepath = os.path.join(base_dir, filename)
            
            if not os.path.exists(filepath):
                logger.warning(f"experience replay buffer for date {date_str} not found: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            self.buffer = deque(save_data['experiences'], maxlen=self.capacity)
            
            logger.info(f"load experience replay buffer by date: {filepath}, "
                       f"date: {save_data.get('date', 'unknown')}, "
                       f"loaded experiences: {len(self.buffer)}")
            return True
            
        except Exception as e:
            logger.error(f"load experience replay buffer by date failed: {e}")
            return False
    
    def get_available_dates(self, base_dir: str = "./rl_modules/SAC_Portfolio_Allocationl/replay_buffer") -> List[str]:
        try:
            import os
            import glob
            
            if not os.path.exists(base_dir):
                return []
            
            files = glob.glob(os.path.join(base_dir, "sac_replay_buffer_*.pkl"))
            
            dates = []
            for file in files:
                filename = os.path.basename(file)
                if filename.startswith("sac_replay_buffer_") and filename.endswith(".pkl"):
                    date_str = filename[len("sac_replay_buffer_"):-4]
                    if len(date_str) == 8 and date_str.isdigit():
                        dates.append(date_str)

            return sorted(dates)
            
        except Exception as e:
            logger.error(f"get available dates failed: {e}")
            return [] 