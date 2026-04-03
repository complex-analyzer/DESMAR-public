#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import logging
import glob
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl_modules.bdq_networks import BDQCore, BDQPolicyHead


def setup_bdq_logger():

    log_dir = os.path.join(os.path.dirname(__file__), 'BDQ_Trade_Execution', 'run_log')
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    log_file = os.path.join(log_dir, 'bdq_agent.log')
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    logger = logging.getLogger(__name__ + ".bdq")
    logger.setLevel(logging.DEBUG)
    # 避免重复添加handler
    if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == file_handler.baseFilename for h in logger.handlers):
        logger.addHandler(file_handler)
    return logger


logger = setup_bdq_logger()

class BDQReplayBuffer:

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.ptr = 0
        self.full = False
        self.storage = []
        self.ts_storage = []  # type: list
        self.last_sampling_debug: Optional[Dict[str, Any]] = None

    def __len__(self):
        return self.capacity if self.full else self.ptr

    def push(self, *transition, timestamp_ns: Optional[int] = None):
        if timestamp_ns is None:
            try:
                timestamp_ns = time.time_ns()
            except Exception:
                timestamp_ns = int(time.time() * 1_000_000_000)
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
            self.ts_storage.append(int(timestamp_ns))
        else:
            self.storage[self.ptr] = transition
            # 若越界，扩展ts_storage
            if len(self.ts_storage) < self.capacity:
                self.ts_storage += [0] * (self.capacity - len(self.ts_storage))
            self.ts_storage[self.ptr] = int(timestamp_ns)
        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size: int):
        idx = np.random.randint(0, len(self), size=batch_size)
        batch = [self.storage[i] for i in idx]
        lob, tr, ap, ar, r, lob2, tr2, d = zip(*batch)
        lob = torch.FloatTensor(np.stack(lob)).to(self.device)
        tr = torch.FloatTensor(np.stack(tr)).to(self.device)
        ap = torch.LongTensor(np.stack(ap)).to(self.device)
        ar = torch.LongTensor(np.stack(ar)).to(self.device)
        r = torch.FloatTensor(np.stack(r)).unsqueeze(1).to(self.device)
        lob2 = torch.FloatTensor(np.stack(lob2)).to(self.device)
        tr2 = torch.FloatTensor(np.stack(tr2)).to(self.device)
        d = torch.FloatTensor(np.stack(d)).unsqueeze(1).to(self.device)
        return lob, tr, ap, ar, r, lob2, tr2, d

    def prioritized_sample(self, batch_size: int, half_life_days: float = 3.0, uniform_mix: float = 0.1):
        buf_len = len(self)
        if buf_len < batch_size:
            raise ValueError(f" ({buf_len}) < ({batch_size})")

        # 边界
        half_life_days = float(max(1e-6, half_life_days))
        uniform_mix = float(min(max(uniform_mix, 0.0), 1.0))

        # 取有效区间的时间戳
        ts_array = np.array([int(self.ts_storage[i]) if i < len(self.ts_storage) else 0 for i in range(buf_len)], dtype=np.int64)
        if not np.any(ts_array > 0):
            now_ns = int(time.time() * 1_000_000_000)
            ts_array[:] = now_ns
        latest_ns = int(np.max(ts_array))
        seconds_per_day = 86400.0
        age_days = (latest_ns - ts_array) / (seconds_per_day * 1_000_000_000.0)
        age_days = np.maximum(age_days, 0.0)

        # 指数半衰权重
        ln2 = np.log(2.0)
        weights = np.exp(-ln2 * (age_days / half_life_days))
        weights = np.maximum(weights, 1e-12)
        wsum = float(np.sum(weights))
        if wsum <= 0.0 or not np.isfinite(wsum):
            probs = np.ones(buf_len, dtype=np.float64) / buf_len
        else:
            probs_recency = weights / wsum
            probs = (1.0 - uniform_mix) * probs_recency + uniform_mix * (1.0 / buf_len)
            probs = probs / probs.sum()

        indices = np.random.choice(buf_len, size=batch_size, p=probs)
        batch = [self.storage[i] for i in indices]
        lob, tr, ap, ar, r, lob2, tr2, d = zip(*batch)

        # 记录调试
        try:
            dates_all = [self._extract_date_from_nanosecond_timestamp(int(ts)) for ts in ts_array]
            dates_unique = sorted(set(dates_all))
            prob_by_date = {}
            for d_i, p in zip(dates_all, probs):
                prob_by_date[d_i] = prob_by_date.get(d_i, 0.0) + float(p)
            from collections import Counter
            cnt = Counter([dates_all[i] for i in indices])
            sample_ratio_by_date = {d_i: cnt[d_i] / float(batch_size) for d_i in cnt}
            self.last_sampling_debug = {
                'latest_date': max(dates_unique) if dates_unique else None,
                'unique_dates': dates_unique,
                'theoretical_ratio_by_date': prob_by_date,
                'sample_ratio_by_date': sample_ratio_by_date,
                'batch_size': int(batch_size),
                'half_life_days': float(half_life_days),
                'uniform_mix': float(uniform_mix),
            }
        except Exception:
            self.last_sampling_debug = None

        lob = torch.FloatTensor(np.stack(lob)).to(self.device)
        tr = torch.FloatTensor(np.stack(tr)).to(self.device)
        ap = torch.LongTensor(np.stack(ap)).to(self.device)
        ar = torch.LongTensor(np.stack(ar)).to(self.device)
        r = torch.FloatTensor(np.stack(r)).unsqueeze(1).to(self.device)
        lob2 = torch.FloatTensor(np.stack(lob2)).to(self.device)
        tr2 = torch.FloatTensor(np.stack(tr2)).to(self.device)
        d = torch.FloatTensor(np.stack(d)).unsqueeze(1).to(self.device)
        return lob, tr, ap, ar, r, lob2, tr2, d

    def clear(self) -> int:
        size = len(self)
        self.storage = []
        self.ts_storage = []
        self.ptr = 0
        self.full = False
        logger.info(f"delete {size} experiences")
        return size

    # --- 与SAC一致的持久化接口（按日期/文件） ---
    def save_to_file(self, filepath: str) -> bool:
        try:
            import pickle
            used_len = len(self)
            data = {
                'experiences': list(self.storage)[:used_len],
                'timestamps': list(self.ts_storage)[:used_len],
                'capacity': self.capacity,
            }
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"replay buffer saved to: {filepath} ({len(data['experiences'])} experiences)")
            return True
        except Exception as e:
            logger.error(f"save replay buffer failed: {e}")
            return False

    def load_from_file(self, filepath: str) -> bool:
        try:
            import pickle
            if not os.path.exists(filepath):
                logger.warning(f"replay buffer not found: {filepath}")
                return False
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            exps = data.get('experiences', [])
            tss = data.get('timestamps', None)
            self.storage = list(exps)[: self.capacity]
            if tss is None:
                now_ns = int(time.time() * 1_000_000_000)
                self.ts_storage = [now_ns] * len(self.storage)
            else:
                self.ts_storage = list(tss)[: len(self.storage)]
            n = len(self.storage)
            self.ptr = n % self.capacity
            self.full = (n >= self.capacity)
            logger.info(f"load replay buffer from: {filepath} ({n} experiences)")
            return True
        except Exception as e:
            logger.error(f"load replay buffer failed: {e}")
            return False

    def _extract_date_from_nanosecond_timestamp(self, ns_timestamp: int) -> str:
        try:
            seconds = ns_timestamp / 1_000_000_000
            import datetime
            dt = datetime.datetime.fromtimestamp(seconds)
            return dt.strftime("%Y%m%d")
        except Exception:
            return "19700101"

    def save_by_date(self, base_dir: str = "./rl_modules/BDQ_Trade_Execution/replay_buffer") -> bool:
        used_len = len(self)
        if used_len == 0:
            logger.warning("BDQ replay buffer is empty, skip saving")
            return False
        try:
            by_date: Dict[str, list] = {}
            for i in range(used_len):
                date_str = self._extract_date_from_nanosecond_timestamp(int(self.ts_storage[i] if i < len(self.ts_storage) else 0))
                by_date.setdefault(date_str, []).append(i)
            latest_date = max(by_date.keys())
            idxs = by_date[latest_date]

            exps = [self.storage[i] for i in idxs]
            tss = [int(self.ts_storage[i] if i < len(self.ts_storage) else 0) for i in idxs]
            data = {
                'experiences': exps,
                'timestamps': tss,
                'capacity': self.capacity,
                'date': latest_date,
                'total_experiences': len(exps),
            }

            os.makedirs(base_dir, exist_ok=True)
            filepath = os.path.join(base_dir, f"bdq_replay_buffer_{latest_date}.pkl")
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"replay buffer saved by date: {filepath}, date: {latest_date}, samples: {len(exps)}")
            return True
        except Exception as e:
            logger.error(f"save replay buffer by date failed: {e}")
            return False

    def load_by_date(self, date_str: str, base_dir: str = "./rl_modules/BDQ_Trade_Execution/replay_buffer") -> bool:
        try:
            filename = f"bdq_replay_buffer_{date_str}.pkl"
            filepath = os.path.join(base_dir, filename)
            return self.load_from_file(filepath)
        except Exception as e:
            logger.error(f"load replay buffer by date failed: {e}")
            return False

    def get_available_dates(self, base_dir: str = "./rl_modules/BDQ_Trade_Execution/replay_buffer") -> list:
        try:
            if not os.path.exists(base_dir):
                return []
            files = glob.glob(os.path.join(base_dir, "bdq_replay_buffer_*.pkl"))
            dates = []
            for f in files:
                name = os.path.basename(f)
                if name.startswith("bdq_replay_buffer_") and name.endswith(".pkl"):
                    d = name[len("bdq_replay_buffer_"):-4]
                    if len(d) == 8 and d.isdigit():
                        dates.append(d)
            return sorted(dates)
        except Exception as e:
            logger.error(f"get available dates for replay buffer failed: {e}")
            return []


class BDQAgent:
    def __init__(
        self,
        lob_depth: int,
        lob_history_len: int,
        ratio_branches: int = 6,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        batch_size: int = 64,
        replay_size: int = 200000,
        device: str = 'auto',
        seed: Optional[int] = 1,
        lstm_hidden_size: int = 128,
        trade_hidden: int = 128,
        fused_dim: int = 256,
        mlp_dropout: float = 0.1,
        trade_gate: float = 1.0,
        eps_start: float = 0.2,
        eps_end: float = 0.01,
        eps_decay_steps: int = 100000,
        # CNN params
        cnn_pair_filters: int = 16,
        cnn_leaky_relu_neg_slope: float = 0.01,
        temporal_kernel: int = 2,
        inception_enabled: bool = True,
        inception_out_channels: int = 32,
        inception_pool_time_kernel: int = 3,
        temporal_repeats: int = 1,
        lstm_input_dropout: float = 0.2,
        lstm_output_dropout: float = 0.0,
    ):
        self.lob_depth = int(lob_depth)
        self.lob_history_len = int(lob_history_len)
        self.lob_feature_dim = 4 * self.lob_depth
        self.price_branches = self.lob_depth + 1
        self.ratio_branches = int(ratio_branches)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        try:
            tg = float(trade_gate)
        except Exception:
            tg = 1.0
        self.trade_gate = max(0.0, min(tg, 1.0))
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        if seed is not None:
            try:
                import random
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

        self.q_main = BDQCore(
            lob_feature_dim=self.lob_feature_dim,
            lob_history_len=self.lob_history_len,
            lob_depth=self.lob_depth,
            price_branch_dim=self.price_branches,
            ratio_branch_dim=self.ratio_branches,
            lstm_hidden_size=lstm_hidden_size,
            trade_hidden=trade_hidden,
            fused_dim=fused_dim,
            mlp_dropout=mlp_dropout,
            trade_gate=self.trade_gate,
            cnn_pair_filters=cnn_pair_filters,
            cnn_leaky_relu_neg_slope=cnn_leaky_relu_neg_slope,
            temporal_kernel=temporal_kernel,
            inception_enabled=inception_enabled,
            inception_out_channels=inception_out_channels,
            inception_pool_time_kernel=inception_pool_time_kernel,
            temporal_repeats=temporal_repeats,
            lstm_input_dropout=lstm_input_dropout,
            lstm_output_dropout=lstm_output_dropout,
        ).to(self.device)
        self.q_target = BDQCore(
            lob_feature_dim=self.lob_feature_dim,
            lob_history_len=self.lob_history_len,
            lob_depth=self.lob_depth,
            price_branch_dim=self.price_branches,
            ratio_branch_dim=self.ratio_branches,
            lstm_hidden_size=lstm_hidden_size,
            trade_hidden=trade_hidden,
            fused_dim=fused_dim,
            mlp_dropout=mlp_dropout,
            trade_gate=self.trade_gate,
            cnn_pair_filters=cnn_pair_filters,
            cnn_leaky_relu_neg_slope=cnn_leaky_relu_neg_slope,
            temporal_kernel=temporal_kernel,
            inception_enabled=inception_enabled,
            inception_out_channels=inception_out_channels,
            inception_pool_time_kernel=inception_pool_time_kernel,
            temporal_repeats=temporal_repeats,
            lstm_input_dropout=lstm_input_dropout,
            lstm_output_dropout=lstm_output_dropout,
        ).to(self.device)
        self.q_target.load_state_dict(self.q_main.state_dict())
        self.policy_head = BDQPolicyHead(self.q_main)

        self.optimizer = optim.Adam(self.q_main.parameters(), lr=lr)
        self.replay = BDQReplayBuffer(replay_size, self.device)

        self.train_steps = 0
        self.total_training_steps = 0
        # epsilon贪心
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay_steps = int(eps_decay_steps)

        # 运行会话与目录
        self.session_timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.history_replay_buffer_days = 10
        self._ensure_dirs()

        logger.info(
            f"BDQ agent initialized | seed: {seed}, lob_depth={self.lob_depth}, history_len={self.lob_history_len}, "
            f"price_branches={self.price_branches}, ratio_branches={self.ratio_branches}, device={self.device}"
        )
        try:
            logger.info(f"trade gate: {self.trade_gate:.3f} (weight ratio of trade features to LOB features, 1.0=equal weight)")
        except Exception:
            pass

    # ---------- inference ----------
    @torch.no_grad()
    def select_actions(self, lob_seq: np.ndarray, trade_vec: np.ndarray) -> Tuple[int, int]:
        self.q_main.eval()
        lob = torch.from_numpy(lob_seq).float().unsqueeze(0).to(self.device)  # (1,K,F)
        tr = torch.from_numpy(trade_vec).float().unsqueeze(0).to(self.device)  # (1,4)
        a_p, a_r = self.policy_head(lob, tr)
        return int(a_p.item()), int(a_r.item())

    @torch.no_grad()
    def select_actions_epsilon_greedy(self, lob_seq: np.ndarray, trade_vec: np.ndarray) -> Tuple[int, int]:
        eps = self._current_epsilon()
        if np.random.rand() < eps:
            ap = np.random.randint(0, self.price_branches)
            ar = np.random.randint(0, self.ratio_branches)
            return int(ap), int(ar)
        return self.select_actions(lob_seq, trade_vec)

    # ---------- add experience ----------
    def add_experience(self, s: Tuple[np.ndarray, np.ndarray], a_price: int, a_ratio: int, r: float, s2: Tuple[np.ndarray, np.ndarray], done: bool, timestamp_ns: Optional[int] = None):
        lob, tr = s
        lob2, tr2 = s2
        if timestamp_ns is None:
            try:
                timestamp_ns = time.time_ns()
            except Exception:
                timestamp_ns = int(time.time() * 1_000_000_000)
        shaped_r = float(r)
        try:
            if bool(getattr(self, 'reward_shaping_enabled', False)):
                filled_next = None
                try:
                    tr2_arr = np.array(tr2)
                    if tr2_arr.size >= 3:
                        filled_next = float(tr2_arr.reshape(-1)[2])
                except Exception:
                    try:
                        filled_next = float(tr2[2]) if len(tr2) >= 3 else None
                    except Exception:
                        filled_next = None

                # 终局连续/分段惩罚（基于成交完成率）
                if bool(done) and (filled_next is not None):
                    mode = str(getattr(self, 'reward_shaping_mode', 'piecewise'))
                    max_penalty = float(getattr(self, 'reward_max_penalty', 99.0))
                    succ_th = float(getattr(self, 'success_filled_threshold', 0.95))
                    if mode == 'piecewise':
                        if filled_next >= succ_th:
                            shaped_r = 0.0
                        else:
                            unfilled_ratio = 1.0 - filled_next
                            scale = (unfilled_ratio - (1.0 - succ_th)) / max(1e-8, (1.0 - (1.0 - succ_th)))
                            shaped_r = -max_penalty * float(min(max(scale, 0.0), 1.0))
                    else:  # 'linear'
                        unfilled_ratio = 1.0 - filled_next
                        shaped_r = -max_penalty * float(min(max(unfilled_ratio, 0.0), 1.0))
                    if shaped_r > 0.0:
                        shaped_r = 0.0
                    if shaped_r < -max_penalty:
                        shaped_r = -max_penalty
        except Exception:
            shaped_r = float(r)

        self.replay.push(lob, tr, a_price, a_ratio, float(shaped_r), lob2, tr2, float(done), timestamp_ns=int(timestamp_ns))
        try:
            self._latest_sim_timestamp_ns = int(timestamp_ns)
        except Exception:
            self._latest_sim_timestamp_ns = None

    def can_train(self) -> bool:
        return len(self.replay) >= self.batch_size

    # ---------- train ----------
    def train_step(self) -> Dict[str, float]:
        if getattr(self, 'prioritized_replay_enabled', False):
            lob, tr, ap, ar, r, lob2, tr2, d = self.replay.prioritized_sample(
                self.batch_size,
                half_life_days=getattr(self, 'prioritized_replay_half_life_days', 3.0),
                uniform_mix=getattr(self, 'prioritized_replay_uniform_mix', 0.1),
            )
            try:
                dbg = self.replay.last_sampling_debug or {}
                latest = dbg.get('latest_date')
                logger.info(
                    f"time-based sampling | latest_date={latest} half_life={dbg.get('half_life_days')} uniform_mix={dbg.get('uniform_mix')} dist={dbg.get('sample_ratio_by_date')}"
                )
            except Exception:
                pass
        else:
            lob, tr, ap, ar, r, lob2, tr2, d = self.replay.sample(self.batch_size)

        # calculate y_d on target network
        with torch.no_grad():
            _, q_p_next, q_r_next = self.q_target(lob2, tr2)
            a_p_next = q_p_next.argmax(dim=-1)
            a_r_next = q_r_next.argmax(dim=-1)
            q_pn = q_p_next.gather(1, a_p_next.view(-1, 1))
            q_rn = q_r_next.gather(1, a_r_next.view(-1, 1))
            y_p = r + (1 - d) * self.gamma * q_pn
            y_r = r + (1 - d) * self.gamma * q_rn

        # current Q on main network
        _, q_p, q_r = self.q_main(lob, tr)
        q_p_sel = q_p.gather(1, ap.view(-1, 1))
        q_r_sel = q_r.gather(1, ar.view(-1, 1))

        loss_p = F.mse_loss(q_p_sel, y_p)
        loss_r = F.mse_loss(q_r_sel, y_r)
        loss = loss_p + loss_r

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_main.parameters(), max_norm=5.0)
        self.optimizer.step()

        # soft update
        with torch.no_grad():
            for tp, p in zip(self.q_target.parameters(), self.q_main.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

        self.train_steps += 1
        return {
            'loss_total': float(loss.item()),
            'loss_price': float(loss_p.item()),
            'loss_ratio': float(loss_r.item()),
        }

    def train(self, steps: int = 1) -> Dict[str, float]:
        stats_last: Dict[str, float] = {}
        steps_done = 0
        t0 = time.perf_counter()
        try:
            logger.info(
                f"training parameters: replay size={len(self.replay)} batch size={self.batch_size} steps={steps} | "
                f"prioritized replay: enabled={getattr(self, 'prioritized_replay_enabled', False)} "
                f"half_life={getattr(self, 'prioritized_replay_half_life_days', 3.0)} "
                f"uniform_mix={getattr(self, 'prioritized_replay_uniform_mix', 0.1)}"
            )
        except Exception:
            pass
        for _ in range(steps):
            if not self.can_train():
                break
            stats_last = self.train_step()
            steps_done += 1
        if steps_done > 0 and stats_last:
            self.total_training_steps += steps_done
            stats_last = dict(stats_last)
            stats_last['num_steps'] = steps_done
            stats_last['train_time'] = float(time.perf_counter() - t0)
            self._log_training_progress(stats_last)
            try:
                logger.info(
                    f"training completed - steps: {steps_done}, total steps: {self.total_training_steps}, "
                    f"loss_total: {stats_last.get('loss_total'):.4f}, "
                    f"loss_price: {stats_last.get('loss_price'):.4f}, "
                    f"loss_ratio: {stats_last.get('loss_ratio'):.4f}, "
                    f"time: {stats_last.get('train_time'):.2f}s"
                )
            except Exception:
                pass
        return stats_last

    # ---------- save/load ----------
    def save(self, prefix: str) -> str:
        path = f"{prefix}_bdq.pth"
        torch.save({
            'q_main': self.q_main.state_dict(),
            'q_target': self.q_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_training_steps': self.total_training_steps,
            'train_steps': self.train_steps,
            'cfg': {
                'lob_depth': self.lob_depth,
                'lob_history_len': self.lob_history_len,
                'ratio_branches': self.ratio_branches,
            }
        }, path)
        return path

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        ckpt = torch.load(path, map_location=self.device)
        self.q_main.load_state_dict(ckpt['q_main'])
        self.q_target.load_state_dict(ckpt['q_target'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        # restore training step state to maintain continuity
        self.total_training_steps = ckpt.get('total_training_steps', 0)
        self.train_steps = ckpt.get('train_steps', 0)
        return True

    def save_models(self, filepath_prefix: str) -> bool:
        try:
            out = self.save(filepath_prefix)
            logger.info(f"BDQ model saved: {out}")
            # export TorchScript inference module for C++ side
            try:
                class BDQInferModule(torch.nn.Module):
                    def __init__(self, core_module: torch.nn.Module):
                        super().__init__()
                        self.lob_encoder = core_module.lob_encoder
                        self.trade_encoder = core_module.trade_encoder
                        self.fusion = core_module.fusion
                        self.value_head = core_module.value_head
                        self.adv_price = core_module.adv_price
                        self.adv_ratio = core_module.adv_ratio
                        # expose branch dimensions for C++ side, avoid hardcoding
                        try:
                            price_dim = int(getattr(self.adv_price[-1], 'out_features', 0))
                        except Exception:
                            price_dim = 0
                        try:
                            ratio_dim = int(getattr(self.adv_ratio[-1], 'out_features', 0))
                        except Exception:
                            ratio_dim = 0
                        self.register_buffer('price_branch_dim', torch.tensor(price_dim, dtype=torch.int64))
                        self.register_buffer('ratio_branch_dim', torch.tensor(ratio_dim, dtype=torch.int64))
                        # expose gate coefficient for C++ side
                        try:
                            trade_gate_val = float(getattr(core_module, 'trade_gate', 1.0))
                            self.register_buffer('trade_gate', torch.tensor(trade_gate_val, dtype=torch.float32))
                        except Exception:
                            pass
                    def forward(self, lob_seq: torch.Tensor, trade_vec: torch.Tensor):
                        lob_feat = self.lob_encoder(lob_seq)
                        trade_feat = self.trade_encoder(trade_vec)
                        shared = self.fusion(lob_feat, trade_feat)
                        V = self.value_head(shared)
                        A_p = self.adv_price(shared)
                        A_r = self.adv_ratio(shared)
                        Q_p = V + (A_p - A_p.mean(dim=-1, keepdim=True))
                        Q_r = V + (A_r - A_r.mean(dim=-1, keepdim=True))
                        a_p = torch.argmax(Q_p, dim=-1)
                        a_r = torch.argmax(Q_r, dim=-1)
                        return a_p, a_r
                self.q_main.eval()
                wrapper = BDQInferModule(self.q_main)
                scripted = torch.jit.script(wrapper)
                ts_path = f"{filepath_prefix}_policy.pt"
                torch.jit.save(scripted, ts_path)
                # print exported gate value
                try:
                    gate_val = float(getattr(self.q_main, 'trade_gate', 1.0))
                    logger.info(f"BDQ TorchScript policy exported: {ts_path} | trade_gate={gate_val:.3f}")
                except Exception:
                    logger.info(f"BDQ TorchScript policy exported: {ts_path}")
            except Exception as e:
                logger.warning(f"export BDQ TorchScript policy failed: {e}")
            return True
        except Exception as e:
            logger.error(f"save BDQ model failed: {e}")
            return False

    def load_models(self, filepath: str) -> bool:
        try:
            ok = self.load(filepath)
            if ok:
                logger.info(f"BDQ model loaded successfully: {filepath}")
            else:
                logger.warning(f"BDQ model loaded failed: {filepath}")
            return ok
        except Exception as e:
            logger.error(f"load BDQ model failed: {e}")
            return False

    # ---------- runtime auxiliary ----------
    def _ensure_dirs(self) -> None:
        base = os.path.join(os.path.dirname(__file__), 'BDQ_Trade_Execution')
        for sub in ['saved_models', 'replay_buffer', 'run_log']:
            os.makedirs(os.path.join(base, sub), exist_ok=True)

    def cleanup(self):
        """cleanup: save final model and current replay buffer."""
        try:
            save_dir = os.path.join('.', 'rl_modules', 'BDQ_Trade_Execution', 'saved_models')
            os.makedirs(save_dir, exist_ok=True)
            # use simulation date for naming
            date_str = self._get_simulation_date_for_naming()
            filepath_prefix = os.path.join(save_dir, f"bdq_agent_final_{date_str}")
            self.save_models(filepath_prefix)
        except Exception as e:
            logger.error(f"save final BDQ model failed: {e}")

        try:
            if len(self.replay) > 0:
                ok = self.replay.save_by_date(base_dir=os.path.join('.', 'rl_modules', 'BDQ_Trade_Execution', 'replay_buffer'))
                if ok:
                    logger.info("BDQ replay buffer saved by date")
                else:
                    logger.warning("BDQ replay buffer save failed")
            else:
                logger.info("BDQ replay buffer is empty, skipping save")
        except Exception as e:
            logger.error(f"save BDQ replay buffer failed: {e}")

    def _log_training_progress(self, stats: Dict[str, Any]) -> None:
        try:
            save_dir = os.path.join('.', 'rl_modules', 'BDQ_Trade_Execution', 'saved_models')
            os.makedirs(save_dir, exist_ok=True)
            # use simulation date for naming training progress file (append to same day)
            date_str = self._get_simulation_date_for_naming()
            filepath = os.path.join(save_dir, f"training_progress_{date_str}.csv")
            file_exists = os.path.exists(filepath)
            with open(filepath, 'a', encoding='utf-8') as f:
                if not file_exists:
                    headers = ['timestamp', 'total_steps'] + list(stats.keys())
                    f.write(','.join(headers) + '\n')
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                values = [timestamp, str(self.total_training_steps)] + [str(v) for v in stats.values()]
                f.write(','.join(values) + '\n')
        except Exception as e:
            logger.error(f"record BDQ training progress failed: {e}")

    def _get_simulation_date_for_naming(self) -> str:
        """use latest simulation timestamp; otherwise infer from replay/saved files; finally revert to system date."""
        # highest priority: learner passed in and recorded simulation timestamp (same as SAC)
        try:
            ts_latest = int(getattr(self, '_latest_sim_timestamp_ns', 0) or 0)
            if ts_latest > 0:
                import datetime
                dt = datetime.datetime.fromtimestamp(ts_latest / 1_000_000_000)
                return dt.strftime("%Y%m%d")
        except Exception:
            pass
        # secondary priority: infer from current replay buffer timestamp
        try:
            used_len = len(self.replay)
            if used_len > 0 and len(self.replay.ts_storage) > 0:
                latest_ns = 0
                for i in range(used_len):
                    try:
                        ts = int(self.replay.ts_storage[i] if i < len(self.replay.ts_storage) else 0)
                    except Exception:
                        ts = 0
                    if ts > latest_ns:
                        latest_ns = ts
                if latest_ns > 0:
                    import datetime
                    dt = datetime.datetime.fromtimestamp(latest_ns / 1_000_000_000)
                    return dt.strftime("%Y%m%d")
        except Exception:
            pass
        # third choice: read saved replay file date
        try:
            base_dir = os.path.join('.', 'rl_modules', 'BDQ_Trade_Execution', 'replay_buffer')
            dates = self.replay.get_available_dates(base_dir=base_dir)
            if dates:
                return dates[-1]
        except Exception:
            pass
        # revert to system current date
        try:
            return time.strftime("%Y%m%d")
        except Exception:
            return "19700101"

    def _auto_load_latest_model(self):
        try:
            save_dir = os.path.join('.', 'rl_modules', 'BDQ_Trade_Execution', 'saved_models')
            pattern = os.path.join(save_dir, "bdq_agent_final_*.pth")
            files = glob.glob(pattern)
            if not files:
                logger.info("BDQ did not find historical model file, starting from scratch")
                return
            # 加载前状态
            try:
                logger.info(
                    f"before loading state: total_training_steps={self.total_training_steps}, replay size={len(self.replay)}"
                )
            except Exception:
                pass
            latest = max(files, key=os.path.getmtime)
            logger.info(f"BDQ found historical model file: {latest}")
            if self.load_models(latest):
                try:
                    logger.info(
                        f"successfully loaded historical model: {latest} | restored state: total_training_steps={self.total_training_steps}, train_steps={self.train_steps}, replay size={len(self.replay)}"
                    )
                    logger.info("BDQ continue training mode enabled, training steps will be recorded continuously")
                except Exception:
                    pass
            else:
                logger.warning(f"load model failed: {latest}")
        except Exception as e:
            logger.error(f"auto load BDQ model failed: {e}")

    def _auto_load_recent_replay_buffers(self):
        try:
            base_dir = os.path.join('.', 'rl_modules', 'BDQ_Trade_Execution', 'replay_buffer')
            dates = self.replay.get_available_dates(base_dir=base_dir)
            if not dates:
                logger.info("BDQ did not find historical replay buffer")
                return
            days_to_load = min(self.history_replay_buffer_days, len(dates))
            recent = dates[-days_to_load:]
            logger.info(f"BDQ preparing to load recent {days_to_load} days of replay: {recent}")
            try:
                logger.info(f"before loading replay size: {len(self.replay)}")
            except Exception:
                pass
            self.replay.clear()
            total = 0
            successful_dates = []
            # use temporary buffer day by day to load and accumulate
            for d in recent:
                try:
                    tmp = BDQReplayBuffer(self.replay.capacity, self.device)
                    if tmp.load_by_date(d, base_dir=base_dir):
                        # append all samples (keep original timestamps)
                        for i, t in enumerate(tmp.storage[: len(tmp)]):
                            tmp_lob, tmp_tr, tmp_ap, tmp_ar, tmp_r, tmp_lob2, tmp_tr2, tmp_d = t
                            # get corresponding original timestamp
                            original_ts = tmp.ts_storage[i] if i < len(tmp.ts_storage) else None
                            self.replay.push(tmp_lob, tmp_tr, tmp_ap, tmp_ar, tmp_r, tmp_lob2, tmp_tr2, tmp_d, timestamp_ns=original_ts)
                        total += len(tmp)
                        logger.info(f"BDQ loaded {d} experience: {len(tmp)}")
                        successful_dates.append(d)
                except Exception as e:
                    logger.warning(f"BDQ load {d} replay failed: {e}")
            logger.info(
                f"BDQ historical replay loaded successfully | successful dates: {successful_dates} | total loaded: {total} | current replay size: {len(self.replay)}"
            )
        except Exception as e:
            logger.error(f"auto load BDQ historical replay failed: {e}")

    # ---------- tools ----------
    def _current_epsilon(self) -> float:
        t = max(0, self.train_steps)
        if self.eps_decay_steps <= 0:
            return self.eps_end
        r = max(0.0, 1.0 - float(t) / float(self.eps_decay_steps))
        return self.eps_end + (self.eps_start - self.eps_end) * r

    @classmethod
    def from_config(cls, config_path: str = "Simulator_configs/config_templates/sac_config.yaml", **overrides):
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        bdq = cfg.get('bdq', {})
        params = dict(
            lob_depth=bdq.get('lob_depth', 10),
            lob_history_len=bdq.get('lob_history_len', 20),
            ratio_branches=bdq.get('ratio_branches', 6),
            gamma=bdq.get('gamma', 0.99),
            tau=bdq.get('tau', 0.005),
            lr=bdq.get('lr', 3e-4),
            batch_size=bdq.get('batch_size', 64),
            replay_size=bdq.get('replay_buffer_size', 200000),
            device=bdq.get('device', cfg.get('device', 'auto')),
            seed=bdq.get('seed', 1), 
            lstm_hidden_size=bdq.get('lstm_hidden_size', 128),
            lstm_input_dropout=bdq.get('lstm_input_dropout', 0.2),
            lstm_output_dropout=bdq.get('lstm_output_dropout', 0.0),
            trade_hidden=bdq.get('trade_hidden', 128),
            fused_dim=bdq.get('fused_dim', 256),
            mlp_dropout=bdq.get('mlp_dropout', 0.1),
            # trade feature weight gate
            trade_gate=float(bdq.get('trade_gate', 1.0)),
            # CNN params
            cnn_pair_filters=bdq.get('cnn_pair_filters', 16),
            cnn_leaky_relu_neg_slope=bdq.get('cnn_leaky_relu_neg_slope', 0.01),
            temporal_kernel=bdq.get('temporal_kernel', 2),
            inception_enabled=bdq.get('inception_enabled', True),
            inception_out_channels=bdq.get('inception_out_channels', 32),
            inception_pool_time_kernel=bdq.get('inception_pool_time_kernel', 3),
            temporal_repeats=bdq.get('temporal_repeats', 1),
            eps_start=bdq.get('eps_start', 0.2),
            eps_end=bdq.get('eps_end', 0.01),
            eps_decay_steps=bdq.get('eps_decay_steps', 100000),
        )
        params.update(overrides)
        agent = cls(**params)
        # history replay buffer days (priority bdq config, then top level)
        try:
            agent.history_replay_buffer_days = int(bdq.get('history_replay_buffer_days', cfg.get('history_replay_buffer_days', 10)))
        except Exception:
            agent.history_replay_buffer_days = 10
        # prioritized replay开关与参数（与SAC字段对齐）
        agent.prioritized_replay_enabled = bool(bdq.get('prioritized_replay_enabled', cfg.get('prioritized_replay_enabled', False)))
        agent.prioritized_replay_half_life_days = float(bdq.get('prioritized_replay_half_life_days', cfg.get('prioritized_replay_half_life_days', 3.0)))
        agent.prioritized_replay_uniform_mix = float(bdq.get('prioritized_replay_uniform_mix', cfg.get('prioritized_replay_uniform_mix', 0.1)))
        # reward shaping configuration
        rs_cfg = bdq.get('reward_shaping', {}) or {}
        agent.reward_shaping_enabled = bool(rs_cfg.get('enabled', False))
        agent.reward_shaping_mode = str(rs_cfg.get('mode', 'piecewise'))
        agent.reward_max_penalty = float(rs_cfg.get('max_penalty', 99.0))
        agent.success_filled_threshold = float(rs_cfg.get('success_filled_threshold', 0.95))
        # print critical initialization configuration to run_log
        try:
            logger.info(
                f"history replay buffer days: {agent.history_replay_buffer_days} | "
                f"prioritized replay: enabled={agent.prioritized_replay_enabled}, half_life_days={agent.prioritized_replay_half_life_days}, uniform_mix={agent.prioritized_replay_uniform_mix}"
            )
            logger.info(
                f"reward shaping: enabled={agent.reward_shaping_enabled}, mode={agent.reward_shaping_mode}, max_penalty={agent.reward_max_penalty}, success_filled_th={agent.success_filled_threshold}"
            )
            # 打印门控机制配置（MI优化相关）
            config_gate = bdq.get('trade_gate', 1.0)
            if config_gate != 1.0:
                logger.info(
                    f"trade feature weight gate enabled: trade_gate={agent.trade_gate:.3f} (config value={config_gate}) | "
                    f"LOB feature: trade feature = 1:{agent.trade_gate:.1f} = {1.0/max(0.001, agent.trade_gate):.1f}:1"
                )
                logger.info("  ↳ purpose: reduce dependency on trade state features (remaining time/remaining quantity), enhance LOB feature focus")
            else:
                logger.info(f"trade feature weight gate: not enabled (trade_gate={agent.trade_gate:.3f}, LOB and trade feature equal weight)")
        except Exception:
            pass
        # auto load model and recent replay
        agent._auto_load_latest_model()
        agent._auto_load_recent_replay_buffers()
        return agent



