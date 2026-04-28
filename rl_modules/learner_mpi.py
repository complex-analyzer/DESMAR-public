#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mpi4py import MPI
import numpy as np
import os
import sys
import time
from typing import List
import yaml

import io
import numpy as np
import threading
import queue
import struct
import torch
import torch.nn.functional as F
from rl_modules.sac_agent import SACAgent
from rl_modules.bdq_agent import BDQAgent
from rl_modules.experience_replay import Experience


LEARNER_EXP_LEN_TAG = 9001
LEARNER_EXP_DATA_TAG = 9002
LEARNER_CTRL_END_TAG = 9003
LEARNER_PARAM_LEN_TAG = 9004
LEARNER_PARAM_DATA_TAG = 9005
LEARNER_COMM_CREATE_TAG = 9106

BATCH_MAGIC = b'BEXP'


def decode_json_basic(js: str):
    import json
    d = json.loads(js)
    agent = d.get('agent', '')
    s = d['s']
    a = d['a']
    r = d['r']
    s_next = d['s_next']
    assets = int(d['assets'])
    window = int(d['window'])
    feat = int(d['feat'])
    return agent, s, a, r, s_next, assets, window, feat


def main():
    LOG_PREFIX = "[learner_mpi] "
    cfg = os.environ.get('DESMAR_SAC_CONFIG', 'Simulator_configs/config_templates/sac_config.yaml')
    agent = None
    bdq_agent = None

    world = MPI.COMM_WORLD
    rank = world.Get_rank()
    scripted_actor_cache = None
    bdq_scripted_policy_cache = None
    try:
        print(f"[learner_mpi] starting rank={rank} size={world.Get_size()}", flush=True)
    except Exception:
        pass

    # IMPORTANT (MPMD compatibility):
    # Do NOT call MPI_COMM_WORLD collectives here. In our MPMD layout, kernel/agent ranks are C++ processes
    # and will not participate in arbitrary Python-side collectives, which would deadlock the whole job.

    if not os.environ.get('DESMAR_KERNEL_RANKS'):
        try:
            send = np.array([-1], dtype='i')
            recv = np.empty(world.Get_size(), dtype='i')
            world.Allgather([send, MPI.INT], [recv, MPI.INT])
            print("[learner_mpi] world Allgather done", flush=True)
        except Exception as e:
            print(f"[learner_mpi] world Allgather(int) failed: {e}", flush=True)
    else:
        print("[learner_mpi] skip world Allgather (DESMAR_KERNEL_RANKS provided)", flush=True)

    members_str = os.environ.get('LEARNER_CROSS_MEMBERS', '')
    if not members_str:
        print('[learner_mpi] LEARNER_CROSS_MEMBERS not provided, exiting')
        return 0
    members = sorted(set(int(x) for x in members_str.split(',') if x))

    lcomm = MPI.COMM_NULL
    try:
        g = MPI.COMM_WORLD.Get_group()
        lgrp = g.Incl(members)
        print(f"[learner_mpi] learnerComm create enter members={members} tag={LEARNER_COMM_CREATE_TAG}", flush=True)
        lcomm = MPI.COMM_WORLD.Create_group(lgrp, LEARNER_COMM_CREATE_TAG)
        print(f"[learner_mpi] learnerComm create exit is_null={lcomm == MPI.COMM_NULL} tag={LEARNER_COMM_CREATE_TAG}", flush=True)
        lgrp.Free(); g.Free()
    except Exception as e:
        print('[learner_mpi] create learnerComm failed:', e)
    if lcomm == MPI.COMM_NULL:
        print('[learner_mpi] learnerComm is NULL; fallback to WORLD (may block)')
        lcomm = MPI.COMM_WORLD
    lx_comm = lcomm
    lx_rank = lx_comm.Get_rank()
    learner_root_local = lx_rank

    try:
        enable_param_bcast = (lcomm != MPI.COMM_WORLD) and (lx_comm.Get_size() > 1)
    except Exception:
        enable_param_bcast = False

    recv_count = 0
    bcast_every_steps = 100
    idle_timeout_sec = 480
    data_recv_timeout_sec = 5
    train_mode = 'both'
    bdq_min_exp = 1000
    bdq_train_freq = 200
    bdq_train_steps = 2
    bdq_experience_count = 0
    bdq_last_training_count = 0
    bdq_bcast_every_steps = -1
    try:
        with open(cfg, 'r', encoding='utf-8') as f:
            _conf = yaml.safe_load(f) or {}
        _dist = (_conf.get('distributed') or {})
        _dist_sac = (_dist.get('sac') or {})
        bcast_every_steps = int(_dist_sac.get('params_broadcast_every_steps', bcast_every_steps))
        train_mode = str(_dist.get('train_mode', train_mode)).lower()
        # BDQ 训练触发
        _bdq = (_conf.get('bdq') or {})
        bdq_min_exp = int(_bdq.get('min_experiences_before_training', bdq_min_exp))
        bdq_train_freq = int(_bdq.get('train_frequency', bdq_train_freq))
        bdq_train_steps = int(_bdq.get('train_steps_per_update', bdq_train_steps))
        # BDQ 广播步长读取 distributed.bdq.params_broadcast_every_steps
        bdq_bcast_every_steps = int((_dist.get('bdq') or {}).get('params_broadcast_every_steps', bdq_bcast_every_steps))
    except Exception:
        pass
    last_bcast_steps = 0
    last_bdq_bcast_steps = 0

    exp_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=1024)
    running = True
    ctrl_end_event = threading.Event()

    def _recv_loop():
        nonlocal running
        len_status = MPI.Status()
        buf_len = np.empty(1, dtype='i')
        req_len = world.Irecv([buf_len, MPI.INT], source=MPI.ANY_SOURCE, tag=LEARNER_EXP_LEN_TAG)
        req_data = None
        data = None
        data_src = None
        data_deadline = 0.0
        while running:
            try:
                try:
                    if world.Iprobe(source=MPI.ANY_SOURCE, tag=LEARNER_CTRL_END_TAG):
                        dummy = np.empty(1, dtype='i')
                        world.Recv([dummy, MPI.INT], source=MPI.ANY_SOURCE, tag=LEARNER_CTRL_END_TAG)
                        try:
                            if req_data is not None:
                                req_data.Cancel(); req_data.Free()
                        except Exception:
                            pass
                        try:
                            if req_len is not None:
                                req_len.Cancel(); req_len.Free()
                        except Exception:
                            pass
                        ctrl_end_event.set()
                        running = False
                        break
                except Exception:
                    pass

                if req_data is None:
                    if not req_len.Test(status=len_status):
                        time.sleep(0.0005)
                        continue
                    length = int(buf_len[0])
                    
                    if length < 0:
                        break
                    if length == 0:
                        buf_len[...] = 0
                        len_status = MPI.Status()
                        req_len = world.Irecv([buf_len, MPI.INT], source=MPI.ANY_SOURCE, tag=LEARNER_EXP_LEN_TAG)
                        continue
                    data_src = len_status.Get_source()
                    data = np.empty(length, dtype='b')
                    req_data = world.Irecv([data, MPI.CHAR], source=data_src, tag=LEARNER_EXP_DATA_TAG)
                    data_deadline = time.time() + float(max(1, data_recv_timeout_sec))
                else:
                    if req_data.Test():
                        try:
                            exp_queue.put(data.tobytes())
                        except Exception:
                            pass
                        req_data = None
                        data = None
                        data_src = None
                        buf_len[...] = 0
                        len_status = MPI.Status()
                        req_len = world.Irecv([buf_len, MPI.INT], source=MPI.ANY_SOURCE, tag=LEARNER_EXP_LEN_TAG)
                        continue
                    if time.time() > data_deadline:
                        try:
                            print(f"[learner_mpi] data recv timeout; src={data_src} len={0 if data is None else data.size} sec={data_recv_timeout_sec}")
                        except Exception:
                            pass
                        try:
                            req_data.Cancel(); req_data.Free()
                        except Exception:
                            pass
                        req_data = None
                        data = None
                        data_src = None
                        buf_len[...] = 0
                        len_status = MPI.Status()
                        req_len = world.Irecv([buf_len, MPI.INT], source=MPI.ANY_SOURCE, tag=LEARNER_EXP_LEN_TAG)
                    else:
                        time.sleep(0.0005)
            except Exception:
                time.sleep(0.001)

    t_recv = threading.Thread(target=_recv_loop, name="LearnerExpRecv", daemon=True)
    t_recv.start()

    training_queue = queue.Queue()
    training_results_queue = queue.Queue()
    
    model_lock = threading.Lock()
    replay_lock = threading.Lock()
    
    def training_worker():
        while running:
            try:
                task = training_queue.get(timeout=0.1)
                if task is None:
                    break
                    
                task_type, agent_obj, extra_params = task
                result = None
                
                try:
                    with replay_lock:
                        with model_lock:
                            if task_type == 'SAC':
                                result = agent_obj.train()
                            elif task_type == 'BDQ':
                                steps = extra_params.get('steps', 2)
                                result = agent_obj.train(steps=steps)
                        
                    training_results_queue.put((task_type, result))
                    
                except Exception as e:
                    if lx_rank == learner_root_local:
                        print(f"[learner_mpi] {task_type} training error: {e}")
                    training_results_queue.put((task_type, {}))
                    
            except queue.Empty:
                continue
            except Exception:
                continue
                
    training_thread = threading.Thread(target=training_worker, daemon=True)
    training_thread.start()

    def _process_single_json(js: str):
        nonlocal agent, bdq_agent, recv_count, last_bcast_steps, last_bdq_bcast_steps, bdq_experience_count, bdq_last_training_count
        import json as _json
        _d = _json.loads(js)
        agent_name_local = _d.get('agent', '') or ''
        ts_ns_local = int(_d.get('ts_ns', 0)) if 'ts_ns' in _d else 0
        if ts_ns_local <= 0:
            ts_ns_local = time.time_ns()

        is_bdq = False
        bdq_keys = ('lob', 'lob_next', 'trade', 'trade_next', 'a_price', 'a_ratio')
        if any(k in _d for k in bdq_keys) or ('bdq' in agent_name_local.lower()):
            is_bdq = True

        try:
            if train_mode == 'sac_only' and is_bdq:
                if lx_rank == learner_root_local:
                    print('[learner_mpi][Gate] drop BDQ experience in sac_only mode')
                return
            if train_mode == 'bdq_only' and (not is_bdq):
                if lx_rank == learner_root_local:
                    print('[learner_mpi][Gate] drop SAC experience in bdq_only mode')
                return
        except Exception:
            pass

        if is_bdq:
            try:
                lob = np.array(_d.get('lob'), dtype=np.float32)
                lob2 = np.array(_d.get('lob_next'), dtype=np.float32)
                trade = np.array(_d.get('trade'), dtype=np.float32)
                trade2 = np.array(_d.get('trade_next'), dtype=np.float32)
                a_price = int(_d.get('a_price', 0))
                a_ratio = int(_d.get('a_ratio', 0))
                reward = float(_d.get('r', 0.0))
                done = bool(_d.get('done', False))

                if bdq_agent is None:
                    try:
                        K = int(lob.shape[0]) if lob.ndim == 2 else None
                        F = int(lob.shape[1]) if lob.ndim == 2 else None
                        D = int(F // 4) if (F is not None and F % 4 == 0) else None
                        overrides = {}
                        if K is not None and D is not None and K > 0 and D > 0:
                            overrides['lob_history_len'] = K
                            overrides['lob_depth'] = D
                        bdq_agent = BDQAgent.from_config(config_path=cfg, **overrides)
                        setattr(bdq_agent, 'min_experiences_before_training', bdq_min_exp)
                        setattr(bdq_agent, 'train_frequency', bdq_train_freq)
                        setattr(bdq_agent, 'train_steps_per_update', bdq_train_steps)
                        if lx_rank == learner_root_local:
                            if overrides:
                                print(f"[learner_mpi] initialized BDQAgent from first experience: lob_depth={bdq_agent.lob_depth}, history_len={bdq_agent.lob_history_len} (overridden)")
                            else:
                                print(f"[learner_mpi] initialized BDQAgent from config (no override): lob_depth={bdq_agent.lob_depth}, history_len={bdq_agent.lob_history_len}")
                    except Exception as e:
                        if lx_rank == learner_root_local:
                            print('[learner_mpi] BDQAgent init error:', e)
                        bdq_agent = None
                        return

                s_bdq = (lob, trade)
                s2_bdq = (lob2, trade2)
                bdq_agent.add_experience(s_bdq, a_price, a_ratio, reward, s2_bdq, done, timestamp_ns=ts_ns_local)
                bdq_experience_count += 1
                if lx_rank == learner_root_local:
                    try:
                        can_train_bdq = bdq_agent.can_train()
                        since_last_bdq = bdq_experience_count - bdq_last_training_count
                        print(f"[learner_mpi][BDQ] recv exp #{bdq_experience_count}: buffer={len(bdq_agent.replay)}, since_last={since_last_bdq}, can_sample={can_train_bdq}")
                    except Exception:
                        pass
                
                if train_mode in ('both', 'bdq_only'):
                    if (bdq_experience_count >= bdq_min_exp) and \
                       (bdq_experience_count - bdq_last_training_count >= bdq_train_freq) and \
                       bdq_agent is not None and bdq_agent.can_train():
                        training_queue.put(('BDQ', bdq_agent, {'steps': bdq_train_steps}))
                        bdq_last_training_count = bdq_experience_count
                        if lx_rank == learner_root_local:
                            print(f"[learner_mpi][BDQ] training task submitted")
            except Exception as e:
                if lx_rank == learner_root_local:
                    print('[learner_mpi][BDQ] decode/add error:', e)
        else:
            agent_name, s, a, r, s_next, assets, window, feat = decode_json_basic(js)
            if agent is None:
                try:
                    agent = SACAgent.from_config(
                        config_path=cfg,
                        low_freq_feature_dim=int(feat),
                        low_freq_seq_len=int(window),
                        num_assets=int(assets)
                    )
                    if lx_rank == learner_root_local:
                        print(f"[learner_mpi] initialized SACAgent from first experience: assets={assets}, window={window}, feat={feat}")
                        try:
                            print(f"{LOG_PREFIX}[learner_mpi] training params: min_exp={agent.min_experiences_before_training}, freq={agent.train_frequency}, steps_per_update={agent.train_steps_per_update}, batch_size={agent.batch_size}")
                        except Exception:
                            pass
                except Exception as e:
                    if lx_rank == learner_root_local:
                        print('[learner_mpi] SACAgent init error:', e)
                    agent = None
                    return
            def reshape(flat):
                arr = np.array(flat, dtype=np.float32)
                expected = assets * window * feat
                if arr.size != expected:
                    if arr.size > expected:
                        arr = arr[:expected]
                    else:
                        pad = np.zeros(expected - arr.size, dtype=np.float32)
                        arr = np.concatenate([arr, pad], axis=0)
                return arr.reshape(assets, window, feat)
            market = reshape(s)
            market_next = reshape(s_next)
            w = np.array(a, dtype=np.float32)
            if w.size != assets or float(w.sum()) <= 1e-8:
                w = np.ones(assets, dtype=np.float32) / max(assets, 1)
            else:
                w = w / float(w.sum())
            state = {'market_features': market, 'portfolio_weights': w}
            next_state = {'market_features': market_next, 'portfolio_weights': w}
            exp = Experience(state=state,
                             action=np.array(a, dtype=np.float32),
                             reward=float(r),
                             next_state=next_state,
                             done=False,
                             agent_name=agent_name,
                             timestamp=int(ts_ns_local))
            if agent is not None:
                with replay_lock:
                    agent.replay_buffer.push(exp)
                    try:
                        agent.experience_count += 1
                    except Exception:
                        pass
                try:
                    buf_size = len(agent.replay_buffer)
                    since_last = agent.experience_count - getattr(agent, 'last_training_count', 0)
                    can_sample = agent.can_train()
                    should_train = agent._should_train()
                    if lx_rank == learner_root_local:
                        print(f"{LOG_PREFIX}[learner_mpi] recv exp #{recv_count + 1}: buffer={buf_size}, exp_count={agent.experience_count}, since_last={since_last}, can_sample={can_sample}, should_train={should_train}", flush=True)
                except Exception:
                    pass
                if train_mode in ('both', 'sac_only') and agent._should_train():
                    training_queue.put(('SAC', agent, {}))
                    agent.last_training_count = agent.experience_count
                    if lx_rank == learner_root_local:
                        print(f"{LOG_PREFIX}[learner_mpi] SAC training task submitted")
        recv_count += 1

    received_any_experience = False
    try:
        last_recv_ts = time.time()
        while True:
            try:
                while True:
                    task_type, result = training_results_queue.get_nowait()
                    if lx_rank == learner_root_local:
                        if task_type == 'SAC':
                            _stats = result
                            try:
                                if _stats:
                                    steps = _stats.get('num_steps', 0)
                                    al = _stats.get('avg_actor_loss', None)
                                    c1 = _stats.get('avg_critic1_loss', None)
                                    c2 = _stats.get('avg_critic2_loss', None)
                                    t = _stats.get('train_time', None)
                                    print(f"{LOG_PREFIX}[learner_mpi] SAC training done: steps={steps}, actor_loss={al}, critic1_loss={c1}, critic2_loss={c2}, time={t}s, total_training_steps={getattr(agent, 'total_training_steps', 0)}")
                                else:
                                    print(f"[learner_mpi] SAC training returned empty stats; can_sample={agent.can_train()}")
                            except Exception:
                                pass
                        elif task_type == 'BDQ':
                            _bstats = result
                            try:
                                if _bstats:
                                    print(f"[learner_mpi][BDQ] training done: steps={_bstats.get('num_steps', 0)} loss={_bstats.get('loss_total')} time={_bstats.get('train_time')}s total_steps={getattr(bdq_agent, 'total_training_steps', 0)}")
                                else:
                                    print(f"[learner_mpi][BDQ] training returned empty stats; can_sample={bdq_agent.can_train()}")
                            except Exception:
                                pass
            except queue.Empty:
                pass  
            except Exception:
                pass
            
            try:
                payload = exp_queue.get(timeout=0.05)
                if lx_rank == learner_root_local:
                    print(f"{LOG_PREFIX}[DEBUG] got payload from queue, size={len(payload) if payload else 0}", flush=True)
            except queue.Empty:
                payload = None
                if lx_rank == learner_root_local:
                    # print(f"{LOG_PREFIX}[DEBUG] queue empty, checking exit conditions", flush=True)
                    pass

            if payload is not None:
                try:
                    if len(payload) >= 8 and payload[:4] == BATCH_MAGIC:
                        cnt = int.from_bytes(payload[4:8], byteorder=sys.byteorder, signed=False)
                        offset = 8
                        for _ in range(cnt):
                            if offset + 4 > len(payload):
                                break
                            ln = int.from_bytes(payload[offset:offset+4], byteorder=sys.byteorder, signed=False)
                            offset += 4
                            if ln <= 0 or offset + ln > len(payload):
                                break
                            js = payload[offset:offset+ln].decode('utf-8')
                            offset += ln
                            _process_single_json(js)
                    else:
                        js = payload.decode('utf-8')
                        if lx_rank == learner_root_local:
                            print(f"{LOG_PREFIX}[DEBUG] processing single JSON, size={len(js)}", flush=True)
                        _process_single_json(js)
                        if lx_rank == learner_root_local:
                            print(f"{LOG_PREFIX}[DEBUG] single JSON processed", flush=True)
                    last_recv_ts = time.time()
                    received_any_experience = True
                except Exception as e:
                    if lx_rank == learner_root_local:
                        print('[learner_mpi] payload decode/add error:', e)
            else:
                if received_any_experience and ((time.time() - last_recv_ts) > idle_timeout_sec):
                    if lx_rank == learner_root_local:
                        try:
                            if agent is not None:
                                buf_size = len(agent.replay_buffer)
                                exp_cnt = getattr(agent, 'experience_count', -1)
                                last_cnt = getattr(agent, 'last_training_count', -1)
                                total_steps = getattr(agent, 'total_training_steps', -1)
                                can_sample = agent.can_train() if agent is not None else False
                                print(f"[learner_mpi] idle timeout; SAC | buffer={buf_size}, exp_count={exp_cnt}, last_training_count={last_cnt}, total_training_steps={total_steps}, can_sample={can_sample}")
                            else:
                                print(f"[learner_mpi] idle timeout; SAC not initialized (no experience)")
                            if bdq_agent is not None:
                                buf_size_b = len(bdq_agent.replay)
                                total_steps_b = getattr(bdq_agent, 'total_training_steps', -1)
                                print(f"[learner_mpi] idle timeout; BDQ | buffer={buf_size_b}, exp_count={bdq_experience_count}, last_training_count={bdq_last_training_count}, total_training_steps={total_steps_b}")
                            else:
                                print(f"[learner_mpi] idle timeout; BDQ not initialized (no experience)")
                            print(f"[learner_mpi] exiting | idle_timeout_sec={idle_timeout_sec}")
                        except Exception:
                            print('[learner_mpi] idle timeout reached; exiting (stats unavailable)')
                    try:
                        sentinel = np.array([-1], dtype='i')
                        world.Send([sentinel, MPI.INT], dest=rank, tag=LEARNER_EXP_LEN_TAG)
                        try:
                            t_recv.join(timeout=2.0)
                        except Exception:
                            pass
                    except Exception:
                        pass
                    break

            if lx_rank == learner_root_local:
                try:
                    if agent is not None and (train_mode in ('both', 'sac_only')):
                        steps = int(getattr(agent, 'total_training_steps', 0))
                        if bcast_every_steps > 0 and steps - last_bcast_steps >= bcast_every_steps and steps > 0:
                            params = None
                            was_training = None
                            try:
                                with model_lock:
                                    was_training = agent.actor.training
                                    agent.actor.eval()
                                    if scripted_actor_cache is None:
                                        class ActorInferModule(torch.nn.Module):
                                            def __init__(self, core: torch.nn.Module, seed: int = 0, eps: float = 1e-6):
                                                super().__init__()
                                                self.market_encoder = core.market_encoder
                                                self.state_fusion = core.state_fusion
                                                self.policy_head = core.policy_head
                                                self.alpha_layer = core.alpha_layer
                                                self.min_concentration = core.min_concentration
                                                try:
                                                    self.register_buffer('sampling_eps', torch.tensor(float(eps), dtype=torch.float32))
                                                except Exception:
                                                    pass
                                                try:
                                                    self.register_buffer('sampling_seed', torch.tensor([int(seed)], dtype=torch.int64))
                                                except Exception:
                                                    pass
                                            def forward(self, market_features: torch.Tensor, portfolio_weights: torch.Tensor):
                                                per_asset_features = self.market_encoder(market_features)
                                                fused_state = self.state_fusion(per_asset_features, portfolio_weights)
                                                policy_features = self.policy_head(fused_state)
                                                alpha_unconstrained = self.alpha_layer(policy_features)
                                                concentration = F.softplus(alpha_unconstrained) + float(self.min_concentration)
                                                mean_action = concentration / concentration.sum(dim=-1, keepdim=True)
                                                return mean_action, concentration
                                        seed_val = int(getattr(agent, 'seed', 0) or 0)
                                        eps_val = float(getattr(agent, 'network_cfg', {}).get('sampling_eps', 1e-6))
                                        wrapper = ActorInferModule(agent.actor, seed=seed_val, eps=eps_val)
                                        scripted_actor_cache = torch.jit.script(wrapper)
                                    else:
                                        src_sd = agent.actor.state_dict()
                                        dst_sd = scripted_actor_cache.state_dict()
                                        mapped = {}
                                        for k_dst in dst_sd.keys():
                                            k_src = k_dst
                                            if k_src.startswith('core.'):
                                                k_src = k_src[len('core.'):] 
                                            if k_src in src_sd and src_sd[k_src].shape == dst_sd[k_dst].shape:
                                                mapped[k_dst] = src_sd[k_src]
                                        scripted_actor_cache.load_state_dict(mapped, strict=False)
                                buf = io.BytesIO()
                                torch.jit.save(scripted_actor_cache, buf)
                                params = buf.getvalue()
                                print(f"{LOG_PREFIX}[learner_mpi] prepared TorchScript actor bytes: {len(params)}B")
                            except Exception as e:
                                print(f"{LOG_PREFIX}[learner_mpi] TorchScript export failed, skip broadcast: {e}")
                                params = None
                            finally:
                                if was_training:
                                    with model_lock:
                                        agent.actor.train()
                            if params is not None and len(params) > 0:
                                if enable_param_bcast:
                                    bell = np.array([1], dtype=np.int32)
                                    lx_comm.Bcast([bell, MPI.INT], root=learner_root_local)
                                    ln = np.array([len(params)], dtype=np.int32)
                                    lx_comm.Bcast([ln, MPI.INT], root=learner_root_local)
                                    buf2 = np.frombuffer(params, dtype=np.byte)
                                    lx_comm.Bcast([buf2, MPI.BYTE], root=learner_root_local)
                                last_bcast_steps = steps
                                try:
                                    print(f"{LOG_PREFIX}[learner_mpi] param broadcast success: total_training_steps={steps}, every={bcast_every_steps}, bytes={len(params)}")
                                except Exception:
                                    pass
                    if bdq_agent is not None and (train_mode in ('both', 'bdq_only')):
                        if bdq_bcast_every_steps > 0:
                            bsteps = int(getattr(bdq_agent, 'total_training_steps', 0))
                            if bsteps - last_bdq_bcast_steps >= bdq_bcast_every_steps and bsteps > 0:
                                params_b = None
                                was_training_bdq = None
                                try:
                                    with model_lock:
                                        core = bdq_agent.q_main
                                        was_training_bdq = core.training
                                        core.eval()
                                        if bdq_scripted_policy_cache is None:
                                            class BDQInferModule(torch.nn.Module):
                                                def __init__(self, core_module: torch.nn.Module):
                                                    super().__init__()
                                                    self.lob_encoder = core_module.lob_encoder
                                                    self.trade_encoder = core_module.trade_encoder
                                                    self.fusion = core_module.fusion
                                                    self.adv_price = core_module.adv_price
                                                    self.adv_ratio = core_module.adv_ratio
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
                                                def forward(self, lob_seq: torch.Tensor, trade_vec: torch.Tensor):
                                                    lob_feat = self.lob_encoder(lob_seq)
                                                    trade_feat = self.trade_encoder(trade_vec)
                                                    shared = self.fusion(lob_feat, trade_feat)
                                                    A_p = self.adv_price(shared)
                                                    A_r = self.adv_ratio(shared)
                                                    a_p = torch.argmax(A_p, dim=-1)
                                                    a_r = torch.argmax(A_r, dim=-1)
                                                    return a_p, a_r
                                            wrapper_b = BDQInferModule(core)
                                            bdq_scripted_policy_cache = torch.jit.script(wrapper_b)
                                        else:
                                            src_sd = core.state_dict()
                                            dst_sd = bdq_scripted_policy_cache.state_dict()
                                            mapped = {}
                                            for k_dst in dst_sd.keys():
                                                k_src = k_dst
                                                if k_src.startswith('core.'):
                                                    k_src = k_src[len('core.'):]
                                                if k_src in src_sd and src_sd[k_src].shape == dst_sd[k_dst].shape:
                                                    mapped[k_dst] = src_sd[k_src]
                                            bdq_scripted_policy_cache.load_state_dict(mapped, strict=False)
                                        buf_b = io.BytesIO()
                                        torch.jit.save(bdq_scripted_policy_cache, buf_b)
                                        params_b = buf_b.getvalue()
                                        print(f"[BDQ][learner_mpi] prepared TorchScript policy bytes: {len(params_b)}B")
                                except Exception as e:
                                    print(f"[BDQ][learner_mpi] TorchScript export failed, skip broadcast: {e}")
                                    params_b = None
                                finally:
                                    if was_training_bdq:
                                        with model_lock:
                                            core.train()
                                if params_b is not None and len(params_b) > 0:
                                    if enable_param_bcast:
                                        bell = np.array([1], dtype=np.int32)
                                        lx_comm.Bcast([bell, MPI.INT], root=learner_root_local)
                                        ln_b = np.array([len(params_b)], dtype=np.int32)
                                        lx_comm.Bcast([ln_b, MPI.INT], root=learner_root_local)
                                        buf2_b = np.frombuffer(params_b, dtype=np.byte)
                                        lx_comm.Bcast([buf2_b, MPI.BYTE], root=learner_root_local)
                                    last_bdq_bcast_steps = bsteps
                                    try:
                                        print(f"[BDQ][learner_mpi] param broadcast success: total_training_steps={bsteps}, every={bdq_bcast_every_steps}, bytes={len(params_b)}")
                                    except Exception:
                                        pass
                except Exception:
                    pass
            try:
                if ctrl_end_event.is_set():
                    try:
                        if enable_param_bcast:
                            bell = np.array([-1], dtype=np.int32)
                            lx_comm.Bcast([bell, MPI.INT], root=learner_root_local)
                            if lx_rank == learner_root_local:
                                print(f"{LOG_PREFIX}[learner_mpi] broadcasted shutdown bell (-1)", flush=True)
                        else:
                            if lx_rank == learner_root_local:
                                print(f"{LOG_PREFIX}[learner_mpi] skip broadcast: enable_param_bcast={enable_param_bcast}", flush=True)
                    except Exception as e:
                        if lx_rank == learner_root_local:
                            print(f"{LOG_PREFIX}[learner_mpi] broadcast shutdown bell failed: {e}", flush=True)
                    running = False
                    try:
                        if t_recv.is_alive():
                            sentinel = np.array([-1], dtype='i')
                            world.Send([sentinel, MPI.INT], dest=rank, tag=LEARNER_EXP_LEN_TAG)
                            try:
                                t_recv.join(timeout=2.0)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    if lx_rank == learner_root_local:
                        print(f"{LOG_PREFIX}[learner_mpi] received CTRL_END; exiting main loop", flush=True)
                    break
            except Exception:
                pass
            time.sleep(0.001)
    finally:
        try:
            if enable_param_bcast and lx_rank == learner_root_local:
                bell = np.array([-1], dtype=np.int32)
                lx_comm.Bcast([bell, MPI.INT], root=learner_root_local)
                print(f"{LOG_PREFIX}[learner_mpi] emergency broadcast shutdown bell (-1) in finally", flush=True)
        except Exception as e:
            if lx_rank == learner_root_local:
                print(f"{LOG_PREFIX}[learner_mpi] emergency broadcast failed: {e}", flush=True)
        
        try:
            running = False
            training_queue.put(None)
            training_thread.join(timeout=5.0)
            if training_thread.is_alive():
                print(f"{LOG_PREFIX}[learner_mpi] warning: training thread still alive after 10s timeout")
            else:
                print(f"{LOG_PREFIX}[learner_mpi] training thread cleaned up successfully")
        except Exception:
            pass
            
        try:
            print(f"{LOG_PREFIX}[learner_mpi] begin agent cleanup", flush=True)
        except Exception:
            pass
        try:
            if agent is not None:
                agent.cleanup()
        except Exception:
            pass
        try:
            print(f"{LOG_PREFIX}[learner_mpi] agent cleanup done", flush=True)
        except Exception:
            pass
        try:
            if bdq_agent is not None:
                bdq_agent.cleanup()
        except Exception:
            pass
        
        try:
            print(f"{LOG_PREFIX}[learner_mpi] begin MPI cleanup", flush=True)
            
            if lcomm != MPI.COMM_NULL and lcomm != MPI.COMM_WORLD:
                try:
                    lcomm.Free()
                    print(f"{LOG_PREFIX}[learner_mpi] learner subcomm freed", flush=True)
                except Exception as e:
                    print(f"{LOG_PREFIX}[learner_mpi] subcomm free failed: {e}", flush=True)
            
            print(f"{LOG_PREFIX}[learner_mpi] calling MPI.Finalize() with 10s timeout", flush=True)
            
            import subprocess
            
            def force_exit_after_timeout():
                import time
                time.sleep(10)
                print(f"{LOG_PREFIX}[learner_mpi] MPI.Finalize() timeout! Force exit", flush=True)
                try:
                    subprocess.run(['kill', '-9', str(os.getpid())], check=False)
                except:
                    pass
                os._exit(0)
            
            timeout_thread = threading.Thread(target=force_exit_after_timeout, daemon=True)
            timeout_thread.start()
            
            try:
                MPI.Finalize()
                print(f"{LOG_PREFIX}[learner_mpi] MPI.Finalize() success", flush=True)
            except Exception as e:
                print(f"{LOG_PREFIX}[learner_mpi] MPI.Finalize() failed: {e}", flush=True)
                os._exit(0)
            
        except Exception as e:
            print(f"{LOG_PREFIX}[learner_mpi] MPI cleanup failed: {e}, force exit", flush=True)
            os._exit(0)

    return 0


if __name__ == '__main__':
    sys.exit(main())


