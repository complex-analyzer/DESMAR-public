#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import torch

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _resolve_device(device: str = 'auto') -> torch.device:
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def _find_latest_pth(search_dir: str, pattern: str) -> str:
    try:
        entries = os.listdir(search_dir)
    except Exception:
        return ''

    def _match(name: str) -> bool:
        if pattern == 'bdq_agent_final_*_bdq.pth':
            return name.startswith('bdq_agent_final_') and name.endswith('_bdq.pth')
        if pattern == 'sac_agent_final_*.pth':
            return name.startswith('sac_agent_final_') and name.endswith('.pth')
        return name.endswith('.pth')

    files = [os.path.join(search_dir, n) for n in entries if _match(n) and os.path.isfile(os.path.join(search_dir, n))]
    if not files:
        return ''
    return max(files, key=os.path.getmtime)


def export_bdq(ckpt_path: str, config_path: str, device_str: str = 'cpu') -> str:
    from rl_modules.bdq_agent import BDQAgent

    if not ckpt_path:
        search_dir = os.path.join('rl_modules', 'BDQ_Trade_Execution', 'saved_models')
        ckpt_path = _find_latest_pth(search_dir, 'bdq_agent_final_*_bdq.pth')
    if not ckpt_path or (not os.path.exists(ckpt_path)):
        raise FileNotFoundError(f"BDQ checkpoint not found: {ckpt_path}")

    agent = BDQAgent.from_config(config_path, device=device_str)
    ok = agent.load_models(ckpt_path)
    if not ok:
        raise RuntimeError(f"load BDQ model failed: {ckpt_path}")

    dirname = os.path.dirname(ckpt_path)
    base = os.path.basename(ckpt_path)
    if base.endswith('_bdq.pth'):
        prefix = base[:-8]
    elif base.endswith('.pth'):
        prefix = base[:-4]
    else:
        prefix = os.path.splitext(base)[0]
    out_path = os.path.join(dirname, f"{prefix}_policy.pt")

    filepath_prefix = os.path.join(dirname, prefix)
    agent.save_models(filepath_prefix)

    if not os.path.exists(out_path):
        class BDQInferModule(torch.nn.Module):
            def __init__(self, core: torch.nn.Module):
                super().__init__()
                self.lob_encoder = core.lob_encoder
                self.trade_encoder = core.trade_encoder
                self.fusion = core.fusion
                self.value_head = core.value_head
                self.adv_price = core.adv_price
                self.adv_ratio = core.adv_ratio
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

        agent.q_main.eval()
        scripted = torch.jit.script(BDQInferModule(agent.q_main))
        torch.jit.save(scripted, out_path)

    return out_path


def export_sac(ckpt_path: str, config_path: str, device_str: str = 'cpu') -> str:
    from rl_modules.sac_agent import SACAgent

    if not ckpt_path:
        search_dir = os.path.join('rl_modules', 'SAC_Portfolio_Allocationl', 'saved_models')
        ckpt_path = _find_latest_pth(search_dir, 'sac_agent_final_*.pth')
    if not ckpt_path or (not os.path.exists(ckpt_path)):
        raise FileNotFoundError(f"SAC checkpoint not found: {ckpt_path}")

    agent = SACAgent.from_config(config_path, device=device_str)
    ok = agent.load_models(ckpt_path)
    if not ok:
        raise RuntimeError(f"load SAC model failed: {ckpt_path}")

    dirname = os.path.dirname(ckpt_path)
    base = os.path.basename(ckpt_path)
    prefix = base[:-4] if base.endswith('.pth') else os.path.splitext(base)[0]
    out_path = os.path.join(dirname, f"{prefix}_actor.pt")

    # 复用类内导出能力
    filepath_prefix = os.path.join(dirname, prefix)
    agent.save_models(filepath_prefix)  # 内部会同时导出 *_actor.pt

    # 兜底：若未导出成功，提示异常
    if not os.path.exists(out_path):
        raise RuntimeError(f"failed to generate TorchScript file: {out_path}")

    return out_path


def main():
    parser = argparse.ArgumentParser(description='re-export TorchScript to adapt runtime version')
    parser.add_argument('--algo', choices=['bdq', 'sac'], required=True, help='choose algorithm: bdq or sac')
    parser.add_argument('--ckpt', type=str, default='', help='specify .pth checkpoint path; default: latest')
    parser.add_argument('--config', type=str, default='Simulator_configs/config_templates/sac_config.yaml', help='config file path, for building network shape and hyperparameters')
    parser.add_argument('--device', type=str, default='auto', help="device: auto/cpu/cuda")
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high') if hasattr(torch, 'set_float32_matmul_precision') else None

    os.makedirs(os.path.join('rl_modules', 'BDQ_Trade_Execution', 'saved_models'), exist_ok=True)
    os.makedirs(os.path.join('rl_modules', 'SAC_Portfolio_Allocationl', 'saved_models'), exist_ok=True)

    try:
        if args.algo == 'bdq':
            out = export_bdq(args.ckpt, args.config, device_str=args.device)
        else:
            out = export_sac(args.ckpt, args.config, device_str=args.device)
        print(f"export success: {out}")
    except Exception as e:
        print(f"export failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


