#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RL modules for DESMAR
"""
# 核心模块
from .experience_replay import ExperienceReplayBuffer
from .policy_networks import PolicyNetwork, QNetwork
from .sac_agent import SACAgent
from .bdq_agent import BDQAgent
from .bdq_networks import BDQCore

# 导出列表
__all__ = [
    'ExperienceReplayBuffer',
    'PolicyNetwork', 
    'QNetwork',
    'SACAgent',
    'BDQAgent',
    'BDQCore'
]

# 版本信息
__version__ = "2.0.0" 