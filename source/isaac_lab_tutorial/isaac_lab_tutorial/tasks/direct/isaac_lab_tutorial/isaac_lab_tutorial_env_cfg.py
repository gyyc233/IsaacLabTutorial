# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaac_lab_tutorial.robots.jetbot import JETBOT_CONFIG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# 环境配置类，定义了强化学习环境的各种参数

@configclass
class IsaacLabTutorialEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2 # 控制动作应用的频率，每2个仿真步骤执行一次动作
    episode_length_s = 5.0 # 每个episode的持续时间
    # - spaces definition
    action_space = 2 # 动作空间维度为2（对应左右轮子的速度控制）
    # observation_space = 9
    observation_space = 3 # 观测空间维度为3（点积、叉积z分量、前向速度）
    state_space = 0 # 状态空间维度为0（不使用额外状态信息）
    # simulation 仿真步长为1/120 s, 渲染间隔与decimation同步
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # robot(s) 机器人配置 使用Jetbot机器人配置，设置USD路径格式
    robot_cfg: ArticulationCfg = JETBOT_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    # scene 场景配置，并行运行100个环境，环境间距为2.0米，启用物理复制以提高性能
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=2.0, replicate_physics=True)
    # 定义需要控制的关节名称
    dof_names = ["left_wheel_joint", "right_wheel_joint"]