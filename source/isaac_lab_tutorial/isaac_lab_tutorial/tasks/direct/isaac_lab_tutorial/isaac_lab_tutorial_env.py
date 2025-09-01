# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from .isaac_lab_tutorial_env_cfg import IsaacLabTutorialEnvCfg

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    # 定义了两个箭头,红色表示目标指令，蓝色+绿色表示机器人当前朝向
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)

class IsaacLabTutorialEnv(DirectRLEnv):
    cfg: IsaacLabTutorialEnvCfg

    def __init__(self, cfg: IsaacLabTutorialEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # 关节索引和关节名称，这里只关心索引，所以用 _ 忽略名称
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)

    # 场景初始化设置
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg) # 创建机器人实体
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate 克隆环境
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene 在场景中添加机器人关节
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.visualization_markers = define_markers() # 可视化标记

        # 初始化变量与命令
        self.up_dir = torch.tensor([0.0, 0.0, 1.0]).cuda() # 定义向上方向向量
        self.yaws = torch.zeros((self.cfg.scene.num_envs, 1)).cuda()

        # 命令初始化和航向计算
        self.commands = torch.randn((self.cfg.scene.num_envs, 3)).cuda() # 为每个环境生成随机的3D目标方向向量
        self.commands[:,-1] = 0.0 # z分量为0，保持在地面平面
        self.commands = self.commands/torch.linalg.norm(self.commands, dim=1, keepdim=True) # 归一化为单位长度
        
        # offsets to account for atan range and keep things on [-pi, pi]
        # 根据命令所在象限添加或减去 pi 来确定yaw
        ratio = self.commands[:,1]/(self.commands[:,0]+1E-8) #in case the x component is zero
        gzero = torch.where(self.commands > 0, True, False)
        lzero = torch.where(self.commands < 0, True, False)
        plus = lzero[:,0]*gzero[:,1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        self.yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1) # 偏航角计算

        # 设置可视化箭头位置
        self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset[:,-1] = 0.5 # 可视化箭头在机器人上方0.5米
        self.forward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
        self.command_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
        

    # 可视化 markers
    def _visualize_markers(self):
        # get marker locations and orientations
        # 获取机器人当前位置和朝向
        self.marker_locations = self.robot.data.root_pos_w
        self.forward_marker_orientations = self.robot.data.root_quat_w

        # 计算目标朝向
        # quat_from_angle_axis 旋转轴和角度生成四元数
        self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()

        # offset markers so they are above the jetbot 使得markers 位于jetbot上方
        loc = self.marker_locations + self.marker_offset
        loc = torch.vstack((loc, loc))
        rots = torch.vstack((self.forward_marker_orientations, self.command_marker_orientations))

        # render the markers
        all_envs = torch.arange(self.cfg.scene.num_envs)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))

        self.visualization_markers.visualize(loc, rots, marker_indices=indices)

    # _pre_physics_step 在每次物理仿真步骤之前执行，接收来自策略网络的动作指令 actions
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 保存动作副本供后续使用
        self.actions = actions.clone()# + torch.ones_like(actions)
        # 在 _pre_physics_step 预物理步骤上调用 _visualize_markers 来使箭头可见
        self._visualize_markers()

    # _apply_action 将动作应用到机器人上，在物理仿真步骤中执行
    def _apply_action(self) -> None:
        # 设置机器人关节速度
        # self.actions: 目标速度值
        # joint_ids=self.dof_idx: 指定要控制的关节索引
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    # 观测空间原则：保持观测空间尽可能小
    def _get_observations(self) -> dict:
        # 获取机器人质心在世界坐标系中的速度，但这个值并未直接用于最终的观测值中
        self.velocity = self.robot.data.root_com_vel_w 
        # 通过将机器人的根节点朝向四元数（root_link_quat_w）应用于其基础前进向量（通常是 [1,0,0]）来计算世界坐标系中的前进方向向量
        self.forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)
        # obs = torch.hstack((self.velocity, self.commands))

        # 观测部分的组成
        # dot 机器人前进方向与目标指令方向的点积。这代表了它们之间夹角的余弦值，表明了两者对齐程度
        dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        # cross 前进向量和指令向量叉积的z分量。这指示了需要旋转的方向（绕z轴的正向或负向）以与指令对齐
        cross = torch.cross(self.forwards, self.commands, dim=-1)[:,-1].reshape(-1,1)
        # forward_speed 提取机器人在机体坐标系中x方向的线速度分量，即前进速度
        forward_speed = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
        # 将这三个分量组合成每个环境的单一观测向量
        obs = torch.hstack((dot, cross, forward_speed))
        
        observations = {"policy": obs}
        # 修改 ``IsaacLabTutorialEnvCfg`` 以将观测空间设置回3 ，这包括点积、叉积的z分量和前向速度
        return observations

    # 奖励原则：尽可能减少和简化奖励函数，这里仅奖励智能体前进且对齐
    def _get_rewards(self) -> torch.Tensor:
        # 当机器人表现如期望的那样时，它将全速朝着控制命令的方向行驶

        # forward_reward 是机器人质心线性速度在机体坐标系中的 x 分量
        forward_reward = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
        #  x 方向是资产的前进方向，因此这应等同于前进向量与世界坐标系中线性速度的内积
        # 对齐项是前进向量与控制命令向量的内积: 当它们指向同一方向时，这一项将为 1，但指向相反方向时，它将为 -1 其它情况在[-1,1]之间
        alignment_reward = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        # 将它们相加以获得组合奖励，最后我们可以开始训练了
        total_reward = forward_reward + alignment_reward
        # total_reward = forward_reward*alignment_reward # 速度与对齐的乘积
        # total_reward = forward_reward*alignment_reward + forward_reward # 乘积加上速度奖励
        # total_reward = forward_reward*torch.exp(alignment_reward) # 速度乘以对齐奖励的指数

        # 当前使用的奖励函数设计简单直观：机器人获得奖励的条件是既要快速移动（forward_reward），又要朝着正确的方向移动（alignment_reward）。这种组合鼓励机器人以最快的速度朝指令方向前进。
        return total_reward

    # 判断每个环境是否应该结束
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # self.episode_length_buf: 记录当前episode已经持续的步数
        # self.max_episode_length: 每个episode的最大步数限制
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # 环境是否因失败而终止（这里没使用）；是否因超时终止
        return False, time_out

    # 重置指定环境状态，通常在episode结束时调用
    def _reset_idx(self, env_ids: Sequence[int] | None):
        # 当一个或多个环境需要重置时（如达到最大步数），此方法会重新初始化这些环境的状态
        if env_ids is None:
            # 如果未指定环境ID，则重置所有环境
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # 生成新的目标指令
        self.commands[env_ids] = torch.randn((len(env_ids), 3)).cuda()
        self.commands[env_ids,-1] = 0.0
        self.commands[env_ids] = self.commands[env_ids]/torch.linalg.norm(self.commands[env_ids], dim=1, keepdim=True)
        
        # 重新计算偏航角
        ratio = self.commands[env_ids][:,1]/(self.commands[env_ids][:,0]+1E-8)
        gzero = torch.where(self.commands[env_ids] > 0, True, False)
        lzero = torch.where(self.commands[env_ids]< 0, True, False)
        plus = lzero[:,0]*gzero[:,1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        self.yaws[env_ids] = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

        # 重置机器人状态
        default_root_state = self.robot.data.default_root_state[env_ids] # 获取机器人的默认根状态
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_state_to_sim(default_root_state, env_ids)
        self._visualize_markers()
