# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import math

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab.envs import ManagerBasedEnv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

def is_success(env: ManagerBasedRLEnv, command_name: str, threshold: float, heading_threshold: float = 0.1, vel_threshold: float = 1.0) -> torch.Tensor:
    """
    Determine whether the target has been reached.

    This function checks if the rover is within a certain threshold distance from the target.
    If the target is reached, a scaled reward is returned based on the remaining time steps.
    """

    # Accessing the target's position
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]

    angle = env.command_manager.get_command(command_name)[:, 3]

    # Calculating the distance and determining if the target is reached
    distance = torch.norm(target_position, p=2, dim=-1)

    # 3. 로봇의 조인트 속도 확인 (정지 여부)
    # "robot"은 SceneCfg에서 정의한 로봇의 이름이어야 합니다.
    joint_vel = env.scene["robot"].data.joint_vel
    joint_vel_norm = torch.norm(joint_vel, p=2, dim=-1) # 모든 조인트 속도의 합계 수준 확인

    return torch.where((distance < threshold) & (torch.abs(angle) < heading_threshold) & (joint_vel_norm < vel_threshold), True, False)


def rover_power_consumption(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # return power = torque * velocity (here actions: joint torques)
        return torch.sum(torch.abs(env.action_manager.action * asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1)


# --- Observation Func ---
def distance_to_target_obs(env: ManagerBasedRLEnv, command_name: str):
    """Calculate the euclidean distance to the target."""
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]
    distance: torch.Tensor = torch.norm(target_position, p=2, dim=-1)
    return distance.unsqueeze(-1)


def angle_to_target_obs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Calculate the angle to the target."""

    # Get vector(x,y) from rover to target, in base frame of the rover.
    target_vector_b = env.command_manager.get_command(command_name)[:, :2]

    # Calculate the angle between the rover's heading [1, 0] and the vector to the target.
    angle = torch.atan2(target_vector_b[:, 1], target_vector_b[:, 0])

    return angle.unsqueeze(-1)


def angle_diff_obs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Calculate the angle difference between the rover's heading and the target."""
    # Get the angle to the target
    heading_angle_diff = env.command_manager.get_command(command_name)[:, 3]

    return heading_angle_diff.unsqueeze(-1)

def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned
    robot frame using an exponential kernel.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def randomize_soil_properties_startup(env: ManagerBasedEnv, env_ids: any, soil_ranges: dict):
    """
    시뮬레이션 시작(startup) 시 전체 환경의 토양 파라미터를 일괄 무작위화합니다.
    (reset을 고려하지 않으므로 env_ids 인자는 무시합니다.)
    """
    num_envs = env.num_envs
    
    # 1. 정의된 범위 내에서 전체 환경에 대한 랜덤 텐서 생성 및 덮어쓰기
    for key, (min_val, max_val) in soil_ranges.items():
        if key in env.p:
            env.p[key] = torch.rand(num_envs, device=env.device) * (max_val - min_val) + min_val

    # 2. 변경된 전체 토양 파라미터를 바탕으로 C1, C2 일괄 재계산
    phi_soil = env.p['phi_soil']
    alpha_b = env.p['alpha_b']

    Xc = (torch.pi / 4.0) - (phi_soil / 2.0)
    def cot(x): return 1.0 / torch.tan(x)
    env.C1 = (cot(Xc) + torch.tan(Xc + phi_soil)) / (1.0 - torch.tan(alpha_b) * torch.tan(Xc + phi_soil))
    env.C2 = (cot(Xc) - torch.tan(alpha_b)) + ((cot(Xc) - torch.tan(alpha_b))**2) / (torch.tan(alpha_b) + cot(phi_soil))

def grid_soil_properties_startup(env: ManagerBasedEnv, env_ids: any, soil_ranges: dict):
    """
    발표 시각화용: 6x6 그리드에 맞춰 X축으로는 kc, Y축으로는 kphi를 점진적으로 증가시킵니다.
    """
    num_envs = env.num_envs
    device = env.device

    # 1. 환경 개수(36)를 바탕으로 그리드 크기(6) 계산
    grid_size = int(math.sqrt(num_envs))
    
    # 만약 num_envs가 완전제곱수(예: 36)가 아니면 경고
    if grid_size * grid_size != num_envs:
        print(f"[경고] 그리드 시각화를 위해서는 num_envs가 완전제곱수(예: 36)여야 합니다. 현재: {num_envs}")

    # 2. 각 로버의 2D 인덱스 계산 (0 ~ 5)
    env_indices = torch.arange(num_envs, device=device)
    rows = torch.div(env_indices, grid_size, rounding_mode='floor')  # Y축 인덱스 (0~5)
    cols = env_indices % grid_size                                   # X축 인덱스 (0~5)

    # 3. 각 파라미터별 세팅
    for key, (min_val, max_val) in soil_ranges.items():
        if key not in env.p:
            continue
            
        if key == 'c_soil':
            # X축(cols)을 따라 kc 증가
            step = (max_val - min_val) / max(1, grid_size - 1)
            env.p[key] = min_val + cols.float() * step
            
        elif key == 'phi_soil':
            # Y축(rows)을 따라 kphi 증가
            step = (max_val - min_val) / max(1, grid_size - 1)
            env.p[key] = min_val + rows.float() * step
            
        else:
            # 나머지 변수들(c_soil, phi_soil 등)은 시각화 통제를 위해 중간값(평균)으로 고정
            mean_val = (min_val + max_val) / 2.0
            env.p[key] = torch.full((num_envs,), mean_val, device=device)

    # 4. 변경된 파라미터를 바탕으로 유도 변수(C1, C2) 재계산
    phi_soil = env.p['phi_soil']
    alpha_b = env.p['alpha_b']

    Xc = (torch.pi / 4.0) - (phi_soil / 2.0)

    def cot(x): return 1.0 / torch.tan(x)

    env.C1 = (cot(Xc) + torch.tan(Xc + phi_soil)) / (1.0 - torch.tan(alpha_b) * torch.tan(Xc + phi_soil))
    env.C2 = (cot(Xc) - torch.tan(alpha_b)) + ((cot(Xc) - torch.tan(alpha_b))**2) / (torch.tan(alpha_b) + cot(phi_soil))

def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)

def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return heading_b.abs() * torch.tanh(distance / 0.6)
