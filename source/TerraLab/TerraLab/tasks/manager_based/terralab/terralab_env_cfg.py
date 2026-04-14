# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg

from . import mdp

##
# Pre-defined configs
##

from TerraLab.robots.rover import ROVER_CONFIG


##
# Scene definition
##


@configclass
class TerralabSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # Terrain
    terrain = TerrainImporterCfg(
        prim_path= "/World",
        terrain_type = 'usd',
        usd_path = r"C:\test\source\test\test\tasks\manager_based\test\Terrain0.5.usd",
        env_spacing = 1.0,
        collision_group=-1,
    )

    # Rover
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Rover",
        spawn=ROVER_CONFIG.spawn,
        debug_vis = False,
        actuator_value_resolution_debug_print=True,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.07),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators=ROVER_CONFIG.actuators,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    Sinkage_scan_FL = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Rover/Rover/Force_FL",
        mesh_prim_paths=["/World/terrain/Terrain/moon"],
        pattern_cfg=patterns.GridPatternCfg(
            resolution=1.0,
            size=(0.0, 0.0),
        ),
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
        max_distance = 0.7,
        debug_vis = True,
        ray_alignment="yaw",
    )

    Sinkage_scan_RL = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Rover/Rover/Force_RL",
            mesh_prim_paths=["/World/terrain/Terrain/moon"],
            pattern_cfg=patterns.GridPatternCfg(
                resolution=1.0,
                size=(0.0, 0.0),
            ),
            offset=RayCasterCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0)
            ),
            max_distance = 0.7,
            debug_vis = True,
            ray_alignment="yaw",
        )
    
    Sinkage_scan_FR = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Rover/Rover/Force_FR",
        mesh_prim_paths=["/World/terrain/Terrain/moon"],
        pattern_cfg=patterns.GridPatternCfg(
            resolution=1.0,
            size=(0.0, 0.0),
        ),
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
        max_distance = 0.7,
        debug_vis = True,
        ray_alignment="yaw",
    )

    Sinkage_scan_RR = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Rover/Rover/Force_RR",
        mesh_prim_paths=["/World/terrain/Terrain/moon"],
        pattern_cfg=patterns.GridPatternCfg(
            resolution=1.0,
            size=(0.0, 0.0),
        ),
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
        max_distance = 0.7,
        debug_vis = True,
        ray_alignment="yaw",
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Torque
    robot = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["Wheel_joint_.*"],
        scale=0.3432,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    # reset_cart_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
    #         "position_range": (-1.0, 1.0),
    #         "velocity_range": (-0.5, 0.5),
    #     },
    # )

    # reset_pole_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
    #         "position_range": (-0.25 * math.pi, 0.25 * math.pi),
    #         "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright
    # pole_pos = RewTerm(
    #     func=mdp.joint_pos_target_l2,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    # )
    # # (4) Shaping tasks: lower cart velocity
    # cart_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.01,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    # )
    # # (5) Shaping tasks: lower pole angular velocity
    # pole_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.005,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    # cart_out_of_bounds = DoneTerm(
    #     func=mdp.joint_pos_out_of_manual_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    # )


##
# Environment configuration
##


@configclass
class TerralabEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: TerralabSceneCfg = TerralabSceneCfg(num_envs=4096, env_spacing=1.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 10
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (5.0, 0.0, 1.0)
        # simulation settings
        self.sim.gravity = (0.0, 0.0, -1.62)
        self.sim.dt = 1 / 500
        self.sim.render_interval = self.decimation


##
# Custom Environment Class
##

class TerralabEnv(ManagerBasedRLEnv):

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.R = 0.035
        self.p = {
            'b': 0.025, 'n': 1.0,
            'kc': 990, 'kphi': 1528000.0,
            'c1': 0.30, 'c2': 0.10,
            'c_soil': 1716.0, 'phi_soil': 0.7086, 'mu_s': 0.8,
            'Kx': 0.005, 'Ky': 0.005,
            'c_s': 15.0, 
            'rho_s': 1600.0,  # 흙의 밀도 (kg/m^3)
            'alpha_b': 0.0    # 흙을 미는 가상 블레이드의 각도
        }
        self.prev_sinkage = torch.zeros((self.num_envs, 4), device=self.device)

        # 불도저 상수 Init
        Xc = (math.pi / 4.0) - (self.p['phi_soil'] / 2.0)
        def cot(x):
            return 1.0 / math.tan(x)

        self.C1 = (cot(Xc) + math.tan(Xc + self.p['phi_soil'])) / (1.0 - math.tan(self.p['alpha_b']  ) * math.tan(Xc + self.p['phi_soil']))
        self.C2 = (cot(Xc) - math.tan(self.p['alpha_b']  )) + ((cot(Xc) - math.tan(self.p['alpha_b']  ))**2) / (math.tan(self.p['alpha_b']  ) + cot(self.p['phi_soil']))
        
        # (주의: step 함수에서 centers_z를 stack한 순서와 똑같아야 합니다!)
        dummy_names = ["Force_FL", "Force_RL", "Force_FR", "Force_RR"]
        wheel_names = ["Wheel_FL", "Wheel_RL", "Wheel_FR", "Wheel_RR"]
        
        # 2. 더미 링크 ID 하나씩 찾아서 순서대로 넣기
        self.dummy_body_ids = []
        for name in dummy_names:
            ids, _ = self.scene["robot"].find_bodies([name])
            self.dummy_body_ids.extend(ids)
            
        # 3. 진짜 바퀴 ID 하나씩 찾아서 순서대로 넣기 (이것도 꼬였을 확률 100%입니다!)
        self.wheel_body_ids = []
        for name in wheel_names:
            ids, _ = self.scene["robot"].find_bodies([name])
            self.wheel_body_ids.extend(ids)
            
        # 4. 최종 병합
        self.Terra_body_ids = self.dummy_body_ids + self.wheel_body_ids
    
    def step(self, action: torch.Tensor):
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1

            # set Terramechanics force apply
            self._apply_terramechanics(dt=self.physics_dt)
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _apply_terramechanics(self, dt: float):
        robot = self.scene["robot"]
        # 4개의 바퀴 데이터를 하나의 텐서로 병합 (Shape: [num_envs, 4])
        # 1. 모든 환경(num_envs)의 바퀴 중심 Z 좌표 추출 (Shape: [num_envs, 4])
        centers_z = torch.stack([
            self.scene["Sinkage_scan_FL"].data.pos_w[:, 2],
            self.scene["Sinkage_scan_RL"].data.pos_w[:, 2],
            self.scene["Sinkage_scan_FR"].data.pos_w[:, 2],
            self.scene["Sinkage_scan_RR"].data.pos_w[:, 2]], dim=1)

        # 2. 모든 환경의 지면(Hit) Z 좌표 추출 (Shape: [num_envs, 4])
        hits_z = torch.stack([
            self.scene["Sinkage_scan_FL"].data.ray_hits_w[:, 0, 2],
            self.scene["Sinkage_scan_RL"].data.ray_hits_w[:, 0, 2],
            self.scene["Sinkage_scan_FR"].data.ray_hits_w[:, 0, 2], 
            self.scene["Sinkage_scan_RR"].data.ray_hits_w[:, 0, 2]], dim=1)
        
        self.Sinkage = torch.clamp(self.R - (centers_z - hits_z), min=1e-9, max=0.034999)

        # 침하량 변화율
        v_c = torch.clamp((self.Sinkage - self.prev_sinkage) / dt, min=0.0)
        self.prev_sinkage = self.Sinkage.clone()

        # 순서 주의 [FL RL FR RR]
        wheel_joint_vel = robot.data.joint_vel[:, 2:6] # [num_envs, 4]

        vel_global = robot.data.body_lin_vel_w[:, self.dummy_body_ids, :]
        quat_global = robot.data.body_quat_w[:, self.dummy_body_ids, :]
        # 2. 역회전(Inverse Rotate)을 통해 바퀴 기준의 로컬 속도를 구합니다.
        vel_local = math_utils.quat_apply_inverse(quat_global, vel_global)

        v_x = vel_local[..., 0]
        v_y = vel_local[..., 1]
        v_z = vel_local[..., 2]

        # 사이드 슬립 앵글
        alpha = torch.atan2(v_y, torch.clamp(torch.abs(v_x), min=1e-6))

        # 슬립 Ratio
        v_diff = self.R * wheel_joint_vel - v_x # 바퀴 회전 선속도 (Rw * w)
        # 4096x4 행렬 연산이 한 방에 일어납니다.
        num = torch.abs(v_diff)
        den = torch.max(torch.abs(self.R * wheel_joint_vel), torch.abs(v_x)) + 1e-6
        self.slip_ratios = torch.clamp(num / den, min=0.0, max=1.0)

        # Wong-Reece Model
        Fn_flat, Ft_n_flat, Ft_s_flat, Fy_flat, T_flat = self.wong_reece_model_torch(self.slip_ratios.view(-1), self.Sinkage.view(-1), v_c.view(-1), alpha.view(-1))

        # 6. 결과값을 다시 원래 구조 [num_envs, 4]로 복구
        Fn = Fn_flat.view(self.num_envs, 4)
        Ft_n = Ft_n_flat.view(self.num_envs, 4)
        Ft_s = Ft_s_flat.view(self.num_envs, 4)
        Fy_mag = Fy_flat.view(self.num_envs, 4)
        T_mag = T_flat.view(self.num_envs, 4)

        # 힘 방향성 결합
        Ft = Ft_s * torch.sign(v_diff) - Ft_n * torch.sign(v_x)
        Fy = -Fy_mag * torch.sign(alpha)
        Torque_res = -T_mag * torch.sign(v_diff)
        
        # 직선 힘(Forces) 텐서 생성
        forces = torch.zeros((self.num_envs, 8, 3), device=self.device)
        forces[:, 0:4, 0] = Ft
        forces[:, 0:4, 1] = Fy
        forces[:, 0:4, 2] = Fn
        
        # 회전력(Torques) 텐서 생성
        torques = torch.zeros((self.num_envs, 8, 3), device=self.device)
        torques[:, 4:8, 2] = Torque_res

        # print(f"[디버그] 조인트속도! 1번 환경 값: {wheel_joint_vel[1]}")
        # print(f"[디버그] 힘! 1번 환경 값: {forces[1, 0:4, 2]}")
        # print(f"[디버그] 슬립율! 1번 환경 값: {self.slip_ratios[1]}")
        # print(f"[디버그] 침하량 텐서 Shape: {self.Sinkage.shape}, 1번 환경 값: {self.Sinkage[1]}")
        
        # 로봇에 외부 힘과 회전력 모두 설정
        robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=self.Terra_body_ids,
            positions = None,
            is_global=False
            )
    
    def wong_reece_model_torch(self, s: torch.Tensor, h: torch.Tensor, v_c: torch.Tensor, alpha: torch.Tensor):

        # 벡터화된 Wong-Reece 테라메카닉스 모델 (PyTorch)
        # s: Slip ratio 텐서 (Shape: [N])
        # h: 침하량 텐서 (Shape: [N])
        # v_c: 침하량 변화율 텐서 (Shape: [N])
        # alpha: Side-Slip angle 텐서 (Shape: [N])

        device = h.device
        
        # 파라미터 언패킹
        Rw, b, n = self.R, self.p['b'], self.p['n']
        kc, kphi = self.p['kc'], self.p['kphi']
        c1, c2 = self.p['c1'], self.p['c2']
        c_soil, phi_soil = self.p['c_soil'], self.p['phi_soil']
        Kx, Ky = self.p['Kx'], self.p['Ky']
        c_s = self.p['c_s']
        mu_s = self.p['mu_s']
        rho_s = self.p['rho_s']
        
        k_eq = kc / b + kphi

        # 1. 각도 계산 (Shape: [N])
        # clamp를 통해 h가 Rw보다 커서 acos 내부에 음수가 들어가는 에러 방지
        theta1 = torch.acos(torch.clamp(1.0 - h / Rw, min=-1.0, max=1.0)) 
        theta_m = (c1 + c2 * s) * theta1

        A_c = b * Rw * theta1 
        A_c_safe = torch.clamp(A_c, min=1e-6) # 0으로 나누기 에러 방지

        p_damping = (c_s * v_c) / A_c_safe
        p_damping_exp = p_damping.unsqueeze(1) # 차원 확장 [N, 1] -> [N, 60] 병렬 연산용
        
        # 🌟 [핵심 트릭] 적분 구간(linspace) 벡터화 🌟
        # 매트랩처럼 각 환경마다 크기가 다른 linspace를 직접 만들 수 없으므로,
        # 0~1 사이의 정규화된 배열(t)을 만들고, 여기에 theta1을 곱해서 [N, 60] 차원으로 확장합니다.
        num_points = 60
        t = torch.linspace(0, 1, num_points, device=device) # [60]
        
        # theta Shape: [N, 60]
        theta = theta1.unsqueeze(1) * t.unsqueeze(0) 
        
        # 연산을 위해 1D 텐서들을 [N, 60] 차원으로 맞춰줌 (Broadcasting)
        theta1_exp = theta1.unsqueeze(1)
        theta_m_exp = theta_m.unsqueeze(1)
        s_exp = s.unsqueeze(1)
        alpha_exp = alpha.unsqueeze(1)
        
        # 2. 구간 마스크 생성
        idx_F = theta >= theta_m_exp
        idx_R = theta < theta_m_exp
        
        p_theta = torch.zeros_like(theta)
        
        # (Front 구간)
        cos_diff_F = torch.cos(theta) - torch.cos(theta1_exp)
        p_F = k_eq * (torch.clamp(Rw * cos_diff_F, min=0.0) ** n) + p_damping_exp
        
        # (Rear 구간) - theta_m이 0일 때 0으로 나누기 에러 방지
        theta_m_safe = torch.where(theta_m_exp == 0, torch.tensor(1e-6, device=device), theta_m_exp)
        theta_eq = theta1_exp - (theta / theta_m_safe) * (theta1_exp - theta_m_exp)
        cos_diff_R = torch.cos(theta_eq) - torch.cos(theta1_exp)
        p_R = k_eq * (torch.clamp(Rw * cos_diff_R, min=0.0) ** n) + p_damping_exp

        # 마스크 적용
        p_theta = torch.where(idx_F, p_F, p_theta)
        p_theta = torch.where(idx_R, p_R, p_theta)
        
        # 4. 전단 응력(Shear stress) 분포 계산
        tau_max = torch.minimum(mu_s * p_theta, c_soil + p_theta * math.tan(phi_soil))
        j_x = Rw * ((theta1_exp - theta) - (1 - s_exp) * (torch.sin(theta1_exp) - torch.sin(theta)))
        j_y = Rw * (1 - s_exp) * (theta1_exp - theta) * torch.tan(alpha_exp)
        tau_x = tau_max * (1 - torch.exp(-j_x / Kx))
        tau_y = tau_max * (1 - torch.exp(-torch.abs(j_y) / Ky))
        
        # 불도저 효과
        h_theta = torch.clamp(Rw * (torch.cos(theta) - torch.cos(theta1_exp)), min=0.0)
        Rb_theta = self.C1 * (h_theta * c_soil + 0.5 * rho_s * (h_theta ** 2) * self.C2)
        
        # 5. 적분 (PyTorch의 trapezoid 함수 사용)
        # y값 먼저 넣고, x값(theta) 넣고, 어느 축(dim=1)으로 적분할지 지정
        Ft_normal_mag = Rw * b * torch.trapezoid(p_theta * torch.sin(theta), x=theta, dim=1)
        Ft_shear_mag  = Rw * b * torch.trapezoid(tau_x * torch.cos(theta), x=theta, dim=1)
        Fn_mag        = Rw * b * torch.trapezoid(p_theta * torch.cos(theta) + tau_x * torch.sin(theta), x=theta, dim=1)
        T_mag         = (Rw ** 2) * b * torch.trapezoid(tau_x, x=theta, dim=1)
        Fy_mag        = Rw * b * torch.trapezoid(tau_y, x=theta, dim=1) + torch.trapezoid(Rb_theta * Rw * torch.cos(theta), x=theta, dim=1) * torch.abs(torch.sin(alpha))
        
        return Fn_mag, Ft_normal_mag, Ft_shear_mag, Fy_mag, T_mag