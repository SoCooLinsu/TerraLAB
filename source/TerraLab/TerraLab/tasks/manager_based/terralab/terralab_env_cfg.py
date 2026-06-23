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
from isaaclab.terrains import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from . import mdp

##
# Pre-defined configs
##

from TerraLab.robots.rover import ROVER_CONFIG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##


@configclass
class TerralabSceneCfg(InteractiveSceneCfg):

    # Terrain
    terrain = TerrainImporterCfg(
        prim_path= "/World",
        terrain_type = 'usd',
        usd_path = r"C:\test\source\test\test\tasks\manager_based\test\Terrain0.5.usd",
        env_spacing = 1.0,
        collision_group=-1,
    )

    #terrain = AssetBaseCfg(
    #    prim_path="/World/terrain/Terrain/moon",
    #    spawn=sim_utils.GroundPlaneCfg(size=(50.0, 50.0)),
    #)

    # terrain = TerrainImporterCfg(
    #     prim_path="/World/terrain/Terrain/moon",
    #     terrain_type="plane",
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     visual_material=sim_utils.MdlFileCfg(
    #         mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
    #         project_uvw=True,
    #         texture_scale=(0.25, 0.25),
    #     ),
    #     debug_vis=False,
    # )

    # terrain = TerrainImporterCfg(
    #     prim_path="/World/terrain/Terrain/moon",
    #     terrain_type="generator", 
    #     terrain_generator=TerrainGeneratorCfg(
    #         seed=42,
    #         size=(6.0, 6.0),          # 각 지형 패치의 크기
    #         border_width=3.0,        # 외곽 여백
    #         num_rows=10,               # 세로 칸 수
    #         num_cols=10,             # 가로 칸 수
    #         horizontal_scale=0.05,     # X, Y축 해상도
    #         vertical_scale=0.005,     # Z축 해상도
    #         # 4가지 지형 혼합 설정
    #         sub_terrains={
    #             "flat": terrain_gen.MeshPlaneTerrainCfg(
    #                 proportion=0.2,
    #             ),
    #             "random_uniform": terrain_gen.HfRandomUniformTerrainCfg(
    #                 proportion=0.2,
    #                 noise_range=(0.005, 0.02),
    #                 noise_step=0.005,
    #             ),
    #             "wave": terrain_gen.HfWaveTerrainCfg(
    #                 proportion=0.3, 
    #                 amplitude_range=(0.025, 0.1),
    #                 num_waves=4,
    #             ),
    #             "pyramid_slopes": terrain_gen.HfPyramidSlopedTerrainCfg(
    #                 proportion=0.3,
    #                 platform_width=0.5,
    #                 slope_range=(0.05, 0.2),
    #             ),
    #         },
    #     ),
    #     collision_group=-1,
    # )

    # Rover
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Rover",
        spawn=ROVER_CONFIG.spawn,
        debug_vis = False,
        actuator_value_resolution_debug_print=True,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 1.0, 0.15),
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
            pos=(0.0, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
        max_distance = 5.0,
        debug_vis = False,
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
            pos=(0.0, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
        max_distance = 5.0,
        debug_vis = False,
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
            pos=(0.0, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
        max_distance = 5.0,
        debug_vis = False,
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
            pos=(0.0, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
        max_distance = 5.0,
        debug_vis = False,
        ray_alignment="yaw",
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Rover/Rover/base_link",
        mesh_prim_paths=["/World/terrain/Terrain/moon"],
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.05,
            size=(0.5, 0.4),
            ordering = 'yx'
        ),
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
        max_distance = 5.0,
        debug_vis= True,
        ray_alignment="yaw",
    )
    height_scanner = None


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
        scale=3.0,
        # clip={"Wheel_joint_.*": (-1.0, 1.0)},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # input task
        #velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        # # (3) Gravity (3 inputs)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        # # (4) Current Rocker joint Position (rocker_l, rocker_r) (2 inputs)
        rocker_position = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Rocker_joint_.*"])}, scale=3 / torch.pi)
        # # (5) Current wheel velocity (fl, rl fr rr) (4 inputs)
        wheel_velocities = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Wheel_joint_.*"])}, scale=1 / 15.0)
        actions = ObsTerm(func=mdp.last_action)
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 1.0),
        # )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    # 토양 성질 도메인 랜덤화
    randomize_soil = EventTerm(
        func=mdp.randomize_soil_properties_startup,
        mode="startup", 
        params={
            "soil_ranges": {
                'kc': (888.0, 1111.0),
                'kphi': (828000.0, 1228000.0),
                'c_soil': (1600.0, 1800.0),
                'phi_soil': (0.6, 0.8),
                'Kx': (0.0003, 0.0006),
                'Ky': (0.0006, 0.0012),
            }
        },
    )

    # 토양 성질 그리드 스윕 (발표 시각화용)
    # grid_sweep_soil = EventTerm(
    #     func=mdp.grid_soil_properties_startup, # 새로 만든 함수로 변경!
    #     mode="startup", 
    #     params={
    #         "soil_ranges": {
    #             # 스윕하고 싶은 최소-최대 범위를 지정하세요
    #             'kc': (490.0, 490.0),       # X축: 좌측에서 우측으로 갈수록 kc 증가
    #             'kphi': (528000.0, 528000.0), # Y축: 아래에서 위로 갈수록 kphi 증가
    #             # 아래 값들은 시각화 통제를 위해 이 범위의 "중간값"으로 일괄 고정됩니다.
    #             'c_soil': (0.0, 3000.0),
    #             'phi_soil': (0.0872665, 0.785398),
    #             'Kx': (0.0006, 0.0006),
    #             'Ky': (0.005, 0.005),
    #         }
    #     },
    # )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # -- task
    #track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=10.0, params={"command_name": "base_velocity", "std": math.sqrt(0.025)})
    #track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.025)})
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=10.0,
        params={"std": 3.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=10.0,
        params={"std": 0.6, "command_name": "pose_command"},
    )
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.5,
        params={"command_name": "pose_command"},
    )
    # -- penalty
    power_consumption = RewTerm(func=mdp.rover_power_consumption, weight=-0.005, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Wheel_joint_.*"])})
    wheel_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-0.000005, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Wheel_joint_.*"])})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    #base_contact = DoneTerm(
    #    func=mdp.illegal_contact,
    #    params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    #)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, time_out=False, params={"limit_angle": 0.8})


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    #base_velocity = mdp.UniformVelocityCommandCfg(
    #    asset_name="robot",
    #    resampling_time_range=(0.0, 10.0),
    #    rel_standing_envs=0.00,
    #    debug_vis=True,
    #    ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #        lin_vel_x=(0.5, 0.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.0, 0.0),
    #    ),
    #)

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)),
    )


##
# Environment configuration
##


@configclass
class TerralabEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: TerralabSceneCfg = TerralabSceneCfg(num_envs=4096, env_spacing=0.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 8
        # viewer settings
        self.viewer.eye = (5.0, 0.0, 5.0)
        # simulation settings
        self.sim.gravity = (0.0, 0.0, -1.62)
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.min_velocity_iteration_count = 1
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt


##
# Custom Environment Class
##

class TerralabEnv(ManagerBasedRLEnv):

    def __init__(self, cfg, **kwargs):
        
        # 1. 부모 클래스 초기화 전에 설정(cfg)에서 디바이스와 환경 개수 추출
        num_envs = cfg.scene.num_envs
        device = cfg.sim.device

        # 2. 토양 파라미터 텐서를 부모 init 호출 "전에" 미리 생성
        # [수정] self.num_envs -> num_envs, self.device -> device 로 변경!
        self.p = {
            'b': torch.full((num_envs,), 0.0375, device=device),
            'n': torch.full((num_envs,), 1.0, device=device),
            'kc': torch.full((num_envs,), 990.0, device=device),
            'kphi': torch.full((num_envs,), 1528000.0, device=device),
            'c1': torch.full((num_envs,), 0.30, device=device),
            'c2': torch.full((num_envs,), 0.10, device=device),
            'c_soil': torch.full((num_envs,), 1716.0, device=device),
            'phi_soil': torch.full((num_envs,), 0.7086, device=device),
            'mu_s': torch.full((num_envs,), 0.8, device=device),
            'Kx': torch.full((num_envs,), 0.0006, device=device),
            'Ky': torch.full((num_envs,), 0.005, device=device),
            'c_s': torch.full((num_envs,), 40.0, device=device),
            'rho_s': torch.full((num_envs,), 3340.0, device=device),
            'alpha_b': torch.full((num_envs,), 0.0, device=device)
        }
        
        # 여기도 마찬가지로 지역 변수 사용
        self.C1 = torch.zeros((num_envs,), device=device)
        self.C2 = torch.zeros((num_envs,), device=device)

        self.R = 0.0555
        self.prev_sinkage = torch.zeros((num_envs, 4), device=device)

        # 3. 부모 클래스 __init__ 호출 
        super().__init__(cfg, **kwargs)

        print("\n" + "="*60)
        print(f"🚀 [디버그] 토양 파라미터 도메인 랜덤화 결과 (총 {self.num_envs}개 환경)")
        # 텐서가 GPU에 있으므로 보기 편하게 리스트로 변환해서 출력합니다.
        print(f" - kphi (Y축 스윕)   : {self.p['kphi'][::6][:5].tolist()}")
        print(f" - phi_soil (마찰각) : {self.p['phi_soil'][:5].tolist()}")
        print(f" - c_soil (점착력)   : {self.p['c_soil'][:5].tolist()}")
        print(f" - kc (변형 계수)    : {self.p['kc'][:5].tolist()}")
        print(f" - C1 (불도징 계수 1): {self.C1[:5].tolist()}")
        print("="*60 + "\n")
        
        # 초기 기본값으로 C1, C2 세팅 (이벤트 매니저가 곧 덮어쓰겠지만 안전을 위해)
        # all_env_ids = torch.arange(self.num_envs, device=self.device)
        # mdp.randomize_soil_properties_startup(self, all_env_ids, {}) # 초기값 기반 연산
        
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

        # 신경망 Set
        self.eps = 1e-6
        self.surrogate_model = torch.jit.load("terra_surrogate_isaac.pt", map_location=self.device)
        self.surrogate_model.eval()
    
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
        
        self.Sinkage = torch.clamp(self.R - (centers_z - hits_z), min=1e-9, max=2*self.R)

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
        alpha = torch.clamp(torch.atan2(v_y, torch.abs(v_x)), min=-((math.pi / 2.0) - 1e-6), max=(math.pi / 2.0) - 1e-6)

        # 슬립 Ratio
        v_diff = self.R * wheel_joint_vel - v_x # 바퀴 회전 선속도 (Rw * w)
        num = torch.abs(v_diff)
        den = torch.max(torch.abs(self.R * wheel_joint_vel), torch.abs(v_x)) + 1e-6
        self.slip_ratios = torch.clamp(num / den, min=0.0, max=1.0)

        # Wong-Reece Model
        Fn_flat, Ft_n_flat, Ft_s_flat, Fy_flat, T_flat = self.wong_reece_model_torch(self.slip_ratios.view(-1), self.Sinkage.view(-1), v_c.view(-1), alpha.view(-1))
        # Fn_flat, Ft_n_flat, Ft_s_flat, Fy_flat, T_flat = self.wong_reece_model_net(self.slip_ratios.view(-1), self.Sinkage.view(-1), v_c.view(-1), torch.abs(alpha.view(-1)))

        # 6. 결과값을 다시 원래 구조 [num_envs, 4]로 복구
        Fn     = Fn_flat.view(self.num_envs, 4)
        Ft_n   = Ft_n_flat.view(self.num_envs, 4)
        Ft_s   = Ft_s_flat.view(self.num_envs, 4)
        Fy_mag = Fy_flat.view(self.num_envs, 4)
        T_mag  = T_flat.view(self.num_envs, 4)

        # 힘 방향성 결합
        Ft = Ft_s * torch.sign(v_diff) - Ft_n * torch.sign(v_x)
        Fy = -Fy_mag * torch.sign(alpha)
        Torque_res = -T_mag * torch.sign(v_diff)
        
        # Forces 텐서
        forces = torch.zeros((self.num_envs, 8, 3), device=self.device)
        forces[:, 0:4, 0] = Ft
        forces[:, 0:4, 1] = Fy
        forces[:, 0:4, 2] = Fn
        
        # Torques 텐서
        torques = torch.zeros((self.num_envs, 8, 3), device=self.device)
        torques[:, 4:8, 2] = Torque_res

        # print(f"[디버그] 조인트속도! 1번 환경 값: {wheel_joint_vel[1]}")
        # print(f"[디버그] 힘! 1번 환경 값: {forces[1, 0:4, 2]}")
        # print(f"[디버그] 힘! 1번 환경 값: {torques[0, 4:8, 2]}")
        # print(f"[디버그] 슬립율! 1번 환경 값: {self.slip_ratios[1]}")
        # print(f"[디버그] 침하량 텐서 Shape: {self.Sinkage.shape}, 1번 환경 값: {self.Sinkage[1]}")
        
        # 로봇에 외부 힘과 회전력 모두 설정
        robot.instantaneous_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=self.Terra_body_ids,
            positions = None,
            is_global=False
            )
    
    def wong_reece_model_torch(self, s: torch.Tensor, h: torch.Tensor, v_c: torch.Tensor, alpha: torch.Tensor):

        device = h.device
        
        # (num_envs,) 텐서를 (num_envs * 4,) 로 확장하는 헬퍼 함수
        def expand_param(param_tensor):
            return param_tensor.unsqueeze(1).expand(-1, 4).reshape(-1)

        # 파라미터 언패킹 및 확장 (Shape: [N] -> [64])
        Rw = self.R
        b = expand_param(self.p['b'])
        n = expand_param(self.p['n'])
        kc = expand_param(self.p['kc'])
        kphi = expand_param(self.p['kphi'])
        c1 = expand_param(self.p['c1'])
        c2 = expand_param(self.p['c2'])
        c_soil = expand_param(self.p['c_soil'])
        phi_soil = expand_param(self.p['phi_soil'])
        Kx = expand_param(self.p['Kx'])
        Ky = expand_param(self.p['Ky'])
        c_s = expand_param(self.p['c_s'])
        mu_s = expand_param(self.p['mu_s'])
        rho_s = expand_param(self.p['rho_s'])
        
        C1_exp = expand_param(self.C1)
        C2_exp = expand_param(self.C2)

        k_eq = kc / b + kphi

        # 1. 각도 계산 (Shape: [N])
        theta1 = torch.acos(1.0 - h / Rw)
        theta_m = (c1 + c2 * s) * theta1

        A_c = b * Rw * theta1 
        A_c_safe = torch.clamp(A_c, min=1e-6)
        p_damping = (c_s * v_c) / A_c_safe
        
        num_points = 240
        t = torch.linspace(0, 1, num_points, device=device)
        
        # theta Shape: [N, 60]
        theta = theta1.unsqueeze(1) * t.unsqueeze(0) 
        
        # 연산을 위해 1D 텐서들을 [N, 1] 차원으로 맞춰줌 (Broadcasting 준비)
        theta1_exp = theta1.unsqueeze(1)
        theta_m_exp = theta_m.unsqueeze(1)
        s_exp = s.unsqueeze(1)
        alpha_exp = alpha.unsqueeze(1)
        p_damping_exp = p_damping.unsqueeze(1)
        
        # [수정] 토양 파라미터들도 [N, 1]로 확장
        n_exp = n.unsqueeze(1)
        k_eq_exp = k_eq.unsqueeze(1)
        c_soil_exp = c_soil.unsqueeze(1)
        phi_soil_exp = phi_soil.unsqueeze(1)
        mu_s_exp = mu_s.unsqueeze(1)
        Kx_exp = Kx.unsqueeze(1)
        Ky_exp = Ky.unsqueeze(1)
        C1_mat = C1_exp.unsqueeze(1)
        C2_mat = C2_exp.unsqueeze(1)
        rho_s_mat = rho_s.unsqueeze(1)

        # 2. 구간 마스크 생성
        idx_F = theta >= theta_m_exp
        idx_R = theta < theta_m_exp
        
        # Normal stress
        cos_diff_F = torch.cos(theta) - torch.cos(theta1_exp)
        p_F = k_eq_exp * (torch.clamp(Rw * cos_diff_F, min=0.0) ** n_exp) + p_damping_exp
        
        theta_m_safe = torch.where(theta_m_exp == 0, torch.tensor(1e-6, device=device), theta_m_exp)
        theta_eq = theta1_exp - (theta / theta_m_safe) * (theta1_exp - theta_m_exp)
        cos_diff_R = torch.cos(theta_eq) - torch.cos(theta1_exp)
        p_R = k_eq_exp * (torch.clamp(Rw * cos_diff_R, min=0.0) ** n_exp) + p_damping_exp

        p_theta = torch.zeros_like(theta)
        p_theta = torch.where(idx_F, p_F, p_theta)
        p_theta = torch.where(idx_R, p_R, p_theta)
        
        # Shear stress [수정: math.tan -> torch.tan]
        tau_max = torch.minimum(mu_s_exp * p_theta, c_soil_exp + p_theta * torch.tan(phi_soil_exp))
        j_x = Rw * ((theta1_exp - theta) - (1 - s_exp) * (torch.sin(theta1_exp) - torch.sin(theta)))
        j_y = Rw * (1 - s_exp) * (theta1_exp - theta) * torch.tan(torch.abs(alpha_exp))
        tau_x = tau_max * (1 - torch.exp(-j_x / Kx_exp))
        tau_y = tau_max * (1 - torch.exp(-j_y / Ky_exp))
        
        # Bulldozing Resistance
        h_theta = torch.clamp(Rw * (torch.cos(theta) - torch.cos(theta1_exp)), min=0.0)
        Rb_theta = C1_mat * (h_theta * c_soil_exp + 0.5 * rho_s_mat * (h_theta ** 2) * C2_mat)
        
        # 5. 적분 (결과값 Shape: [N])
        Fn_mag        = Rw * b * torch.trapezoid(p_theta * torch.cos(theta) + tau_x * torch.sin(theta), x=theta, dim=1)
        Ft_normal_mag = Rw * b * torch.trapezoid(p_theta * torch.sin(theta), x=theta, dim=1)
        Ft_shear_mag  = Rw * b * torch.trapezoid(tau_x * torch.cos(theta), x=theta, dim=1)
        Fy_mag        = Rw * b * torch.trapezoid(tau_y, x=theta, dim=1) + torch.trapezoid(Rb_theta * Rw * torch.cos(theta), x=theta, dim=1) * torch.abs(torch.sin(alpha))
        T_mag         = (Rw ** 2) * b * torch.trapezoid(tau_x, x=theta, dim=1)
        
        return Fn_mag, Ft_normal_mag, Ft_shear_mag, Fy_mag, T_mag
    
    def wong_reece_model_net(self, s: torch.Tensor, h: torch.Tensor, v_c: torch.Tensor, alpha: torch.Tensor):
        # 1. 입력 텐서 정규화 (alpha 분모 주의!)
        x_input = torch.stack([
            s, 
            h / (0.035 - self.eps), 
            v_c / 2.0, 
            alpha / ((math.pi / 2.0) - self.eps),
        ], dim=1)

        # 2. 대리 모델 추론 (JIT 모델이 역정규화까지 알아서 다 해줍니다!)
        with torch.no_grad():
            preds_real = self.surrogate_model(x_input) # 바로 물리량(Newton) 튀어나옴
            preds_real[h <= self.eps] = 0.0
            preds_real[alpha <= self.eps, 3] = 0.0

        # 3. 결과 언패킹
        Fn_flat   = preds_real[:, 0]
        Ft_n_flat = preds_real[:, 1]
        Ft_s_flat = preds_real[:, 2]
        Fy_flat   = preds_real[:, 3]
        T_flat    = preds_real[:, 4]

        return Fn_flat, Ft_n_flat, Ft_s_flat, Fy_flat, T_flat