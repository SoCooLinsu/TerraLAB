# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# ... (주석 생략)

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--save_name", type=str, default="trajectory", help="Prefix for the saved trajectory file.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import os
import time  # 🌟 [추가] 정밀 시간 측정을 위한 모듈

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import TerraLab.tasks  # noqa: F401

def main():
    # 🌟 [매우 중요] 결정론적 환경을 위한 완벽한 시드 고정
    torch.manual_seed(42)
    
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    
    env.reset()
    
    trajectory_history = []
    MAX_STEPS = 5000
    step_count = 0
    
    print(f"🚀 궤적 데이터 수집 시작... (총 {MAX_STEPS} 스텝)")

    # 🌟 [추가] 측정 시작 직전 타이머 가동 (GPU 동기화 포함)
    if env.unwrapped.device == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    while simulation_app.is_running() and step_count < MAX_STEPS:
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)
            
            base_pos = env.unwrapped.scene["robot"].data.root_pos_w.clone()
            trajectory_history.append(base_pos.cpu())
            
        step_count += 1
        if step_count % 1000 == 0: # 출력이 너무 많아지지 않게 500 단위로 조정
            print(f"진행 상황: {step_count}/{MAX_STEPS} 스텝 완료")

    # 🌟 [추가] 측정 종료 타이머 가동 및 소요 시간 계산
    if env.unwrapped.device == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    elapsed_time = end_time - start_time
    sps = step_count / elapsed_time  # 초당 처리 스텝 수 (Steps Per Second)

    # 데이터 병합 및 파일 저장
    final_trajectory_tensor = torch.stack(trajectory_history)
    
    save_path = f"{args_cli.save_name}.pt"
    torch.save(final_trajectory_tensor, save_path)
    
    print("\n=======================================================")
    print(f"✅ 수집 완료! 위치 데이터가 '{save_path}' 에 저장되었습니다.")
    print(f"📦 저장된 텐서 형태 (Shape): {final_trajectory_tensor.shape}")
    print(f"⏱️ 총 시뮬레이션 소요 시간: {elapsed_time:.4f} 초")
    print(f"⚡ 초당 스텝 처리량 (SPS): {sps:.2f} steps/sec")
    print("=======================================================\n")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()