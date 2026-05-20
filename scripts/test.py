import torch

# 두 데이터 로드
traj_math = torch.load("math_model.pt") # Shape: [Steps, Envs, 3]
traj_net = torch.load("net_model.pt")

# 🌟 MSE를 먼저 구한 뒤, 루트(sqrt)를 씌워 RMSE로 변환
mse_loss = torch.nn.functional.mse_loss(traj_math, traj_net)
rmse_loss = torch.sqrt(mse_loss)

# 최대 오차 계산 (이것도 매우 직관적인 지표입니다)
max_error = torch.max(torch.abs(traj_math - traj_net))

print("📊 [물리 수식 vs 인공지능 대리 모델 궤적 비교 결과]")
print(f"총 스텝 수: {traj_math.shape[0]}, 환경 수: {traj_math.shape[1]}")
print(f"평균 궤적 오차 (RMSE): {rmse_loss.item():.10f} m")
print(f"단일 로버 기준 최대 위치 이탈 거리: {max_error.item():.6f} m")