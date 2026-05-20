import time, torch, math
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from WongReeceTrain import WongReeceTeacher
from scipy.stats import qmc
import joblib

# --- (generate_dataset 함수 부분은 질문자님 코드 그대로 사용) ---
def generate_dataset(num_samples=1000000, device='cuda'):
    print(f"[{num_samples}개] 라틴 하이퍼큐브 샘플링(LHS) 데이터 생성 중...")
    
    eps = 1e-6
    sampler = qmc.LatinHypercube(d=4)
    sample = sampler.random(n=num_samples) 
    
    lhs_tensor = torch.tensor(sample, dtype=torch.float32, device=device)
    
    slip = lhs_tensor[:, 0]                                
    sinkage = eps + lhs_tensor[:, 1] * (0.035 - eps)                    
    v_c = lhs_tensor[:, 2] * 2.0                           
    alpha = (lhs_tensor[:, 3] * 2 - 1) * (math.pi / 2.0)
    
    # 신경망 입력용 0~1 정규화 (X)
    X = torch.stack([
        slip, 
        (sinkage - eps) / (0.035 - eps),
        v_c / 2.0, 
        alpha / (math.pi / 2.0)
    ], dim=1)
    
    teacher = WongReeceTeacher(device)
    Fn, Ftn, Fts, Fy, T = teacher.wong_reece_model_torch(slip, sinkage, v_c, alpha)
    Y_raw = torch.stack([Fn, Ftn, Fts, Fy, T], dim=1)
    
    Y_mean = Y_raw.mean(dim=0)
    Y_std = Y_raw.std(dim=0)
    Y_norm = (Y_raw - Y_mean) / (Y_std + 1e-8)
    
    return X, Y_norm, Y_mean, Y_std


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    X, Y_norm, Y_mean, Y_std = generate_dataset(num_samples=500000, device=device)
    
    split_idx = int(len(X) * 0.9)
    X_train_np = X[:split_idx].cpu().numpy()
    Y_train_np = Y_norm[:split_idx].cpu().numpy()
    X_val_np = X[split_idx:].cpu().numpy()
    Y_val_np = Y_norm[split_idx:].cpu().numpy()

    # ---------------------------------------------------------
    # 베이스라인 모델: LightGBM 다중 출력 회귀 모델
    # ---------------------------------------------------------
    print("\n🌲 [Baseline] LightGBM 모델 학습 시작...")
    start_time = time.time()

    # 🌟 최적화: n_jobs=-1 (전체 코어 사용), verbose=-1 (경고 로그 숨김)
    lgbm_base = LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    baseline_model = MultiOutputRegressor(lgbm_base)

    baseline_model.fit(X_train_np, Y_train_np)

    train_time = time.time() - start_time
    print(f"✅ LightGBM 학습 완료! (소요 시간: {train_time:.2f}초)")

    # ---------------------------------------------------------
    # 검증 데이터 평가 및 추론 속도 측정
    # ---------------------------------------------------------
    print("\n📊 검증 데이터 평가 및 추론 속도 측정 중...")
    infer_start = time.time()
    Y_pred_np = baseline_model.predict(X_val_np)
    infer_time = time.time() - infer_start

    # 1. 정규화된 상태에서의 성능 (학습 안정성 지표)
    mse_norm = mean_squared_error(Y_val_np, Y_pred_np)
    r2 = r2_score(Y_val_np, Y_pred_np)
    
    print(f"🔸 정규화 검증 MSE: {mse_norm:.6f}")
    print(f"🔸 검증 R² Score (1.0에 가까울수록 완벽): {r2:.4f}")

    # 2. 🌟 역정규화를 통한 실제 물리량 기준 오차 계산
    # Y_mean과 Y_std를 NumPy 배열로 변환하여 계산
    y_mean_np = Y_mean.cpu().numpy()
    y_std_np = Y_std.cpu().numpy()
    
    Y_val_real = Y_val_np * y_std_np + y_mean_np
    Y_pred_real = Y_pred_np * y_std_np + y_mean_np
    
    mae_real = mean_absolute_error(Y_val_real, Y_pred_real)
    print(f"🔸 실제 단위(Newton/Nm) 평균 절대 오차(MAE): 약 {mae_real:.2f}")

    # 3. 속도 측정
    samples_count = len(X_val_np)
    time_per_10k = (infer_time / samples_count) * 10000 * 1000 # 밀리초(ms) 단위
    print(f"🔸 추론 속도 (10,000개 샘플 동시 추론 기준): {time_per_10k:.2f} ms")
    print("-" * 60)

    print("📦 LightGBM 모델 및 파라미터 저장 중...")
    joblib.dump(baseline_model, "lightgbm_surrogate.pkl")
    np.save("y_mean.npy", Y_mean.cpu().numpy())
    np.save("y_std.npy", Y_std.cpu().numpy())
    print("✅ 저장 완료! ('lightgbm_surrogate.pkl', 'y_mean.npy', 'y_std.npy')")