import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import joblib
from WongReeceTrain import WongReeceTeacher

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_path = "lightgbm_surrogate.pkl"
    mean_path = "y_mean.npy"
    std_path = "y_std.npy"
    
    if not os.path.exists(model_path):
        print(f"❌ 오류: '{model_path}' 파일을 찾을 수 없습니다.")
        print("학습 스크립트에서 joblib.dump()로 모델을 먼저 저장해주세요.")
        return

    print(f"✅ '{model_path}' 및 정규화 파라미터 로드 중...")
    
    # 🌟 LightGBM 모델 및 역정규화 파라미터 로드
    lgbm_model = joblib.load(model_path)
    y_mean_np = np.load(mean_path)
    y_std_np = np.load(std_path)
    
    teacher = WongReeceTeacher(device)
    
    # 테스트용 데이터셋 생성 (Slip Ratio 0.0 ~ 1.0 Sweep)
    num_points = 100
    s_test = torch.linspace(0.0, 1.0, num_points, device=device)
    h_fixed = torch.full((num_points,), 0.002, device=device)       # 침하량
    vc_fixed = torch.full((num_points,), 0.2, device=device)          # 침하 속도
    alpha_fixed = torch.full((num_points,), 0.00, device=device)      # 측면 슬립각
    
    print("🧠 Teacher(물리 연산) 및 LightGBM 추론 진행 중...")
    
    # 1. 정답 도출 (Physics)
    with torch.no_grad():
        Fn_t, Ftn_t, Fts_t, Fy_t, T_t = teacher.wong_reece_model_torch(s_test, h_fixed, vc_fixed, alpha_fixed)
    
    # 2. 대리 모델 입력 정규화 (🌟 alpha 분모 pi/2 적용)
    X_test = torch.stack([
        s_test, 
        h_fixed / 0.035, 
        vc_fixed / 2.0, 
        alpha_fixed / (math.pi / 2.0) 
    ], dim=1)
    
    # 🌟 LightGBM은 NumPy 배열을 입력으로 받습니다.
    X_test_np = X_test.cpu().numpy()
    
    # LightGBM 추론
    preds_norm_np = lgbm_model.predict(X_test_np)
    
    # 🌟 출력 정규화 해제 (실제 Newton 단위 복구)
    preds_real_np = preds_norm_np * y_std_np + y_mean_np
    
    # CPU로 내리고 Numpy 변환 (Teacher)
    s_np = s_test.cpu().numpy()
    teacher_forces = [Fn_t.cpu().numpy(), Ftn_t.cpu().numpy(), Fts_t.cpu().numpy(), Fy_t.cpu().numpy(), T_t.cpu().numpy()]
    student_forces = [preds_real_np[:, 0], preds_real_np[:, 1], preds_real_np[:, 2], preds_real_np[:, 3], preds_real_np[:, 4]]
    
    # ---------------------------------------------------------
    # 4. Matplotlib 플롯 생성
    # ---------------------------------------------------------
    titles = ['Normal Force (Fn)', 'Normal-driven Tract (Ftn)', 'Shear-driven Tract (Fts)', 'Lateral Force (Fy)', 'Wheel Torque (T)']
    y_labels = ['Force (N)', 'Force (N)', 'Force (N)', 'Force (N)', 'Torque (Nm)']
    
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle("Terramechanics Validation: Physics vs. LightGBM Surrogate", fontsize=18, fontweight='bold')
    
    for i in range(5):
        axes[i].plot(s_np, teacher_forces[i], 'b-', linewidth=3, alpha=0.7, label='Physics (Teacher)')
        axes[i].plot(s_np, student_forces[i], 'r--', linewidth=2, label='LightGBM (Student)')
        axes[i].set_title(titles[i], fontsize=12)
        axes[i].set_xlabel('Slip Ratio (s)', fontsize=10)
        axes[i].set_ylabel(y_labels[i], fontsize=10)
        axes[i].grid(True, linestyle=':', alpha=0.6)
        axes[i].legend()
        
    plt.tight_layout()
    print("📊 렌더링 완료! 그래프 창을 띄웁니다.")
    plt.show()

    # 속도 벤치마크 실행
    benchmark_speed(lgbm_model, device)


def benchmark_speed(lgbm_model, device='cuda'):
    num_batch = int(16384)
    print(f"\n🚀 [속도 벤치마크] {num_batch}개 로버 바퀴 동시 연산 속도 측정 중...")

    teacher = WongReeceTeacher(device)
    
    # 1. 더미 배치 데이터 생성 (Teacher용 PyTorch 텐서)
    s = torch.rand(num_batch, device=device)
    h = torch.rand(num_batch, device=device) * 0.035
    v_c = torch.rand(num_batch, device=device) * 2.0
    alpha = (torch.rand(num_batch, device=device) * 2 - 1) * (math.pi / 2.0)
    
    # 2. LightGBM용 NumPy 배열 정규화 입력 생성
    X_batch = torch.stack([s, h/0.035, v_c/2.0, alpha/(math.pi/2.0)], dim=1)
    X_batch_np = X_batch.cpu().numpy()
    
    # ========================================================
    # 🌟 GPU 예열 (Teacher 연산용)
    # ========================================================
    with torch.no_grad():
        _ = teacher.wong_reece_model_torch(s, h, v_c, alpha)
    if device == 'cuda':
        torch.cuda.synchronize()

    # ========================================================
    # [1] 🌲 LightGBM 모델 시간 측정 (CPU 연산)
    # ========================================================
    start_time = time.perf_counter()
    _ = lgbm_model.predict(X_batch_np)
    # LightGBM은 CPU에서 돌아가므로 동기화(synchronize)가 필요 없습니다.
    time_lgbm_batch = time.perf_counter() - start_time

    # ========================================================
    # [2] 📐 원본 전산 모델 (Physics) 시간 측정 (GPU 연산)
    # ========================================================
    start_time = time.perf_counter()
    with torch.no_grad():
        _ = teacher.wong_reece_model_torch(s, h, v_c, alpha)
    if device == 'cuda':
        torch.cuda.synchronize()
    time_orig_batch = time.perf_counter() - start_time

    # ========================================================
    # 3. 결과 출력
    # ========================================================
    speedup_ratio = time_orig_batch / time_lgbm_batch
    
    print('===================================================================')
    print(f'⏱️ {num_batch}개 데이터 동시 연산(Batch Processing) 속도 비교')
    print(f'🌲 LightGBM 대리 모델 (CPU 트리 앙상블): {time_lgbm_batch:.6f} 초')
    print(f'📐 원본 수식 모델 (GPU 수치 적분): {time_orig_batch:.6f} 초')
    
    if speedup_ratio > 1:
        print(f'🚀 진짜 결과: LightGBM이 ** {speedup_ratio:.2f}배 ** 더 빠릅니다!')
    else:
        print(f'⚠️ 결과: GPU를 100% 활용하는 원본 수식 모델이 LightGBM보다 ** {1/speedup_ratio:.2f}배 ** 빠릅니다.')
        print('   (트리 모델은 순차적 연산 특성상, 배치 크기가 커질수록 GPU 기반 딥러닝이나 벡터 연산보다 밀릴 수 있습니다.)')
    print('===================================================================')

if __name__ == "__main__":
    main()