import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from WongReeceTrain import WongReeceTeacher, TerraSurrogateNet
from scipy.stats import qmc

import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from WongReeceTrain import WongReeceTeacher
from sklearn.metrics import r2_score

def main():
    eps = 1e-6
    max_h = 0.035
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 저장된 JIT 모델 파일 경로
    model_path = "terra_surrogate_isaac.pt" 
    
    if not os.path.exists(model_path):
        print(f"❌ 오류: '{model_path}' 파일을 찾을 수 없습니다.")
        return

    print(f"✅ '{model_path}' 로드 중...")
    
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    
    teacher = WongReeceTeacher(device)

    #---------------R2 Score--------------------
    print("\n📈 [전체 물리 공간 R² Score 평가 중...]")

    n_eval = int(150000)
    sampler = qmc.LatinHypercube(d=4)
    sample = sampler.random(n=n_eval) 
    
    lhs_tensor = torch.tensor(sample, dtype=torch.float32, device=device)

    # 🌟 집중도(Concentration) 조절 파라미터 (1.0 = 균일, 클수록 0 근처에 집중)
    n = 1.5
    sinkage_dense = lhs_tensor[:, 1] ** n
    vc_dense = lhs_tensor[:, 2] ** n
    
    s_eval = lhs_tensor[:, 0]                                        # 0 ~ 1 [number]
    h_eval = eps + sinkage_dense * (0.035 - eps)                     # eps ~ R [m]
    vc_eval = vc_dense * 2.0                                         # 0 ~ 2 [m/s]
    alpha_eval = lhs_tensor[:, 3] * ((math.pi / 2.0) - eps)          # 0 ~ (pi/2 - eps) [rad]
    
    with torch.no_grad():
        # Teacher 정답 계산
        Fn_e, Ftn_e, Fts_e, Fy_e, T_e = teacher.wong_reece_model_torch(s_eval, h_eval, vc_eval, alpha_eval)
        Y_true = torch.stack([Fn_e, Ftn_e, Fts_e, Fy_e, T_e], dim=1).cpu().numpy() 

        # Student 예측 계산
        X_eval = torch.stack([
            s_eval, 
            (h_eval - eps) / (max_h - eps),
            vc_eval / 2.0, 
            alpha_eval / ((math.pi / 2.0) - eps)
        ], dim=1)
        Y_pred = model(X_eval).cpu().numpy()

    print("---------------------------------------------------")
    force_names = ['Fn (수직항력)', 'Ftn (법선 추력)', 'Fts (전단 추력)', 'Fy (측면력)', 'T (구동 토크)']
    for i in range(5):
        r2 = r2_score(Y_true[:, i], Y_pred[:, i])
        print(f"🔸 {force_names[i]:<18} R² Score: {r2:.6f}")
    print("---------------------------------------------------\n")

    
    # =========================================================
    # 🌟 다중 축(Multi-Axis) 시각화 블록
    # =========================================================
    h_values = torch.linspace(1e-6, max_h, 5, device=device)
    num_points = 100

    # 반복되는 그래프 생성 코드를 우아하게 처리하기 위한 헬퍼 함수
    def create_plot_window(x_tensor, x_label, fixed_s, fixed_vc, fixed_alpha, title):
        x_np = x_tensor.cpu().numpy()
        
        fig, axes = plt.subplots(2, 5, figsize=(24, 10))
        fig.suptitle(title, fontsize=18, fontweight='bold')
        
        titles = ['Normal Force (Fn)', 'Normal-driven Tract (Ftn)', 'Shear-driven Tract (Fts)', 'Lateral Force (Fy)', 'Wheel Torque (T)']
        y_labels_top = ['Force (N)', 'Force (N)', 'Force (N)', 'Force (N)', 'Torque (Nm)']
        y_labels_bottom = ['Abs Error (N)', 'Abs Error (N)', 'Abs Error (N)', 'Abs Error (N)', 'Abs Error (Nm)']
        
        colors = plt.cm.jet(np.linspace(0, 1, len(h_values)))
        
        for idx, h_val in enumerate(h_values):
            # 1. 현재 축(X-axis) 설정에 맞게 동적으로 텐서 할당
            h_t = torch.full((num_points,), h_val.item(), device=device)
            s_t = x_tensor if x_label == 'Slip Ratio (s)' else torch.full((num_points,), fixed_s, device=device)
            vc_t = x_tensor if x_label == 'Sinkage Rate (v_c, m/s)' else torch.full((num_points,), fixed_vc, device=device)
            alpha_t = x_tensor if x_label == 'Slip Angle (alpha, rad)' else torch.full((num_points,), fixed_alpha, device=device)
            
            # 2. Teacher (물리 모델) 추론
            with torch.no_grad():
                Fn_t, Ftn_t, Fts_t, Fy_t, T_t = teacher.wong_reece_model_torch(s_t, h_t, vc_t, alpha_t)
            
            # 3. Student (신경망) 정규화 및 추론 
            # 🚨 학습 때 사용했던 분모(2.0, max_h 등)를 그대로 유지
            X_test = torch.stack([
                s_t, 
                (h_t - eps) / (max_h - eps),
                vc_t / 2.0, 
                alpha_t / ((math.pi / 2.0) - eps)
            ], dim=1)
            
            with torch.no_grad():
                preds_real = model(X_test)
                
            teacher_forces = [Fn_t.cpu().numpy(), Ftn_t.cpu().numpy(), Fts_t.cpu().numpy(), Fy_t.cpu().numpy(), T_t.cpu().numpy()]
            student_forces = [preds_real[:, 0].cpu().numpy(), preds_real[:, 1].cpu().numpy(), preds_real[:, 2].cpu().numpy(), preds_real[:, 3].cpu().numpy(), preds_real[:, 4].cpu().numpy()]
            
            h_mm = h_val.item() * 1000
            
            # 4. 플롯 그리기
            for i in range(5):
                axes[0, i].plot(x_np, teacher_forces[i], linestyle='-', color=colors[idx], linewidth=4, alpha=0.4, 
                             label=f'Physics (h={h_mm:.1f}mm)' if i==0 else "")
                axes[0, i].plot(x_np, student_forces[i], linestyle='--', color=colors[idx], linewidth=2, 
                             label=f'NN (h={h_mm:.1f}mm)' if i==0 else "")
                
                abs_error = np.abs(teacher_forces[i] - student_forces[i])
                axes[1, i].plot(x_np, abs_error, linestyle='-', color=colors[idx], linewidth=2,
                             label=f'Error (h={h_mm:.1f}mm)' if i==0 else "")

        # 5. 축 세팅 및 디자인
        for i in range(5):
            axes[0, i].set_title(titles[i], fontsize=12)
            axes[0, i].set_ylabel(y_labels_top[i], fontsize=10)
            axes[0, i].grid(True, linestyle=':', alpha=0.6)
            if i == 0: axes[0, i].legend(fontsize=9, loc='upper left')
            
            axes[1, i].set_xlabel(x_label, fontsize=10)
            axes[1, i].set_ylabel(y_labels_bottom[i], fontsize=10, color='red')
            axes[1, i].grid(True, linestyle=':', alpha=0.6)
            axes[1, i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            if i == 0: axes[1, i].legend(fontsize=9, loc='upper left')
                
        plt.tight_layout()

    print("🧠 3개의 개별 윈도우 렌더링을 준비 중입니다...")

    # --- 🪟 창 1: X축 = Slip Ratio (기존) ---
    s_test = torch.linspace(0.0, 1.0, num_points, device=device)
    create_plot_window(x_tensor=s_test, x_label='Slip Ratio (s)', 
                       fixed_s=None, fixed_vc=0.01, fixed_alpha=eps/2,
                       title="Validation: X-Axis = Slip Ratio (Sweep Sinkage)")

    # --- 🪟 창 2: X축 = Slip Angle (알파) ---
    alpha_test = torch.linspace(1e-5, math.pi/2.0, num_points, device=device)
    create_plot_window(x_tensor=alpha_test, x_label='Slip Angle (alpha, rad)', 
                       fixed_s=0.2, fixed_vc=0.01, fixed_alpha=None, # 고정 슬립율을 0.2로 세팅
                       title="Validation: X-Axis = Slip Angle (Sweep Sinkage)")

    # --- 🪟 창 3: X축 = Sinkage Rate (침하 속도) ---
    vc_test = torch.linspace(0.0, 2.0, num_points, device=device)
    create_plot_window(x_tensor=vc_test, x_label='Sinkage Rate (v_c, m/s)', 
                       fixed_s=0.2, fixed_vc=None, fixed_alpha=0.1, 
                       title="Validation: X-Axis = Sinkage Rate (Sweep Sinkage)")

    print("📊 렌더링 완료! 3개의 그래프 창을 동시에 띄웁니다.")
    plt.show() # 🌟 이 명령어 한 번으로 3개의 창이 동시에 팝업됩니다.

    # 벤치마크 호출
    benchmark_speed()


def benchmark_speed(model_path="terra_surrogate_isaac.pt", device='cuda'):
    num_batch = int(64)
    iterations = 100  # 🌟 100번 반복하여 평균을 냅니다
    print(f"\n🚀 [속도 벤치마크] {num_batch}개 로버 바퀴 동시 연산 (100회 반복 평균) 측정 중...")

    if not os.path.exists(model_path):
        print(f"❌ 오류: '{model_path}' 파일을 찾을 수 없습니다.")
        return

    print(f"✅ '{model_path}' 로드 중...")
    model = torch.jit.load(model_path, map_location=device)
    model.eval() 
    
    teacher = WongReeceTeacher(device)
    
    s = torch.rand(num_batch, device=device)
    h = torch.rand(num_batch, device=device) * 0.035 # 튜닝된 범위
    v_c = torch.rand(num_batch, device=device) * 0.2
    alpha = (torch.rand(num_batch, device=device) * 2 - 1) * (math.pi / 2.0)
    
    X_batch = torch.stack([s, h/0.010, v_c/0.2, alpha/(math.pi/2.0)], dim=1)
    
    # ========================================================
    # 🌟 [매우 중요] GPU 예열 (Warm-up) - 여러 번 돌려서 GPU를 완전히 깨웁니다
    # ========================================================
    print("🔥 GPU 예열 중...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(X_batch)
            _ = teacher.wong_reece_model_torch(s, h, v_c, alpha)
    if device == 'cuda':
        torch.cuda.synchronize()

    # ========================================================
    # [1] 🤖 신경망 대리 모델 (Surrogate) 시간 측정
    # ========================================================
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(X_batch)
    if device == 'cuda':
        torch.cuda.synchronize() 
    time_pinn_batch = (time.perf_counter() - start_time) / iterations # 1회 평균

    # ========================================================
    # [2] 📐 원본 전산 모델 (Physics) 시간 측정
    # ========================================================
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = teacher.wong_reece_model_torch(s, h, v_c, alpha)
    if device == 'cuda':
        torch.cuda.synchronize()
    time_orig_batch = (time.perf_counter() - start_time) / iterations # 1회 평균

    # ========================================================
    # 3. 결과 출력
    # ========================================================
    speedup_ratio = time_orig_batch / time_pinn_batch
    
    print('===================================================================')
    print(f'⏱️ {num_batch}개 데이터 동시 연산 (1회 평균 속도)')
    print(f'🤖 신경망 대리 모델 (순수 행렬곱): {time_pinn_batch:.6f} 초')
    print(f'📐 원본 수식 모델 (60구간 수치 적분): {time_orig_batch:.6f} 초')
    
    if speedup_ratio > 1:
        print(f'🚀 결과: 신경망 대리 모델이 ** {speedup_ratio:.2f}배 ** 더 빠릅니다!')
    else:
        print(f'⚠️ 결과: 원본 수식 모델이 ** {1/speedup_ratio:.2f}배 ** 더 빠릅니다.')
    print('===================================================================')

if __name__ == "__main__":
    main()