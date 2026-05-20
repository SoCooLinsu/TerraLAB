import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import itertools
import time
from scipy.stats import qmc
import numpy as np
import math
import time
import math
import os

# ========================================================
# 1. 다항식 피처 생성 네트워크 (Polynomial Regressor)
# ========================================================
class PolynomialRegressor(nn.Module):
    def __init__(self, num_inputs, num_outputs, degree=3):
        super(PolynomialRegressor, self).__init__()
        self.degree = degree
        
        # 주어진 차수(degree)까지의 모든 조합 생성 (중복 조합)
        self.combos = []
        for d in range(1, degree + 1):
            self.combos.extend(list(itertools.combinations_with_replacement(range(num_inputs), d)))
        
        self.num_features = len(self.combos)
        print(f"🔹 다항식 차수: {degree} | 확장된 입력 피처 개수: {self.num_features}개")
        
        # 선형 회귀 레이어 (다항식 피처 -> 출력)
        # bias=True를 통해 상수항(y절편)을 학습합니다.
        self.linear = nn.Linear(self.num_features, num_outputs, bias=True)

    def forward(self, x):
        # x shape: (Batch, num_inputs)
        poly_features = []
        
        # 배치 데이터에 대해 다항식 피처를 실시간 연산
        for combo in self.combos:
            # 선택된 인덱스들의 값을 곱함 (예: x_0 * x_1)
            feat = torch.prod(x[:, combo], dim=1, keepdim=True)
            poly_features.append(feat)
            
        # 연산된 피처들을 가로로 병합
        x_poly = torch.cat(poly_features, dim=1)
        
        # 선형 결합 (W * x_poly + b)
        return self.linear(x_poly)
    
    # ========================================================
# 📊 [헬퍼 함수] R2 스코어 계산 함수 (PyTorch Tensor용)
# ========================================================
def calculate_r2(y_true, y_pred):
    # 각 출력 차원별로 R2 스코어 계산
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
    ss_tot = torch.sum((y_true - torch.mean(y_true, dim=0)) ** 2, dim=0)
    
    # 0으로 나누는 것 방지
    ss_tot_safe = torch.where(ss_tot == 0, torch.tensor(1e-8, device=y_true.device), ss_tot)
    r2 = 1.0 - (ss_res / ss_tot_safe)
    return r2

# ========================================================
# ⏱️ [벤치마크 함수] 다항함수 모델 vs 원본 수치적분 모델
# ========================================================
def benchmark_speed(model, teacher, device='cuda', num_batch=16384, iterations=100):
    print(f"\n🚀 [속도 벤치마크] {num_batch}개 로버 바퀴 동시 연산 ({iterations}회 반복 평균) 측정 중...")

    model.eval() 
    
    # 임의의 테스트 데이터 생성
    s = torch.rand(num_batch, device=device)
    h = torch.rand(num_batch, device=device) * 0.035  # 튜닝된 범위
    v_c = torch.rand(num_batch, device=device) * 0.2
    alpha = (torch.rand(num_batch, device=device) * 2 - 1) * (math.pi / 2.0)
    
    # 다항함수 모델 입력용 텐서 조합 (정규화 가정)
    X_batch = torch.stack([s, h/0.035, v_c/0.2, alpha/(math.pi/2.0)], dim=1)
    
    # ========================================================
    # 🌟 [매우 중요] GPU 예열 (Warm-up)
    # ========================================================
    print("🔥 GPU 예열 중...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(X_batch)
            _ = teacher.wong_reece_model_torch(s, h, v_c, alpha)
    if device == 'cuda':
        torch.cuda.synchronize()

    # ========================================================
    # [1] 🤖 다항함수 대리 모델 (Polynomial Surrogate) 시간 측정
    # ========================================================
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(X_batch)
    if device == 'cuda':
        torch.cuda.synchronize() 
    time_pinn_batch = (time.perf_counter() - start_time) / iterations

    # ========================================================
    # [2] 📐 원본 전산 모델 (Physics Teacher) 시간 측정
    # ========================================================
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = teacher.wong_reece_model_torch(s, h, v_c, alpha)
    if device == 'cuda':
        torch.cuda.synchronize()
    time_orig_batch = (time.perf_counter() - start_time) / iterations

    # ========================================================
    # [3] 결과 출력
    # ========================================================
    speedup_ratio = time_orig_batch / time_pinn_batch
    
    print('===================================================================')
    print(f'⏱️ {num_batch}개 데이터 동시 연산 (1회 평균 속도)')
    print(f'🤖 다항함수 대리 모델 (순수 행렬곱): {time_pinn_batch:.6f} 초')
    print(f'📐 원본 수식 모델 (60구간 수치 적분): {time_orig_batch:.6f} 초')
    
    if speedup_ratio > 1:
        print(f'🚀 결과: 다항함수 대리 모델이 ** {speedup_ratio:.2f}배 ** 더 빠릅니다!')
    else:
        print(f'⚠️ 결과: 원본 수식 모델이 ** {1/speedup_ratio:.2f}배 ** 더 빠릅니다.')
    print('===================================================================')

# ---------------------------------------------------------
# 2. 물리 엔진 정답지 (Teacher)
# ---------------------------------------------------------
class WongReeceTeacher:
    def __init__(self, device='cuda'):
        self.device = device
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
        Xc = (math.pi / 4.0) - (self.p['phi_soil'] / 2.0)
        def cot(x): return 1.0 / math.tan(x)
        self.C1 = (cot(Xc) + math.tan(Xc + self.p['phi_soil'])) / (1.0 - math.tan(self.p['alpha_b']) * math.tan(Xc + self.p['phi_soil']))
        self.C2 = (cot(Xc) - math.tan(self.p['alpha_b'])) + ((cot(Xc) - math.tan(self.p['alpha_b']))**2) / (math.tan(self.p['alpha_b']) + cot(self.p['phi_soil']))
    
    def wong_reece_model_torch(self, s: torch.Tensor, h: torch.Tensor, v_c: torch.Tensor, alpha: torch.Tensor):

        # 벡터화된 Wong-Reece 테라메카닉스 모델 (PyTorch)
        # s: Slip ratio 텐서 (Shape: [N]) [ 0 ~ 1 ]
        # h: 침하량 텐서 (Shape: [N]) [ 0 ~ R/2 ]
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
        theta1 = torch.acos(1.0 - h / Rw)
        theta_m = (c1 + c2 * s) * theta1

        A_c = b * Rw * theta1 
        A_c_safe = torch.clamp(A_c, min=1e-6) # 0으로 나누기 에러 방지
        p_damping = (c_s * v_c) / A_c_safe
        
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
        p_damping_exp = p_damping.unsqueeze(1)
        
        # 2. 구간 마스크 생성
        idx_F = theta >= theta_m_exp
        idx_R = theta < theta_m_exp
        
        # Normal stress
        # (Front 구간)
        cos_diff_F = torch.cos(theta) - torch.cos(theta1_exp)
        p_F = k_eq * (torch.clamp(Rw * cos_diff_F, min=0.0) ** n) + p_damping_exp
        
        # (Rear 구간) - theta_m이 0일 때 0으로 나누기 에러 방지
        theta_m_safe = torch.where(theta_m_exp == 0, torch.tensor(1e-6, device=device), theta_m_exp)
        theta_eq = theta1_exp - (theta / theta_m_safe) * (theta1_exp - theta_m_exp)
        cos_diff_R = torch.cos(theta_eq) - torch.cos(theta1_exp)
        p_R = k_eq * (torch.clamp(Rw * cos_diff_R, min=0.0) ** n) + p_damping_exp

        # 마스크 적용
        p_theta = torch.zeros_like(theta)
        p_theta = torch.where(idx_F, p_F, p_theta)
        p_theta = torch.where(idx_R, p_R, p_theta)
        
        # Shear stress
        tau_max = torch.minimum(mu_s * p_theta, c_soil + p_theta * math.tan(phi_soil))
        j_x = Rw * ((theta1_exp - theta) - (1 - s_exp) * (torch.sin(theta1_exp) - torch.sin(theta)))
        j_y = Rw * (1 - s_exp) * (theta1_exp - theta) * torch.tan(torch.abs(alpha_exp))
        tau_x = tau_max * (1 - torch.exp(-j_x / Kx))
        tau_y = tau_max * (1 - torch.exp(-j_y / Ky))
        
        # Bulldozing Resistance
        h_theta = torch.clamp(Rw * (torch.cos(theta) - torch.cos(theta1_exp)), min=0.0)
        Rb_theta = self.C1 * (h_theta * c_soil + 0.5 * rho_s * (h_theta ** 2) * self.C2)
        
        # 5. 적분 (PyTorch의 trapezoid 함수 사용)
        # y값 먼저 넣고, x값(theta) 넣고, 어느 축(dim=1)으로 적분할지 지정
        Fn_mag        = Rw * b * torch.trapezoid(p_theta * torch.cos(theta) + tau_x * torch.sin(theta), x=theta, dim=1)
        Ft_normal_mag = Rw * b * torch.trapezoid(p_theta * torch.sin(theta), x=theta, dim=1)
        Ft_shear_mag  = Rw * b * torch.trapezoid(tau_x * torch.cos(theta), x=theta, dim=1)
        Fy_mag        = Rw * b * torch.trapezoid(tau_y, x=theta, dim=1) + torch.trapezoid(Rb_theta * Rw * torch.cos(theta), x=theta, dim=1) * torch.abs(torch.sin(alpha))
        T_mag         = (Rw ** 2) * b * torch.trapezoid(tau_x, x=theta, dim=1)
        
        return Fn_mag, Ft_normal_mag, Ft_shear_mag, Fy_mag, T_mag

# ---------------------------------------------------------
# 3. 데이터셋 생성 (LHS + -pi/2 ~ pi/2)
# ---------------------------------------------------------
def generate_dataset(num_samples=500000, device='cuda'):
    print(f"[{num_samples}개] 라틴 하이퍼큐브 샘플링(LHS) 데이터 생성 중...")
    teacher = WongReeceTeacher(device)

    num_boundary = int(num_samples * 0.1) 
    num_main = num_samples - (num_boundary * 3)
    eps = 1e-6
    max_h = 0.035
    max_alpha = (math.pi / 2.0)     # 90도 (약 1.57 rad)
    n = 2.0                     # 0 근처 데이터 집중도 (클수록 0에 몰림)

    # 🛠️ [헬퍼 함수] 텐서 정규화 (모든 입력을 완벽한 0.0 ~ 1.0 범위로 매핑)
    def get_norm_X(s, h, vc, a):
        return torch.stack([
            s, 
            (h - eps) / (max_h - eps), 
            vc / 2.0, 
            a / (max_alpha - eps)
        ], dim=1)

# ========================================================
    # 🌟 1. Main Cloud (비선형 집중 샘플링)
    # ========================================================
    sampler_main = qmc.LatinHypercube(d=4)
    lhs_main = torch.tensor(sampler_main.random(n=num_main), dtype=torch.float32, device=device)
    
    s_main = lhs_main[:, 0]                                
    h_main = eps + (lhs_main[:, 1] ** n) * (max_h - eps)          
    vc_main = (lhs_main[:, 2] ** n) * 2.0                           
    alpha_main = lhs_main[:, 3] * (max_alpha - eps)
    
    X_main = get_norm_X(s_main, h_main, vc_main, alpha_main)
    with torch.no_grad():
        Y_main = torch.stack(teacher.wong_reece_model_torch(s_main, h_main, vc_main, alpha_main), dim=1)
    
    print(f"   ✔️ 메인 데이터셋 생성 완료 ({num_main}개)")

    # ========================================================
    # 🌟 2. Boundary 1: 바닥 경계 (Sinkage h = eps)
    # ========================================================
    sampler_b1 = qmc.LatinHypercube(d=3)
    lhs_b1 = torch.tensor(sampler_b1.random(n=num_boundary), dtype=torch.float32, device=device)
    
    s_b1 = lhs_b1[:, 0]
    h_b1 = torch.full((num_boundary,), eps, device=device)
    vc_b1 = (lhs_b1[:, 1] ** n) * 2.0
    alpha_b1 = lhs_b1[:, 2] * (max_alpha - eps)
    
    X_b1 = get_norm_X(s_b1, h_b1, vc_b1, alpha_b1)
    with torch.no_grad():
        Y_b1 = torch.stack(teacher.wong_reece_model_torch(s_b1, h_b1, vc_b1, alpha_b1), dim=1)

    # ========================================================
    # 🌟 3. Boundary 2: 직진 경계 (Slip Angle alpha = eps)
    # ========================================================
    sampler_b2 = qmc.LatinHypercube(d=3)
    lhs_b2 = torch.tensor(sampler_b2.random(n=num_boundary), dtype=torch.float32, device=device)
    
    s_b2 = lhs_b2[:, 0]
    h_b2 = eps + (lhs_b2[:, 1] ** n) * (max_h - eps)
    vc_b2 = (lhs_b2[:, 2] ** n) * 2.0
    alpha_b2 = torch.full((num_boundary,), 0, device=device)
    
    X_b2 = get_norm_X(s_b2, h_b2, vc_b2, alpha_b2)
    with torch.no_grad():
        Y_b2 = torch.stack(teacher.wong_reece_model_torch(s_b2, h_b2, vc_b2, alpha_b2), dim=1)

    # ========================================================
    # 🌟 4. Boundary 3: 횡슬립 한계 (Slip Angle alpha = max_alpha)
    # ========================================================
    sampler_b3 = qmc.LatinHypercube(d=3)
    lhs_b3 = torch.tensor(sampler_b3.random(n=num_boundary), dtype=torch.float32, device=device)
    
    s_b3 = lhs_b3[:, 0]
    h_b3 = eps + (lhs_b3[:, 1] ** n) * (max_h - eps)
    vc_b3 = (lhs_b3[:, 2] ** n) * 2.0
    alpha_b3 = torch.full((num_boundary,), max_alpha - eps, device=device)
    
    X_b3 = get_norm_X(s_b3, h_b3, vc_b3, alpha_b3)
    with torch.no_grad():
        Y_b3 = torch.stack(teacher.wong_reece_model_torch(s_b3, h_b3, vc_b3, alpha_b3), dim=1)

    print(f"   ✔️ 3대 극단 경계(Boundary) 데이터 생성 완료 ({num_boundary * 3}개)")

    # ========================================================
    # 🌟 5. 병합(Concat) 및 데이터 셔플링
    # ========================================================
    X = torch.cat([X_main, X_b1, X_b2, X_b3], dim=0)
    Y = torch.cat([Y_main, Y_b1, Y_b2, Y_b3], dim=0)
    
    indices = torch.randperm(X.size(0), device=device)
    X = X[indices]
    Y = Y[indices]

    # ========================================================
    # 🌟 6. [핵심] Y값 Z-Score 정규화 (Standardization)
    # ========================================================
    print("⚖️ 타겟 변수(Y) 통계량 추출 및 Z-Score 정규화 수행 중...")
    Y_mean = Y.mean(dim=0)
    Y_std = Y.std(dim=0)
    
    # 만약 std가 0인 구간이 있을 경우(발생 확률은 극히 낮음) 0으로 나누기 방지
    Y_std_safe = torch.where(Y_std < 1e-8, torch.tensor(1.0, device=device), Y_std)
    
    Y_norm = (Y - Y_mean) / Y_std_safe

    print('===================================================================')
    print(f"🎉 총 {X.shape[0]}개의 데이터셋이 완성되었습니다!")
    print(f"   ✔️ X_shape: {X.shape}, Y_norm_shape: {Y_norm.shape}")
    print(f"   ✔️ Y_mean: {Y_mean.cpu().numpy()}")
    print(f"   ✔️ Y_std:  {Y_std.cpu().numpy()}")
    print('===================================================================')
    
    # 🌟 질문자님의 파이프라인과 완벽하게 호환되는 출력 형태!
    return X, Y_norm, Y_mean, Y_std

# ========================================================
# 2. 학습 파이프라인 함수
# ========================================================
def fit_polynomial(X, Y_norm, Y_mean, Y_std, degree=3, epochs=100, batch_size=8192, lr=0.01):
    device = X.device
    num_inputs = X.shape[1]   # 4차원 (s, h, vc, alpha)
    num_outputs = Y_norm.shape[1] # 타겟 변수 개수
    
    # 모델 초기화
    model = PolynomialRegressor(num_inputs, num_outputs, degree=degree).to(device)
    
    # 데이터 로더 설정 (미니 배치)
    dataset = TensorDataset(X, Y_norm)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 손실 함수 및 옵티마이저 (Adam 사용 시 수렴이 빠름)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("\n🚀 다항함수 피팅을 시작합니다...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_X, batch_Y in dataloader:
            optimizer.zero_grad()
            
            # 예측 및 손실 계산
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y)
            
            # 역전파 및 가중치 업데이트
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_X.size(0)
            
        avg_loss = total_loss / len(dataset)
        
        # 10 에포크마다 진행 상황 출력
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss (MSE): {avg_loss:.6f}")

    print(f"✅ 학습 완료! (소요 시간: {time.time() - start_time:.2f}초)")
    
    return model

# ========================================================
# 3. 실제 사용 예시 (질문자님의 코드 뒷부분에 추가)
# ========================================================
if __name__ == "__main__":
    # 데이터 생성 함수 호출 (이미 작성하신 함수)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X, Y_norm, Y_mean, Y_std = generate_dataset()
    
    # 다항식 차수를 3차(degree=3)로 설정하고 피팅 진행
    model = fit_polynomial(X, Y_norm, Y_mean, Y_std, degree=3, epochs=50, batch_size=8192, lr=0.01)
    
    # ========================================================
    # 3. 🎯 검증 (R2 스코어 및 오차 확인)
    # ========================================================
    print("\n🔍 전체 데이터셋에 대한 모델 성능 검증 중...")
    model.eval()
    with torch.no_grad():
        # 메모리 초과를 방지하기 위해 전체 데이터 중 10만 개 정도만 무작위 샘플링하여 평가
        eval_size = min(100000, X.shape[0])
        eval_indices = torch.randperm(X.shape[0])[:eval_size]
        
        test_X = X[eval_indices]
        actual_Y_norm = Y_norm[eval_indices]
        
        # 모델 예측
        pred_Y_norm = model(test_X)
        
        # 원래 스케일 복원
        pred_Y = (pred_Y_norm * Y_std) + Y_mean
        actual_Y = (actual_Y_norm * Y_std) + Y_mean
        
        # R2 Score 계산
        r2_scores = calculate_r2(actual_Y, pred_Y)
        
        # RMSE 계산
        rmse_scores = torch.sqrt(torch.mean((actual_Y - pred_Y) ** 2, dim=0))
        
        print('===================================================================')
        print("🎯 [모델 피팅 정확도 결과]")
        for i in range(actual_Y.shape[1]):
            print(f"   ✔️ 타겟 {i+1} | R² 스코어: {r2_scores[i].item():.4f} | RMSE: {rmse_scores[i].item():.6f}")
        print('===================================================================')

    # ========================================================
    # 4. 🚀 속도 벤치마크 실행
    # ========================================================

    teacher = WongReeceTeacher(device)
    benchmark_speed(model, teacher, device=device, num_batch=16384, iterations=100)