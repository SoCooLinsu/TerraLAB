import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import qmc
import numpy as np
import math

# ---------------------------------------------------------
# 1. 대리 모델 아키텍처
# ---------------------------------------------------------
class TerraSurrogateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        return self.net(x)

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
def generate_dataset(num_samples=1000000, device='cuda'):
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

# ---------------------------------------------------------
# 4. Isaac Sim 전용 JIT 래퍼 클래스
# ---------------------------------------------------------
class TerraIsaacWrapper(nn.Module):
    def __init__(self, core_model, y_mean, y_std):
        super().__init__()
        self.core_model = core_model
        # JIT 컴파일 시 파라미터가 모델 안에 완전히 박제(Bake)됩니다.
        self.register_buffer('y_mean', y_mean)
        self.register_buffer('y_std', y_std)
        
    def forward(self, x_norm):
        # x_norm: [s, h_norm, vc_norm, alpha_norm]
        y_norm = self.core_model(x_norm)
        # 역정규화 진행 -> 즉시 실제 물리량(Newton) 반환
        return y_norm * self.y_std + self.y_mean

# ---------------------------------------------------------
# 5. 메인 학습 및 JIT 익스포트
# ---------------------------------------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    X, Y_norm, Y_mean, Y_std = generate_dataset(num_samples=700000, device=device)
    
    split_idx = int(len(X) * 0.9)
    train_dataset = TensorDataset(X[:split_idx], Y_norm[:split_idx])
    val_dataset = TensorDataset(X[split_idx:], Y_norm[split_idx:])
    
    train_loader = DataLoader(train_dataset, batch_size=8192, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8192, shuffle=False)
    
    model = TerraSurrogateNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.5)
    criterion = nn.L1Loss()
    
    epochs = 5000
    target_loss = 1e-5
    
    print("🚀 대리 모델 학습 시작!")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_preds = model(val_x)
                val_loss += criterion(val_preds, val_y).item()
                
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_train_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_core_model.pth")
            
            # 래퍼 모델에 현재 메모리에 있는 최고 상태의 model을 바로 탑재
            inference_model = TerraIsaacWrapper(model, Y_mean, Y_std).to(device)
            inference_model.eval() # JIT 컴파일을 위해 평가 모드로 전환
            
            # TorchScript JIT 컴파일 및 저장
            jit_model = torch.jit.script(inference_model)
            jit_model.save("terra_surrogate_isaac.pt")
            
            # 🚨 [매우 중요] 다음 에포크의 정상적인 학습을 위해 코어 모델을 다시 학습 모드로 복구
            model.train()
            
        if (epoch + 1) % 1 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.9f} | Val Loss: {avg_val_loss:.9f} | LR: {optimizer.param_groups[0]['lr']:.12f}")

        if avg_val_loss <= target_loss:
            print(f"\n🎯 [조기 종료] 목표 검증 오차 도달! (Val Loss: {avg_val_loss:.9f})")
            break
            
    print("✅ 모델 학습 완료!")