import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# 로버의 구체적인 설정 정의
ROVER_CONFIG = ArticulationCfg(
    # 1. USD 파일 경로 설정 (파일이 있는 경로로 수정하세요)
    spawn=sim_utils.UsdFileCfg(
        usd_path= r"C:\TerraLab\source\TerraLab\TerraLab\robots\Rover.usd", # 실제 Rover_usd.usd 경로를 입력하세요
        activate_contact_sensors=True),
    
    # 2. 액추에이터(모터) 설정
    actuators={
        # 바퀴 조인트 제어 (정규표현식으로 Wheel이 들어간 조인트만 선택)
        "wheel_acts": ImplicitActuatorCfg(
            joint_names_expr=["Wheel_joint_.*"], 
            stiffness=0.0,
            damping=0.0,
            velocity_limit_sim=4.8171,   #rad/s
            effort_limit_sim=0.3432,  #Nm
        ),
        "passive_rocker": ImplicitActuatorCfg(
            joint_names_expr=["Rocker_joint_.*"],
            stiffness=0.0,
            damping=None,
        ),
    },
)