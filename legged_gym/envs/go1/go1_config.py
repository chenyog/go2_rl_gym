from legged_gym.envs.go2.go2_config import (
    GO2Cfg,
    GO2CfgACMoECTS,
    GO2CfgCTS,
    GO2CfgDualMoECTS,
    GO2CfgMCPCTS,
    GO2CfgMoECTS,
    GO2CfgMoENGCTS,
    GO2CfgPPO,
)


class GO1Cfg(GO2Cfg):
    class init_state(GO2Cfg.init_state):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = {
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }

    class env(GO2Cfg.env):
        num_actions = 12
        num_observations = 45
        num_privileged_obs = 45 + 3 + 4 + 12 + 12 + 187

    class control(GO2Cfg.control):
        stiffness = {"joint": 20.0}
        damping = {"joint": 0.5}
        action_scale = 0.25

    class rewards(GO2Cfg.rewards):
        base_height_target = 0.38

    class asset(GO2Cfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf"
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter


class GO1CfgPPO(GO2CfgPPO):
    class runner(GO2CfgPPO.runner):
        experiment_name = "go1_ppo"


class GO1CfgCTS(GO2CfgCTS):
    class runner(GO2CfgCTS.runner):
        experiment_name = "go1_cts"


class GO1CfgMoENGCTS(GO2CfgMoENGCTS):
    class runner(GO2CfgMoENGCTS.runner):
        experiment_name = "go1_moe_no_goal_cts"


class GO1CfgMCPCTS(GO2CfgMCPCTS):
    class runner(GO2CfgMCPCTS.runner):
        experiment_name = "go1_mcp_cts"


class GO1CfgACMoECTS(GO2CfgACMoECTS):
    class runner(GO2CfgACMoECTS.runner):
        experiment_name = "go1_ac_moe_cts"


class GO1CfgDualMoECTS(GO2CfgDualMoECTS):
    class runner(GO2CfgDualMoECTS.runner):
        experiment_name = "go1_dual_moe_cts"


class GO1CfgMoECTS(GO2CfgMoECTS):
    class runner(GO2CfgMoECTS.runner):
        experiment_name = "go1_moe_cts"
