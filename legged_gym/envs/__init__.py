from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go1.go1_env import Go1Robot
from legged_gym.envs.go1.go1_config import (
    GO1Cfg,
    GO1CfgACMoECTS,
    GO1CfgCTS,
    GO1CfgDualMoECTS,
    GO1CfgMCPCTS,
    GO1CfgMoECTS,
    GO1CfgMoENGCTS,
    GO1CfgPPO,
)
from legged_gym.envs.go2.go2_env import Go2Robot
from legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgPPO, GO2CfgCTS, GO2CfgMoECTS, GO2CfgMoENGCTS, GO2CfgMCPCTS, GO2CfgACMoECTS, GO2CfgDualMoECTS
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register("go1", Go1Robot, GO1Cfg(), GO1CfgPPO())
task_registry.register("go1_cts", Go1Robot, GO1Cfg(), GO1CfgCTS())
task_registry.register("go1_moe_cts", Go1Robot, GO1Cfg(), GO1CfgMoECTS())
task_registry.register("go1_moe_ng_cts", Go1Robot, GO1Cfg(), GO1CfgMoENGCTS())
task_registry.register("go1_mcp_cts", Go1Robot, GO1Cfg(), GO1CfgMCPCTS())
task_registry.register("go1_ac_moe_cts", Go1Robot, GO1Cfg(), GO1CfgACMoECTS())
task_registry.register("go1_dual_moe_cts", Go1Robot, GO1Cfg(), GO1CfgDualMoECTS())

task_registry.register("go2", Go2Robot, GO2Cfg(), GO2CfgPPO())
task_registry.register("go2_cts", Go2Robot, GO2Cfg(), GO2CfgCTS())
task_registry.register("go2_moe_cts", Go2Robot, GO2Cfg(), GO2CfgMoECTS())
task_registry.register("go2_moe_ng_cts", Go2Robot, GO2Cfg(), GO2CfgMoENGCTS())
task_registry.register("go2_mcp_cts", Go2Robot, GO2Cfg(), GO2CfgMCPCTS())
task_registry.register("go2_ac_moe_cts", Go2Robot, GO2Cfg(), GO2CfgACMoECTS())
task_registry.register("go2_dual_moe_cts", Go2Robot, GO2Cfg(), GO2CfgDualMoECTS())
