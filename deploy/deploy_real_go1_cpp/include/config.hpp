#pragma once

#include <array>
#include <filesystem>
#include <map>
#include <string>

namespace go1_deploy {

struct DeployConfig {
    std::filesystem::path root_dir;
    std::string initial_policy = "up";
    std::map<std::string, std::string> policies;
    double fsm_dt = 0.002;
    double control_dt = 0.02;
    std::string sdk_mode = "low";
    std::string connection = "low_wifi";
    std::array<int, 12> joint2motor_idx{};
    std::array<float, 12> default_angles{};
    std::array<float, 12> kps{};
    std::array<float, 12> kds{};
    std::array<float, 12> stand_kps{};
    std::array<float, 12> stand_kds{};
    float obs_scales_ang_vel = 0.25f;
    float obs_scales_dof_pos = 1.0f;
    float obs_scales_dof_vel = 0.05f;
    std::array<float, 3> command_scale{2.0f, 2.0f, 0.25f};
    float action_scale = 0.25f;
    int num_obs = 45;
    int num_actions = 12;
    double state_timeout_s = 0.1;
    double move_to_default_s = 2.0;
    bool remote_enabled = true;
    bool remote_cmd_enabled = true;
    float remote_deadzone = 0.05f;
};

DeployConfig load_config(const std::filesystem::path& config_path);

} // namespace go1_deploy
