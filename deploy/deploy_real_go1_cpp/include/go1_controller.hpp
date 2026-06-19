#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "config.hpp"
#include "onnx_policy.hpp"

#include "ucl/low_cmd.hpp"
#include "ucl/low_state.hpp"
#include "ucl/unitree_connection.hpp"

namespace go1_deploy {

enum class ControllerState {
    Passive,
    MoveToDefault,
    DefaultStand,
    PolicyRun,
    ExitDamping,
};

struct RemoteState {
    std::array<int, 16> buttons{};
    std::array<int, 16> last_buttons{};
    float lx = 0.0f;
    float ly = 0.0f;
    float rx = 0.0f;
    float ry = 0.0f;
};

struct RuntimeOptions {
    std::array<float, 3> command{0.0f, 0.0f, 0.0f};
    std::string connection_override;
    bool dry_run = false;
    bool print_state = false;
    bool print_remote = false;
};

class Go1Controller {
public:
    Go1Controller(DeployConfig config, RuntimeOptions options);
    ~Go1Controller();

    void run();
    void dry_run();
    void request_stop();

private:
    ucl::ConnectionSettings connection_settings() const;
    bool receive_latest_state();
    bool wait_for_low_state();
    void send_cmd();
    void send_damping();
    void validate_timing() const;
    void print_keyboard_help() const;
    void handle_keyboard(char key);
    void update_remote();
    void handle_remote();
    void load_policies();
    void switch_policy(const std::string& name);
    void run_policy_once();
    void transition_to(ControllerState next);
    void tick_passive();
    void tick_move_to_default();
    void tick_default_stand();
    void tick_policy_run();
    void tick_exit_damping();
    std::array<float, 12> current_joint_positions() const;
    void fill_servo_cmd(const std::array<float, 12>& q, const std::array<float, 12>& kp, const std::array<float, 12>& kd);
    std::vector<float> build_observation() const;
    void print_current_state() const;
    void print_remote_bytes() const;

    DeployConfig config_;
    RuntimeOptions options_;
    std::map<std::string, std::unique_ptr<OnnxPolicy>> policies_;
    OnnxPolicy* active_policy_ = nullptr;
    std::string active_policy_name_;
    std::unique_ptr<ucl::unitreeConnection> conn_;
    ucl::lowCmd low_cmd_;
    ucl::lowState low_state_;
    RemoteState remote_;
    std::array<float, 3> command_{};
    std::array<float, 12> last_action_{};
    std::array<float, 12> move_start_q_{};
    std::array<float, 12> policy_target_q_{};
    ControllerState state_ = ControllerState::Passive;
    std::atomic<bool> stop_requested_{false};
    std::chrono::steady_clock::time_point last_state_time_{};
    std::chrono::steady_clock::time_point state_enter_time_{};
    std::chrono::steady_clock::time_point next_policy_time_{};
    std::chrono::steady_clock::time_point next_debug_print_time_{};
    int exit_damping_ticks_ = 0;
};

} // namespace go1_deploy
