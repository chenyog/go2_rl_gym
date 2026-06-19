#include "go1_controller.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <utility>

#include <sys/select.h>
#include <termios.h>
#include <unistd.h>

#include <yaml-cpp/yaml.h>

#include "math_utils.hpp"
#include "ucl/enums.hpp"

namespace go1_deploy {
namespace {
constexpr std::array<const char*, 4> kPolicyNames = {"up", "down", "left", "right"};
constexpr int kStart = 2;
constexpr int kSelect = 3;
constexpr int kL2 = 5;
constexpr int kA = 8;
constexpr int kB = 9;
constexpr int kUp = 12;
constexpr int kRight = 13;
constexpr int kDown = 14;
constexpr int kLeft = 15;

class NonBlockingKeyboard {
public:
    NonBlockingKeyboard() {
        enabled_ = ::isatty(STDIN_FILENO) == 1;
        if (!enabled_) {
            return;
        }
        if (::tcgetattr(STDIN_FILENO, &old_term_) != 0) {
            enabled_ = false;
            return;
        }
        auto raw = old_term_;
        raw.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        if (::tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) {
            enabled_ = false;
        }
    }

    ~NonBlockingKeyboard() {
        if (enabled_) {
            ::tcsetattr(STDIN_FILENO, TCSANOW, &old_term_);
        }
    }

    char poll() const {
        if (!enabled_) {
            return '\0';
        }
        fd_set set;
        FD_ZERO(&set);
        FD_SET(STDIN_FILENO, &set);
        timeval timeout{};
        const int ready = ::select(STDIN_FILENO + 1, &set, nullptr, nullptr, &timeout);
        if (ready <= 0 || !FD_ISSET(STDIN_FILENO, &set)) {
            return '\0';
        }
        char key = '\0';
        const auto nread = ::read(STDIN_FILENO, &key, 1);
        if (nread == 1) {
            return key;
        }
        return '\0';
    }

private:
    bool enabled_ = false;
    termios old_term_{};
};

template <typename T, std::size_t N>
std::array<T, N> read_array(const YAML::Node& node, const std::string& key) {
    if (!node[key] || !node[key].IsSequence() || node[key].size() != N) {
        throw std::runtime_error("Expected " + key + " to be a sequence of length " + std::to_string(N));
    }
    std::array<T, N> out{};
    for (std::size_t i = 0; i < N; ++i) {
        out[i] = node[key][i].as<T>();
    }
    return out;
}

std::filesystem::path infer_root_dir(const std::filesystem::path& config_path) {
    auto abs = std::filesystem::absolute(config_path);
    auto path = abs.parent_path();
    for (int i = 0; i < 3; ++i) {
        path = path.parent_path();
    }
    return path;
}

std::string replace_root(std::string value, const std::filesystem::path& root) {
    const std::string token = "{LEGGED_GYM_ROOT_DIR}";
    auto pos = value.find(token);
    if (pos != std::string::npos) {
        value.replace(pos, token.size(), root.string());
    }
    return value;
}

float remote_float(const std::array<std::uint8_t, 40>& data, std::size_t offset) {
    float value = 0.0f;
    std::memcpy(&value, data.data() + offset, sizeof(float));
    return value;
}

float dz(float value, float deadzone) {
    return std::abs(value) < deadzone ? 0.0f : value;
}

std::array<float, 12> joint_lower_limits() {
    return {-0.863f, -0.686f, -2.818f, -0.863f, -0.686f, -2.818f,
            -0.863f, -0.686f, -2.818f, -0.863f, -0.686f, -2.818f};
}

std::array<float, 12> joint_upper_limits() {
    return {0.863f, 4.501f, -0.888f, 0.863f, 4.501f, -0.888f,
            0.863f, 4.501f, -0.888f, 0.863f, 4.501f, -0.888f};
}

const char* state_name(ControllerState state) {
    switch (state) {
    case ControllerState::Passive:
        return "Passive/Damping";
    case ControllerState::MoveToDefault:
        return "MoveToDefault";
    case ControllerState::DefaultStand:
        return "DefaultStand";
    case ControllerState::PolicyRun:
        return "PolicyRun";
    case ControllerState::ExitDamping:
        return "ExitDamping";
    }
    return "Unknown";
}
} // namespace

DeployConfig load_config(const std::filesystem::path& config_path) {
    YAML::Node yaml = YAML::LoadFile(config_path.string());
    DeployConfig cfg;
    cfg.root_dir = infer_root_dir(config_path);
    if (!yaml["initial_policy"] || !yaml["policies"] || !yaml["policies"].IsMap()) {
        throw std::runtime_error("Go1 deploy config requires initial_policy and policies map.");
    }
    cfg.initial_policy = yaml["initial_policy"].as<std::string>();
    for (const auto& item : yaml["policies"]) {
        cfg.policies[item.first.as<std::string>()] = replace_root(item.second.as<std::string>(), cfg.root_dir);
    }
    for (const auto* name : kPolicyNames) {
        if (!cfg.policies.count(name)) {
            throw std::runtime_error("Go1 deploy config missing policy slot: " + std::string(name));
        }
    }
    if (!cfg.policies.count(cfg.initial_policy)) {
        throw std::runtime_error("initial_policy is not listed in policies: " + cfg.initial_policy);
    }
    cfg.fsm_dt = yaml["fsm_dt"] ? yaml["fsm_dt"].as<double>() : cfg.fsm_dt;
    cfg.control_dt = yaml["control_dt"].as<double>();
    cfg.sdk_mode = yaml["sdk_mode"].as<std::string>();
    cfg.connection = yaml["connection"].as<std::string>();
    cfg.joint2motor_idx = read_array<int, 12>(yaml, "joint2motor_idx");
    cfg.default_angles = read_array<float, 12>(yaml, "default_angles");
    cfg.kps = read_array<float, 12>(yaml, "kps");
    cfg.kds = read_array<float, 12>(yaml, "kds");
    cfg.stand_kps = read_array<float, 12>(yaml, "stand_kps");
    cfg.stand_kds = read_array<float, 12>(yaml, "stand_kds");
    cfg.obs_scales_ang_vel = yaml["obs_scales_ang_vel"].as<float>();
    cfg.obs_scales_dof_pos = yaml["obs_scales_dof_pos"].as<float>();
    cfg.obs_scales_dof_vel = yaml["obs_scales_dof_vel"].as<float>();
    cfg.command_scale = read_array<float, 3>(yaml, "command_scale");
    cfg.action_scale = yaml["action_scale"].as<float>();
    cfg.num_obs = yaml["num_obs"].as<int>();
    cfg.num_actions = yaml["num_actions"].as<int>();
    cfg.state_timeout_s = yaml["state_timeout_s"].as<double>();
    cfg.move_to_default_s = yaml["move_to_default_s"].as<double>();
    cfg.remote_enabled = yaml["remote_enabled"] ? yaml["remote_enabled"].as<bool>() : cfg.remote_enabled;
    cfg.remote_cmd_enabled = yaml["remote_cmd_enabled"] ? yaml["remote_cmd_enabled"].as<bool>() : cfg.remote_cmd_enabled;
    cfg.remote_deadzone = yaml["remote_deadzone"] ? yaml["remote_deadzone"].as<float>() : cfg.remote_deadzone;
    return cfg;
}

Go1Controller::Go1Controller(DeployConfig config, RuntimeOptions options)
    : config_(std::move(config)), options_(std::move(options)), command_(options_.command) {
    load_policies();
    switch_policy(config_.initial_policy);
}

Go1Controller::~Go1Controller() {
    request_stop();
    if (conn_) {
        send_damping();
        conn_->stopRecv();
    }
}

void Go1Controller::request_stop() {
    stop_requested_.store(true);
}

void Go1Controller::validate_timing() const {
    if (config_.fsm_dt <= 0.0) {
        throw std::runtime_error("fsm_dt must be positive.");
    }
    if (config_.control_dt <= 0.0) {
        throw std::runtime_error("control_dt must be positive.");
    }
    const double ratio = config_.control_dt / config_.fsm_dt;
    const double rounded = std::round(ratio);
    if (std::abs(ratio - rounded) > 1e-6) {
        throw std::runtime_error("control_dt must be an integer multiple of fsm_dt.");
    }
}

void Go1Controller::print_keyboard_help() const {
    std::cout << "\nGo1 deploy keyboard controls:\n"
              << "  s: move to default stand\n"
              << "  r: run policy from default stand\n"
              << "  1/2/3/4: switch policy up/down/left/right\n"
              << "  p: passive damping\n"
              << "  q: damping and quit\n"
              << "  h: print this help\n"
              << "Remote: L2+A=stand, L2+B=passive, select=quit, start+dpad=switch/run policy\n"
              << "Current state: " << state_name(state_) << ", policy=" << active_policy_name_ << "\n\n";
}

void Go1Controller::load_policies() {
    for (const auto* name : kPolicyNames) {
        const auto& path = config_.policies.at(name);
        policies_[name] = std::make_unique<OnnxPolicy>(path, config_.num_obs, config_.num_actions);
        std::vector<float> obs(static_cast<std::size_t>(config_.num_obs), 0.0f);
        auto action = policies_[name]->run(obs);
        if (action.size() != 12) {
            throw std::runtime_error("Policy slot " + std::string(name) + " action is not 12-dimensional.");
        }
        policies_[name]->reset();
    }
}

void Go1Controller::switch_policy(const std::string& name) {
    auto it = policies_.find(name);
    if (it == policies_.end()) {
        throw std::runtime_error("Unknown policy slot: " + name);
    }
    active_policy_ = it->second.get();
    active_policy_name_ = name;
    active_policy_->reset();
    auto q = current_joint_positions();
    for (int i = 0; i < 12; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        last_action_[idx] = clamp((q[idx] - config_.default_angles[idx]) / config_.action_scale, -1.0f, 1.0f);
        policy_target_q_[idx] = clamp(config_.default_angles[idx] + last_action_[idx] * config_.action_scale, joint_lower_limits()[idx], joint_upper_limits()[idx]);
    }
    std::cout << "Active policy: " << active_policy_name_ << "\n";
}

ucl::ConnectionSettings Go1Controller::connection_settings() const {
    const std::string selected = options_.connection_override.empty() ? config_.connection : options_.connection_override;
    if (selected == "low_wifi") {
        return ucl::LOW_WIFI_DEFAULTS;
    }
    if (selected == "low_wired") {
        return ucl::LOW_WIRED_DEFAULTS;
    }
    throw std::runtime_error("Unsupported Go1 connection: " + selected);
}

bool Go1Controller::receive_latest_state() {
    bool got_state = false;
    auto packets = conn_->getData();
    for (const auto& packet : packets) {
        if (low_state_.parseData(packet)) {
            got_state = true;
            last_state_time_ = std::chrono::steady_clock::now();
        }
    }
    return got_state;
}

bool Go1Controller::wait_for_low_state() {
    std::cout << "Waiting for Go1 lowState packets...\n";
    while (!stop_requested_.load()) {
        if (receive_latest_state()) {
            std::cout << "Successfully connected to Go1 lowState.\n";
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return false;
}

void Go1Controller::send_cmd() {
    if (conn_) {
        conn_->send(low_cmd_.buildCmd(false));
    }
}

void Go1Controller::send_damping() {
    ucl::motorCmdArray commands;
    for (int motor = 0; motor < 20; ++motor) {
        commands.setMotorCmd(static_cast<std::size_t>(motor), ucl::motorCmd(ucl::MotorModeLow::Damping, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f));
    }
    low_cmd_.motorCmds = commands;
    send_cmd();
}

std::array<float, 12> Go1Controller::current_joint_positions() const {
    std::array<float, 12> q{};
    for (int i = 0; i < 12; ++i) {
        q[static_cast<std::size_t>(i)] =
            low_state_.motorStates[static_cast<std::size_t>(config_.joint2motor_idx[static_cast<std::size_t>(i)])].q;
    }
    return q;
}

void Go1Controller::fill_servo_cmd(const std::array<float, 12>& q, const std::array<float, 12>& kp, const std::array<float, 12>& kd) {
    ucl::motorCmdArray commands;
    for (int motor = 0; motor < 20; ++motor) {
        commands.setMotorCmd(static_cast<std::size_t>(motor), ucl::motorCmd(ucl::MotorModeLow::Damping, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f));
    }
    for (int i = 0; i < 12; ++i) {
        const int motor_idx = config_.joint2motor_idx[static_cast<std::size_t>(i)];
        commands.setMotorCmd(
            static_cast<std::size_t>(motor_idx),
            ucl::motorCmd(ucl::MotorModeLow::Servo, q[static_cast<std::size_t>(i)], 0.0f, 0.0f, kp[static_cast<std::size_t>(i)], kd[static_cast<std::size_t>(i)]));
    }
    low_cmd_.motorCmds = commands;
}

std::vector<float> Go1Controller::build_observation() const {
    std::vector<float> obs(static_cast<std::size_t>(config_.num_obs), 0.0f);
    for (int i = 0; i < 3; ++i) {
        obs[static_cast<std::size_t>(i)] = low_state_.imuData.gyroscope[static_cast<std::size_t>(i)] * config_.obs_scales_ang_vel;
    }
    auto gravity = gravity_orientation(low_state_.imuData.quaternion);
    for (int i = 0; i < 3; ++i) {
        obs[static_cast<std::size_t>(3 + i)] = gravity[static_cast<std::size_t>(i)];
        obs[static_cast<std::size_t>(6 + i)] = command_[static_cast<std::size_t>(i)] * config_.command_scale[static_cast<std::size_t>(i)];
    }
    for (int i = 0; i < 12; ++i) {
        const auto motor_idx = static_cast<std::size_t>(config_.joint2motor_idx[static_cast<std::size_t>(i)]);
        const auto obs_i = static_cast<std::size_t>(i);
        obs[9 + obs_i] = (low_state_.motorStates[motor_idx].q - config_.default_angles[obs_i]) * config_.obs_scales_dof_pos;
        obs[21 + obs_i] = low_state_.motorStates[motor_idx].dq * config_.obs_scales_dof_vel;
        obs[33 + obs_i] = last_action_[obs_i];
    }
    return obs;
}

void Go1Controller::handle_keyboard(char key) {
    if (key == '\0') {
        return;
    }
    if (key >= '1' && key <= '4') {
        switch_policy(kPolicyNames[static_cast<std::size_t>(key - '1')]);
        if (state_ == ControllerState::DefaultStand || state_ == ControllerState::PolicyRun) {
            transition_to(ControllerState::PolicyRun);
        }
        return;
    }
    switch (key) {
    case 'h':
    case 'H':
        print_keyboard_help();
        break;
    case 'q':
    case 'Q':
        transition_to(ControllerState::ExitDamping);
        break;
    case 'p':
    case 'P':
        transition_to(ControllerState::Passive);
        break;
    case 's':
    case 'S':
        if (state_ == ControllerState::Passive || state_ == ControllerState::DefaultStand) {
            transition_to(ControllerState::MoveToDefault);
        } else {
            std::cout << "Ignoring 's' in state " << state_name(state_) << ".\n";
        }
        break;
    case 'r':
    case 'R':
        if (state_ == ControllerState::DefaultStand) {
            transition_to(ControllerState::PolicyRun);
        } else {
            std::cout << "Policy can only start from DefaultStand. Current state: " << state_name(state_) << "\n";
        }
        break;
    default:
        break;
    }
}

void Go1Controller::update_remote() {
    remote_.last_buttons = remote_.buttons;
    const auto& r = low_state_.wirelessRemote;
    const auto keys = static_cast<std::uint16_t>(r[2] | (static_cast<std::uint16_t>(r[3]) << 8));
    for (int i = 0; i < 16; ++i) {
        remote_.buttons[static_cast<std::size_t>(i)] = (keys >> i) & 1;
    }
    remote_.lx = remote_float(r, 4);
    remote_.rx = remote_float(r, 8);
    remote_.ry = remote_float(r, 12);
    remote_.ly = remote_float(r, 20);
}

void Go1Controller::handle_remote() {
    if (!config_.remote_enabled) {
        return;
    }
    auto on = [&](int key) {
        return remote_.buttons[static_cast<std::size_t>(key)] &&
               !remote_.last_buttons[static_cast<std::size_t>(key)];
    };
    if (on(kSelect)) {
        transition_to(ControllerState::ExitDamping);
        return;
    } else if (remote_.buttons[kL2] && on(kA) && state_ == ControllerState::Passive) {
        transition_to(ControllerState::MoveToDefault);
    } else if (remote_.buttons[kL2] && on(kB)) {
        transition_to(ControllerState::Passive);
    }

    const std::pair<int, const char*> slots[] = {{kUp, "up"}, {kDown, "down"}, {kLeft, "left"}, {kRight, "right"}};
    for (const auto& slot : slots) {
        if (remote_.buttons[kStart] && on(slot.first)) {
            switch_policy(slot.second);
            if (state_ == ControllerState::DefaultStand || state_ == ControllerState::PolicyRun) {
                transition_to(ControllerState::PolicyRun);
            }
        }
    }
}

void Go1Controller::transition_to(ControllerState next) {
    if (state_ == ControllerState::ExitDamping) {
        return;
    }
    if (state_ == next) {
        return;
    }

    std::cout << "FSM: " << state_name(state_) << " -> " << state_name(next) << "\n";
    state_ = next;
    state_enter_time_ = std::chrono::steady_clock::now();

    switch (state_) {
    case ControllerState::Passive:
        send_damping();
        break;
    case ControllerState::MoveToDefault:
        move_start_q_ = current_joint_positions();
        break;
    case ControllerState::DefaultStand:
        fill_servo_cmd(config_.default_angles, config_.stand_kps, config_.stand_kds);
        break;
    case ControllerState::PolicyRun:
        switch_policy(active_policy_name_);
        next_policy_time_ = std::chrono::steady_clock::now();
        fill_servo_cmd(policy_target_q_, config_.kps, config_.kds);
        break;
    case ControllerState::ExitDamping:
        exit_damping_ticks_ = std::max(1, static_cast<int>(std::ceil(0.1 / config_.fsm_dt)));
        send_damping();
        break;
    }
}

void Go1Controller::tick_passive() {
    send_damping();
}

void Go1Controller::tick_move_to_default() {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - state_enter_time_).count();
    const double duration = std::max(config_.move_to_default_s, config_.fsm_dt);
    const float alpha = static_cast<float>(std::min(1.0, elapsed / duration));

    std::array<float, 12> q{};
    for (int i = 0; i < 12; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        q[idx] = move_start_q_[idx] * (1.0f - alpha) + config_.default_angles[idx] * alpha;
    }
    fill_servo_cmd(q, config_.stand_kps, config_.stand_kds);
    send_cmd();

    if (alpha >= 1.0f) {
        transition_to(ControllerState::DefaultStand);
    }
}

void Go1Controller::tick_default_stand() {
    fill_servo_cmd(config_.default_angles, config_.stand_kps, config_.stand_kds);
    send_cmd();
}

void Go1Controller::run_policy_once() {
    if (config_.remote_enabled && config_.remote_cmd_enabled) {
        command_ = {dz(remote_.ly, config_.remote_deadzone), dz(-remote_.lx, config_.remote_deadzone), dz(-remote_.rx, config_.remote_deadzone)};
    }
    const auto lower = joint_lower_limits();
    const auto upper = joint_upper_limits();
    auto action = active_policy_->run(build_observation());
    if (action.size() != 12) {
        std::cerr << "Policy action is not 12-dimensional. Entering damping.\n";
        transition_to(ControllerState::ExitDamping);
        return;
    }
    for (int i = 0; i < 12; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        last_action_[idx] = clamp(action[idx], -1.0f, 1.0f);
        policy_target_q_[idx] = clamp(config_.default_angles[idx] + last_action_[idx] * config_.action_scale, lower[idx], upper[idx]);
    }
}

void Go1Controller::tick_policy_run() {
    const auto now = std::chrono::steady_clock::now();
    const double age = std::chrono::duration<double>(now - last_state_time_).count();
    if (age > config_.state_timeout_s) {
        std::cerr << "lowState timeout: " << age << " s. Entering damping.\n";
        transition_to(ControllerState::Passive);
        return;
    }

    if (now >= next_policy_time_) {
        run_policy_once();
        if (state_ == ControllerState::ExitDamping) {
            return;
        }
        next_policy_time_ = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(config_.control_dt));
    }

    fill_servo_cmd(policy_target_q_, config_.kps, config_.kds);
    send_cmd();
}

void Go1Controller::tick_exit_damping() {
    send_damping();
    --exit_damping_ticks_;
    if (exit_damping_ticks_ <= 0) {
        stop_requested_.store(true);
    }
}

void Go1Controller::print_current_state() const {
    auto rpy = rpy_from_quat(low_state_.imuData.quaternion);
    std::cout << "rpy=(" << rpy[0] << ", " << rpy[1] << ", " << rpy[2] << ") "
              << "gyro=(" << low_state_.imuData.gyroscope[0] << ", " << low_state_.imuData.gyroscope[1] << ", " << low_state_.imuData.gyroscope[2] << ")\n";
}

void Go1Controller::print_remote_bytes() const {
    std::cout << "remote:";
    for (auto byte : low_state_.wirelessRemote) {
        std::cout << " " << static_cast<int>(byte);
    }
    std::cout << " | buttons:";
    for (auto button : remote_.buttons) {
        std::cout << button;
    }
    std::cout << " axes=(" << remote_.lx << ", " << remote_.ly << ", " << remote_.rx << ", " << remote_.ry << ")"
              << " cmd=(" << command_[0] << ", " << command_[1] << ", " << command_[2] << ")\n";
}

void Go1Controller::dry_run() {
    validate_timing();
    std::vector<float> obs(static_cast<std::size_t>(config_.num_obs), 0.0f);
    std::vector<float> action;
    for (auto& item : policies_) {
        item.second->reset();
        action = item.second->run(obs);
        if (action.size() != 12) {
            throw std::runtime_error("Dry run policy slot " + item.first + " action is not 12-dimensional.");
        }
    }
    std::array<float, 12> target_q{};
    for (int i = 0; i < 12; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        target_q[idx] = config_.default_angles[idx] + clamp(action[idx], -1.0f, 1.0f) * config_.action_scale;
    }
    fill_servo_cmd(target_q, config_.kps, config_.kds);
    auto bytes = low_cmd_.buildCmd(false);
    std::cout << "Dry run OK. policies=" << policies_.size()
              << ", active=" << active_policy_name_
              << ", input dim=" << active_policy_->input_dim()
              << ", history=" << active_policy_->history_length()
              << ", lowCmd bytes=" << bytes.size() << "\n";
}

void Go1Controller::run() {
    validate_timing();
    if (options_.dry_run) {
        dry_run();
        return;
    }

    conn_ = std::make_unique<ucl::unitreeConnection>(connection_settings());
    conn_->startRecv();
    send_damping();
    if (!wait_for_low_state()) {
        return;
    }

    NonBlockingKeyboard keyboard;
    state_ = ControllerState::Passive;
    state_enter_time_ = std::chrono::steady_clock::now();
    next_debug_print_time_ = state_enter_time_;
    print_keyboard_help();

    auto next_tick = std::chrono::steady_clock::now();
    while (!stop_requested_.load()) {
        receive_latest_state();
        update_remote();
        const auto now = std::chrono::steady_clock::now();
        const double age = std::chrono::duration<double>(now - last_state_time_).count();
        if (age > config_.state_timeout_s &&
            state_ != ControllerState::Passive &&
            state_ != ControllerState::ExitDamping) {
            std::cerr << "lowState timeout: " << age << " s. Entering damping.\n";
            transition_to(ControllerState::Passive);
        }

        handle_remote();
        handle_keyboard(keyboard.poll());

        if ((options_.print_state || options_.print_remote) && now >= next_debug_print_time_) {
            if (options_.print_state) {
                print_current_state();
            }
            if (options_.print_remote) {
                print_remote_bytes();
            }
            next_debug_print_time_ =
                now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(config_.control_dt));
        }

        switch (state_) {
        case ControllerState::Passive:
            tick_passive();
            break;
        case ControllerState::MoveToDefault:
            tick_move_to_default();
            break;
        case ControllerState::DefaultStand:
            tick_default_stand();
            break;
        case ControllerState::PolicyRun:
            tick_policy_run();
            break;
        case ControllerState::ExitDamping:
            tick_exit_damping();
            break;
        }

        next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(config_.fsm_dt));
        std::this_thread::sleep_until(next_tick);
    }

    send_damping();
}

} // namespace go1_deploy
