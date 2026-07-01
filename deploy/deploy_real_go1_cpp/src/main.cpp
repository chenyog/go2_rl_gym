#include <csignal>
#include <atomic>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "config.hpp"
#include "go1_controller.hpp"

namespace {
std::atomic<go1_deploy::Go1Controller*> g_controller{nullptr};

void signal_handler(int) {
    auto* controller = g_controller.load();
    if (controller != nullptr) {
        controller->request_stop();
    }
}

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [--config path] [--connection low_wifi|low_wired] "
              << "[--dry-run] [--print-state]\n";
    std::cout << "Runtime keys: S=stand, r=run policy, 1/2/3/4=up/down/left/right policy, "
              << "p=passive damping, Q=quit damping, h=help\n";
    std::cout << "Keyboard cmd source: w/s or up/down=vx, a/d or left/right=vy, q/e=yaw, space=zero\n";
    std::cout << "Remote: L2+A=stand, L2+B=passive, select=quit, start+dpad=switch/run policy\n";
}
} // namespace

int main(int argc, char** argv) {
    std::filesystem::path config_path = "deploy/deploy_real_go1_cpp/config/go1.yaml";
    go1_deploy::RuntimeOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--connection" && i + 1 < argc) {
            options.connection_override = argv[++i];
        } else if (arg == "--dry-run") {
            options.dry_run = true;
        } else if (arg == "--print-state") {
            options.print_state = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            print_usage(argv[0]);
            return 2;
        }
    }

    try {
        auto config = go1_deploy::load_config(config_path);
        go1_deploy::Go1Controller controller(config, options);
        g_controller.store(&controller);
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);
        controller.run();
        g_controller.store(nullptr);
    } catch (const std::exception& exc) {
        std::cerr << "go1_deploy error: " << exc.what() << "\n";
        return 1;
    }

    return 0;
}
