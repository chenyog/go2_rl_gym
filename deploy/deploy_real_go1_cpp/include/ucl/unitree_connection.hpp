#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ucl {

struct ConnectionSettings {
    int listenPort;
    std::string addr;
    int sendPort;
    std::string localIP;
};

inline const ConnectionSettings LOW_WIRED_DEFAULTS = {8090, "192.168.123.10", 8007, "192.168.123.16"};
inline const ConnectionSettings LOW_WIFI_DEFAULTS = {8090, "192.168.123.10", 8007, "192.168.12.14"};
inline const ConnectionSettings HIGH_WIRED_DEFAULTS = {8090, "192.168.123.161", 8082, "192.168.123.16"};
inline const ConnectionSettings HIGH_WIFI_DEFAULTS = {8090, "192.168.12.1", 8082, "192.168.12.14"};

class unitreeConnection {
public:
    explicit unitreeConnection(const ConnectionSettings& settings = HIGH_WIFI_DEFAULTS)
        : settings_(settings), sock_(-1), run_recv_(false) {
        initSocket();
    }

    ~unitreeConnection() {
        stopRecv();
        if (sock_ >= 0) {
            close(sock_);
        }
    }

    void startRecv() {
        if (run_recv_) {
            return;
        }
        run_recv_ = true;
        recv_thread_ = std::thread(&unitreeConnection::recvThread, this);
    }

    void stopRecv() {
        if (!run_recv_) {
            return;
        }
        run_recv_ = false;
        if (recv_thread_.joinable()) {
            recv_thread_.join();
        }
    }

    bool send(const std::vector<std::uint8_t>& cmd) const {
        if (sock_ < 0) {
            return false;
        }
        sockaddr_in addr;
        std::memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(settings_.sendPort);
        if (inet_pton(AF_INET, settings_.addr.c_str(), &addr.sin_addr) != 1) {
            return false;
        }
        auto sent = sendto(sock_, cmd.data(), cmd.size(), 0, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        return sent == static_cast<ssize_t>(cmd.size());
    }

    std::vector<std::vector<std::uint8_t>> getData() {
        std::lock_guard<std::mutex> lock(data_mutex_);
        auto ret = data_;
        data_.clear();
        return ret;
    }

private:
    void recvThread() {
        while (run_recv_) {
            std::vector<std::uint8_t> buffer(2048);
            sockaddr_in from_addr;
            socklen_t from_len = sizeof(from_addr);
            ssize_t received = recvfrom(sock_, buffer.data(), buffer.size(), 0, reinterpret_cast<sockaddr*>(&from_addr), &from_len);
            if (received > 0) {
                buffer.resize(static_cast<std::size_t>(received));
                std::lock_guard<std::mutex> lock(data_mutex_);
                data_.push_back(std::move(buffer));
            }
        }
    }

    void initSocket() {
        sock_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock_ < 0) {
            throw std::runtime_error("socket failed: " + std::string(std::strerror(errno)));
        }

        int reuse = 1;
        setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

        sockaddr_in local;
        std::memset(&local, 0, sizeof(local));
        local.sin_family = AF_INET;
        local.sin_port = htons(settings_.listenPort);
        if (inet_pton(AF_INET, settings_.localIP.c_str(), &local.sin_addr) != 1) {
            throw std::runtime_error("Invalid local IP: " + settings_.localIP);
        }
        if (bind(sock_, reinterpret_cast<sockaddr*>(&local), sizeof(local)) < 0) {
            throw std::runtime_error("bind " + settings_.localIP + ":" + std::to_string(settings_.listenPort) + " failed: " + std::string(std::strerror(errno)));
        }

        timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    }

    ConnectionSettings settings_;
    int sock_;
    std::thread recv_thread_;
    bool run_recv_;
    std::mutex data_mutex_;
    std::vector<std::vector<std::uint8_t>> data_;
};

} // namespace ucl
