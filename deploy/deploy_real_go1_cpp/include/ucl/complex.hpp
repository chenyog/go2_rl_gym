#pragma once

#include "ucl/common.hpp"
#include "ucl/enums.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

namespace ucl {

struct cartesian {
    float x;
    float y;
    float z;
};

struct bmsState {
    std::uint8_t version_h;
    std::uint8_t version_l;
    std::uint8_t bms_status;
    std::uint8_t SOC;
    int current;
    std::uint16_t cycle;
    std::array<std::uint8_t, 2> BQ_NTC;
    std::array<std::uint8_t, 2> MCU_NTC;
    std::vector<int> cell_vol;
};

class bmsCmd {
public:
    std::uint8_t off;
    std::array<std::uint8_t, 3> reserve;

    bmsCmd(std::uint8_t off = 0, std::array<std::uint8_t, 3> reserve = {0, 0, 0})
        : off(off), reserve(reserve) {}

    std::array<std::uint8_t, 4> getBytes() const {
        return {off, reserve[0], reserve[1], reserve[2]};
    }

    void fromBytes(const std::uint8_t* data) {
        off = data[0];
        reserve = {data[1], data[2], data[3]};
    }
};

class led {
public:
    std::uint8_t r;
    std::uint8_t g;
    std::uint8_t b;

    led(std::uint8_t r = 0, std::uint8_t g = 0, std::uint8_t b = 0) : r(r), g(g), b(b) {}

    std::array<std::uint8_t, 4> getBytes() const {
        return {r, g, b, 0};
    }
};

struct motorState {
    std::uint8_t mode;
    float q;
    float dq;
    float ddq;
    float tauEst;
    float q_raw;
    float dq_raw;
    float ddq_raw;
    std::uint8_t temperature;
    std::array<std::uint32_t, 2> reserve;
};

struct imu {
    std::array<float, 4> quaternion;
    std::array<float, 3> gyroscope;
    std::array<float, 3> accelerometer;
    std::array<float, 3> rpy;
    std::uint8_t temperature;
};

class motorCmd {
public:
    std::uint8_t mode;
    float q;
    float dq;
    float tau;
    float Kp;
    float Kd;
    std::array<std::uint32_t, 3> reserve;

    motorCmd(MotorModeLow mode = MotorModeLow::Servo,
             float q = 0.0f,
             float dq = 0.0f,
             float tau = 0.0f,
             float Kp = 0.0f,
             float Kd = 0.0f,
             std::array<std::uint32_t, 3> reserve = {0, 0, 0})
        : mode(static_cast<std::uint8_t>(mode)),
          q(q),
          dq(dq),
          tau(tau),
          Kp(Kp),
          Kd(Kd),
          reserve(reserve) {}

    std::array<std::uint8_t, 27> getBytes() const {
        std::array<std::uint8_t, 27> out = {};
        out[0] = mode;

        auto q_bytes = float_to_hex(q);
        auto dq_bytes = float_to_hex(dq);
        auto tau_bytes = tau_to_hex(tau);
        auto kp_bytes = kp_to_hex(Kp);
        auto kd_bytes = kd_to_hex(Kd);

        std::copy(q_bytes.begin(), q_bytes.end(), out.begin() + 1);
        std::copy(dq_bytes.begin(), dq_bytes.end(), out.begin() + 5);
        std::copy(tau_bytes.begin(), tau_bytes.end(), out.begin() + 9);
        std::copy(kp_bytes.begin(), kp_bytes.end(), out.begin() + 11);
        std::copy(kd_bytes.begin(), kd_bytes.end(), out.begin() + 13);

        std::size_t offset = 15;
        for (std::size_t i = 0; i < reserve.size(); ++i) {
            std::uint32_t val = reserve[i];
            out[offset++] = static_cast<std::uint8_t>(val & 0xFF);
            out[offset++] = static_cast<std::uint8_t>((val >> 8) & 0xFF);
            out[offset++] = static_cast<std::uint8_t>((val >> 16) & 0xFF);
            out[offset++] = static_cast<std::uint8_t>((val >> 24) & 0xFF);
        }
        return out;
    }

    void fromBytes(const std::uint8_t* data, bool should_print = false) {
        mode = data[0];
        q = hex_to_float(data + 1);
        dq = hex_to_float(data + 5);
        tau = hex_to_tau(data + 9);
        Kp = hex_to_kp(data + 11);
        Kd = hex_to_kd(data + 13);

        reserve[0] = static_cast<std::uint32_t>(data[15]) |
                     (static_cast<std::uint32_t>(data[16]) << 8) |
                     (static_cast<std::uint32_t>(data[17]) << 16) |
                     (static_cast<std::uint32_t>(data[18]) << 24);
        reserve[1] = static_cast<std::uint32_t>(data[19]) |
                     (static_cast<std::uint32_t>(data[20]) << 8) |
                     (static_cast<std::uint32_t>(data[21]) << 16) |
                     (static_cast<std::uint32_t>(data[22]) << 24);
        reserve[2] = static_cast<std::uint32_t>(data[23]) |
                     (static_cast<std::uint32_t>(data[24]) << 8) |
                     (static_cast<std::uint32_t>(data[25]) << 16) |
                     (static_cast<std::uint32_t>(data[26]) << 24);

        if (should_print) {
            std::cout << "Mcmd:\t" << std::hex << static_cast<int>(mode) << std::dec << "\n";
            std::cout << "q:\t" << byte_print(data + 1, 4) << "\n";
            std::cout << "dq:\t" << byte_print(data + 5, 4) << "\n";
            std::cout << "tau:\t" << byte_print(data + 9, 2) << "\n";
            std::cout << "Kp:\t" << byte_print(data + 11, 2) << "\n";
            std::cout << "Kd:\t" << byte_print(data + 13, 2) << "\n";
            std::cout << "res:\t" << byte_print(data + 15, 4) << ", "
                      << byte_print(data + 19, 4) << ", "
                      << byte_print(data + 23, 4) << "\n";
        }
    }
};

class motorCmdArray {
public:
    std::array<motorCmd, 20> motors;

    motorCmdArray() {
        for (std::size_t i = 0; i < motors.size(); ++i) {
            motors[i] = motorCmd();
        }
    }

    void setMotorCmd(std::size_t motorIndex, const motorCmd& cmd) {
        if (motorIndex >= motors.size()) {
            return;
        }
        motors[motorIndex] = cmd;
    }

    std::vector<std::uint8_t> getBytes() const {
        std::vector<std::uint8_t> out;
        out.reserve(motors.size() * 27);
        for (const auto& motor : motors) {
            auto bytes = motor.getBytes();
            out.insert(out.end(), bytes.begin(), bytes.end());
        }
        return out;
    }

    void fromBytes(const std::vector<std::uint8_t>& data) {
        if (data.size() < motors.size() * 27) {
            return;
        }
        for (std::size_t i = 0; i < motors.size(); ++i) {
            motors[i].fromBytes(data.data() + (i * 27));
        }
    }
};

} // namespace ucl
