#pragma once

#include <array>
#include <cmath>

namespace go1_deploy {

inline std::array<float, 3> gravity_orientation(const std::array<float, 4>& q) {
    const float qw = q[0];
    const float qx = q[1];
    const float qy = q[2];
    const float qz = q[3];
    return {
        2.0f * (-qz * qx + qw * qy),
        -2.0f * (qz * qy + qw * qx),
        1.0f - 2.0f * (qw * qw + qz * qz),
    };
}

inline std::array<float, 3> rpy_from_quat(const std::array<float, 4>& q) {
    const float w = q[0];
    const float x = q[1];
    const float y = q[2];
    const float z = q[3];

    const float sinr_cosp = 2.0f * (w * x + y * z);
    const float cosr_cosp = 1.0f - 2.0f * (x * x + y * y);
    const float roll = std::atan2(sinr_cosp, cosr_cosp);

    const float sinp = 2.0f * (w * y - z * x);
    float pitch = 0.0f;
    if (std::abs(sinp) >= 1.0f) {
        pitch = std::copysign(1.57079632679f, sinp);
    } else {
        pitch = std::asin(sinp);
    }

    const float siny_cosp = 2.0f * (w * z + x * y);
    const float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
    const float yaw = std::atan2(siny_cosp, cosy_cosp);
    return {roll, pitch, yaw};
}

inline float clamp(float value, float lo, float hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

} // namespace go1_deploy
