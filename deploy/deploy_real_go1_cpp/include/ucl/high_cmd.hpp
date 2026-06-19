#pragma once

#include "ucl/common.hpp"
#include "ucl/complex.hpp"
#include "ucl/enums.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

namespace ucl {

class highCmd {
public:
    std::array<std::uint8_t, 2> head;
    std::uint8_t levelFlag;
    std::uint8_t frameReserve;
    std::array<std::uint8_t, 8> SN;
    std::array<std::uint8_t, 8> version;
    std::array<std::uint8_t, 2> bandWidth;
    MotorModeHigh mode;
    GaitType gaitType;
    SpeedLevel speedLevel;
    float footRaiseHeight;
    float bodyHeight;
    std::array<float, 2> position;
    std::array<float, 3> euler;
    std::array<float, 2> velocity;
    float yawSpeed;
    bmsCmd bms;
    led ledStatus;
    std::array<std::uint8_t, 40> wirelessRemote;
    std::array<std::uint8_t, 4> reserve;
    bool encrypt;

    highCmd()
        : head{0xFE, 0xEF},
          levelFlag(0x00),
          frameReserve(0),
          SN{},
          version{},
          bandWidth{},
          mode(MotorModeHigh::IDLE),
          gaitType(GaitType::IDLE),
          speedLevel(SpeedLevel::LOW_SPEED),
          footRaiseHeight(0.0f),
          bodyHeight(0.0f),
          position{0.0f, 0.0f},
          euler{0.0f, 0.0f, 0.0f},
          velocity{0.0f, 0.0f},
          yawSpeed(0.0f),
          bms(0, {0, 0, 0}),
          ledStatus(0, 0, 0),
          wirelessRemote{},
          reserve{},
          encrypt(false) {}

    std::vector<std::uint8_t> buildCmd(bool debug = false) const {
        std::vector<std::uint8_t> cmd(129, 0);
        cmd[0] = head[0];
        cmd[1] = head[1];
        cmd[2] = levelFlag;
        cmd[3] = frameReserve;

        std::copy(SN.begin(), SN.end(), cmd.begin() + 4);
        std::copy(version.begin(), version.end(), cmd.begin() + 12);
        cmd[20] = bandWidth[0];
        cmd[21] = bandWidth[1];

        cmd[22] = static_cast<std::uint8_t>(mode);
        cmd[23] = static_cast<std::uint8_t>(gaitType);
        cmd[24] = static_cast<std::uint8_t>(speedLevel);

        auto foot_bytes = float_to_hex(footRaiseHeight);
        auto body_bytes = float_to_hex(bodyHeight);
        std::copy(foot_bytes.begin(), foot_bytes.end(), cmd.begin() + 25);
        std::copy(body_bytes.begin(), body_bytes.end(), cmd.begin() + 29);

        auto pos_x = float_to_hex(position[0]);
        auto pos_y = float_to_hex(position[1]);
        std::copy(pos_x.begin(), pos_x.end(), cmd.begin() + 33);
        std::copy(pos_y.begin(), pos_y.end(), cmd.begin() + 37);

        auto euler_x = float_to_hex(euler[0]);
        auto euler_y = float_to_hex(euler[1]);
        auto euler_z = float_to_hex(euler[2]);
        std::copy(euler_x.begin(), euler_x.end(), cmd.begin() + 41);
        std::copy(euler_y.begin(), euler_y.end(), cmd.begin() + 45);
        std::copy(euler_z.begin(), euler_z.end(), cmd.begin() + 49);

        auto vel_x = float_to_hex(velocity[0]);
        auto vel_y = float_to_hex(velocity[1]);
        std::copy(vel_x.begin(), vel_x.end(), cmd.begin() + 53);
        std::copy(vel_y.begin(), vel_y.end(), cmd.begin() + 57);

        auto yaw_bytes = float_to_hex(yawSpeed);
        std::copy(yaw_bytes.begin(), yaw_bytes.end(), cmd.begin() + 61);

        auto bms_bytes = bms.getBytes();
        std::copy(bms_bytes.begin(), bms_bytes.end(), cmd.begin() + 65);

        auto led_bytes = ledStatus.getBytes();
        std::copy(led_bytes.begin(), led_bytes.end(), cmd.begin() + 69);

        std::copy(wirelessRemote.begin(), wirelessRemote.end(), cmd.begin() + 73);
        std::copy(reserve.begin(), reserve.end(), cmd.begin() + 113);

        std::vector<std::uint8_t> crc_data(cmd.begin(), cmd.begin() + (cmd.size() - 5));
        auto crc = genCrc(crc_data);
        if (encrypt) {
            crc = encryptCrc(crc);
        }
        cmd[cmd.size() - 4] = crc[0];
        cmd[cmd.size() - 3] = crc[1];
        cmd[cmd.size() - 2] = crc[2];
        cmd[cmd.size() - 1] = crc[3];

        if (debug) {
            std::cout << "Send Data (" << cmd.size() << "): " << byte_print(cmd) << "\n";
        }

        return cmd;
    }
};

} // namespace ucl
