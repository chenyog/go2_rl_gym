#pragma once

#include "ucl/common.hpp"
#include "ucl/complex.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

namespace ucl {

class lowCmd {
public:
    std::array<std::uint8_t, 2> head;
    std::uint8_t levelFlag;
    std::uint8_t frameReserve;
    std::array<std::uint8_t, 8> SN;
    std::array<std::uint8_t, 8> version;
    std::array<std::uint8_t, 2> bandWidth;
    motorCmdArray motorCmds;
    bmsCmd bms;
    std::array<std::uint8_t, 40> wirelessRemote;
    std::array<std::uint8_t, 4> reserve;
    bool encrypt;

    lowCmd()
        : head{0xFE, 0xEF},
          levelFlag(0xFF),
          frameReserve(0),
          SN{},
          version{},
          bandWidth{0x3A, 0xC0},
          motorCmds(),
          bms(0, {0, 0, 0}),
          wirelessRemote{},
          reserve{},
          encrypt(true) {}

    std::vector<std::uint8_t> buildCmd(bool debug = false) const {
        std::vector<std::uint8_t> cmd(614, 0);
        cmd[0] = head[0];
        cmd[1] = head[1];
        cmd[2] = levelFlag;
        cmd[3] = frameReserve;

        std::copy(SN.begin(), SN.end(), cmd.begin() + 4);
        std::copy(version.begin(), version.end(), cmd.begin() + 12);
        cmd[20] = bandWidth[0];
        cmd[21] = bandWidth[1];

        auto motor_bytes = motorCmds.getBytes();
        std::copy(motor_bytes.begin(), motor_bytes.end(), cmd.begin() + 22);

        auto bms_bytes = bms.getBytes();
        std::copy(bms_bytes.begin(), bms_bytes.end(), cmd.begin() + 562);

        std::copy(wirelessRemote.begin(), wirelessRemote.end(), cmd.begin() + 566);

        std::vector<std::uint8_t> crc_data(cmd.begin(), cmd.begin() + (cmd.size() - 6));
        auto crc = genCrc(crc_data);
        if (encrypt) {
            crc = encryptCrc(crc);
        }
        cmd[cmd.size() - 4] = crc[0];
        cmd[cmd.size() - 3] = crc[1];
        cmd[cmd.size() - 2] = crc[2];
        cmd[cmd.size() - 1] = crc[3];

        if (debug) {
            std::cout << "Length: " << cmd.size() << "\n";
            std::cout << "Data: " << byte_print(cmd) << "\n";
        }

        return cmd;
    }
};

} // namespace ucl
