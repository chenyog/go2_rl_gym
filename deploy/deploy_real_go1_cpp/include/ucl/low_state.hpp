#pragma once

#include "ucl/common.hpp"
#include "ucl/complex.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace ucl {

class lowState {
public:
    std::array<std::uint8_t, 2> head;
    std::uint8_t levelFlag;
    std::uint8_t frameReserve;
    std::array<std::uint8_t, 8> SN;
    std::array<std::uint8_t, 8> version;
    std::array<std::uint8_t, 4> bandWidth;
    imu imuData;
    std::vector<motorState> motorStates;
    bmsState bms;
    std::array<std::uint16_t, 4> footForce;
    std::array<std::uint16_t, 4> footForceEst;
    std::array<std::uint8_t, 4> tick;
    std::array<std::uint8_t, 40> wirelessRemote;
    std::array<std::uint8_t, 4> reserve;
    std::array<std::uint8_t, 4> crc;

    lowState()
        : head{0, 0},
          levelFlag(0),
          frameReserve(0),
          SN{},
          version{},
          bandWidth{},
          imuData{{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, 0},
          motorStates(20),
          bms{0, 0, 0, 0, 0, 0, {0, 0}, {0, 0}, {}},
          footForce{0, 0, 0, 0},
          footForceEst{0, 0, 0, 0},
          tick{},
          wirelessRemote{},
          reserve{},
          crc{} {}

    bool parseData(const std::vector<std::uint8_t>& data) {
        if (data.size() < 807) {
            return false;
        }
        head = {data[0], data[1]};
        levelFlag = data[2];
        frameReserve = data[3];
        std::copy(data.begin() + 4, data.begin() + 12, SN.begin());
        std::copy(data.begin() + 12, data.begin() + 20, version.begin());
        std::copy(data.begin() + 20, data.begin() + 24, bandWidth.begin());

        imuData = dataToImu(data.data() + 22);
        motorStates.clear();
        for (int i = 0; i < 20; ++i) {
            motorStates.push_back(dataToMotorState(data.data() + (i * 32) + 75));
        }

        bms = dataToBmsState(data.data() + 715);
        footForce = {static_cast<std::uint16_t>(data[739] | (data[740] << 8)),
                     static_cast<std::uint16_t>(data[751] | (data[752] << 8)),
                     static_cast<std::uint16_t>(data[753] | (data[754] << 8)),
                     static_cast<std::uint16_t>(data[755] | (data[756] << 8))};
        footForceEst = {static_cast<std::uint16_t>(data[747] | (data[748] << 8)),
                        static_cast<std::uint16_t>(data[759] | (data[760] << 8)),
                        static_cast<std::uint16_t>(data[761] | (data[762] << 8)),
                        static_cast<std::uint16_t>(data[763] | (data[764] << 8))};

        std::copy(data.begin() + 759, data.begin() + 799, wirelessRemote.begin());
        std::copy(data.begin() + 799, data.begin() + 803, reserve.begin());
        std::copy(data.begin() + 803, data.begin() + 807, crc.begin());
        return true;
    }

private:
    bmsState dataToBmsState(const std::uint8_t* data) const {
        bmsState state;
        state.version_h = data[0];
        state.version_l = data[1];
        state.bms_status = data[2];
        state.SOC = data[3];
        state.current = static_cast<int>(
            static_cast<std::int32_t>(data[4]) |
            (static_cast<std::int32_t>(data[5]) << 8) |
            (static_cast<std::int32_t>(data[6]) << 16) |
            (static_cast<std::int32_t>(data[7]) << 24));
        state.cycle = static_cast<std::uint16_t>(data[8]) | (static_cast<std::uint16_t>(data[9]) << 8);
        state.BQ_NTC = {data[10], data[11]};
        state.MCU_NTC = {data[12], data[13]};
        state.cell_vol.clear();
        for (int i = 0; i < 10; ++i) {
            state.cell_vol.push_back(static_cast<int>(data[14 + i]) * 32);
        }
        return state;
    }

    imu dataToImu(const std::uint8_t* data) const {
        imu out;
        out.quaternion = {hex_to_float(data + 0), hex_to_float(data + 4), hex_to_float(data + 8), hex_to_float(data + 12)};
        out.gyroscope = {hex_to_float(data + 16), hex_to_float(data + 20), hex_to_float(data + 24)};
        out.accelerometer = {hex_to_float(data + 28), hex_to_float(data + 32), hex_to_float(data + 36)};
        out.rpy = {hex_to_float(data + 40), hex_to_float(data + 44), hex_to_float(data + 48)};
        out.temperature = data[52];
        return out;
    }

    motorState dataToMotorState(const std::uint8_t* data) const {
        motorState st;
        st.mode = data[0];
        st.q = hex_to_float(data + 1);
        st.dq = hex_to_float(data + 5);
        std::int16_t ddq_raw = static_cast<std::int16_t>(data[9] | (data[10] << 8));
        st.ddq = static_cast<float>(ddq_raw);
        std::int16_t tau_raw = static_cast<std::int16_t>(data[11] | (data[12] << 8));
        st.tauEst = static_cast<float>(tau_raw) * 0.00390625f;
        st.q_raw = hex_to_float(data + 13);
        st.dq_raw = hex_to_float(data + 17);
        std::int16_t ddq_raw2 = static_cast<std::int16_t>(data[21] | (data[22] << 8));
        st.ddq_raw = static_cast<float>(ddq_raw2);
        st.temperature = data[24];
        st.reserve = {
            static_cast<std::uint32_t>(data[24]) |
                (static_cast<std::uint32_t>(data[25]) << 8) |
                (static_cast<std::uint32_t>(data[26]) << 16) |
                (static_cast<std::uint32_t>(data[27]) << 24),
            static_cast<std::uint32_t>(data[28]) |
                (static_cast<std::uint32_t>(data[29]) << 8) |
                (static_cast<std::uint32_t>(data[30]) << 16) |
                (static_cast<std::uint32_t>(data[31]) << 24)
        };
        return st;
    }
};

} // namespace ucl
