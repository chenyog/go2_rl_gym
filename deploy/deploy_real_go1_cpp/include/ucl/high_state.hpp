#pragma once

#include "ucl/common.hpp"
#include "ucl/complex.hpp"
#include "ucl/enums.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace ucl {

class highState {
public:
    std::array<std::uint8_t, 2> head;
    std::uint8_t levelFlag;
    std::uint8_t frameReserve;
    std::array<std::uint8_t, 8> SN;
    std::array<std::uint8_t, 8> version;
    std::array<std::uint8_t, 4> bandWidth;
    imu imuData;
    std::vector<motorState> motorstate;
    bmsState bms;
    std::array<std::uint16_t, 4> footForce;
    std::array<std::uint16_t, 4> footForceEst;
    MotorModeHigh mode;
    float progress;
    GaitType gaitType;
    float footRaiseHeight;
    std::array<float, 3> position;
    float bodyHeight;
    std::array<float, 3> velocity;
    float yawSpeed;
    std::array<float, 4> rangeObstacle;
    std::vector<cartesian> footPosition2Body;
    std::vector<cartesian> footSpeed2Body;
    SpeedLevel speedLevel;
    std::array<std::uint8_t, 40> wirelessRemote;
    std::array<std::uint8_t, 4> reserve;
    std::array<std::uint8_t, 4> crc;

    highState()
        : head{0, 0},
          levelFlag(0),
          frameReserve(0),
          SN{},
          version{},
          bandWidth{},
          imuData{{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, 0},
          motorstate(20),
          bms{0, 0, 0, 0, 0, 0, {0, 0}, {0, 0}, {}},
          footForce{0, 0, 0, 0},
          footForceEst{0, 0, 0, 0},
          mode(MotorModeHigh::IDLE),
          progress(0.0f),
          gaitType(GaitType::IDLE),
          footRaiseHeight(0.0f),
          position{0.0f, 0.0f, 0.0f},
          bodyHeight(0.0f),
          velocity{0.0f, 0.0f, 0.0f},
          yawSpeed(0.0f),
          rangeObstacle{0.0f, 0.0f, 0.0f, 0.0f},
          footPosition2Body(4),
          footSpeed2Body(4),
          speedLevel(SpeedLevel::LOW_SPEED),
          wirelessRemote{},
          reserve{},
          crc{} {}

    void parseData(const std::vector<std::uint8_t>& data) {
        if (data.size() < 1087) {
            return;
        }
        head = {data[0], data[1]};
        levelFlag = data[2];
        frameReserve = data[3];
        std::copy(data.begin() + 4, data.begin() + 12, SN.begin());
        std::copy(data.begin() + 12, data.begin() + 20, version.begin());
        std::copy(data.begin() + 20, data.begin() + 24, bandWidth.begin());

        imuData = dataToImu(data.data() + 22);
        motorstate.clear();
        for (int i = 0; i < 20; ++i) {
            motorstate.push_back(dataToMotorState(data.data() + (i * 32) + 75));
        }

        bms = dataToBmsState(data.data() + 835);
        footForce = {static_cast<std::uint16_t>(data[869] | (data[870] << 8)),
                     static_cast<std::uint16_t>(data[871] | (data[872] << 8)),
                     static_cast<std::uint16_t>(data[873] | (data[874] << 8)),
                     static_cast<std::uint16_t>(data[875] | (data[876] << 8))};
        footForceEst = {static_cast<std::uint16_t>(data[877] | (data[878] << 8)),
                        static_cast<std::uint16_t>(data[879] | (data[880] << 8)),
                        static_cast<std::uint16_t>(data[881] | (data[882] << 8)),
                        static_cast<std::uint16_t>(data[883] | (data[884] << 8))};

        mode = static_cast<MotorModeHigh>(data[885]);
        progress = hex_to_float(data.data() + 886);
        gaitType = static_cast<GaitType>(data[890]);
        footRaiseHeight = hex_to_float(data.data() + 891);
        position = {hex_to_float(data.data() + 895), hex_to_float(data.data() + 899), hex_to_float(data.data() + 903)};
        bodyHeight = hex_to_float(data.data() + 907);
        velocity = {hex_to_float(data.data() + 911), hex_to_float(data.data() + 915), hex_to_float(data.data() + 919)};
        yawSpeed = hex_to_float(data.data() + 923);
        rangeObstacle = {hex_to_float(data.data() + 927), hex_to_float(data.data() + 931),
                         hex_to_float(data.data() + 935), hex_to_float(data.data() + 939)};

        footPosition2Body.clear();
        for (int i = 0; i < 4; ++i) {
            footPosition2Body.push_back({
                hex_to_float(data.data() + (i * 12) + 943),
                hex_to_float(data.data() + (i * 12) + 947),
                hex_to_float(data.data() + (i * 12) + 951)
            });
        }

        footSpeed2Body.clear();
        for (int i = 0; i < 4; ++i) {
            footSpeed2Body.push_back({
                hex_to_float(data.data() + (i * 12) + 991),
                hex_to_float(data.data() + (i * 12) + 995),
                hex_to_float(data.data() + (i * 12) + 999)
            });
        }

        std::copy(data.begin() + 1039, data.begin() + 1079, wirelessRemote.begin());
        std::copy(data.begin() + 1079, data.begin() + 1083, reserve.begin());
        std::copy(data.begin() + 1083, data.begin() + 1087, crc.begin());
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
            int idx = 13 + (i * 2);
            int cell = static_cast<int>(data[idx + 1]) << 8 | static_cast<int>(data[idx]);
            state.cell_vol.push_back(cell);
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
        st.ddq = hex_to_float(data + 9);
        st.tauEst = hex_to_float(data + 13);
        st.q_raw = hex_to_float(data + 17);
        st.dq_raw = hex_to_float(data + 21);
        st.ddq_raw = hex_to_float(data + 25);
        st.temperature = data[29];
        st.reserve = {static_cast<std::uint32_t>(data[30]), static_cast<std::uint32_t>(data[31])};
        return st;
    }
};

} // namespace ucl
