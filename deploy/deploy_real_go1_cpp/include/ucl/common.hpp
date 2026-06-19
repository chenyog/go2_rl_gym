#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ucl {

inline std::string lib_version() {
    return "0.2";
}

inline std::pair<std::string, std::string> decode_sn(const std::vector<std::uint8_t>& data) {
    if (data.size() < 6) {
        return {"UNKNOWN_UNKNOWN", "0-0-0[0]"};
    }
    std::string type_name = "UNKNOWN";
    switch (data[0]) {
        case 1: type_name = "Laikago"; break;
        case 2: type_name = "Aliengo"; break;
        case 3: type_name = "A1"; break;
        case 4: type_name = "Go1"; break;
        case 5: type_name = "B1"; break;
        default: break;
    }
    std::string model_name = "UNKNOWN";
    switch (data[1]) {
        case 1: model_name = "AIR"; break;
        case 2: model_name = "PRO"; break;
        case 3: model_name = "EDU"; break;
        case 4: model_name = "PC"; break;
        case 5: model_name = "XX"; break;
        default: break;
    }

    std::string product_name = type_name + "_" + model_name;
    std::ostringstream id;
    id << static_cast<int>(data[2]) << "-" << static_cast<int>(data[3]) << "-"
       << static_cast<int>(data[4]) << "[" << static_cast<int>(data[5]) << "]";
    return {product_name, id.str()};
}

inline std::pair<std::string, std::string> decode_version(const std::vector<std::uint8_t>& data) {
    if (data.size() < 6) {
        return {"0.0.0", "0.0.0"};
    }
    std::ostringstream hw;
    std::ostringstream sw;
    hw << static_cast<int>(data[0]) << "." << static_cast<int>(data[1]) << "." << static_cast<int>(data[2]);
    sw << static_cast<int>(data[3]) << "." << static_cast<int>(data[4]) << "." << static_cast<int>(data[5]);
    return {hw.str(), sw.str()};
}

inline int getVoltage(const std::vector<int>& cellVoltages) {
    int sum = 0;
    for (int v : cellVoltages) {
        sum += v;
    }
    return sum;
}

inline std::array<std::uint8_t, 4> float_to_hex(float value) {
    std::uint32_t raw = 0;
    std::memcpy(&raw, &value, sizeof(raw));
    std::array<std::uint8_t, 4> out = {
        static_cast<std::uint8_t>(raw & 0xFF),
        static_cast<std::uint8_t>((raw >> 8) & 0xFF),
        static_cast<std::uint8_t>((raw >> 16) & 0xFF),
        static_cast<std::uint8_t>((raw >> 24) & 0xFF)
    };
    return out;
}

inline float hex_to_float(const std::uint8_t* hex_bytes) {
    std::uint32_t raw = static_cast<std::uint32_t>(hex_bytes[0]) |
                        (static_cast<std::uint32_t>(hex_bytes[1]) << 8) |
                        (static_cast<std::uint32_t>(hex_bytes[2]) << 16) |
                        (static_cast<std::uint32_t>(hex_bytes[3]) << 24);
    float value = 0.0f;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

inline std::uint8_t fraction_to_hex(float fraction, bool neg) {
    if (fraction == 0.0f) {
        neg = false;
    }
    int hex_value = static_cast<int>(fraction * 256.0f);
    if (neg) {
        hex_value = 255 + hex_value + 1;
    }
    return static_cast<std::uint8_t>(hex_value & 0xFF);
}

inline std::array<std::uint8_t, 2> tau_to_hex(float tau) {
    float rounded = std::round(tau * 100.0f) / 100.0f;
    int integer_part = static_cast<int>(rounded);
    float fractional_part = rounded - static_cast<float>(integer_part);
    bool neg = false;
    if (rounded < 0.0f) {
        neg = true;
        integer_part = 255 + integer_part;
    }
    std::array<std::uint8_t, 2> out = {
        fraction_to_hex(fractional_part, neg),
        static_cast<std::uint8_t>(integer_part & 0xFF)
    };
    return out;
}

inline float hex_to_fraction(std::uint8_t hex_byte, bool neg) {
    if (neg) {
        return -1.0f + std::round((static_cast<float>(hex_byte) / 256.0f) * 100.0f) / 100.0f;
    }
    return std::round((static_cast<float>(hex_byte) / 256.0f) * 100.0f) / 100.0f;
}

inline float hex_to_tau(const std::uint8_t* hex_bytes) {
    int int_val = static_cast<int>(hex_bytes[1]);
    bool neg = false;
    if (int_val > 126) {
        neg = true;
        int_val = -255 + int_val;
    }
    return static_cast<float>(int_val) + hex_to_fraction(hex_bytes[0], neg);
}

inline std::array<std::uint8_t, 2> kp_to_hex(float kp) {
    float base_f;
    float frac_f = std::modf(kp, &base_f);
    int base = static_cast<int>(base_f);
    int frac = static_cast<int>(std::round(frac_f * 10.0f));

    int val = 0;
    if (frac < 5) {
        val = (base * 32) + frac * 3;
    } else {
        val = (base * 32) + ((frac - 1) * 3) + 4;
    }

    std::uint16_t v = static_cast<std::uint16_t>(val);
    std::array<std::uint8_t, 2> out = {
        static_cast<std::uint8_t>(v & 0xFF),
        static_cast<std::uint8_t>((v >> 8) & 0xFF)
    };
    return out;
}

inline float hex_to_kp(const std::uint8_t* hex_bytes) {
    std::uint16_t val = static_cast<std::uint16_t>(hex_bytes[0]) |
                        (static_cast<std::uint16_t>(hex_bytes[1]) << 8);
    int base = val / 32;
    int remainder = val % 32;
    float frac = 0.0f;
    if (remainder < 15) {
        frac = static_cast<float>(remainder) / 3.0f;
    } else {
        frac = (static_cast<float>(remainder - 4) / 3.0f) + 1.0f;
    }
    return static_cast<float>(base) + (std::round(frac) / 10.0f);
}

inline char get_hex_frac_char(float frac) {
    if (frac == 0.0f) return '0';
    if (frac == 0.1f) return '1';
    if (frac == 0.2f) return '3';
    if (frac == 0.3f) return '4';
    if (frac == 0.4f) return '6';
    if (frac == 0.5f) return '8';
    if (frac == 0.6f) return '9';
    if (frac == 0.7f) return 'b';
    if (frac == 0.8f) return 'c';
    if (frac == 0.9f) return 'e';
    return '0';
}

inline std::array<std::uint8_t, 2> kd_to_hex(float kd) {
    int integer_part = static_cast<int>(kd);
    float fractional_part = std::round((kd - static_cast<float>(integer_part)) * 10.0f) / 10.0f;
    char frac_char = get_hex_frac_char(fractional_part);

    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(3) << integer_part << frac_char;
    std::string hex = oss.str();
    std::uint8_t high = static_cast<std::uint8_t>(std::stoul(hex.substr(0, 2), nullptr, 16));
    std::uint8_t low = static_cast<std::uint8_t>(std::stoul(hex.substr(2, 2), nullptr, 16));

    return {low, high};
}

inline float get_frac_hex(char frac) {
    switch (frac) {
        case '0': return 0.0f;
        case '1': return 0.1f;
        case '3': return 0.2f;
        case '4': return 0.3f;
        case '6': return 0.4f;
        case '8': return 0.5f;
        case '9': return 0.6f;
        case 'b': return 0.7f;
        case 'c': return 0.8f;
        case 'e': return 0.9f;
        default: return 0.0f;
    }
}

inline float hex_to_kd(const std::uint8_t* hex_bytes) {
    std::uint16_t val = static_cast<std::uint16_t>(hex_bytes[0]) |
                        (static_cast<std::uint16_t>(hex_bytes[1]) << 8);
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(4) << val;
    std::string hex = oss.str();
    int int_part = std::stoi(hex.substr(0, 3), nullptr, 16);
    float frac_part = get_frac_hex(hex[3]);
    return static_cast<float>(int_part) + frac_part;
}

inline std::array<std::uint8_t, 4> genCrc(const std::vector<std::uint8_t>& data) {
    std::uint32_t crc = 0xFFFFFFFFu;
    std::size_t len = data.size() / 4;
    for (std::size_t i = 0; i < len; ++i) {
        std::uint32_t j = static_cast<std::uint32_t>(data[i * 4]) |
                          (static_cast<std::uint32_t>(data[i * 4 + 1]) << 8) |
                          (static_cast<std::uint32_t>(data[i * 4 + 2]) << 16) |
                          (static_cast<std::uint32_t>(data[i * 4 + 3]) << 24);
        for (int b = 0; b < 32; ++b) {
            std::uint32_t x = (crc >> 31) & 1u;
            crc <<= 1;
            crc &= 0xFFFFFFFFu;
            if (x ^ ((j >> (31 - b)) & 1u)) {
                crc ^= 0x04c11db7u;
            }
        }
    }
    std::array<std::uint8_t, 4> out = {
        static_cast<std::uint8_t>(crc & 0xFF),
        static_cast<std::uint8_t>((crc >> 8) & 0xFF),
        static_cast<std::uint8_t>((crc >> 16) & 0xFF),
        static_cast<std::uint8_t>((crc >> 24) & 0xFF)
    };
    return out;
}

inline std::array<std::uint8_t, 4> encryptCrc(const std::array<std::uint8_t, 4>& crc) {
    std::uint32_t crc_val = static_cast<std::uint32_t>(crc[0]) |
                            (static_cast<std::uint32_t>(crc[1]) << 8) |
                            (static_cast<std::uint32_t>(crc[2]) << 16) |
                            (static_cast<std::uint32_t>(crc[3]) << 24);
    std::uint32_t val = crc_val ^ 0xedcab9deu;
    std::array<std::uint8_t, 4> data = {
        static_cast<std::uint8_t>(val & 0xFF),
        static_cast<std::uint8_t>((val >> 8) & 0xFF),
        static_cast<std::uint8_t>((val >> 16) & 0xFF),
        static_cast<std::uint8_t>((val >> 24) & 0xFF)
    };
    return {data[1], data[2], data[3], data[0]};
}

inline std::string byte_print(const std::vector<std::uint8_t>& bytes) {
    std::ostringstream oss;
    for (std::uint8_t byte : bytes) {
        oss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(byte);
    }
    return oss.str();
}

inline std::string byte_print(const std::uint8_t* bytes, std::size_t len) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < len; ++i) {
        oss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(bytes[i]);
    }
    return oss.str();
}

} // namespace ucl
