#pragma once

#include <string>
#include <variant>
#include <iostream>

namespace MetalFloat16Accelerator {

enum class ErrorCode {
    SUCCESS = 0,
    ERR_INITIALIZATION_FAILED,
    ERR_DEVICE_NOT_FOUND,
    ERR_POOL_EXHAUSTED,
    ERR_THERMAL_THROTTLE,
    ERR_DIMENSION_MISMATCH,
    ERR_GPU_EXECUTION_FAILED,
    ERR_UNSUPPORTED_HARDWARE
};

inline std::string errorToString(ErrorCode code) {
    switch (code) {
        case ErrorCode::SUCCESS: return "Success";
        case ErrorCode::ERR_INITIALIZATION_FAILED: return "Initialization Failed";
        case ErrorCode::ERR_DEVICE_NOT_FOUND: return "Device Not Found";
        case ErrorCode::ERR_POOL_EXHAUSTED: return "Memory Pool Exhausted";
        case ErrorCode::ERR_THERMAL_THROTTLE: return "Thermal Throttling Active";
        case ErrorCode::ERR_DIMENSION_MISMATCH: return "Matrix Dimension Mismatch";
        case ErrorCode::ERR_GPU_EXECUTION_FAILED: return "GPU Execution Failed";
        case ErrorCode::ERR_UNSUPPORTED_HARDWARE: return "Unsupported Hardware";
        default: return "Unknown Error";
    }
}

/**
 * @brief Result pattern (C++20 style) for deterministic error handling.
 */
template <typename T>
class Result {
private:
    std::variant<T, ErrorCode> data_;

public:
    Result(T value) : data_(value) {}
    Result(ErrorCode error) : data_(error) {}

    bool is_ok() const { return std::holds_alternative<T>(data_); }
    bool is_err() const { return std::holds_alternative<ErrorCode>(data_); }

    T unwrap() const {
        if (is_err()) {
            throw std::runtime_error("Attempted to unwrap an error result: " + errorToString(get_error()));
        }
        return std::get<T>(data_);
    }

    ErrorCode get_error() const {
        return std::get<ErrorCode>(data_);
    }
};

} // namespace MetalFloat16Accelerator
