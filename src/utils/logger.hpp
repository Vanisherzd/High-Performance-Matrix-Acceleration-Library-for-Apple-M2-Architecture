#pragma once

#include <string>
#include <iostream>
#include <chrono>
#include <mutex>
#include <vector>
#include <optional>

namespace MetalFloat16Accelerator {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    HARDWARE
};

struct LogEntry {
    LogLevel level;
    std::string message;
    std::chrono::system_clock::time_point timestamp;
};

class SystemLogger {
private:
    std::mutex mutex_;
    std::vector<LogEntry> logs_;
    static SystemLogger* instance_;
    
    SystemLogger() = default;

public:
    static SystemLogger& getInstance() {
        static SystemLogger instance;
        return instance;
    }

    void log(LogLevel level, const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::system_clock::now();
        logs_.push_back({level, message, now});
        
        // Print to console in dmesg-style format
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        char buf[20];
        std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&time_t_now));
        
        std::cout << "[" << buf << "] ";
        switch (level) {
            case LogLevel::DEBUG:    std::cout << "[DEBUG] "; break;
            case LogLevel::INFO:     std::cout << "[INFO]  "; break;
            case LogLevel::WARNING:  std::cout << "[WARN]  "; break;
            case LogLevel::ERROR:    std::cout << "[ERROR] "; break;
            case LogLevel::HARDWARE: std::cout << "[HW]    "; break;
        }
        std::cout << message << std::endl;
    }

    // Hardware Telemetry Simulation (In a real system, would use IOKit or SMC)
    double getThermalStatus() { return 42.5; } // Example Celsius
    double getGPUOccupancy() { return 0.75; }  // 75% Load
};

#define LOG_INFO(msg) SystemLogger::getInstance().log(LogLevel::INFO, msg)
#define LOG_ERROR(msg) SystemLogger::getInstance().log(LogLevel::ERROR, msg)
#define LOG_HW(msg) SystemLogger::getInstance().log(LogLevel::HARDWARE, msg)

} // namespace MetalFloat16Accelerator
