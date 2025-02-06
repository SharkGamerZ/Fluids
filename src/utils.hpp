#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <utility>

namespace Utils {
    /// ANSI color codes
    enum Color { RED, GRN, YLW, BLU, MAG, CYN, WHT, RST };
    std::string color(Color color);
    /// Logging levels
    enum LogLevel { DEBUG, INFO, WARN, ERROR };
    /// Log message to console
    void log(LogLevel level, std::ostream &ostream, const std::string &message);
    /// Read file contents
    std::optional<std::string> readFile(const std::string &filePath);
}; // namespace Utils
