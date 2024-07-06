#include "Utils.hpp"

std::string Utils::color(Utils::Color color) {
    switch (color) {
        using enum Utils::Color;
        case RED: return "\033[1;31m";
        case GRN: return "\033[1;32m";
        case YLW: return "\033[1;33m";
        case BLU: return "\033[1;34m";
        case MAG: return "\033[1;35m";
        case CYN: return "\033[1;36m";
        case WHT: return "\033[1;37m";
        case RST: return "\033[0m";
    }
    return "\033[0m"; // To silence compiler warning
}

std::string Utils::readFile(const std::string &filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        Utils::log(Utils::LogLevel::ERROR, std::cerr, "Unable to open file ", filePath);
        return "";
    }

    std::stringstream stream;

    // Read file's buffer contents into stream
    stream << file.rdbuf();

    // Close file handler
    file.close();

    // Convert stream into string
    return stream.str();
}
