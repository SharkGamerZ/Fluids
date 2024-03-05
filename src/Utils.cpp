#include "Utils.hpp"

std::string Utils::color(Utils::Color color) {
    switch (color) {
        case Color::RED: return "\033[1;31m";
        case Color::GRN: return "\033[1;32m";
        case Color::YLW: return "\033[1;33m";
        case Color::BLU: return "\033[1;34m";
        case Color::MAG: return "\033[1;35m";
        case Color::CYN: return "\033[1;36m";
        case Color::WHT: return "\033[1;37m";
        case Color::RST: return "\033[0m";
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
