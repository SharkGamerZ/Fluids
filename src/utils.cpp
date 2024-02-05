#include "utils.hpp"

const bool DEBUG_MODE = true;

[[maybe_unused]] void tracePrint(const std::string &message) {
    std::cout << BOLD GREEN << "[TRACE] " << RESET << message << std::endl;
}

[[maybe_unused]] void debugPrint(const std::string &message) {
    if (DEBUG_MODE) {
        std::cout << BOLD YELLOW << "[DEBUG] " << RESET << message << std::endl;
    }
}

[[maybe_unused]] void errorPrint(const std::string &message) {
    std::cout << BOLD RED << "[ERROR] " << RESET << message << std::endl;
}

std::string readFile(const std::string &filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        errorPrint("Unable to open file " + filePath);
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
