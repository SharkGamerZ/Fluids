#include "utils.hpp"

namespace Utils {
    std::string color(const Color color) {
        switch (color) {
            using enum Color;
            case RED: return "\033[31m";
            case GRN: return "\033[32m";
            case YLW: return "\033[33m";
            case BLU: return "\033[34m";
            case MAG: return "\033[35m";
            case CYN: return "\033[36m";
            case WHT: return "\033[37m";
            case RST: return "\033[0m";
        }
        std::unreachable();
    }

    void log(const LogLevel level, std::ostream &ostream, const std::string &message) {
        std::string logPrefix;
        std::string coloredLogPrefix;
        const bool isOfstream = typeid(ostream) == typeid(std::ofstream);

        switch (level) {
            using enum LogLevel;
            using enum Color;

            case DEBUG:
                logPrefix = "[DEBUG] ";
                coloredLogPrefix = std::format("{}[DEBUG] {}", color(YLW), color(RST));
                break;
            case INFO:
                logPrefix = "[INFO] ";
                coloredLogPrefix = std::format("{}[INFO] {}", color(GRN), color(RST));
                break;
            case WARN:
                logPrefix = "[WARN] ";
                coloredLogPrefix = std::format("{}[WARN] {}", color(MAG), color(RST));
                break;
            case ERROR:
                logPrefix = "[ERROR] ";
                coloredLogPrefix = std::format("{}[ERROR] {}", color(RED), color(RST));
                break;
        }

        ostream << (isOfstream ? logPrefix : coloredLogPrefix) << message << std::endl;

        if (isOfstream) {
            std::cout << coloredLogPrefix << message << std::endl;
        }
    }

    std::optional<std::string> readFile(const std::string &filePath) {
        try {
            std::ifstream file(filePath, std::ios::binary | std::ios::ate);
            if (!file.is_open()) {
                log(ERROR, std::cerr, std::format("Failed to open file: {}", filePath));
                return std::nullopt;
            }

            const std::streamsize size = file.tellg();
            if (size <= 0) {
                log(ERROR, std::cerr, std::format("File is empty or invalid: {}", filePath));
                return std::nullopt;
            }

            file.seekg(0, std::ios::beg);
            std::string buffer(size, '\0');

            if (!file.read(buffer.data(), size)) {
                log(ERROR, std::cerr, std::format("Failed to read file: {}", filePath));
                return std::nullopt;
            }

            return buffer;
        } catch (const std::exception &e) {
            log(ERROR, std::cerr, std::format("Exception occurred: {}", e.what()));
            return std::nullopt;
        }
    }
} // namespace Utils
