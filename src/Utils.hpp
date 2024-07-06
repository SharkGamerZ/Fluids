#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

class Utils {
private:
    /**
     * ANSI color codes.
     */
    enum class Color { RED, GRN, YLW, BLU, MAG, CYN, WHT, RST };

    /**
     * Returns the ANSI color code for the specified color.
     */
    static std::string color(Color color);

public:
    Utils() = delete; ///< Default constructor - deleted

    Utils(const Utils &other) = delete; ///< Copy constructor - deleted

    Utils(Utils &&other) = delete; ///< Move constructor - deleted

    Utils &operator=(const Utils &other) = delete; ///< Copy assignment operator - deleted

    Utils &operator=(Utils &&other) = delete; ///< Move assignment operator - deleted

    ~Utils() = delete; ///< Destructor - deleted

    /**
     * Available logging levels.
     */
    enum class LogLevel { DEBUG, TRACE, ALERT, ERROR };

    /**
     * Prints a log message to the specified output stream with the specified debug level.
     * @param level the debug level
     * @param ostream the output stream
     * @param contents the contents to print
     */
    template<typename... Contents>
    [[maybe_unused]] static void log(Utils::LogLevel level, std::ostream &ostream, Contents... contents);

    /**
     * Read a file and return its content as a string
     * @param filePath file path
     * @return file content as string
     */
    static std::string readFile(const std::string &filePath);
};

template<typename... Contents>
void Utils::log(Utils::LogLevel level, std::ostream &ostream, Contents... contents) {
    std::ostringstream oss;
    (oss << ... << contents);

    switch (level) {
        using enum Utils::LogLevel;
        using enum Utils::Color;
        case DEBUG: ostream << color(YLW) << "[DEBUG] " << color(RST); break;
        case TRACE: ostream << color(GRN) << "[TRACE] " << color(RST); break;
        case ALERT: ostream << color(MAG) << "[ALERT] " << color(RST); break;
        case ERROR: ostream << color(RED) << "[ERROR] " << color(RST); break;
    }

    ostream << oss.str() << std::endl;
}
