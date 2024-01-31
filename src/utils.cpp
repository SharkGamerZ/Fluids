#include "utils.h"

std::string readFile(const std::string &filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Unable to open file " << filePath << std::endl;
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
