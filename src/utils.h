#ifndef FLUIDS_UTILS_H
#define FLUIDS_UTILS_H

#include <string>
#include <fstream>
#include <sstream>

/**
 * Read a file and return its content as a string
 * @param filePath file path
 * @return file content as string
 */
std::string readFile(const std::string& filePath);

#endif //FLUIDS_UTILS_H
