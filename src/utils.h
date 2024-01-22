#ifndef FLUIDS_UTILS_H
#define FLUIDS_UTILS_H

#include <string>
#include <fstream>
#include <sstream>
#include <chrono>


#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define BOLD    "\033[1m"
#define ITALIC "\033[3m"
#define RESET   "\033[0m"


/**
 * Read a file and return its content as a string
 * @param filePath file path
 * @return file content as string
 */
std::string readFile(const std::string& filePath);

#endif //FLUIDS_UTILS_H
