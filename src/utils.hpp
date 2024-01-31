#ifndef FLUIDS_UTILS_H
#define FLUIDS_UTILS_H

#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iostream>


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


/**
 * Measure the execution time of a function
 * Usage: measure<>::execution(function, args...)
 * @tparam TimeT
 */
template<typename TimeT = std::chrono::milliseconds> struct measure
{
    template<typename F, typename ...Args>
    static typename TimeT::rep execution(F func, Args&&... args)
    {
        auto start = std::chrono::high_resolution_clock::now();
        func(std::forward<Args>(args)...);
        auto duration = std::chrono::duration_cast< TimeT>
                (std::chrono::high_resolution_clock::now() - start);
        return duration.count();
    }
};



#endif //FLUIDS_UTILS_H
