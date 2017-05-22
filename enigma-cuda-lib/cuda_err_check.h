#pragma once
#include <iostream>
#include <sstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define CUDA_CHECK(ans) { CudaErrorCheck((ans), __FILE__, __LINE__); }

inline void CudaErrorCheck(cudaError_t code, const char *file, int line)
{
	if (code == cudaSuccess) return;

	std::stringstream msg;
	msg << "CUDA error: " << file << ":" << line << ":  " << cudaGetErrorString(code);	
    std::string str = msg.str();
	std::cerr << str;
	throw std::runtime_error(str);
}
