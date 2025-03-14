#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <cufft.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>	//thrust::cuda::par.on(stream)

class Command
{
public:
	virtual ~Command() {};
	virtual void execute(int turn) = 0;

	double s = -1;
	std::string name = "empty";
protected:

private:

};


//class Invoker
//{
//public:
//	void setCommand(Command* command) {
//		this->command = command;
//	}
//
//	void executeCommand() {
//		command->execute();
//	}
//
//private:
//	Command* command;
//};


inline void checkCudaError(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess)
	{
		//std::cerr << "\nError: cuda error: " << cudaGetErrorString(cudaGetLastError()) << ", in " << file << ", at line " << line << "\n";
		spdlog::get("logger")->error("cuda error: {}, in {}, at line {}", cudaGetErrorString(cudaGetLastError()), file, line);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}
inline void checkCufftError(cufftResult err, const char* file, int line)
{
	if (err != CUFFT_SUCCESS)
	{
		//std::cerr << "\nError: cufft error number: " << err << ", in " << file << ", at line " << line << "\n";
		spdlog::get("logger")->error("cufft error: {}, in {}, at line {}", cudaGetErrorString(cudaGetLastError()), file, line);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

inline void checkCusparse(cusparseStatus_t err, const char* file, int line)
{
	if (err != CUSPARSE_STATUS_SUCCESS)
	{
		//std::cerr << "\nError: cusparse error number: " << err << ", in " << file << ", at line " << line << "\n";
		spdlog::get("logger")->error("cufft cusparse: {}, in {}, at line {}", cudaGetErrorString(cudaGetLastError()), file, line);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

inline void checkCusolver(cusolverStatus_t err, const char* file, int line)
{
	if (err != CUSOLVER_STATUS_SUCCESS)
	{
		//std::cerr << "\nError: cusolver error number: " << err << ", in " << file << ", at line " << line << "\n";
		spdlog::get("logger")->error("cusolver cusparse: {}, in {}, at line {}", cudaGetErrorString(cudaGetLastError()), file, line);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

inline void checkCublas(cublasStatus_t err, const char* file, int line)
{
	if (err != CUBLAS_STATUS_SUCCESS)
	{
		//std::cerr << "\nError: cublas error number: " << err << ", in " << file << ", at line " << line << "\n";
		spdlog::get("logger")->error("cublas cusparse: {}, in {}, at line {}", cudaGetErrorString(cudaGetLastError()), file, line);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

#ifndef callCuda
#define callCuda(call) (checkCudaError(call, __FILE__, __LINE__));
#endif // !callCuda

#ifndef callCufft
#define callCufft(call) (checkCufftError(call, __FILE__, __LINE__));
#endif // !callCufft

#ifndef callCusparse
#define callCusparse(call) (checkCusparse(call, __FILE__, __LINE__));
#endif // !callCusparse

#ifndef callCusolver
#define callCusolver(call) (checkCusolver(call, __FILE__, __LINE__));
#endif // !callCusolver

#ifndef callCublas
#define callCublas(call) (checkCublas(call, __FILE__, __LINE__));
#endif // !callCublas