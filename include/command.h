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
	virtual void execute() = 0;

	double s = -1;
	std::string name;
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


inline void checkCudaError(cudaError_t err, const char* file, int line, std::shared_ptr<spdlog::logger>& logger)
{
	if (err != cudaSuccess)
	{
		//std::cerr << "\nError: cuda error: " << cudaGetErrorString(cudaGetLastError()) << ", in " << file << ", at line " << line << "\n";
		logger->error("cuda error: {}, in {}, at line {}", err, file, line);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}
inline void checkCufftError(cufftResult err, const char* file, int line, std::shared_ptr<spdlog::logger>& logger)
{
	if (err != CUFFT_SUCCESS)
	{
		//std::cerr << "\nError: cufft error number: " << err << ", in " << file << ", at line " << line << "\n";
		logger->error("cufft error: {}, in {}, at line {}", err, file, line);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

inline void checkCusparse(cusparseStatus_t err, const char* file, int line, std::shared_ptr<spdlog::logger>& logger)
{
	if (err != CUSPARSE_STATUS_SUCCESS)
	{
		//std::cerr << "\nError: cusparse error number: " << err << ", in " << file << ", at line " << line << "\n";
		logger->error("cufft cusparse: {}, in {}, at line {}", err, file, line);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

inline void checkCusolver(cusolverStatus_t err, const char* file, int line, std::shared_ptr<spdlog::logger>& logger)
{
	if (err != CUSOLVER_STATUS_SUCCESS)
	{
		//std::cerr << "\nError: cusolver error number: " << err << ", in " << file << ", at line " << line << "\n";
		logger->error("cusolver cusparse: {}, in {}, at line {}", err, file, line);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

inline void checkCublas(cublasStatus_t err, const char* file, int line, std::shared_ptr<spdlog::logger>& logger)
{
	if (err != CUBLAS_STATUS_SUCCESS)
	{
		//std::cerr << "\nError: cublas error number: " << err << ", in " << file << ", at line " << line << "\n";
		logger->error("cublas cusparse: {}, in {}, at line {}", err, file, line);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

#ifndef callCuda
#define callCuda(call, logger) (checkCudaError(call, __FILE__, __LINE__, logger));
#endif // !callCuda

#ifndef callCufft
#define callCufft(call, logger) (checkCufftError(call, __FILE__, __LINE__, logger));
#endif // !callCufft

#ifndef callCusparse
#define callCusparse(call, logger) (checkCusparse(call, __FILE__, __LINE__, logger));
#endif // !callCusparse

#ifndef callCusolver
#define callCusolver(call, logger) (checkCusolver(call, __FILE__, __LINE__, logger));
#endif // !callCusolver

#ifndef callCublas
#define callCublas(call, logger) (checkCublas(call, __FILE__, __LINE__, logger));
#endif // !callCublas