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
#include <cudss.h>


class Command
{
public:
	virtual ~Command() {};
	virtual void execute(int turn) = 0;
	virtual std::string get_name() const = 0;
	virtual std::string get_commandType()const = 0;
	virtual double get_s() const = 0;

protected:

private:

};


template <typename Receiver>
class ConcreteCommand : public Command {
public:
	explicit ConcreteCommand(std::unique_ptr<Receiver> receiver)
		: receiver_(std::move(receiver)) {
	}

	void execute(int turn) override {
		//spdlog::get("logger")->debug("[Command] turn: {}, execute command: {} at {}.", turn, receiver_->name, receiver_->s);
		receiver_->execute(turn);
	}
	std::string get_name() const override {
		return receiver_->name;
	}
	std::string get_commandType() const override {
		return receiver_->commandType;
	}
	double get_s() const override {
		return receiver_->s;
	}

private:
	std::unique_ptr<Receiver> receiver_;
};


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

inline void checkCudss(cudssStatus_t err, const char* file, int line)
{
	if (err != CUDSS_STATUS_SUCCESS)
	{
		//std::cerr << "\nError: cudss error number: " << err << ", in " << file << ", at line " << line << "\n";
		spdlog::get("logger")->error("cudss error: {}, in {}, at line {}", cudaGetErrorString(cudaGetLastError()), file, line);
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

#ifndef callCudss
#define callCudss(call) (checkCudss(call, __FILE__, __LINE__));
#endif // !callCudss


#define USE_CUDA_DEBUG_MODE 1

#if USE_CUDA_DEBUG_MODE
#define callKernel(...) LAUNCH_KERNEL_DEBUG(__VA_ARGS__)
#else
#define callKernel(...) LAUNCH_KERNEL_RELEASE(__VA_ARGS__)
#endif

#define LAUNCH_KERNEL_DEBUG(...) \
    do { \
        { __VA_ARGS__; } \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
			spdlog::get("logger")->error("[CUDA kernel launch error] {}:{}, {}",__FILE__, __LINE__, cudaGetErrorString(__err)); \
            exit(EXIT_FAILURE); \
        } \
        __err = cudaDeviceSynchronize(); \
        if (__err != cudaSuccess) { \
			spdlog::get("logger")->error("[CUDA kernel execution error] {}:{}, {}",__FILE__, __LINE__, cudaGetErrorString(__err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define LAUNCH_KERNEL_RELEASE(...) \
    do { \
        { __VA_ARGS__; } \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
			spdlog::get("logger")->error("[CUDA kernel error] {}:{}, {}",__FILE__, __LINE__, cudaGetErrorString(__err)); \
            exit(EXIT_FAILURE); \
        } \
	} while (0)