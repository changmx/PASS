#pragma once

#include <iostream>
#include "cuda_runtime.h"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

class ParallelPlan1d
{

public:
	ParallelPlan1d() = default;
	ParallelPlan1d(int threads_per_block, int max_calls_per_thread, int Np)
		: threads_per_block_(validate_threads(threads_per_block)),
		max_calls_per_thread_(validate_calls(max_calls_per_thread)),
		Np_(validate_Np(Np)),
		blocks_x_(calculate_blocks_x(threads_per_block_, max_calls_per_thread_, Np_)),
		total_threads_(blocks_x_* threads_per_block_) {
	}

	~ParallelPlan1d()
	{
		//spdlog::get("logger")->debug(
		//	"[ParallelPlan1d] class destructor: threads per block={}, blocks x={}, max call time per thread={}, Np={}, total threads ={}",
		//	threads_per_block_, blocks_x_, max_calls_per_thread_, Np_, total_threads_);
	};

	void print() const {
		spdlog::get("logger")->info(
			"[ParallelPlan1d] threads per block={}, blocks x={}, max call time per thread={}, Np={}, total threads={}",
			threads_per_block_, blocks_x_, max_calls_per_thread_, Np_, total_threads_);
	}

	int get_threads_per_block() const { return threads_per_block_; }
	int get_blocks_x() const { return blocks_x_; }

private:

	int calculate_blocks_x(int threads, int calls, int n) {
		const int total_work = threads * calls;
		if (total_work <= 0) return 0; // 实际由构造函数参数校验保证
		return (n + total_work - 1) / total_work;
	}

	int validate_threads(int value) {
		if (value <= 0) {
			spdlog::get("logger")->error("[ParallelPlan1d] Threads per block must be positive, but now = {}.", value);
			std::exit(EXIT_FAILURE);
		}
		return value;
	}

	int validate_calls(int value) {
		if (value <= 0) {
			spdlog::get("logger")->error("[ParallelPlan1d] Max calls per thread must be positive, but now = {}.", value);
			std::exit(EXIT_FAILURE);
		}
		return value;
	}

	int validate_Np(int value) {
		if (value <= 0) {
			spdlog::get("logger")->error("[ParallelPlan1d] Number of particles must be positive, but now = {}.", value);
			std::exit(EXIT_FAILURE);
		}
		return value;
	}

	const int threads_per_block_;
	const int max_calls_per_thread_;
	const int Np_;
	const int blocks_x_;
	const int total_threads_;
};


class ParallelPlan2d
{

public:
	ParallelPlan2d() = default;
	ParallelPlan2d(int threads_per_block_x, int threads_per_block_y,
		int max_calls_per_thread, int Np)
		: threads_x_(validate_threads(threads_per_block_x)),  // 假设x维度最大1024
		threads_y_(validate_threads(threads_per_block_y)),  // 假设y维度最大1024
		max_calls_per_thread_(validate_calls(max_calls_per_thread)),
		Np_(validate_Np(Np)),
		blocks_x_(calculate_blocks_x()),
		blocks_y_(calculate_blocks_y()),
		total_threads_(blocks_x_* blocks_y_* threads_x_* threads_y_) {
	}

	~ParallelPlan2d()
	{
		spdlog::get("logger")->info(
			"[ParallelPlan2d] class destructor: threads per block x/y={}/{}, blocks x/y={}/{}, max call time per thread={}, Np={}, total threads ={}",
			threads_x_, threads_y_, blocks_x_, blocks_y_, max_calls_per_thread_, Np_, total_threads_);
	};


	void print()
	{
		spdlog::get("logger")->info(
			"[ParallelPlan2d] threads per block x/y={}/{}, blocks x/y={}/{}, max call time per thread={}, Np={}, total threads ={}",
			threads_x_, threads_y_, blocks_x_, blocks_y_, max_calls_per_thread_, Np_, total_threads_);
	}

	int get_threads_x() const { return threads_x_; }
	int get_threads_y() const { return threads_y_; }
	int get_blocks_x() const { return blocks_x_; }
	int get_blocks_y() const { return blocks_y_; }

private:

	int calculate_total_blocks() const {
		cudaDeviceProp prop; //"cuda_runtime.h"
		cudaGetDeviceProperties(&prop, 0);
		int maxThreadsPerBlock = prop.maxThreadsPerBlock;

		if (threads_x_ * threads_y_ > maxThreadsPerBlock) {
			spdlog::get("logger")->error("[ParallelPlan2d] Threads per block x/y={}/{} exceed the maximum allowed ({})",
				threads_x_, threads_y_, maxThreadsPerBlock);
			std::exit(EXIT_FAILURE);
		}

		const int threads_per_block = threads_x_ * threads_y_;
		const int Np_per_block = threads_per_block * max_calls_per_thread_;
		if (Np_per_block <= 0) return 0;  // 由参数校验保证不会触发
		return (Np_ + Np_per_block - 1) / Np_per_block;
	}

	int calculate_blocks_x() const {
		const int total_blocks = calculate_total_blocks();

		// 目标：寻找接近平方根的块布局（优化GPU利用率）
		const int ideal_dim = static_cast<int>(std::ceil(std::sqrt(total_blocks)));
		return std::min(ideal_dim, total_blocks);  // 防止ideal_dim^2远大于实际需求
	}

	int calculate_blocks_y() const {
		const int total_blocks = calculate_total_blocks();

		const int blocks_x = calculate_blocks_x();
		return (total_blocks + blocks_x - 1) / blocks_x;  // 向上取整
	}


	int validate_threads(int value) {
		if (value <= 0) {
			spdlog::get("logger")->error("[ParallelPlan2d] Threads per block must be positive, but now = {}.", value);
			std::exit(EXIT_FAILURE);
		}
		return value;
	}

	int validate_calls(int value) {
		if (value <= 0) {
			spdlog::get("logger")->error("[ParallelPlan2d] Max calls per thread must be positive, but now = {}.", value);
			std::exit(EXIT_FAILURE);
		}
		return value;
	}

	int validate_Np(int value) {
		if (value <= 0) {
			spdlog::get("logger")->error("[ParallelPlan2d] Number of particles must be positive, but now = {}.", value);
			std::exit(EXIT_FAILURE);
		}
		return value;
	}

	const int threads_x_;
	const int threads_y_;
	const int max_calls_per_thread_;
	const int Np_;
	const int blocks_x_;
	const int blocks_y_;
	const int total_threads_;
};