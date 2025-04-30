#pragma once

#include "command.h"
#include "particle.h"
#include "parameter.h"
#include "general.h"
#include "parallelPlan.h"


class Monitor
{
public:

	virtual ~Monitor() = default;

	double s = -1;
	std::string commandType = "Monitor";
	std::string name = "MonitorObj";

	virtual void execute(int turn) = 0;
	virtual void print() = 0;

};


class DistMonitor :public Monitor
{
public:
	DistMonitor(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name, TimeEvent& timeevent);

	// 禁用拷贝构造函数与拷贝赋值运算符
	DistMonitor(const DistMonitor&) = delete;
	DistMonitor& operator=(const DistMonitor&) = delete;

	// 添加移动构造函数
	DistMonitor(DistMonitor&& other)noexcept
		:Np(other.Np), bunchId(other.bunchId), saveDir(other.saveDir), saveName_part(other.saveName_part), simTime(other.simTime),
		dev_bunch(other.dev_bunch), host_bunch(other.host_bunch)
	{
		other.host_bunch = nullptr;
		other.dev_bunch = nullptr;
		spdlog::get("logger")->debug("[DistMonitor] class move constructor: {}.", name);
	}

	// 添加移动赋值运算符
	DistMonitor& operator=(DistMonitor&& other)noexcept
	{
		if (this != &other)
		{
			Np = other.Np;
			bunchId = other.bunchId;
			saveDir = other.saveDir;
			saveName_part = other.saveName_part;
			simTime = other.simTime;

			delete[] host_bunch; // 释放原有的内存
			host_bunch = other.host_bunch;
			other.host_bunch = nullptr;

			dev_bunch = other.dev_bunch;
			other.dev_bunch = nullptr;

			spdlog::get("logger")->debug("[DistMonitor] class move assignment operator: {}.", name);
		}
		return *this;
	}

	~DistMonitor() override {
		delete[] host_bunch;
		spdlog::get("logger")->debug("[DistMonitor] class destructor: {}.", name);
	};

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[Distribution Monitor] print");
	}

private:
	int Np = 0;

	int bunchId = 0;
	std::filesystem::path saveDir;
	std::string saveName_part;

	TimeEvent& simTime;

	Particle* dev_bunch = nullptr;
	Particle* host_bunch = nullptr;
};


class StatMonitor :public Monitor
{
public:
	StatMonitor(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name, const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	// 禁用拷贝构造函数与拷贝赋值运算符
	StatMonitor(const StatMonitor&) = delete;
	StatMonitor& operator=(const StatMonitor&) = delete;

	// 添加移动构造函数
	StatMonitor(StatMonitor&& other)noexcept
		:Nstat(other.Nstat), pitch_statistic(other.pitch_statistic), Np(other.Np), thread_x(other.thread_x), block_x(other.block_x), bunchId(other.bunchId),
		saveDir(other.saveDir), saveName_part(other.saveName_part), simTime(other.simTime),
		dev_bunch(other.dev_bunch), dev_statistic(other.dev_statistic), host_statistic(other.host_statistic)
	{
		other.dev_bunch = nullptr;
		other.dev_statistic = nullptr;
		other.host_statistic = nullptr;
		spdlog::get("logger")->debug("[StatMonitor] class move constructor: {}.", name);
	}

	// 添加移动赋值运算符
	StatMonitor& operator=(StatMonitor&& other)noexcept
	{
		if (this != &other)
		{
			Nstat = other.Nstat;
			pitch_statistic = other.pitch_statistic;
			Np = other.Np;
			thread_x = other.thread_x;
			block_x = other.block_x;
			bunchId = other.bunchId;
			saveDir = other.saveDir;
			saveName_part = other.saveName_part;
			simTime = other.simTime;

			callCuda(cudaFreeHost(host_statistic)); // 释放原有的内存（对于cpu端new的数组，使用delete[]的方法没问题，但对于gpu端声明的数组，尚不清楚使用cudaFree是否可行，存疑。不过目前还不会用到移动赋值运算符）
			host_statistic = other.host_statistic;
			other.host_statistic = nullptr;

			callCuda(cudaFree(dev_statistic));
			dev_statistic = other.dev_statistic;
			other.dev_statistic = nullptr;

			dev_bunch = other.dev_bunch;
			other.dev_bunch = nullptr;

			spdlog::get("logger")->debug("[StatMonitor] class move assignment operator: {}.", name);
		}
		return *this;
	}

	~StatMonitor() override {
		callCuda(cudaFreeHost(host_statistic));
		callCuda(cudaFree(dev_statistic));
		spdlog::get("logger")->debug("[StatMonitor] class destructor: {}.", name);
	};

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[StatMonitor Monitor] print");
	}

private:
	int Nstat = 22;
	// host_statistic[22]
	// 0:x, 1:x^2, 2:x*px, 3:px^2
	// 4:y, 5:y^2, 6:y*py, 7:py^2
	// 8:beam loss
	// 9:z^2, 10:pz^2
	// 11:z, 12:pz
	// 13:x', 14:y'
	// 15:xz, 16:xy, 17:yz
	// 18:x^3, 19:x^4, 20:y^3, 21:y^4

	size_t pitch_statistic;

	int Np = 0;

	int thread_x = 0;
	int block_x = 0;

	int bunchId = 0;
	std::filesystem::path saveDir;
	std::string saveName_part;

	TimeEvent& simTime;

	Particle* dev_bunch = nullptr;
	double* dev_statistic = nullptr;
	double* host_statistic = nullptr;
};


__global__ void cal_statistic_perblock(Particle* dev_bunch, double* dev_statistic, size_t pitch_statistic, int NpPerBunch);

__global__ void cal_statistic_allblock_2(double* dev_statistic, size_t pitch_statistic, int gridDimX, int NpInit);

void save_bunchInfo_statistic(double* host_statistic, int Np, std::filesystem::path saveDir, std::string saveName_part, int turn);