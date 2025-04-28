#pragma once

#include "command.h"
#include "particle.h"
#include "parameter.h"
#include "general.h"


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