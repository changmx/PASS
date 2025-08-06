#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <filesystem>

#include <cmdline/cmdline.h>
#include <tabulate/tabulate.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

#include "parameter.h"
#include "particle.h"

#ifndef STRNCASECMP
#define STRNCASECMP _strnicmp
#endif

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define MY_SYSTEM "Windows"
#elif __linux__
#define MY_SYSTEM "Linux"
#elif __APPLE__
printf("\nSorry, we don't support Apple system now.\n");
exit(1);
#elif __unix__
printf("\nSorry, we don't support Unix system now.\n");
exit(1);
#else
printf("\nUnknown runtime system.\n");
exit(1);
#endif

void print_logo(const Parameter& Para);

void print_copyright(const Parameter& Para);

void get_cmd_input(int argc, char** argv, std::vector<std::filesystem::path>& path_input_para, std::string& yearMonDay, std::string& hourMinSec);

void print_config_parameter(const Parameter&);

void print_beam_parameter(const Parameter& Para, const std::vector<Bunch>& Beam0, const std::vector<Bunch>& Beam1);

void create_logger(const Parameter& Para);

void show_device_info();

std::string timeStamp();

class TimeEvent
{
public:

	// Save the time used in the following steps
	// In units of ms.
	float allocate2gridSC = 0;
	float calBoundarySC = 0;
	float calPotentialSC = 0;
	float calElectricSC = 0;
	float calKickSC = 0;

	float allocate2gridBB = 0;
	float calBoundaryBB = 0;
	float calPotentialBB = 0;
	float calElectricBB = 0;
	float calKickBB = 0;

	float sort = 0;
	float slice = 0;
	float hourGlass = 0;
	float calPhase = 0;
	float statistic = 0;
	float crossingAngle = 0;
	float crabCavity = 0;
	float floatWaist = 0;
	float oneTurnMap = 0;
	float transferFixPoint = 0;
	float saveStatistic = 0;
	float savePhase = 0;
	float saveBunch = 0;
	float saveFixpoint = 0;
	float saveLuminosity = 0;

	float twiss = 0;
	float transferElement = 0;

	float total = 0;

	float turn = 0;

	bool isTime;

	cudaEvent_t start, stop;
	cudaEvent_t startPerTurn, stopPerTurn;

	void initial(int deviceid, bool timeOrNot);
	void free(int deviceid);
	TimeEvent& add(const TimeEvent& rhs);
	void print(int totalTurn, double cpuTime, int deviceid);

private:

};


// 判断值是否存在于任意一个循环范围中
bool is_value_in_turn_ranges(int value, const std::vector<CycleRange>& ranges);

// 判断值是否存在于任意一个循环范围中，并返回所在循环的索引
bool is_value_in_turn_ranges(int value, const std::vector<CycleRange>& ranges, int& index);

// 判断值是否为任意一个循环范围的起始点
bool is_value_firstPoint_in_turn_ranges(int value, const std::vector<CycleRange>& ranges);

// 判断值是否为任意一个循环范围的结束点
bool is_value_lastPoint_in_turn_ranges(int value, const std::vector<CycleRange>& ranges);

void print_cycleRange(const std::vector<CycleRange>& ranges);

std::string ms_to_timeString(double ms);