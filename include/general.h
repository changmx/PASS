#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <stdlib.h>
#include <filesystem>
#include <cmath>
#include <type_traits>
#include <cassert>
#include <cctype>
#include <functional>

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

std::vector<std::vector<double>> loadtxt(const std::string& filename, char delimiter = ',', int skiprows = 0, const std::vector<int>& usecols = {});

std::vector<std::string> split_line(const std::string& line);

std::vector<std::vector<double>> read_file_data(const std::string& file_path);

// 判断是否相等（对于浮点数，使用相对误差进行比较；对于其他类型，直接比较）。eps是相对误差的容忍度，默认为1e-10。
template<typename T1, typename T2>
inline bool approx_equal(T1 a, T2 b, std::common_type_t<T1, T2> eps = 1e-12)
{
	using Float = std::common_type_t<T1, T2, double>;
	if constexpr (std::is_floating_point_v<Float>) {
		Float fa = static_cast<Float>(a);
		Float fb = static_cast<Float>(b);
		return std::fabs(fa - fb) <= eps * std::max({ Float(1), std::fabs(fa), std::fabs(fb) });
	}
	else {
		return a == b;
	}
}

// 线性插值函数，输入为x和y的向量，以及需要插值的x0，输出为对应的y0。y可以是多维的，即每个x对应一个y向量，输出的y0也是一个向量。例如x为时间，y为多个参数。
template<typename XType, typename YType, typename XQueryType>
std::vector<YType> linearInterpolate(
	const std::vector<XType>& xs,
	const std::vector<std::vector<YType>>& ys,
	XQueryType x0)
{
	if (xs.size() < 2)
		throw std::invalid_argument("linearInterpolate: need at least two points");

	if (xs.size() != ys.size())
		throw std::invalid_argument("linearInterpolate: xs and ys size mismatch");

	if (ys.empty() || ys[0].empty())
		return {};

	const size_t dim = ys[0].size();
	for (const auto& yv : ys) {
		if (yv.size() != dim)
			throw std::invalid_argument("linearInterpolate: inconsistent y dimensions");
	}

	assert(std::is_sorted(xs.begin(), xs.end()));

	using Float = std::common_type_t<XType, XQueryType, double>;

	const Float x = static_cast<Float>(x0);
	const Float xMin = static_cast<Float>(xs.front());
	const Float xMax = static_cast<Float>(xs.back());

	if (x < xMin || x > xMax)
		throw std::out_of_range("linearInterpolate: x0 out of range");

	if (approx_equal(x, xMin))
		return ys.front();

	if (approx_equal(x, xMax))
		return ys.back();

	auto it = std::lower_bound(
		xs.begin(), xs.end(), x0,
		[](const XType& lhs, const XQueryType& rhs) {
			return lhs < rhs;
		});

	size_t idx = static_cast<size_t>(it - xs.begin());

	if (idx < xs.size() && approx_equal(static_cast<Float>(xs[idx]), x))
		return ys[idx];

	if (idx == 0 || idx >= xs.size())
		throw std::logic_error("linearInterpolate: invalid lower_bound result");

	size_t left = idx - 1;
	size_t right = idx;

	const Float xL = static_cast<Float>(xs[left]);
	const Float xR = static_cast<Float>(xs[right]);

	if (approx_equal(xL, xR))
		return ys[left];

	const Float t = (x - xL) / (xR - xL);

	const auto& yL = ys[left];
	const auto& yR = ys[right];

	std::vector<YType> result(dim);
	for (size_t i = 0; i < dim; ++i) {
		result[i] = static_cast<YType>(
			yL[i] + t * (yR[i] - yL[i])
			);
	}

	return result;
}


std::string to_lower(const std::string& str);

std::string to_upper(const std::string& str);

const double brent(const std::function<double(double)>& func, const double x1, const double x2, const double tol = 1e-10, const int iter_max = 1000);