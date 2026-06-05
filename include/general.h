#pragma once

#include <cmdline/cmdline.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <stdlib.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

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

void print_logo();

void print_copyright();

void get_cmd_input(int argc, char** argv, std::vector<std::filesystem::path>& path_input_para, std::string& yearMonDay, std::string& hourMinSec);

void create_logger(std::string logfile_path);

void info_centered(std::shared_ptr<spdlog::logger> logger, const std::string& text, int total_width = 60);

void show_device_info();

std::string timeStamp();

struct CycleRange
{
	int start;
	int end;
	int step;
	int totalPoints;

	// data validation
	CycleRange(int s, int e, int st) : start(s), end(e), step(st)
	{
		if (st == 0)
		{
			throw std::invalid_argument("Step cannot be zero");
		}
		if ((st > 0 && s > e) || (st < 0 && s < e))
		{
			throw std::invalid_argument("Invalid range direction");
		}

		if (step > 0)
		{
			int diff = end - start;
			totalPoints = diff / step + 1;
			end = start + (totalPoints - 1) * step;	 // Ensure end is the last point in the range
		}
		else
		{
			int diff = start - end;
			totalPoints = diff / (-step) + 1;
			end = start + (totalPoints - 1) * step;	 // Ensure end is the last point in the range
		}
	}

	// Check if a value is within the range and is one of the points in the range
	bool contains(int value) const
	{
		if (step > 0)
		{
			if (value < start || value > end) return false;
		}
		else
		{
			if (value > start || value < end) return false;
		}
		return (value - start) % step == 0;
	}

	// Check if a value is the last point in the range
	bool isLastPoint(int value) const { return contains(value) && (value == end); }

	// Check if a value is the first point in the range
	bool isFirstPoint(int value) const { return contains(value) && (value == start); }
};

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

// Decide whether the value is in any of the given turn ranges
bool is_value_in_turn_ranges(int value, const std::vector<CycleRange>& ranges);

// Decide whether the value is in any of the given turn ranges and return the index of the range
bool is_value_in_turn_ranges(int value, const std::vector<CycleRange>& ranges, int& index);

// Decide whether the value is the first point in any of the given turn ranges
bool is_value_firstPoint_in_turn_ranges(int value, const std::vector<CycleRange>& ranges);

// Decide whether the value is the last point in any of the given turn ranges
bool is_value_lastPoint_in_turn_ranges(int value, const std::vector<CycleRange>& ranges);

void print_cycleRange(const std::vector<CycleRange>& ranges);

std::string ms_to_timeString(double ms);

std::vector<std::vector<double>> loadtxt(const std::string& filename, char delimiter = ',', int skiprows = 0, const std::vector<int>& usecols = {});

std::vector<std::string> split_line(const std::string& line);

std::vector<std::vector<double>> read_file_data(const std::string& file_path);

// Decide whether two values are approximately equal, using relative error for floating-point numbers and direct comparison for other types. The eps
// parameter specifies the tolerance, defaulting to 1e-12.
template <typename T1, typename T2>
inline bool approx_equal(T1 a, T2 b, std::common_type_t<T1, T2> eps = 1e-12)
{
	using Float = std::common_type_t<T1, T2, double>;
	if constexpr (std::is_floating_point_v<Float>)
	{
		Float fa = static_cast<Float>(a);
		Float fb = static_cast<Float>(b);
		return std::fabs(fa - fb) <= eps * std::max({Float(1), std::fabs(fa), std::fabs(fb)});
	}
	else
	{
		return a == b;
	}
}

// Linear interpolation function, input is x, y arrays and the x0 to be interpolated, output is the corresponding y0. y is multi-dimensional, each x
// corresponds to a y vector, so y0 is also a vector. Generally, x is time and y is particle beam parameters.
template <typename XType, typename YType, typename XQueryType>
std::vector<YType> linearInterpolate(const std::vector<XType>& xs, const std::vector<std::vector<YType>>& ys, XQueryType x0)
{
	if (xs.size() < 2) throw std::invalid_argument("linearInterpolate: need at least two points");

	if (xs.size() != ys.size()) throw std::invalid_argument("linearInterpolate: xs and ys size mismatch");

	if (ys.empty() || ys[0].empty()) return {};

	const size_t dim = ys[0].size();
	for (const auto& yv : ys)
	{
		if (yv.size() != dim) throw std::invalid_argument("linearInterpolate: inconsistent y dimensions");
	}

	assert(std::is_sorted(xs.begin(), xs.end()));

	using Float = std::common_type_t<XType, XQueryType, double>;

	const Float x = static_cast<Float>(x0);
	const Float xMin = static_cast<Float>(xs.front());
	const Float xMax = static_cast<Float>(xs.back());

	if (x < xMin || x > xMax) throw std::out_of_range("linearInterpolate: x0 out of range");

	if (approx_equal(x, xMin)) return ys.front();

	if (approx_equal(x, xMax)) return ys.back();

	auto it = std::lower_bound(xs.begin(), xs.end(), x0, [](const XType& lhs, const XQueryType& rhs) { return lhs < rhs; });

	size_t idx = static_cast<size_t>(it - xs.begin());

	if (idx < xs.size() && approx_equal(static_cast<Float>(xs[idx]), x)) return ys[idx];

	if (idx == 0 || idx >= xs.size()) throw std::logic_error("linearInterpolate: invalid lower_bound result");

	size_t left = idx - 1;
	size_t right = idx;

	const Float xL = static_cast<Float>(xs[left]);
	const Float xR = static_cast<Float>(xs[right]);

	if (approx_equal(xL, xR)) return ys[left];

	const Float t = (x - xL) / (xR - xL);

	const auto& yL = ys[left];
	const auto& yR = ys[right];

	std::vector<YType> result(dim);
	for (size_t i = 0; i < dim; ++i)
	{
		result[i] = static_cast<YType>(yL[i] + t * (yR[i] - yL[i]));
	}

	return result;
}

std::string to_lower(const std::string& str);

std::string to_upper(const std::string& str);

const double brent(const std::function<double(double)>& func, const double x1, const double x2, const double tol = 1e-10, const int iter_max = 1000);

// Use trapezoidal method to calculate the integral of 1D numerical function. The arguments are the function, lower limit, upper limit, and the
// intervals' number.
const double trapz(const std::function<double(double)>& func, const double a, const double b, const int n = 1000);

// Use trapezoidal method to calculate the integral of 2D numerical function. The arguments are the function, lower limit, upper limit, and the
// intervals' number.
const double trapz2d(const std::function<double(double, double)>& func, const std::function<double(double)>& funcy1,
					 const std::function<double(double)>& funcy2, const double a, const double b, const int n = 1000);
