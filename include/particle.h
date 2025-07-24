#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

#include "config.h"
#include "parameter.h"


//// Align the memory to 64 bytes or 128 bytes
//#ifdef PASS_CAL_PHASE
//class alignas(128) Particle
//#else
//class alignas(64) Particle
//#endif


class Particle
{

public:
	Particle() = default;
	__host__ __device__ ~Particle() {};

	__host__ __device__ bool operator>(const Particle& other) const
	{
		return this->z > other.z;
	}

	double x = 0;
	double px = 0;
	double y = 0;
	double py = 0;
	double z = 0;
	double pz = 0;

	int tag = 0;		// lost flag, >0 means not lost, <0 means lost
	int lostTurn = -1;	// lost turn, -1 means not lost
	double lostPos = -1;	// lost position, -1 means not lost

#ifdef PASS_CAL_PHASE

	double last_x = 0;
	double last_y = 0;
	double last_px = 0;
	double last_py = 0;
	double phase_x = 0;
	double phase_y = 0;

#endif // PASS_CAL_PHASE

private:

};

//// 静态断言验证
//#ifdef PASS_CAL_PHASE
//static_assert(sizeof(Particle) == 128, "Size mismatch");
//#else
//static_assert(sizeof(Particle) == 64, "Size mismatch");
//#endif


struct Slice;	// Forward declaration of Slice struct (in cutSlice.h)
class FieldSolver;	// Forward declaration of FieldSolver class (in pic.h)

struct CycleRange {
	int start;
	int end;
	int step;
	int totalPoints;

	// 添加构造函数进行数据校验
	CycleRange(int s, int e, int st) : start(s), end(e), step(st) {
		if (st == 0) {
			throw std::invalid_argument("Step cannot be zero");
		}
		if ((st > 0 && s > e) || (st < 0 && s < e)) {
			throw std::invalid_argument("Invalid range direction");
		}

		if (step > 0)
		{
			int diff = end - start;
			totalPoints = diff / step + 1;
			end = start + (totalPoints - 1) * step;  // 确保end是最后一个点
		}
		else
		{
			int diff = start - end;
			totalPoints = diff / (-step) + 1;
			end = start + (totalPoints - 1) * step;  // 确保end是最后一个点
		}
	}

	// 检查值是否在范围内
	bool contains(int value) const {
		if (step > 0) {
			if (value < start || value > end) return false;
		}
		else {
			if (value > start || value < end) return false;
		}
		return (value - start) % step == 0;
	}

	// 判断是否是范围内的最后一个点
	bool isLastPoint(int value) const {
		return contains(value) && (value == end);
	}
};


class Bunch
{
public:
	Bunch() = default;
	Bunch(const Parameter& para, int input_beamId, int input_bunchId);
	~Bunch();

	int bunchId = 0;

	double Nrp = 0;	// Number of real particles
	int Np = 0;		// Number of macro particles
	int Np_sur = 0;	// Number of surviving macro particles
	double ratio = 0;	// Value of Nrp/Np

	int Nproton = 0;	// Number of protons per real particle
	int Nneutron = 0;	// Number of neutrons per real particle
	int Ncharge = 0;	// Number of charges per real particle. e.g. for electron beam, charge = -1, for positron/proton beam, charge = 1

	double m0 = 0;	// Static mass of a nucleon (eV/c2/u))
	double Ek = 0;	// Kinetic energy of a nucleon (eV)
	double p0 = 0;	// Momentum of a nucleon (eV/c/u)
	double p0_kg = 0;	// Momentum of a nucleon (kg*m/s/u)
	double beta = 0, gamma = 0;	// Relativistic velocity and Lorentz factor

	double mass = 0;	// Mass of a macro particle
	double charge = 0;	// Charge of a macro particle

	double Brho = 0;

	//double emitx = 0, emity = 0;	// Geometric emittance (rad'm)
	//double emitx_norm = 0, emity_norm = 0;	// Normalized emittance (rad'm)
	////double emitx_equi = 0, emity_equi = 0;	// Equilibrium emittance, used in synchrotron radiation (rad'm)

	//double alphax = 0, alphay = 0;	// Twiss parameters
	//double betax = 0, betay = 0;	// Twiss parameters
	//double gammax = 0, gammay = 0;	// Twiss parameters

	//double sigmax = 0, sigmay = 0, sigmaz = 0;	// RMS value of horizontal, vertical bunch size and bunch length (m)
	//double sigmapx = 0, sigmapy = 0, dp = 0;	// RMS value of horizontal, vertical divergence (rad) and deltap/p

	//double Qx = 0, Qy = 0, Qz = 0;	// Tunes
	//double chromx = 0, chromy = 0;	// Chromaticity

	double gammat = 0;

	// sigmaz and dp will be set in Injection initialize function
	double sigmaz = 0;
	double dp = 0;

	//int dampTurn = 0;	// Transverse damping turns

	Particle* dev_bunch = nullptr;
	Particle* dev_bunch_tmp = nullptr;	// Temporary buffer for sorting particles

	std::string dist_transverse;
	std::string dist_longitudinal;

	/********************** Parameters for sorting and cutting slice **********************/
	bool is_slice_for_sc = false;		// Whether to slice for space-charge effect
	bool is_slice_for_bb = false;		// Whether to slice for beam-beam effect

	double* dev_sort_z = nullptr;		// particle z position
	int* dev_sort_index = nullptr;		// particle index

	int Nslice_sc = 0;	// Number of slices in longitudinal distribution
	int Nslice_bb = 0;	// Number of slices in longitudinal distribution

	Slice* dev_slice_sc = nullptr;	// slice information
	Slice* dev_slice_bb = nullptr;	// slice information

	int* dev_survive_flags;   // mark survive（tag>=0）
	int* dev_survive_prefix;  // sum of prefix of survive particles
	void* dev_cub_temp = nullptr;	// CUB tmporary memory
	size_t cub_temp_bytes = 0;      // size of CUB temporary memory


	/********************** Parameters for particle monitor (PM) **********************/
	bool is_enableParticleMonitor = false;	// Whether to use particle monitor to save particles

	std::vector<CycleRange> saveTurn_PM;

	int Np_PM = 0;	// Number of particles to be saved
	int Nobs_PM = 0;	// Number of observe points
	int Nturn_PM = 0;	// Number of turns to save particles

	Particle* dev_PM = nullptr;


	/********************** Parameters for space charge (sc) **********************/
	bool is_enable_spaceCharge = false;

	std::vector<std::shared_ptr<FieldSolver>> solver_sc;

private:

};
