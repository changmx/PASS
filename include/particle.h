#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

#include "config.h"
#include "parameter.h"
#include "command.h"


class Particle
{
public:
	// Particle attributes
	double* __restrict__ x = nullptr;
	double* __restrict__ px = nullptr;
	double* __restrict__ y = nullptr;
	double* __restrict__ py = nullptr;
	double* __restrict__ z = nullptr;
	double* __restrict__ pz = nullptr;
	double* __restrict__ lostPos = nullptr;
	int* __restrict__ tag = nullptr;
	int* __restrict__ lostTurn = nullptr;
	int* __restrict__ sliceId = nullptr;

#ifdef PASS_CAL_PHASE
	double* __restrict__ last_x = nullptr;
	double* __restrict__ last_y = nullptr;
	double* __restrict__ last_px = nullptr;
	double* __restrict__ last_py = nullptr;
	double* __restrict__ phase_x = nullptr;
	double* __restrict__ phase_y = nullptr;
#endif

	~Particle() {};

	// Memory manegement
	__host__ void mem_allocate_gpu(size_t n);
	__host__ void mem_allocate_cpu(size_t n);
	__host__ void mem_free_gpu();
	__host__ void mem_free_cpu();

private:

};

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

	// 判断是否是范围内的第一个点
	bool isFirstPoint(int value) const {
		return contains(value) && (value == start);
	}
};


class Bunch
{
public:
	Bunch() = default;
	Bunch(const Parameter& para, int input_beamId, int input_bunchId);
	~Bunch();

	Particle dev_particle;
	Particle dev_particle_tmp;

	int bunchId = 0;

	double Nrp = 0;	// Number of real particles
	int Np = 0;		// Number of macro particles
	int Np_sur = 0;	// Number of surviving macro particles
	double ratio = 0;	// Value of Nrp/Np
	double qm_ratio = 0;	// Charge/mass ratio of a particle (e/m)

	/* Example of Nproton, Nneutron, Ncharge:
	*		for electron beam: Nproton = 0, Nneutron = 0, Ncharge = -1
	*		for positron beam: Nproton = 0, Nneutron = 0, Ncharge = +1
	*		for proton   beam: Nproton = 1, Nneutron = 0, Ncharge = +1
	*		for 238U35+  beam: Nproton = 92, Nneutron = 146, Ncharge = +35
	*/
	int Nproton = 0;	// Number of protons per real particle
	int Nneutron = 0;	// Number of neutrons per real particle
	int Ncharge = 0;	// Number of charges per real particle. charge of a macro particle is Ncharge*ratio, charge of a nucluon is Ncharge*qm_ration*ratio

	double m0 = 0;	// Static mass of a nucleon (eV/c2/u))
	double Ek = 0;	// Kinetic energy of a nucleon (eV/u)
	double p0 = 0;	// Momentum of a nucleon (eV/c/u)
	double p0_kg = 0;	// Momentum of a nucleon (kg*m/s/u)
	double beta = 0, gamma = 0;	// Relativistic velocity and Lorentz factor

	double Brho = 0;

	double gammat = 0;

	// sigmaz and dp will be set in Injection initialize function
	double sigmaz = 0;
	double dp = 0;

	//int dampTurn = 0;	// Transverse damping turns

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

	Particle dev_PM;


	/********************** Parameters for space charge (sc) **********************/
	bool is_enable_spaceCharge = false;

	std::vector<std::shared_ptr<FieldSolver>> solver_sc;

private:

};


void particle_copy(Particle dst, Particle src, size_t n, cudaMemcpyKind kink, std::string type);