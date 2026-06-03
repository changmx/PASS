#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "command.h"
#include "config.h"
#include "general.h"
#include "parameter.h"

class Particle
{
   public:
	// Particle attributes
	double* __restrict__ x = nullptr;		 // horizontal position (m)
	double* __restrict__ px = nullptr;		 // horizontal normalized momentum (rad, px/p0)
	double* __restrict__ y = nullptr;		 // vertical position (m)
	double* __restrict__ py = nullptr;		 // vertical normalized momentum (rad, py/p0)
	double* __restrict__ z = nullptr;		 // longitudinal position relative to the ideal particle (m)
	double* __restrict__ pz = nullptr;		 // relative momentum deviation (dp/p0)
	double* __restrict__ lostPos = nullptr;	 // position where the particle is lost (m)
	int* __restrict__ tag = nullptr;		 // particle tag, start from 1, each particle has a unique tag
	int* __restrict__ lostTurn = nullptr;	 // turn when the particle is lost
	int* __restrict__ sliceId = nullptr;	 // slice ID of the particle in longitudinal direction

#ifdef PASS_CAL_PHASE
	double* __restrict__ last_x = nullptr;	 // horizontal position at the phase advance monitor at the previous turn (m)
	double* __restrict__ last_px = nullptr;	 // horizontal normalized momentum at the phase advance monitor at the previous turn (rad, px/p0)
	double* __restrict__ last_y = nullptr;	 // vertical position at the phase advance monitor at the previous turn (m)
	double* __restrict__ last_py = nullptr;	 // vertical normalized momentum at the phase advance monitor at the previous turn (rad, py/p0)
	double* __restrict__ phase_x = nullptr;	 // accumulated horizontal phase advance
	double* __restrict__ phase_y = nullptr;	 // accumulated vertical phase advance
#endif

	~Particle() {};

	// Memory manegement
	__host__ void mem_allocate_gpu(size_t n);
	__host__ void mem_allocate_cpu(size_t n);
	__host__ void mem_free_gpu();
	__host__ void mem_free_cpu();

   private:
};

void particle_copy(Particle dst, Particle src, size_t n, cudaMemcpyKind kink, std::string type, int dst_offset = 0, int src_offset = 0);