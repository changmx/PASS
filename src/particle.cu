#include <cub/cub.cuh>
#include <fstream>
#include <iostream>

#include "command.h"
#include "constant.h"
#include "cutSlice.h"
#include "parameter.h"
#include "particle.h"
#include "pic.h"

// #include "amgx_c.h"

__host__ void Particle::mem_allocate_gpu(size_t n)
{
	// Allocate memory on gpu device
	callCuda(cudaMalloc((void**)&x, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&px, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&y, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&py, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&z, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&pz, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&lostPos, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&tag, n * sizeof(int)));
	callCuda(cudaMalloc((void**)&lostTurn, n * sizeof(int)));
	callCuda(cudaMalloc((void**)&sliceId, n * sizeof(int)));

	callCuda(cudaMemset(x, 0, n * sizeof(double)));
	callCuda(cudaMemset(px, 0, n * sizeof(double)));
	callCuda(cudaMemset(y, 0, n * sizeof(double)));
	callCuda(cudaMemset(py, 0, n * sizeof(double)));
	callCuda(cudaMemset(z, 0, n * sizeof(double)));
	callCuda(cudaMemset(pz, 0, n * sizeof(double)));
	callCuda(cudaMemset(lostPos, -1, n * sizeof(double)));	// Initialize lostPos to -1
	callCuda(cudaMemset(tag, 0, n * sizeof(int)));			// Initialize tag to 0
	callCuda(cudaMemset(lostTurn, -1, n * sizeof(int)));	// Initialize lostTurn to -1
	callCuda(cudaMemset(sliceId, 0, n * sizeof(int)));		// Initialize sliceId to 0

#ifdef PASS_CAL_PHASE
	callCuda(cudaMalloc((void**)&last_x, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&last_y, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&last_px, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&last_py, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&phase_x, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&phase_y, n * sizeof(double)));

	callCuda(cudaMemset(last_x, 0, n * sizeof(double)));
	callCuda(cudaMemset(last_y, 0, n * sizeof(double)));
	callCuda(cudaMemset(last_px, 0, n * sizeof(double)));
	callCuda(cudaMemset(last_py, 0, n * sizeof(double)));
	callCuda(cudaMemset(phase_x, 0, n * sizeof(double)));
	callCuda(cudaMemset(phase_y, 0, n * sizeof(double)));
#endif
}

__host__ void Particle::mem_allocate_cpu(size_t n)
{
	// Allocate memory on host
	x = new double[n]{};
	px = new double[n]{};
	y = new double[n]{};
	py = new double[n]{};
	z = new double[n]{};
	pz = new double[n]{};
	lostPos = new double[n]{};
	tag = new int[n]{};
	lostTurn = new int[n]{};
	sliceId = new int[n]{};
#ifdef PASS_CAL_PHASE
	last_x = new double[n]{};
	last_y = new double[n]{};
	last_px = new double[n]{};
	last_py = new double[n]{};
	phase_x = new double[n]{};
	phase_y = new double[n]{};
#endif
}

__host__ void Particle::mem_free_gpu()
{
	// Release device memory
	callCuda(cudaFree(x));
	callCuda(cudaFree(px));
	callCuda(cudaFree(y));
	callCuda(cudaFree(py));
	callCuda(cudaFree(z));
	callCuda(cudaFree(pz));
	callCuda(cudaFree(lostPos));
	callCuda(cudaFree(tag));
	callCuda(cudaFree(lostTurn));
	callCuda(cudaFree(sliceId));

#ifdef PASS_CAL_PHASE
	callCuda(cudaFree(last_x));
	callCuda(cudaFree(last_y));
	callCuda(cudaFree(last_px));
	callCuda(cudaFree(last_py));
	callCuda(cudaFree(phase_x));
	callCuda(cudaFree(phase_y));
#endif
}

__host__ void Particle::mem_free_cpu()
{
	// Release host memory
	delete[] x;
	delete[] px;
	delete[] y;
	delete[] py;
	delete[] z;
	delete[] pz;
	delete[] lostPos;
	delete[] tag;
	delete[] lostTurn;
	delete[] sliceId;

#ifdef PASS_CAL_PHASE
	delete[] last_x;
	delete[] last_y;
	delete[] last_px;
	delete[] last_py;
	delete[] phase_x;
	delete[] phase_y;
#endif
}

void particle_copy(Particle dst, Particle src, size_t n, cudaMemcpyKind kind, std::string type, int dst_offset, int src_offset)
{
	if ("all" == type)
	{
		callCuda(cudaMemcpy(dst.x + dst_offset, src.x + src_offset, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.px + dst_offset, src.px + src_offset, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.y + dst_offset, src.y + src_offset, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.py + dst_offset, src.py + src_offset, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.z + dst_offset, src.z + src_offset, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.pz + dst_offset, src.pz + src_offset, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.lostPos + dst_offset, src.lostPos + src_offset, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.tag + dst_offset, src.tag + src_offset, n * sizeof(int), kind));
		callCuda(cudaMemcpy(dst.lostTurn + dst_offset, src.lostTurn + src_offset, n * sizeof(int), kind));
		callCuda(cudaMemcpy(dst.sliceId + dst_offset, src.sliceId + src_offset, n * sizeof(int), kind));
#ifdef PASS_CAL_PHASE
		callCuda(cudaMemcpy(dst.last_x + dst_offset, src.last_x + src_offset, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.last_y + dst_offset, src.last_y + src_offset, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.last_px + dst_offset, src.last_px + src_offset, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.last_py + dst_offset, src.last_py + src_offset, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.phase_x + dst_offset, src.phase_x + src_offset, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.phase_y + dst_offset, src.phase_y + src_offset, n * sizeof(double), kind));
#endif
	}
	else if ("dist" == type)
	{
		callCuda(cudaMemcpy(dst.x + dst_offset, src.x + src_offset, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.px + dst_offset, src.px + src_offset, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.y + dst_offset, src.y + src_offset, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.py + dst_offset, src.py + src_offset, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.z + dst_offset, src.z + src_offset, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.pz + dst_offset, src.pz + src_offset, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.lostPos + dst_offset, src.lostPos + src_offset, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.tag + dst_offset, src.tag + src_offset, n * sizeof(int), kind));
		callCuda(cudaMemcpy(dst.lostTurn + dst_offset, src.lostTurn + src_offset, n * sizeof(int), kind));
		callCuda(cudaMemcpy(dst.sliceId + dst_offset, src.sliceId + src_offset, n * sizeof(int), kind));
	}
	else if ("phase" == type)
	{
		callCuda(cudaMemcpy(dst.tag + dst_offset, src.tag + src_offset, n * sizeof(int), kind));
#ifdef PASS_CAL_PHASE
		callCuda(cudaMemcpy(dst.phase_x + dst_offset, src.phase_x + src_offset, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.phase_y + dst_offset, src.phase_y + src_offset, n * sizeof(double), kind));
#endif
	}
	else
	{
		spdlog::get("logger")->error("[particle_copy] Unknown particle copy type: {}, we support all, dist and phase now.", type);
		std::exit(EXIT_FAILURE);
	}
}