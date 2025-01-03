#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "config.h"

class Particle
{
public:
	Particle() = default;
	__host__ __device__ ~Particle() {};

	__host__ __device__ bool operator<(const Particle& a) const
	{
		return z > a.z;
	}

	double x = 0;
	double px = 0;
	double y = 0;
	double py = 0;
	double z = 0;
	double pz = 0;

	int tag = 0;
	int lostTurn = 0;

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

Particle::Particle()
{
}

Particle::~Particle()
{
}


