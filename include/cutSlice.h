#pragma once

#include "command.h"
#include "particle.h"
#include "parameter.h"
#include "parallelPlan.h"
#include "general.h"


class SortBunch
{
public:
	SortBunch(const Parameter& para, int input_beamId, Bunch& Bunch, std::string obj_name, const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~SortBunch() = default;

	double s = -1;
	std::string commandType = "SortBunch";
	std::string name = "SortBunchObj";

	void execute(int turn);

	void print();


private:
	Particle* dev_bunch = nullptr;
	Particle* dev_bunch_tmp = nullptr;	// temporary particle array for sorting
	TimeEvent& simTime;
	Bunch& bunchRef;

	int Np = 0;
	int Np_sur = 0;

	double* dev_sort_z = nullptr;	// particle z position
	int* dev_sort_index = nullptr;	// particle index
	Slice* dev_slice = nullptr;		// slice information

	std::string sort_purpose;
	std::string slice_model;
	int Nslice = 0;

	int thread_x = 0;
	int block_x = 0;

	int* dev_survive_flags;   // mark survive£¨tag>=0£©
	int* dev_survive_prefix;  // sum of prefix of survive particles
	void* dev_cub_temp = nullptr;	// CUB tmporary memory
	size_t cub_temp_bytes = 0;      // size of CUB temporary memory

};


__global__ void reduction_z_avg(const Particle* dev_bunch, Slice* dev_slice, int Nslice);

__global__ void find_slice_indices(const double* sorted_z, int Np_sur, Slice* slices, int Nslice);

__global__ void mark_survive_particles(Particle* dev_bunch, int* flags, int Np);

__global__ void stable_partition(Particle* src, Particle* dst, int* valid_prefix, int Np, int Np_sur);

__global__ void setup_slice_euqal_particle(const Particle* dev_bunch, Slice* dev_slice, int Np_sur, int Nslice);

__global__ void setup_slice_euqal_length(const Particle* dev_bunch, Slice* dev_slice, int Np_sur, int Nslice);

__global__ void test_change_particle_tag(Particle* dev_bunch, int Np, int turn);

__global__ void show_slice_info(const Slice* dev_slice, int Nslice);

__device__ int find_slice_index(const Slice* dev_slice, int Nslice, int particle_index);