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

};


__global__ void reduction_z_avg(const Particle* dev_bunch, Slice* dev_slice, int Nslice);