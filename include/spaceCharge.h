#pragma once

#include "command.h"
#include "particle.h"
#include "parameter.h"
#include "parallelPlan.h"
#include "general.h"
#include "constant.h"
#include "pic.h"
#include "aperture.h"


class SpaceCharge {
public:
	SpaceCharge(const Parameter& para, int input_beamId, Bunch& Bunch, std::string obj_name, const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~SpaceCharge() = default;

	double s = -1;
	std::string commandType = "SpaceCharge";
	std::string name = "SpaceChargeObj";

	void execute(int turn);

	void print();

private:
	Particle* dev_bunch = nullptr;
	TimeEvent& simTime;
	Bunch& bunchRef;
	Slice* dev_slice = nullptr;

	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool is_enable_spaceCharge = false;		// Flag to enable/disable space charge
	double sc_length = 0.0;			// Length of the space charge region
	int Ncharge = 0;	// Number of charges per real particle
	double ratio = 0.0;		// Nrp/Np
	double qm_ratio = 0.0;	// Charge/mass ratio of a particle (q/m)

	std::shared_ptr<FieldSolver> solver = nullptr;

};


__global__ void cal_spaceCharge_kick(Particle* dev_bunch, const double2* dev_E, const Slice* dev_slice,
	int Np_sur, int Nx, int Ny, double Lx, double Ly, int Nslice, double sc_factor);