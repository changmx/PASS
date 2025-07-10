#pragma once

#include "command.h"
#include "particle.h"
#include "parameter.h"
#include "parallelPlan.h"
#include "general.h"
#include "constant.h"
#include "pic.h"
#include "aperture.h"
#include "pic.h"


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

	int Np_sur = 0;
	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool is_enable_spaceCharge = false;		// Flag to enable/disable space charge
	double scLength = 0.0;			// Length of the space charge region

	std::shared_ptr<PicConfig> picConfig = nullptr;

};


