#pragma once

#include "command.h"
#include "particle.h"
#include "parameter.h"
#include "parallelPlan.h"
#include "general.h"
#include "constant.h"

class Twiss
{
public:
	Twiss(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name, const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	double s = -1;
	std::string commandType = "Twiss";
	std::string name = "TwissObj";

	void execute(int turn);

	void print();

private:
	Particle dev_particle;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	double s_previous = -1;

	std::string longitudinal_transfer = "off";
	// longitudinal parameters for point-to-point transfer
	double gamma = 0;
	double gammat = 0;
	// longitudinal parameters for one-turn map
	double sigmaz = 0;
	double dp = 0;

	int Np = 0;
	double circumference = 0;

	// Twiss parameters of current position
	double alphax = 0;
	double alphay = 0;

	double betax = 0;
	double betay = 0;

	double mux = 0;
	double muy = 0;
	double muz = 0;

	double Dx = 0;
	double Dpx = 0;

	double DQx = 0;
	double DQy = 0;

	// Twiss parameters of previos position
	double alphax_previous = 0;
	double alphay_previous = 0;

	double betax_previous = 0;
	double betay_previous = 0;

	double mux_previous = 0;
	double muy_previous = 0;
	double muz_previous = 0;

	double Dx_previous = 0;
	double Dpx_previous = 0;

	double phi_x = 0;
	double phi_y = 0;
	double phi_z = 0;

	double m11_z = 0, m12_z = 0, m21_z = 0, m22_z = 0;

	int thread_x = 0;
	int block_x = 0;
};


__global__ void transfer_matrix_6D(Particle dev_particle, int Np, double circumference, int turn, double s,
	double betax, double betax_previous, double alphax, double alphax_previous,
	double betay, double betay_previous, double alphay, double alphay_previous,
	double phix, double phiy, double DQx, double DQy,
	double Dx, double Dx_previous, double Dpx, double Dpx_previous,
	double m11_z, double m12_z, double m21_z, double m22_z);