#pragma once

#include "command.h"
#include "particle.h"
#include "parameter.h"
#include "parallelPlan.h"

class Twiss
{
public:
	Twiss(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name, const ParallelPlan1d& plan1d);

	double s = -1;
	std::string name = "Twiss";

	void run(int turn);

	void print();

private:
	Particle* dev_bunch = nullptr;

	double s_previous = -1;

	std::string logitudinal_transfer = "off";
	// logitudinal parameters for point-to-point transfer
	double gamma = 0;
	double gammat = 0;
	// logitudinal parameters for one-turn map
	double sigmaz = 0;
	double dp = 0;

	int Np = 0;

	// Twiss parameters of current position
	double alphax = 0;
	double alphay = 0;

	double betax = 0;
	double betay = 0;

	double mux = 0;
	double muy = 0;

	double muz = 0;

	//double Dx = 0;

	// Twiss parameters of previos position
	double alphax_previous = 0;
	double alphay_previous = 0;

	double betax_previous = 0;
	double betay_previous = 0;

	double mux_previous = 0;
	double muy_previous = 0;

	double m11_x = 0, m12_x = 0, m21_x = 0, m22_x = 0;	// transfer matrix elements;
	double m11_y = 0, m12_y = 0, m21_y = 0, m22_y = 0;
	double m11_z = 0, m12_z = 0, m21_z = 0, m22_z = 0;

	int thread_x = 0;
	int block_x = 0;
};

class TwissCommand : public Command
{
public:
	~TwissCommand() {};

	explicit TwissCommand(Twiss* twi) {
		twiss = twi;
		s = twi->s;
		name = twi->name;
	}

	void execute(int turn) override {
		twiss->run(turn);
	}

private:
	Twiss* twiss = nullptr;
};


__global__ void transfer_matrix_4D(Particle* dev_bunch, int Np,
	double m11_x, double m12_x, double m21_x, double m22_x,
	double m11_y, double m12_y, double m21_y, double m22_y);

__global__ void transfer_matrix_6D(Particle* dev_bunch, int Np,
	double m11_x, double m12_x, double m21_x, double m22_x,
	double m11_y, double m12_y, double m21_y, double m22_y,
	double m11_z, double m12_z, double m21_z, double m22_z);