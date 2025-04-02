#pragma once

#include "command.h"
#include "particle.h"
#include "parameter.h"

class Twiss
{
public:
	Twiss(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name);

	double s = -1;
	std::string name = "Twiss";

	void run(int turn);

	void print();

private:
	Particle* dev_bunch = NULL;

	int Np = 0;

	// Twiss parameters of current position
	double alphax = 0;
	double alphay = 0;
	double alphaz = 0;

	double betax = 0;
	double betay = 0;
	double betaz = 0;

	double mux = 0;
	double muy = 0;
	double muz = 0;

	double Dx = 0;

	// Twiss parameters of previos position
	double alphax_previous = 0;
	double alphay_previous = 0;

	double betax_previous = 0;
	double betay_previous = 0;

	double mux_previous = 0;
	double muy_previous = 0;
};


class TwissCommand : public Command
{
public:
	~TwissCommand() {};

	TwissCommand(Twiss* twi) {
		twiss = twi;
		s = twi->s;
		name = twi->name;
	}

	void execute(int turn) override {
		twiss->run(turn);
	}

private:
	Twiss* twiss;
};