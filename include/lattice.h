#pragma once

#include "command.h"
#include "particle.h"
#include "parameter.h"

class Lattice
{
public:
	Lattice();
	~Lattice();

	double s = -1;
	std::string name = "Lattice";

	void run(int turn);

private:

};

Lattice::Lattice()
{
}

Lattice::~Lattice()
{
}