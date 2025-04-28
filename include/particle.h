#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

#include "config.h"
#include "parameter.h"

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

	/*The integer part of lossTurn is the number of turns in which the loss occured,
	and the decimal part is the loss position (in unit of m) divided by 1e6*/
	double lostTurn = -1;

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


class Bunch
{
public:
	Bunch() = default;
	Bunch(const Parameter& para, int input_beamId, int input_bunchId);
	~Bunch() = default;

	void init_memory();
	void free_memory();

	int bunchId = 0;

	double Nrp = 0;	// Number of real particles
	int Np = 0;	// Number of macro particles
	double ratio = 0;	// Value of Nrp/Np

	int Nproton = 0;	// Number of protons per real particle
	int Nneutron = 0;	// Number of neutrons per real particle
	int Ncharge = 0;	// Number of charges per real particle. e.g. for electron beam, charge = -1, for positron/proton beam, charge = 1

	double m0 = 0;	// Static mass of a real particle (eV/c2))
	double Ek = 0;	// Kinetic energy of a real particle (eV)
	double p0 = 0;	// Momentum of an ideal real particle (eV/c)
	double p0_kg = 0;	// Momentum of an ideal real particle (kg*m/s)
	double beta = 0, gamma = 0;	// Relativistic velocity and Lorentz factor

	double Brho = 0;

	//double emitx = 0, emity = 0;	// Geometric emittance (rad'm)
	//double emitx_norm = 0, emity_norm = 0;	// Normalized emittance (rad'm)
	////double emitx_equi = 0, emity_equi = 0;	// Equilibrium emittance, used in synchrotron radiation (rad'm)

	//double alphax = 0, alphay = 0;	// Twiss parameters
	//double betax = 0, betay = 0;	// Twiss parameters
	//double gammax = 0, gammay = 0;	// Twiss parameters

	//double sigmax = 0, sigmay = 0, sigmaz = 0;	// RMS value of horizontal, vertical bunch size and bunch length (m)
	//double sigmapx = 0, sigmapy = 0, dp = 0;	// RMS value of horizontal, vertical divergence (rad) and deltap/p

	//double Qx = 0, Qy = 0, Qz = 0;	// Tunes
	//double chromx = 0, chromy = 0;	// Chromaticity

	double gammat = 0;

	// sigmaz and dp will be set in Injection initialize function
	double sigmaz = 0;
	double dp = 0;

	//int dampTurn = 0;	// Transverse damping turns

	Particle* dev_bunch = NULL;

	std::string dist_transverse;
	std::string dist_logitudinal;

private:

};
