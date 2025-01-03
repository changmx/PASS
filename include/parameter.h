#pragma once

#include <string>
#include <nlohmann/json.hpp>

class Parameter
{
public:
	Parameter() = default;
	Parameter(int argc, char** argv);
	~Parameter() {
		//std::cout << "Parameter class destructor: " << beam_name << std::endl;
	};

	//std::vector<int> Nrp;	// Number of real particles bunch
	//std::vector<int> Np;	// Number of macro particles per bunch
	//std::vector<double> ratio;	// The value of Nrp/Np
	//std::vector<int> Nproton;	// Number of protons per real particle
	//std::vector<int> Nneutron;	// Number of neutrons per real particle
	//std::vector<int> Ncharge;	// Number of charge per real particle. e.g. for electron beam, charge = -1, for positron/proton beam, charge = 1

	int Nbeam = 0;	// Number of beams
	std::vector<int> beamId;	// 0 for beam0.json and 1 for beam1.json
	std::vector<int> Nbunch;	// Number of bunches per beam
	int Nturn = 0;	// Number of simulation turns
	int Ncollision = 0;	// Number of interaction points

	std::vector<double> Qx;	// Qx, will not be used if a madx file is loaded
	std::vector<double> Qy;	// Qy, will not be used if a madx file is loaded
	std::vector<double> Qz;	// Qz, will not be used if a madx file is loaded

	std::vector<double> chromx;	// chromaticity in x direction
	std::vector<double> chromy;	// chromaticity in y direction

	std::vector<double> gammaT;	// gamma transition

	int Ngpu = 0;	// Number of GPU devices
	std::vector<int> gpuId;	// GPU device Id

	bool is_beambeam = false;
	bool is_spacecharge = false;

	bool is_plot = false;

	std::vector<std::string> beam_name;

	std::vector<std::filesystem::path> path_input_para;

	std::filesystem::path dir_output;
	std::filesystem::path dir_output_statistic;
	std::filesystem::path dir_output_distribution;
	std::filesystem::path dir_output_tuneSpread;
	std::filesystem::path dir_output_chargeDensity;
	std::filesystem::path dir_output_plot;

	std::string yearMonDay;
	std::string hourMinSec;

private:

};