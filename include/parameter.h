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

	int Nbeam = 0;	// Number of beams
	std::vector<int> beamId;	// 0 for beam0.json and 1 for beam1.json
	std::vector<std::string> beam_name;	// beam name, used to distinguish beams and store data
	std::vector<int> Nbunch;	// Number of bunches per beam
	int Nturn = 0;	// Number of simulation turns
	int Ncollision = 0;	// Number of interaction points

	double circumference = 0;	// in unit of m

	int Ngpu = 0;	// Number of GPU devices
	std::vector<int> gpuId;	// GPU device Id

	bool is_plot = false;

	bool is_beambeam = false;
	bool is_spacecharge = false;


	std::vector<std::filesystem::path> path_input_para;

	std::filesystem::path path_logfile;

	std::filesystem::path dir_output;
	std::filesystem::path dir_output_statistic;
	std::filesystem::path dir_output_distribution;
	std::filesystem::path dir_output_tuneSpread;
	std::filesystem::path dir_output_chargeDensity;
	std::filesystem::path dir_output_plot;
	std::filesystem::path dir_output_particle;

	std::filesystem::path dir_load_distribution;

	std::string yearMonDay;
	std::string hourMinSec;

private:

};
