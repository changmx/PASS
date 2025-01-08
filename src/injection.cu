#include "injection.h"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <iostream>
#include <fstream>
#include <sstream>

Injection::Injection(const Parameter& para, int input_beamId, const Bunch& Bunch) {

	dev_particle = Bunch.dev_particle;

	Np = Bunch.Np;

	alphax = Bunch.alphax;
	alphay = Bunch.alphay;
	betax = Bunch.betax;
	betay = Bunch.betay;
	gammax = Bunch.gammax;
	gammay = Bunch.gammay;

	emitx = Bunch.emitx;
	emity = Bunch.emity;

	sigmaz = Bunch.sigmaz;
	dp = Bunch.dp;

	dir_load_distribution = para.dir_load_distribution;

	beam_name = para.beam_name[input_beamId];

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	std::string key_bunch = "bunch" + std::to_string(Bunch.bunchId);

	try
	{
		s = data.at("Sequence").at("Injection").at("S (m)");
		name = data.at("Sequence").at("Injection").at("Command");

		injection_mode = data.at("Sequence").at("Injection").at(key_bunch).at("Mode");
		dist_transverse = data.at("Sequence").at("Injection").at(key_bunch).at("Transverse dist");
		dist_logitudinal = data.at("Sequence").at("Injection").at(key_bunch).at("Logitudinal dist");

		is_offset_x = data.at("Sequence").at("Injection").at(key_bunch).at("Offset x").at("Is offset");
		is_offset_y = data.at("Sequence").at("Injection").at(key_bunch).at("Offset y").at("Is offset");

		offset_x = data.at("Sequence").at("Injection").at(key_bunch).at("Offset x").at("Offset (m)");
		offset_y = data.at("Sequence").at("Injection").at(key_bunch).at("Offset y").at("Offset (m)");

		is_load_dist = data.at("Sequence").at("Injection").at(key_bunch).at("Is load distribution");
		filename_load_dist = data.at("Sequence").at("Injection").at(key_bunch).at("Name of loaded file");

		for (size_t i = 0; i < data.at("Sequence").at("Injection").at(key_bunch).at("Inject turns").size(); i++)
		{
			inject_turns.push_back(data.at("Sequence").at("Injection").at(key_bunch).at("Inject turns")[i]);
		}
	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}
}

void Injection::execute() {
	auto logger = spdlog::get("logger");

	logger->debug("Injection action");

	if (injection_mode == "1turn1time")
	{
		logger->debug("Start: 1-turn and 1-time injection");

		if (is_load_dist)
		{
			load_distribution();
		}
	}
	else if (injection_mode == "1turnxtime")
	{
		logger->error("Sorry, we don't support: 1-turn and multi-time injection");
		std::exit(EXIT_FAILURE);
	}
	else if (injection_mode == "xturnxtime")
	{
		logger->error("Sorry, we don't support: multi-turn and multi-time injection");
		std::exit(EXIT_FAILURE);
	}
	else
	{
		logger->error("Input wrong injection mode value: ", injection_mode);
	}
}

void Injection::load_distribution() {

	std::filesystem::path dist_path = dir_load_distribution / filename_load_dist;

	if (std::filesystem::exists(dist_path))
	{
		if (filename_load_dist.find(beam_name) == std::string::npos)
			spdlog::get("logger")->warn("Please be careful to confirm that the file is {} distribution: {}", beam_name, dist_path.string());
		if (filename_load_dist.find(dist_transverse) == std::string::npos)
			spdlog::get("logger")->warn("Please be careful to confirm that the file is {} distribution: {}", dist_transverse, dist_path.string());
		if (filename_load_dist.find(std::to_string(Np)) == std::string::npos)
			spdlog::get("logger")->warn("Please be careful to confirm that the file contain {} particles: {}", Np, dist_path.string());

		spdlog::get("logger")->info("... loading distribution file: {}", dist_path.string());

		Particle* host_bunch = new Particle[Np];

		std::ifstream input(dist_path);

		std::string line;
		int j = 0;

		double a[7];
		std::string tmp;
		int row = 0;
		int skiprows = 0;
		while (std::getline(input, line))
		{
			std::stringstream sline(line);
			//std::cout << line << std::endl;
			int k = 0;
			if (row != skiprows)
			{
				while (std::getline(sline, tmp, ','))
				{
					//std::cout << tmp << std::endl;
					a[k] = std::stod(tmp);
					//std::cout << j << a[j] << std::endl;
					++k;
				}
				//std::cout << a[0] << "," << a[1] << std::endl;
				//spdlog::get("logger")->debug("row [{}] a[0] = {}, a[1] = {}", row, a[0], a[1]);

				int offset = j;
				host_bunch[offset].x = a[0];
				host_bunch[offset].px = a[1];
				host_bunch[offset].y = a[2];
				host_bunch[offset].py = a[3];
				host_bunch[offset].z = a[4];
				host_bunch[offset].pz = a[5];
				host_bunch[offset].tag = a[6];

				j++;
			}
			++row;
		}

		if (j != (Np - 1))
		{
			spdlog::get("logger")->warn("We only load {}/{} particles from file {}", j, Np, dist_path.string());
		}

		input.close();

		//callCuda(cudaMalloc())

		delete[] host_bunch;

		spdlog::get("logger")->info("... distribution file has been loadded successfully");
	}
	else
	{
		spdlog::get("logger")->error("We don't find distribution file: {}", dist_path.string());
		std::exit(EXIT_FAILURE);
	}
}
