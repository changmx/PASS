#include "injection.h"
#include "constant.h"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include <random>

Injection::Injection(const Parameter& para, int input_beamId, Bunch& Bunch) {
	std::cout << "pointer 2 " << std::hex << Bunch.dev_bunch << std::endl;
	std::cout << "pointer 3 " << std::hex << dev_bunch << std::endl;

	dev_bunch = Bunch.dev_bunch;
	std::cout << "pointer 4 " << std::hex << dev_bunch << std::endl;

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

	beamId = input_beamId;
	bunchId = Bunch.bunchId;

	dir_load_distribution = para.dir_load_distribution;
	dir_save_distribution = para.dir_output_distribution;
	hourMinSec = para.hourMinSec;

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

		is_save_initial_dist = data.at("Sequence").at("Injection").at(key_bunch).at("Is save initial distribution");

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
		else if (dist_transverse == "kv")
		{
			generate_KV_distribution();
		}
		else
		{
			logger->error("Sorry, we don't support distribution type {}.", dist_transverse);
			std::exit(EXIT_FAILURE);
		}


		if (is_save_initial_dist)
		{
			save_initial_distribution();
		}

	}
	else if (injection_mode == "1turnxtime")
	{
		logger->error("Sorry, we don't support: 1-turn and multi-time injection.");
		std::exit(EXIT_FAILURE);
	}
	else if (injection_mode == "xturnxtime")
	{
		logger->error("Sorry, we don't support: multi-turn and multi-time injection.");
		std::exit(EXIT_FAILURE);
	}
	else
	{
		logger->error("Input wrong injection mode value: {}.", injection_mode);
	}
}

void Injection::load_distribution() {

	std::filesystem::path dist_path = dir_load_distribution / filename_load_dist;

	if (std::filesystem::exists(dist_path))
	{
		if (filename_load_dist.find(beam_name) == std::string::npos)
			spdlog::get("logger")->warn("Please be careful to confirm that the file is {} distribution: {}.", beam_name, dist_path.string());
		if (filename_load_dist.find(dist_transverse) == std::string::npos)
			spdlog::get("logger")->warn("Please be careful to confirm that the file is {} distribution: {}.", dist_transverse, dist_path.string());
		if (filename_load_dist.find(std::to_string(Np)) == std::string::npos)
			spdlog::get("logger")->warn("Please be careful to confirm that the file contain {} particles: {}.", Np, dist_path.string());

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
			spdlog::get("logger")->warn("We only load {}/{} particles from file {}.", j, Np, dist_path.string());
		}

		input.close();

		//Particle* dev_bunch2;
		//callCuda(cudaMalloc(&dev_bunch2, Np * sizeof(double)));
		callCuda(cudaMemcpy(dev_bunch, host_bunch, Np * sizeof(Particle), cudaMemcpyHostToDevice));

		delete[] host_bunch;
		//callCuda(cudaFree(dev_bunch2));

		spdlog::get("logger")->info("... distribution file has been loadded successfully.");
	}
	else
	{
		spdlog::get("logger")->error("We don't find distribution file: {}.", dist_path.string());
		std::exit(EXIT_FAILURE);
	}
}


void Injection::generate_KV_distribution() {

	/* This menthod is derived from "Particle - in - cell code BEAMPATH for beam dynamics simulations in linear accelerators and beamlines"*/
	// The two beams shoule have different seed values to generate different random values.
	// This is 4-D generator.

	spdlog::get("logger")->info("... generating initial KV distribution of beam-{} bunch-{}.", beam_name, bunchId);

	double emittence_x = emitx;
	double emittence_y = emity;
	double alpha_x_twiss = alphax;
	double alpha_y_twiss = alphay;
	double beta_x_twiss = betax;
	double beta_y_twiss = betay;
	double gamma_x_twiss = gammax;
	double gamma_y_twiss = gammay;

	double sigmax = sqrt(emittence_x * betax);
	double sigmay = sqrt(emittence_y * betay);

	// [-1考, 1考] = 0.6826894921370859, [-4考, 4考] = 0.9999366575163338
	// [-2考, 2考] = 0.9544997361036416, [-5考, 5考] = 0.9999994266968562
	// [-3考, 3考] = 0.9973002039367398, [-6考, 6考] = 0.9999999980268246
	double x_max = 4 * sigmax;
	double x_min = -4 * sigmax;
	double y_max = 4 * sigmay;
	double y_min = -4 * sigmay;

	int rank = 0;
	int i = bunchId;
	int beam_label = beamId;

	std::default_random_engine e1;
	e1.seed(time(NULL) + rank * 10000 + (i + 1) * 100 + beam_label);
	std::uniform_real_distribution<> u1(1e-15, 1.0 - 1e-15);

	std::default_random_engine e2;
	e2.seed(time(NULL) + rank * 10000 + (i + 1) * 100 + 10 + beam_label);
	std::uniform_real_distribution<> u2(0, 1);

	Particle* host_bunch = new Particle[Np];

	for (int j = 0; j < Np; ++j)
	{
		double nu, x, px, y, py;
		double X1, X2, Y1, Y2;
		double sigma11_x, sigma11_y, sigma12_x, sigma12_y, sigma22_x, sigma22_y;
		double ax, axpx, ay, aypy;
		double zeta_x, zeta_y, zeta_x_square, zeta_y_square;
		double phi_x, phi_y;
		double beta_x, beta_y;
		double pi = PassConstant::PI;

		double random_zeta = u1(e1);
		double random_beta_x = u2(e2);
		double random_beta_y = u2(e2);

		double F = emittence_x;

		nu = emittence_x / emittence_y;

		sigma11_x = emittence_x * beta_x_twiss;
		sigma12_x = -emittence_x * alpha_x_twiss;
		sigma22_x = emittence_x * gamma_x_twiss;

		sigma11_y = emittence_y * beta_y_twiss;
		sigma12_y = -emittence_y * alpha_y_twiss;
		sigma22_y = emittence_y * gamma_y_twiss;

		phi_x = 0.5 * atan2(2 * sigma12_x, sigma22_x - sigma11_x);
		phi_y = 0.5 * atan2(2 * sigma12_y, sigma22_y - sigma11_y);

		X1 = sqrt(2) * emittence_x / sqrt((sigma11_x + sigma22_x) + sqrt(pow((sigma22_x - sigma11_x), 2) + 4 * pow(sigma12_x, 2)));
		X2 = sqrt(2) * emittence_x / sqrt((sigma11_x + sigma22_x) - sqrt(pow((sigma22_x - sigma11_x), 2) + 4 * pow(sigma12_x, 2)));
		Y1 = sqrt(2) * emittence_y / sqrt((sigma11_y + sigma22_y) + sqrt(pow((sigma22_y - sigma11_y), 2) + 4 * pow(sigma12_y, 2)));
		Y2 = sqrt(2) * emittence_y / sqrt((sigma11_y + sigma22_y) - sqrt(pow((sigma22_y - sigma11_y), 2) + 4 * pow(sigma12_y, 2)));

		ax = sqrt((X1 / X2) * pow(cos(phi_x), 2) + (X2 / X1) * pow(sin(phi_x), 2));
		axpx = (X1 / X2 - X2 / X1) * sin(2 * phi_x) / (2 * ax);
		ay = sqrt((Y1 / Y2) * pow(cos(phi_y), 2) + (Y2 / Y1) * pow(sin(phi_y), 2));
		aypy = (Y1 / Y2 - Y2 / Y1) * sin(2 * phi_y) / (2 * ay);

		zeta_x_square = F * random_zeta;
		zeta_x = sqrt(zeta_x_square);
		zeta_y_square = (F - zeta_x_square) / nu;
		zeta_y = sqrt(zeta_y_square);
		beta_x = 2 * pi * random_beta_x;
		beta_y = 2 * pi * random_beta_y;

		x = zeta_x * ax * cos(beta_x) * 2;
		px = zeta_x * (axpx * cos(beta_x) - sin(beta_x) / ax) * 2;
		y = zeta_y * ay * cos(beta_y) * 2;
		py = zeta_y * (aypy * cos(beta_y) - sin(beta_y) / ay) * 2;
		/*x = zeta_x * ax * cos(beta_x);
		px = zeta_x * (axpx * cos(beta_x) - sin(beta_x) / ax);
		y = zeta_y * ay * cos(beta_y);
		py = zeta_y * (aypy * cos(beta_y) - sin(beta_y) / ay);*/

		if (x > x_min && x < x_max && y > y_min && y < y_max)
		{
			host_bunch[j].x = x;
			host_bunch[j].px = px;
			host_bunch[j].y = y;
			host_bunch[j].py = py;
			host_bunch[j].tag = j + 1;
		}
		else
		{
			--j;
		}
	}

	callCuda(cudaMemcpy(dev_bunch, host_bunch, Np * sizeof(Particle), cudaMemcpyHostToDevice));

	delete[] host_bunch;
	//std::cout << "initial KV distribution of " << beam.beamName << " has been genetated successfully." << std::endl;
	spdlog::get("logger")->info("... initial KV distribution of {} beam bunch-{} has been genetated successfully.", beam_name, bunchId);
}

void Injection::save_initial_distribution() {

	Particle* host_bunch = new Particle[Np];

	callCuda(cudaMemcpy(host_bunch, dev_bunch, Np * sizeof(Particle), cudaMemcpyDeviceToHost));

	std::filesystem::path path_tmp = dir_save_distribution / (hourMinSec + "_" + dist_transverse + "_" + beam_name +
		"_bunch" + std::to_string(bunchId) + "_" + std::to_string(Np) + "_initial.csv");
	std::ofstream file(path_tmp);

	file << "x" << "," << "px" << "," << "y" << "," << "py" << "," << "z" << "," << "pz" << "," << "tag" << "," << "lostTurn" << std::endl;

	for (int j = 0; j < Np; j++) {
		file << std::setprecision(10)
			<< (host_bunch + j)->x << ","
			<< (host_bunch + j)->px << ","
			<< (host_bunch + j)->y << ","
			<< (host_bunch + j)->py << ","
			<< (host_bunch + j)->z << ","
			<< (host_bunch + j)->pz << ","
			<< (host_bunch + j)->tag << ","
			<< (host_bunch + j)->lostTurn << "\n";
	}
	file.close();
	delete[]host_bunch;

	spdlog::get("logger")->info("... initial {} distribution of {} beam bunch-{} has been saved to {}.", dist_transverse, beam_name, bunchId, path_tmp.string());
}