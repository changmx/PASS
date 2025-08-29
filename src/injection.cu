#include "injection.h"
#include "constant.h"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include <random>

Injection::Injection(const Parameter& para, int input_beamId, Bunch& Bunch, std::string obj_name) {
	//std::cout << "pointer 2 " << std::hex << Bunch.dev_bunch << std::endl;
	//std::cout << "pointer 3 " << std::hex << dev_bunch << std::endl;
	name = obj_name;
	dev_particle = Bunch.dev_particle;
	//std::cout << "pointer 4 " << std::hex << dev_bunch << std::endl;

	Np = Bunch.Np;

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
		//name = data.at("Sequence").at("Injection").at("Command");
		if (fabs(s) > 1e-10)
		{
			spdlog::get("logger")->error("[Injection] The position of injection point (simulation start point) should be 0, but now is : {}.", s);
			std::exit(EXIT_FAILURE);
		}

		if (1 == data.at("Sequence").at("Injection").at(key_bunch).at("Inject turns").size())
		{
			startTurn = data.at("Sequence").at("Injection").at(key_bunch).at("Inject turns")[0];
			endTurn = data.at("Sequence").at("Injection").at(key_bunch).at("Inject turns")[0];
		}
		else if (2 == data.at("Sequence").at("Injection").at(key_bunch).at("Inject turns").size())
		{
			startTurn = data.at("Sequence").at("Injection").at(key_bunch).at("Inject turns")[0];
			endTurn = data.at("Sequence").at("Injection").at(key_bunch).at("Inject turns")[1];
		}
		else
		{
			spdlog::get("logger")->error("[Injection] The number of parameters of 'Inject turns' should be 1 or 2, but now is: {}.",
				data.at("Sequence").at("Injection").at(key_bunch).at("Inject turns").size());
			std::exit(EXIT_FAILURE);
		}

		alphax = data.at("Sequence").at("Injection").at(key_bunch).at("Alpha x");
		alphay = data.at("Sequence").at("Injection").at(key_bunch).at("Alpha y");

		betax = data.at("Sequence").at("Injection").at(key_bunch).at("Beta x (m)");
		betay = data.at("Sequence").at("Injection").at(key_bunch).at("Beta y (m)");

		gammax = (1 + alphax * alphax) / betax;
		gammay = (1 + alphay * alphay) / betay;

		emitx = data.at("Sequence").at("Injection").at(key_bunch).at("Emittance x (m'rad)");
		emity = data.at("Sequence").at("Injection").at(key_bunch).at("Emittance y (m'rad)");

		Dx = data.at("Sequence").at("Injection").at(key_bunch).at("Dx (m)");
		Dpx = data.at("Sequence").at("Injection").at(key_bunch).at("Dpx");

		emitx_norm = emitx * Bunch.gamma * Bunch.beta;
		emity_norm = emity * Bunch.gamma * Bunch.beta;

		sigmaz = data.at("Sequence").at("Injection").at(key_bunch).at("Sigma z (m)");
		dp = data.at("Sequence").at("Injection").at(key_bunch).at("DeltaP/P");

		sigmax = sqrt(betax * emitx);
		sigmay = sqrt(betay * emity);

		sigmapx = sqrt(gammax * emitx);
		sigmapy = sqrt(gammay * emity);

		injection_mode = data.at("Sequence").at("Injection").at(key_bunch).at("Mode");
		dist_transverse = data.at("Sequence").at("Injection").at(key_bunch).at("Transverse dist");
		dist_longitudinal = data.at("Sequence").at("Injection").at(key_bunch).at("Longitudinal dist");

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

		if (data.at("Sequence").at("Injection").at(key_bunch).at("Particle coordinate").size() > 0) {
			is_set_specified_coordinate = true;

			for (size_t i = 0; i < data.at("Sequence").at("Injection").at(key_bunch).at("Particle coordinate").size(); i++)
			{
				double x_tmp = data.at("Sequence").at("Injection").at(key_bunch).at("Particle coordinate")[i][0];
				double px_tmp = data.at("Sequence").at("Injection").at(key_bunch).at("Particle coordinate")[i][1];
				double y_tmp = data.at("Sequence").at("Injection").at(key_bunch).at("Particle coordinate")[i][2];
				double py_tmp = data.at("Sequence").at("Injection").at(key_bunch).at("Particle coordinate")[i][3];
				double z_tmp = data.at("Sequence").at("Injection").at(key_bunch).at("Particle coordinate")[i][4];
				double dp_tmp = data.at("Sequence").at("Injection").at(key_bunch).at("Particle coordinate")[i][5];

				specified_coordinate.push_back({ x_tmp, px_tmp, y_tmp, py_tmp, z_tmp, dp_tmp });
			}
		}
	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}

	Bunch.sigmaz = sigmaz;
	Bunch.dp = dp;

	Bunch.dist_transverse = dist_transverse;
	Bunch.dist_longitudinal = dist_longitudinal;
}

void Injection::execute(int turn) {
	auto logger = spdlog::get("logger");

	//logger->debug("Injection action");

	if ("1turn1time" == injection_mode)
	{
		//logger->debug("Start: 1-turn and 1-time injection");
		if (startTurn != endTurn)
		{
			logger->warn("[Injection] In the 1-turn 1-time injection mode, we only inject 1 turn, but input parameters is from turn {} to turn {}. We will only inject at turn {}.",
				startTurn, endTurn, startTurn);
		}

		if (turn == startTurn)
		{
			if (is_load_dist)
			{
				load_distribution();
			}
			else
			{
				if ("kv" == dist_transverse)
				{
					generate_transverse_KV_distribution();
				}
				else if ("gaussian" == dist_transverse)
				{
					generate_transverse_Gaussian_distribution();
				}
				else if ("uniform" == dist_transverse)
				{
					generate_transverse_uniform_distribution();
				}
				else
				{
					logger->error("[Injection] Sorry, we don't support transverse distribution type {}.", dist_transverse);
					std::exit(EXIT_FAILURE);
				}

				if ("gaussian" == dist_longitudinal)
				{
					generate_longitudinal_Gaussian_distribution();
				}
				else if ("uniform" == dist_longitudinal)
				{
					generate_longitudinal_uniform_distribution();
				}
				else
				{
					logger->error("[Injection] Sorry, we don't support longitudinal distribution type {}.", dist_longitudinal);
					std::exit(EXIT_FAILURE);
				}

				add_Dx();
				add_offset();

			}

			if (is_save_initial_dist)
			{
				save_initial_distribution();
			}
		}
	}

	else if ("1turnxtime" == injection_mode)
	{
		logger->error("[Injection] Sorry, we don't support: 1-turn and multi-time injection.");
		std::exit(EXIT_FAILURE);
	}

	else if ("xturnxtime" == injection_mode)
	{
		logger->error("[Injection] Sorry, we don't support: multi-turn and multi-time injection.");
		std::exit(EXIT_FAILURE);
	}

	else
	{
		logger->error("[Injection] Input wrong injection mode value: {}.", injection_mode);
		std::exit(EXIT_FAILURE);
	}
}

void Injection::load_distribution() {

	std::filesystem::path dist_path = dir_load_distribution / filename_load_dist;

	if (std::filesystem::exists(dist_path))
	{
		if (filename_load_dist.find(beam_name) == std::string::npos)
			spdlog::get("logger")->warn("[Injection] Please be careful to confirm that the file is {} distribution: {}.", beam_name, dist_path.string());
		if (filename_load_dist.find(dist_transverse) == std::string::npos)
			spdlog::get("logger")->warn("[Injection] Please be careful to confirm that the file is {} distribution: {}.", dist_transverse, dist_path.string());
		if (filename_load_dist.find(std::to_string(Np)) == std::string::npos)
			spdlog::get("logger")->warn("[Injection] Please be careful to confirm that the file contain {} particles: {}.", Np, dist_path.string());

		spdlog::get("logger")->info("[Injection] Loading distribution file: {}", dist_path.string());

		Particle host_particle;
		host_particle.mem_allocate_cpu(Np);

		std::ifstream input(dist_path);

		std::string line;
		int j = 0;

		double a[6] = { 0,0,0,0,0,0 };
		int a_tag = 0, a_lostTurn = -1;
		int a_sliceId = 0;
		double a_lostPos = -1.0;
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
					if (k < 6)
					{
						//std::cout << tmp << std::endl;
						a[k] = std::stod(tmp);
						//std::cout << j << a[j] << std::endl;
					}
					else if (k == 6)
					{
						a_tag = std::stoi(tmp);
					}
					else if (k == 7)
					{
						a_sliceId = std::stoi(tmp);
					}
					else if (k == 8)
					{
						a_lostTurn = std::stoi(tmp);
					}
					else if (k == 9)
					{
						a_lostPos = std::stod(tmp);
					}
					++k;
				}
				//std::cout << a[0] << "," << a[1] << std::endl;
				//spdlog::get("logger")->debug("row [{}] a[0] = {}, a[1] = {}", row, a[0], a[1]);

				int offset = j;
				if (offset < Np)
				{
					host_particle.x[offset] = a[0];
					host_particle.px[offset] = a[1];
					host_particle.y[offset] = a[2];
					host_particle.py[offset] = a[3];
					host_particle.z[offset] = a[4];
					host_particle.pz[offset] = a[5];
					host_particle.tag[offset] = a_tag;
					host_particle.sliceId[offset] = a_sliceId;
					host_particle.lostTurn[offset] = a_lostTurn;
					host_particle.lostPos[offset] = a_lostPos;

					j++;
				}

			}
			++row;
		}

		if (j != Np)
		{
			spdlog::get("logger")->warn("[Injection] We only load {}/{} particles from file {}.", j, Np, dist_path.string());
		}

		input.close();

		if (is_set_specified_coordinate) {

			for (size_t i = 0; i < specified_coordinate.size(); i++)
			{
				host_particle.x[i] = specified_coordinate[i][0];
				host_particle.px[i] = specified_coordinate[i][1];
				host_particle.y[i] = specified_coordinate[i][2];
				host_particle.py[i] = specified_coordinate[i][3];
				host_particle.z[i] = specified_coordinate[i][4];
				host_particle.pz[i] = specified_coordinate[i][5];
			}
		}

		particle_copy(dev_particle, host_particle, Np, cudaMemcpyHostToDevice, "dist");

		host_particle.mem_free_cpu();

		spdlog::get("logger")->info("[Injection] Distribution file {} has been loadded successfully to {} beam-{} bunch-{}.",
			dist_path.string(), beam_name, beamId, bunchId);
	}
	else
	{
		spdlog::get("logger")->error("[Injection] We don't find distribution file: {}.", dist_path.string());
		std::exit(EXIT_FAILURE);
	}
}


void Injection::generate_transverse_KV_distribution() {

	/* This menthod is derived from "Particle - in - cell code BEAMPATH for beam dynamics simulations in linear accelerators and beamlines"*/
	// The two beams shoule have different seed values to generate different random values.
	// This is 4-D generator.

	spdlog::get("logger")->info("[Injection] The initial transverse KV distribution of {} beam-{} bunch-{} is begin generated ...",
		beam_name, beamId, bunchId);

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

	int i = bunchId;
	int beam_label = beamId;

	std::default_random_engine e1;
	e1.seed(curTime + beam_label * 10000019 + (callTime++) * 1000 + (i + 1) * 1);
	std::uniform_real_distribution<> u1(1e-15, 1.0 - 1e-15);

	std::default_random_engine e2;
	e2.seed(curTime + beam_label * 10000019 + (callTime++) * 1000 + (i + 1) * 1);
	std::uniform_real_distribution<> u2(0, 1);

	Particle host_particle;
	host_particle.mem_allocate_cpu(Np);

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

		phi_x = 0.5 * atan2(2 * alpha_x_twiss, gamma_x_twiss - beta_x_twiss);	// https://agenda.linearcollider.org/event/6258/contributions/29168/attachments/24202/37474/linear_dynamics.pdf
		phi_y = 0.5 * atan2(2 * alpha_y_twiss, gamma_y_twiss - beta_y_twiss);

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
			host_particle.x[j] = x;
			host_particle.px[j] = px;
			host_particle.y[j] = y;
			host_particle.py[j] = py;
			host_particle.tag[j] = j + 1;
			host_particle.sliceId[j] = 0;
			host_particle.lostTurn[j] = -1;
			host_particle.lostPos[j] = -1.0;
		}
		else
		{
			--j;
		}
	}

	if (is_set_specified_coordinate) {

		for (size_t i = 0; i < specified_coordinate.size(); i++)
		{
			host_particle.x[i] = specified_coordinate[i][0];
			host_particle.px[i] = specified_coordinate[i][1];
			host_particle.y[i] = specified_coordinate[i][2];
			host_particle.py[i] = specified_coordinate[i][3];
		}
	}

	//callCuda(cudaMemcpy(dev_bunch, host_bunch, Np * sizeof(Particle), cudaMemcpyHostToDevice));
	particle_copy(dev_particle, host_particle, Np, cudaMemcpyHostToDevice, "dist");

	host_particle.mem_free_cpu();
	//std::cout << "initial KV distribution of " << beam.beamName << " has been genetated successfully." << std::endl;
	spdlog::get("logger")->info("[Injection] The initial transverse KV distribution of {} beam-{} bunch-{} has been genetated successfully.",
		beam_name, beamId, bunchId);
}


void Injection::generate_transverse_Gaussian_distribution() {

	spdlog::get("logger")->info("[Injection] The initial transverse Gaussian distribution of {} beam-{} bunch-{} is begin generated ...",
		beam_name, beamId, bunchId);

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

	int i = bunchId;
	int beam_label = beamId;

	std::default_random_engine e1;
	e1.seed(curTime + beam_label * 10000019 + (callTime++) * 1000 + (i + 1) * 1);
	std::uniform_real_distribution<> u1(1e-15, 1.0 - 1e-15);

	Particle host_particle;
	host_particle.mem_allocate_cpu(Np);
	//Particle* host_bunch;
	//cudaHostAlloc((void**)&host_bunch, Np * sizeof(Particle), cudaHostAllocDefault);


	for (int j = 0; j < Np; ++j)
	{

		double x, px, y, py;
		double Xm, thetaXm, Ym, thetaYm;
		double a_x, a_y, u_x, u_y, v_x, v_y;
		double alpha_x, alpha_y, chi_x, chi_y;
		double pi = PassConstant::PI;

		double random_s1_x = u1(e1); //range (0,1)
		double random_s1_y = u1(e1); //range (0,1)
		double random_s2_x = u1(e1); //range (0,1)
		double random_s2_y = u1(e1); //range (0,1)

		Xm = 2 * sqrt(emittence_x * beta_x_twiss);
		thetaXm = 2 * sqrt(emittence_x * gamma_x_twiss);
		a_x = sqrt(2) / 2 * sqrt(-log(random_s1_x));
		alpha_x = 2 * pi * random_s2_x;
		chi_x = -1 * atan(alpha_x_twiss);
		u_x = a_x * cos(alpha_x);
		v_x = a_x * sin(alpha_x);
		x = Xm * u_x;
		px = thetaXm * (u_x * sin(chi_x) + v_x * cos(chi_x));

		Ym = 2 * sqrt(emittence_y * beta_y_twiss);
		thetaYm = 2 * sqrt(emittence_y * gamma_y_twiss);
		a_y = sqrt(2) / 2 * sqrt(-log(random_s1_y));
		alpha_y = 2 * pi * random_s2_y;
		chi_y = -1 * atan(alpha_y_twiss);
		u_y = a_y * cos(alpha_y);
		v_y = a_y * sin(alpha_y);
		y = Ym * u_y;
		py = thetaYm * (u_y * sin(chi_y) + v_y * cos(chi_y));

		if (x > x_min && x < x_max && y > y_min && y < y_max)
		{
			host_particle.x[j] = x;
			host_particle.px[j] = px;
			host_particle.y[j] = y;
			host_particle.py[j] = py;
			host_particle.tag[j] = j + 1;
			host_particle.sliceId[j] = 0;
			host_particle.lostTurn[j] = -1;
			host_particle.lostPos[j] = -1.0;
		}
		else
		{
			--j;
		}
		/*std::cout << "tag: " << beam.tag(i) << std::endl;
		std::cout << beam.x(i) << " " << beam.px(i) << " " << beam.y(i) << " " << beam.py(i) << std::endl;*/
	}

	if (is_set_specified_coordinate) {

		for (size_t i = 0; i < specified_coordinate.size(); i++)
		{
			host_particle.x[i] = specified_coordinate[i][0];
			host_particle.px[i] = specified_coordinate[i][1];
			host_particle.y[i] = specified_coordinate[i][2];
			host_particle.py[i] = specified_coordinate[i][3];
		}
	}

	//callCuda(cudaMemcpy(dev_bunch, host_bunch, Np * sizeof(Particle), cudaMemcpyHostToDevice));
	particle_copy(dev_particle, host_particle, Np, cudaMemcpyHostToDevice, "dist");

	host_particle.mem_free_cpu();
	//cudaFreeHost(host_bunch);
	//std::cout << "initial Gaussian distribution of " << beam.beamName << " has been genetated successfully." << std::endl;
	spdlog::get("logger")->info("[Injection] The initial transverse Gaussian distribution of {} beam-{} bunch-{} has been genetated successfully.",
		beam_name, beamId, bunchId);

}

void Injection::generate_transverse_uniform_distribution() {

	spdlog::get("logger")->info("[Injection] The initial transverse uniform distribution of {} beam-{} bunch-{} is begin generated ...",
		beam_name, beamId, bunchId);

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

	int i = bunchId;
	int beam_label = beamId;

	std::default_random_engine e1;
	e1.seed(curTime + beam_label * 10000019 + (callTime++) * 1000 + (i + 1) * 1);
	std::uniform_real_distribution<> u1(1e-15, 1.0 - 1e-15);

	Particle host_particle;
	host_particle.mem_allocate_cpu(Np);

	for (int j = 0; j < Np; ++j)
	{

		double m = 1;
		double x, px, y, py;
		double Xm, thetaXm, Ym, thetaYm;
		double Xl, Xlpx, Yl, Ylpy;
		double a_x, a_y, u_x, u_y, v_x, v_y;
		double alpha_x, alpha_y, chi_x, chi_y;
		double pi = PassConstant::PI;

		double random_s1_x = u1(e1); //range (0,1)
		double random_s2_x = u1(e1); //range (0,1)
		double random_s1_y = u1(e1); //range (0,1)
		double random_s2_y = u1(e1); //range (0,1)

		Xm = 2 * sqrt(emittence_x * beta_x_twiss);
		thetaXm = 2 * sqrt(emittence_x * gamma_x_twiss);
		Xl = sqrt((m + 1) / 2) * Xm;
		Xlpx = sqrt((m + 1) / 2) * thetaXm;
		a_x = sqrt(1 - pow(random_s1_x, 1 / m));
		alpha_x = 2 * pi * random_s2_x;
		chi_x = -1 * atan(alpha_x_twiss);
		u_x = a_x * cos(alpha_x);
		v_x = a_x * sin(alpha_x);

		x = Xl * u_x;
		px = Xlpx * (u_x * sin(chi_x) + v_x * cos(chi_x));
		//printf("Xm = %f, thetaXm = %f, Xl = %f, Xlpx = %f, alphaX = %f, ux = %f, vx = %f, x = %f, px = %f\n", Xm, thetaXm, Xl, Xlpx, alpha_x, u_x, v_x, x, px);

		Ym = 2 * sqrt(emittence_y * beta_y_twiss);
		thetaYm = 2 * sqrt(emittence_y * gamma_y_twiss);
		Yl = sqrt((m + 1) / 2) * Ym;
		Ylpy = sqrt((m + 1) / 2) * thetaYm;
		a_y = sqrt(1 - pow(random_s1_y, 1 / m));
		alpha_y = 2 * pi * random_s2_y;
		chi_y = -1 * atan(alpha_y_twiss);
		u_y = a_y * cos(alpha_y);
		v_y = a_y * sin(alpha_y);

		y = Yl * u_y;
		py = Ylpy * (u_y * sin(chi_y) + v_y * cos(chi_y));

		if (x > x_min && x < x_max && y > y_min && y < y_max)
		{
			host_particle.x[j] = x;
			host_particle.px[j] = px;
			host_particle.y[j] = y;
			host_particle.py[j] = py;
			host_particle.tag[j] = j + 1;
			host_particle.sliceId[j] = 0;
			host_particle.lostTurn[j] = -1;
			host_particle.lostPos[j] = -1.0;
		}
		else
		{
			--j;
		}
		/*std::cout << "tag: " << beam.tag(i) << std::endl;
		std::cout << beam.x(i) << " " << beam.px(i) << " " << beam.y(i) << " " << beam.py(i) << std::endl;*/
	}

	if (is_set_specified_coordinate) {

		for (size_t i = 0; i < specified_coordinate.size(); i++)
		{
			host_particle.x[i] = specified_coordinate[i][0];
			host_particle.px[i] = specified_coordinate[i][1];
			host_particle.y[i] = specified_coordinate[i][2];
			host_particle.py[i] = specified_coordinate[i][3];
		}
	}

	//callCuda(cudaMemcpy(dev_bunch, host_bunch, Np * sizeof(Particle), cudaMemcpyHostToDevice));
	particle_copy(dev_particle, host_particle, Np, cudaMemcpyHostToDevice, "dist");

	host_particle.mem_free_cpu();
	//std::cout << "initial Uniform distribution of " << beam.beamName << " has been genetated successfully." << std::endl;
	spdlog::get("logger")->info("[Injection] The initial transverse uniform distribution of {} beam-{} bunch-{} has been genetated successfully.",
		beam_name, beamId, bunchId);
}


void Injection::generate_longitudinal_Gaussian_distribution() {

	//	Generate particle's z position and momentum.
	//	Here we think the correlation coefficient of 2D Gaussian distribution rho = 0

	spdlog::get("logger")->info("[Injection] The initial longitudinal Gaussian distribution of {} beam-{} bunch-{} is begin generated ...",
		beam_name, beamId, bunchId);

	int i = bunchId;
	int beam_label = beamId;

	double sigma_z = sigmaz;
	double sigma_pz = dp;
	//double rho = 0;
	double tmp_z, tmp_pz;

	std::default_random_engine e1;
	e1.seed(curTime + beam_label * 10000019 + (callTime++) * 1000 + (i + 1) * 1);
	std::normal_distribution<> n1(0, sigma_z);

	std::default_random_engine e2;
	e2.seed(curTime + beam_label * 10000019 + (callTime++) * 1000 + (i + 1) * 1);
	std::normal_distribution<> n2(0, sigma_pz);

	Particle host_particle;
	host_particle.mem_allocate_cpu(Np);

	//callCuda(cudaMemcpy(host_bunch, dev_bunch, Np * sizeof(Particle), cudaMemcpyDeviceToHost));
	particle_copy(host_particle, dev_particle, Np, cudaMemcpyDeviceToHost, "dist");

	for (int j = 0; j < Np; ++j)
	{
		tmp_z = n1(e1);
		tmp_pz = n2(e2);

		if (tmp_z >= (-4 * sigma_z) && tmp_z <= (4 * sigma_z))
		{
			host_particle.z[j] = tmp_z;
			host_particle.pz[j] = tmp_pz;
		}
		else
		{
			--j;
		}
	}

	if (is_set_specified_coordinate) {

		for (size_t i = 0; i < specified_coordinate.size(); i++)
		{
			host_particle.z[i] = specified_coordinate[i][4];
			host_particle.pz[i] = specified_coordinate[i][5];
		}
	}

	//callCuda(cudaMemcpy(dev_bunch, host_bunch, Np * sizeof(Particle), cudaMemcpyHostToDevice));
	particle_copy(dev_particle, host_particle, Np, cudaMemcpyHostToDevice, "dist");

	host_particle.mem_free_cpu();
	spdlog::get("logger")->info("[Injection] The initial longitudinal Gaussian distribution of {} beam-{} bunch-{} has been genetated successfully.",
		beam_name, beamId, bunchId);

}


void Injection::generate_longitudinal_uniform_distribution() {

	//	Generate particle's z position and momentum.
	//	Here we think z follows a uniform distribution and pz follows a Gaussian distribution.
	//	The uniform distribution of z ranges from 0 to sigmaz.

	spdlog::get("logger")->info("[Injection] The initial longitudinal uniform distribution of {} beam-{} bunch-{} is begin generated ...",
		beam_name, beamId, bunchId);

	int i = bunchId;
	int beam_label = beamId;

	double sigma_z = sigmaz;	// Acctually, this is the range of uniform distribution, not RMS value
	double sigma_pz = dp;
	//double rho = 0;
	double tmp_z, tmp_pz;

	std::default_random_engine e1;
	e1.seed(curTime + beam_label * 10000019 + (callTime++) * 1000 + (i + 1) * 1);
	std::uniform_real_distribution<> n1(-0.5 * sigma_z, 0.5 * sigma_z);

	std::default_random_engine e2;
	e2.seed(curTime + beam_label * 10000019 + (callTime++) * 1000 + (i + 1) * 1);
	std::normal_distribution<> n2(0, sigma_pz);

	Particle host_particle;
	host_particle.mem_allocate_cpu(Np);

	//callCuda(cudaMemcpy(host_bunch, dev_bunch, Np * sizeof(Particle), cudaMemcpyDeviceToHost));
	particle_copy(host_particle, dev_particle, Np, cudaMemcpyDeviceToHost, "dist");

	for (int j = 0; j < Np; ++j)
	{
		tmp_z = n1(e1);
		tmp_pz = n2(e2);

		if (tmp_z >= (-0.5 * sigma_z) && tmp_z <= (0.5 * sigma_z))
		{
			host_particle.z[j] = tmp_z;
			host_particle.pz[j] = tmp_pz;
		}
		else
		{
			--j;
		}
	}

	if (is_set_specified_coordinate) {

		for (size_t i = 0; i < specified_coordinate.size(); i++)
		{
			host_particle.z[i] = specified_coordinate[i][4];
			host_particle.pz[i] = specified_coordinate[i][5];
		}
	}

	//callCuda(cudaMemcpy(dev_bunch, host_bunch, Np * sizeof(Particle), cudaMemcpyHostToDevice));
	particle_copy(dev_particle, host_particle, Np, cudaMemcpyHostToDevice, "dist");

	host_particle.mem_free_cpu();
	spdlog::get("logger")->info("[Injection] The initial longitudinal uniform distribution of {} beam-{} bunch-{} has been genetated successfully.",
		beam_name, beamId, bunchId);

}


void Injection::save_initial_distribution() {

	Particle host_particle;
	host_particle.mem_allocate_cpu(Np);

	//callCuda(cudaMemcpy(host_bunch, dev_bunch, Np * sizeof(Particle), cudaMemcpyDeviceToHost));
	particle_copy(host_particle, dev_particle, Np, cudaMemcpyDeviceToHost, "dist");

	std::filesystem::path path_tmp = dir_save_distribution / (hourMinSec + "_beam" + std::to_string(beamId) + "_" + beam_name + "_bunch" + std::to_string(bunchId)
		+ "_" + std::to_string(Np) + "_hor_" + dist_transverse + "_longi_" + dist_longitudinal
		+ "_Dx_" + std::to_string(Dx) + "_injection.csv");
	std::ofstream file(path_tmp);

	file << "x" << "," << "px" << "," << "y" << "," << "py" << "," << "z" << "," << "pz" << ","
		<< "tag" << "," << "sliceId" << "," << "lostTurn" << "," << "lostPos" << std::endl;

	for (int j = 0; j < Np; j++) {
		file << std::setprecision(10)
			<< host_particle.x[j] << ","
			<< host_particle.px[j] << ","
			<< host_particle.y[j] << ","
			<< host_particle.py[j] << ","
			<< host_particle.z[j] << ","
			<< host_particle.pz[j] << ","
			<< host_particle.tag[j] << ","
			<< host_particle.sliceId[j] << ","
			<< host_particle.lostTurn[j] << ","
			<< host_particle.lostPos[j] << "\n";
	}
	file.close();
	host_particle.mem_free_cpu();

	spdlog::get("logger")->info("[Injection] Initial {} distribution of {} beam-{} bunch-{} has been saved to {}.",
		dist_transverse, beam_name, beamId, bunchId, path_tmp.string());

}

void Injection::add_Dx() {

	if (fabs(Dx) < 1e-10 && fabs(Dpx) < 1e-10)
	{
		return;
	}

	spdlog::get("logger")->info("[Injection] Dx = {}, Dpx = {} of {} beam-{} bunch-{} is begin added ...",
		Dx, Dpx, beam_name, beamId, bunchId);

	Particle host_particle;
	host_particle.mem_allocate_cpu(Np);

	//callCuda(cudaMemcpy(host_bunch, dev_bunch, Np * sizeof(Particle), cudaMemcpyDeviceToHost));
	particle_copy(host_particle, dev_particle, Np, cudaMemcpyDeviceToHost, "dist");

	for (int j = 0; j < Np; ++j)
	{
		host_particle.x[j] += Dx * host_particle.pz[j];
		host_particle.px[j] += Dpx * host_particle.pz[j];
	}

	//callCuda(cudaMemcpy(dev_bunch, host_bunch, Np * sizeof(Particle), cudaMemcpyHostToDevice));
	particle_copy(dev_particle, host_particle, Np, cudaMemcpyHostToDevice, "dist");

	host_particle.mem_free_cpu();
	spdlog::get("logger")->info("[Injection] Dispersion = {} of {} beam-{} bunch-{} has been genetated successfully.",
		Dx, beam_name, beamId, bunchId);
}


void Injection::add_offset() {

	if (is_offset_x)
	{
		spdlog::get("logger")->info("[Injection] Offset x = {} of {} beam-{} bunch-{} is begin added ...",
			offset_x, beam_name, beamId, bunchId);

		Particle host_particle;
		host_particle.mem_allocate_cpu(Np);

		//callCuda(cudaMemcpy(host_bunch, dev_bunch, Np * sizeof(Particle), cudaMemcpyDeviceToHost));
		particle_copy(host_particle, dev_particle, Np, cudaMemcpyDeviceToHost, "dist");

		for (int j = 0; j < Np; ++j)
		{
			host_particle.x[j] += offset_x;
		}

		//callCuda(cudaMemcpy(dev_bunch, host_bunch, Np * sizeof(Particle), cudaMemcpyHostToDevice));
		particle_copy(dev_particle, host_particle, Np, cudaMemcpyHostToDevice, "dist");

		host_particle.mem_free_cpu();
		spdlog::get("logger")->info("[Injection] Offset x = {} of {} beam-{} bunch-{} has been genetated successfully.",
			offset_x, beam_name, beamId, bunchId);
	}

	if (is_offset_y)
	{
		spdlog::get("logger")->info("[Injection] Offset y = {} of {} beam-{} bunch-{} is begin added ...",
			offset_y, beam_name, beamId, bunchId);

		Particle host_particle;
		host_particle.mem_allocate_cpu(Np);

		//callCuda(cudaMemcpy(host_bunch, dev_bunch, Np * sizeof(Particle), cudaMemcpyDeviceToHost));
		particle_copy(host_particle, dev_particle, Np, cudaMemcpyDeviceToHost, "dist");

		for (int j = 0; j < Np; ++j)
		{
			host_particle.y[j] += offset_y;
		}

		//callCuda(cudaMemcpy(dev_bunch, host_bunch, Np * sizeof(Particle), cudaMemcpyHostToDevice));
		particle_copy(dev_particle, host_particle, Np, cudaMemcpyHostToDevice, "dist");

		host_particle.mem_free_cpu();
		spdlog::get("logger")->info("[Injection] Offset y = {} of {} beam-{} bunch-{} has been genetated successfully.",
			offset_y, beam_name, beamId, bunchId);
	}
}

void Injection::print_config() {


}