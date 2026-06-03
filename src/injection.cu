#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

#include "constant.h"
#include "general.h"
#include "injection.h"

Injection::Injection(const Parameter& para, int input_beamId, Bunch& Bunch, std::string obj_name) : bunchRef(Bunch)
{
	// std::cout << "pointer 2 " << std::hex << Bunch.dev_bunch << std::endl;
	// std::cout << "pointer 3 " << std::hex << dev_bunch << std::endl;
	name = obj_name;
	dev_particle = Bunch.dev_particle;
	// std::cout << "pointer 4 " << std::hex << dev_bunch << std::endl;

	Np = Bunch.Np;

	beamId = input_beamId;
	bunchId = Bunch.bunchId;

	dir_load_distribution = para.dir_load_distribution;
	dir_save_distribution = para.dir_output_distribution;
	hourMinSec = para.hourMinSec;

	beam_name = para.beam_name[input_beamId];
	rho = para.circumference / (2 * PassConstant::PI);
	gammat = Bunch.gammat;
	gamma = Bunch.gamma;
	beta = Bunch.beta;
	t0 = Bunch.t0;
	Ek = Bunch.Ek;
	m0 = Bunch.m0;
	qm_ratio = Bunch.qm_ratio;
	circum = para.circumference;

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	std::string key_bunch = "bunch" + std::to_string(Bunch.bunchId);

	try
	{
		s = data.at("Sequence").at("Injection").at("S (m)");
		// name = data.at("Sequence").at("Injection").at("Command");
		if (fabs(s) > 1e-10)
		{
			std::string error_msg =
				"[Injection] The position of injection point (simulation start point) should be 0, but now is : " + std::to_string(s);
			throw std::invalid_argument(error_msg);
		}

		startTurn = 1;	// The first turn start from 1 not 0.
		endTurn = data.at("Sequence").at("Injection").at(key_bunch).at("Total Injection Turns");
		endTurn += 1;
		intervalTurn = data.at("Sequence").at("Injection").at(key_bunch).at("Injection Interval");
		for (int t = startTurn; t < endTurn; t += intervalTurn)
		{
			inject_turns.push_back(t);
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
		dp = data.at("Sequence").at("Injection").at(key_bunch).at("Sigma dp/p");

		sigmax = sqrt(betax * emitx);
		sigmay = sqrt(betay * emity);

		sigmapx = sqrt(gammax * emitx);
		sigmapy = sqrt(gammay * emity);

		dist_transverse = data.at("Sequence").at("Injection").at(key_bunch).at("Transverse dist");
		dist_longitudinal = data.at("Sequence").at("Injection").at(key_bunch).at("Longitudinal dist");

		rf_voltage = data.at("Sequence").at("Injection").at(key_bunch).at("RF Voltage (V)");
		rf_phi = data.at("Sequence").at("Injection").at(key_bunch).at("RF Phase (rad)");
		harmonic_num = data.at("Sequence").at("Injection").at(key_bunch).at("Harmonic Number");
		harmonic_id = data.at("Sequence").at("Injection").at(key_bunch).at("Harmonic ID of this bunch");
		rf_delta_dist = data.at("Sequence").at("Injection").at(key_bunch).at("RF S Position Refer to Inj. Point (m)");

		z_shift = circum / harmonic_num * harmonic_id;

		is_offset_x = data.at("Sequence").at("Injection").at(key_bunch).at("Offset x").at("Is Offset");
		is_offset_y = data.at("Sequence").at("Injection").at(key_bunch).at("Offset y").at("Is Offset");
		if (is_offset_x)
		{
			is_offset_x_fromFile = data.at("Sequence").at("Injection").at(key_bunch).at("Offset x").at("Is Load From File");

			if (is_offset_x_fromFile)
			{
				offset_x_timekind = data.at("Sequence").at("Injection").at(key_bunch).at("Offset x").at("File Time Kind");
				offset_x_filepath = data.at("Sequence").at("Injection").at(key_bunch).at("Offset x").at("File Path");
				std::vector<std::vector<double>> offset_data = read_file_data(offset_x_filepath);
				offset_time_x = offset_data[0];
				offset_x = offset_data[1];
				offset_px = offset_data[2];
			}
			else
			{
				double offset_x_tmp = data.at("Sequence").at("Injection").at(key_bunch).at("Offset x").at("Offset Position (m)");
				double offset_px_tmp = data.at("Sequence").at("Injection").at(key_bunch).at("Offset x").at("Offset Momentum (rad)");
				offset_x.push_back(offset_x_tmp);
				offset_px.push_back(offset_px_tmp);
			}
		}
		if (is_offset_y)
		{
			is_offset_y_fromFile = data.at("Sequence").at("Injection").at(key_bunch).at("Offset y").at("Is Load From File");

			if (is_offset_y_fromFile)
			{
				offset_y_timekind = data.at("Sequence").at("Injection").at(key_bunch).at("Offset y").at("File Time Kind");
				offset_y_filepath = data.at("Sequence").at("Injection").at(key_bunch).at("Offset y").at("File Path");
				std::vector<std::vector<double>> offset_data = read_file_data(offset_y_filepath);
				offset_time_y = offset_data[0];
				offset_y = offset_data[1];
				offset_py = offset_data[2];
			}
			else
			{
				double offset_y_tmp = data.at("Sequence").at("Injection").at(key_bunch).at("Offset y").at("Offset Position (m)");
				double offset_py_tmp = data.at("Sequence").at("Injection").at(key_bunch).at("Offset y").at("Offset Momentum (rad)");
				offset_y.push_back(offset_y_tmp);
				offset_py.push_back(offset_py_tmp);
			}
		}

		is_load_dist = data.at("Sequence").at("Injection").at(key_bunch).at("Is Load Distribution from File");
		load_dist_filepath = data.at("Sequence").at("Injection").at(key_bunch).at("Distribution File Path");

		is_save_initial_dist = data.at("Sequence").at("Injection").at(key_bunch).at("Is Save Initial Distribution");

		if (data.at("Sequence").at("Injection").at(key_bunch).at("Insert Particle Coordinate").size() > 0)
		{
			is_set_specified_coordinate = true;

			for (size_t i = 0; i < data.at("Sequence").at("Injection").at(key_bunch).at("Insert Particle Coordinate").size(); i++)
			{
				double x_tmp = data.at("Sequence").at("Injection").at(key_bunch).at("Insert Particle Coordinate")[i][0];
				double px_tmp = data.at("Sequence").at("Injection").at(key_bunch).at("Insert Particle Coordinate")[i][1];
				double y_tmp = data.at("Sequence").at("Injection").at(key_bunch).at("Insert Particle Coordinate")[i][2];
				double py_tmp = data.at("Sequence").at("Injection").at(key_bunch).at("Insert Particle Coordinate")[i][3];
				double z_tmp = data.at("Sequence").at("Injection").at(key_bunch).at("Insert Particle Coordinate")[i][4];
				double dp_tmp = data.at("Sequence").at("Injection").at(key_bunch).at("Insert Particle Coordinate")[i][5];

				specified_coordinate.push_back({x_tmp, px_tmp, y_tmp, py_tmp, z_tmp, dp_tmp});
			}
		}
	}
	catch (json::exception e)
	{
		std::string error_msg = "[Injection] Error when parsing injection parameters: " + std::string(e.what());
		throw std::runtime_error(error_msg);
	}

	Bunch.sigmaz = sigmaz;
	Bunch.dp = dp;
	Bunch.z_shift = z_shift;

	Bunch.dist_transverse = dist_transverse;
	Bunch.dist_longitudinal = dist_longitudinal;

	std::random_device rd;	// get system real random
	e1.seed(rd());
}

void Injection::execute(int turn)
{
	auto logger = spdlog::get("logger");

	// logger->debug("Injection action");

	auto it = std::find(inject_turns.begin(), inject_turns.end(), turn);
	if (it == inject_turns.end())
	{
		return;
	}

	Ek = bunchRef.Ek;
	t0 = bunchRef.t0;
	beta = bunchRef.beta;
	gamma = bunchRef.gamma;

	Np_inj_curTurn = int(Np / inject_turns.size());
	if (turn == 1 && Np_inj_curTurn * inject_turns.size() != Np)
	{
		Np_inj_curTurn += (Np - Np_inj_curTurn * inject_turns.size());
		logger->info(
			"[Injection] Since the total number of particles {} cannot be divided exactly by the number of injection turns {}, we will inject {} "
			"particles in the first turn and {} particles in the rest turns.",
			Np, inject_turns.size(), Np_inj_curTurn, int(Np / inject_turns.size()));
	}

	if (is_load_dist)
	{
		load_distribution();
	}
	else
	{
		if ("kv" == to_lower(dist_transverse))
		{
			generate_transverse_KV_distribution();
		}
		else if ("gaussian" == to_lower(dist_transverse))
		{
			generate_transverse_Gaussian_distribution();
		}
		else if ("uniform" == to_lower(dist_transverse))
		{
			generate_transverse_uniform_distribution();
		}
		else
		{
			std::string error_msg = "[Injection] Sorry, we don't support transverse distribution type " + dist_transverse;
			throw std::invalid_argument(error_msg);
		}

		if ("gaussian" == to_lower(dist_longitudinal))
		{
			generate_longitudinal_Gaussian_distribution();
		}
		else if ("coasting" == to_lower(dist_longitudinal))
		{
			generate_longitudinal_coasting_distribution();
		}
		else if ("matchZ" == to_lower(dist_longitudinal))
		{
			generate_longitudinal_matchZ_distribution();
		}
		else if ("matchDp" == to_lower(dist_longitudinal))
		{
			generate_longitudinal_matchDp_distribution();
		}
		else
		{
			std::string error_msg = "[Injection] Sorry, we don't support longitudinal distribution type " + dist_longitudinal;
			throw std::invalid_argument(error_msg);
		}
	}

	add_offset(turn, t0);

	Np_inj_total += Np_inj_curTurn;

	if (turn == inject_turns.back())
	{
		insert_particle();

		if (is_save_initial_dist)
		{
			save_initial_distribution();
		}
	}
}

void Injection::load_distribution()
{
	/*
	Read particle coordinate from file.

	The delimeter of the file could be ",", " ", or "\t".

	The order of the columns must be x, px, y, py, z, dp/p, tag, sliceId, lostTurn, lostPos. If the file only has 1 colums, the data will be assigned
	to x value, and the other particle coordiantes will be set to default value.
		- The tag is an integer starting from 1 and each particle has a unique tag.
		- The sliceId is an integer starting from 0 and indicates which slice the particle belongs to in longitudinal direction.
		- The lostTurn is an integer indicating the turn when the particle is lost. If the particle is not lost, the value of lostTurn should be -1.
		- The lostPos is a double indicating the position where the particle is lost. If the particle is not lost, the value of lostPos should be -1.

	If there is a char or string in row, this row will be skipped.
	*/

	std::filesystem::path dist_path(load_dist_filepath);

	if (std::filesystem::exists(dist_path))
	{
		if (load_dist_filepath.find(beam_name) == std::string::npos)
			spdlog::get("logger")->warn("[Injection] Please be careful to confirm that the file is {} distribution: {}.", beam_name,
										dist_path.string());
		if (load_dist_filepath.find(dist_transverse) == std::string::npos)
			spdlog::get("logger")->warn("[Injection] Please be careful to confirm that the file is {} distribution: {}.", dist_transverse,
										dist_path.string());
		if (load_dist_filepath.find(std::to_string(Np)) == std::string::npos)
			spdlog::get("logger")->warn("[Injection] Please be careful to confirm that the file contain {} particles: {}.", Np, dist_path.string());

		spdlog::get("logger")->info("[Injection] Loading distribution file: {}", dist_path.string());

		Particle host_particle;
		host_particle.mem_allocate_cpu(Np_inj_curTurn);

		std::ifstream input(dist_path);
		if (!input.is_open())
		{
			std::string error_msg = "[Injection] Cannot open file: " + dist_path.string();
			throw std::runtime_error(error_msg);
		}

		std::string line;
		int n_loaded = 0;
		int start_index = Np_inj_total;
		int current_index = 0;

		int row = 0;

		while (std::getline(input, line))
		{
			row++;

			auto data_line = split_line(line);
			if (data_line.empty()) continue;

			double a[6] = {0.};
			int a_tag = current_index + 1, a_lostTurn = -1, a_sliceId = 0;
			double a_lostPos = -1.0;
			bool parse_ok = true;

			try
			{
				for (int k = 0; k < 6; ++k)
				{
					if (k < data_line.size()) a[k] = std::stod(data_line[k]);
				}
				if (data_line.size() > 6) a_tag = std::stoi(data_line[6]);
				if (data_line.size() > 7) a_sliceId = std::stoi(data_line[7]);
				if (data_line.size() > 8) a_lostTurn = std::stoi(data_line[8]);
				if (data_line.size() > 9) a_lostPos = std::stod(data_line[9]);
			}
			catch (const std::exception& e)
			{
				spdlog::get("logger")->warn("[Injection] Error parsing line {} in distribution file: {}. Error message: {}", row, dist_path.string(),
											e.what());
				parse_ok = false;
			}

			if (current_index >= start_index && current_index < (start_index + Np_inj_curTurn))
			{
				if (parse_ok)
				{
					host_particle.x[n_loaded] = a[0];
					host_particle.px[n_loaded] = a[1];
					host_particle.y[n_loaded] = a[2];
					host_particle.py[n_loaded] = a[3];
					host_particle.z[n_loaded] = a[4];
					host_particle.pz[n_loaded] = a[5];
					host_particle.tag[n_loaded] = a_tag;
					host_particle.sliceId[n_loaded] = a_sliceId;
					host_particle.lostTurn[n_loaded] = a_lostTurn;
					host_particle.lostPos[n_loaded] = a_lostPos;
				}
				else
				{
					host_particle.x[n_loaded] = 0.;
					host_particle.px[n_loaded] = 0.;
					host_particle.y[n_loaded] = 0.;
					host_particle.py[n_loaded] = 0.;
					host_particle.z[n_loaded] = 0.;
					host_particle.pz[n_loaded] = 0.;
					host_particle.tag[n_loaded] =
						-1 * (current_index + 1);  // load failed particle with negative tag to distinguish from normal particle
					host_particle.sliceId[n_loaded] = 0;
					host_particle.lostTurn[n_loaded] = -1;
					host_particle.lostPos[n_loaded] = 0.;

					spdlog::get("logger")->warn(
						"[Injection] Since there is an error parsing line {} in distribution file {}, we set the tag of this particle to {}.", row,
						dist_path.string(), -1 * (current_index + 1));
				}
				n_loaded++;
			}
			current_index++;
		}

		input.close();

		if (n_loaded != Np_inj_curTurn)
		{
			std::string error_msg = "[Injection] Number of loaded particles (" + std::to_string(n_loaded) +
									") does not match expected number of particles (" + std::to_string(Np_inj_curTurn) + "). Start index is (" +
									std::to_string(start_index) + ")";
			throw std::runtime_error(error_msg);
		}

		particle_copy(dev_particle, host_particle, Np_inj_curTurn, cudaMemcpyHostToDevice, "dist", start_index, 0);

		host_particle.mem_free_cpu();

		spdlog::get("logger")->info("[Injection] Distribution file {} has been loadded successfully to {} beam-{} bunch-{}.", dist_path.string(),
									beam_name, beamId, bunchId);
	}
	else
	{
		std::string error_msg = "[Injection] We don't find distribution file: " + dist_path.string();
		throw std::runtime_error(error_msg);
	}
}

void Injection::generate_transverse_KV_distribution()
{
	/* This menthod is derived from "Particle - in - cell code BEAMPATH for beam dynamics simulations in linear accelerators and beamlines"*/
	// The two beams shoule have different seed values to generate different random values.
	// This is 4-D generator.

	spdlog::get("logger")->info("[Injection] The initial transverse KV distribution of {} beam-{} bunch-{} is begin generated ...", beam_name, beamId,
								bunchId);

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

	// [-1��, 1��] = 0.6826894921370859, [-4��, 4��] = 0.9999366575163338
	// [-2��, 2��] = 0.9544997361036416, [-5��, 5��] = 0.9999994266968562
	// [-3��, 3��] = 0.9973002039367398, [-6��, 6��] = 0.9999999980268246
	double x_max = 4 * sigmax;
	double x_min = -4 * sigmax;
	double y_max = 4 * sigmay;
	double y_min = -4 * sigmay;

	int i = bunchId;
	int beam_label = beamId;

	std::uniform_real_distribution<> u1(1e-15, 1.0 - 1e-15);
	std::uniform_real_distribution<> u2(0, 1);

	Particle host_particle;
	host_particle.mem_allocate_cpu(Np_inj_curTurn);

	for (int j = 0; j < Np_inj_curTurn; ++j)
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
		double random_beta_x = u2(e1);
		double random_beta_y = u2(e1);

		double F = emittence_x;

		nu = emittence_x / emittence_y;

		sigma11_x = emittence_x * beta_x_twiss;
		sigma12_x = -emittence_x * alpha_x_twiss;
		sigma22_x = emittence_x * gamma_x_twiss;

		sigma11_y = emittence_y * beta_y_twiss;
		sigma12_y = -emittence_y * alpha_y_twiss;
		sigma22_y = emittence_y * gamma_y_twiss;

		phi_x =
			0.5 *
			atan2(2 * alpha_x_twiss,
				  gamma_x_twiss -
					  beta_x_twiss);  // https://agenda.linearcollider.org/event/6258/contributions/29168/attachments/24202/37474/linear_dynamics.pdf
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

	int start_index = Np_inj_total;
	particle_copy(dev_particle, host_particle, Np_inj_curTurn, cudaMemcpyHostToDevice, "dist", start_index, 0);

	host_particle.mem_free_cpu();

	spdlog::get("logger")->info("[Injection] The initial transverse KV distribution of {} beam-{} bunch-{} has been genetated successfully.",
								beam_name, beamId, bunchId);
}

void Injection::generate_transverse_Gaussian_distribution()
{
	spdlog::get("logger")->info("[Injection] The initial transverse Gaussian distribution of {} beam-{} bunch-{} is begin generated ...", beam_name,
								beamId, bunchId);

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

	// [-1��, 1��] = 0.6826894921370859, [-4��, 4��] = 0.9999366575163338
	// [-2��, 2��] = 0.9544997361036416, [-5��, 5��] = 0.9999994266968562
	// [-3��, 3��] = 0.9973002039367398, [-6��, 6��] = 0.9999999980268246
	double x_max = 4 * sigmax;
	double x_min = -4 * sigmax;
	double y_max = 4 * sigmay;
	double y_min = -4 * sigmay;

	int i = bunchId;
	int beam_label = beamId;

	std::uniform_real_distribution<> u1(1e-15, 1.0 - 1e-15);

	Particle host_particle;
	host_particle.mem_allocate_cpu(Np_inj_curTurn);

	for (int j = 0; j < Np_inj_curTurn; ++j)
	{
		double x, px, y, py;
		double Xm, thetaXm, Ym, thetaYm;
		double a_x, a_y, u_x, u_y, v_x, v_y;
		double alpha_x, alpha_y, chi_x, chi_y;
		double pi = PassConstant::PI;

		double random_s1_x = u1(e1);  // range (0,1)
		double random_s1_y = u1(e1);  // range (0,1)
		double random_s2_x = u1(e1);  // range (0,1)
		double random_s2_y = u1(e1);  // range (0,1)

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
	}

	int start_index = Np_inj_total;
	particle_copy(dev_particle, host_particle, Np_inj_curTurn, cudaMemcpyHostToDevice, "dist", start_index, 0);

	host_particle.mem_free_cpu();

	spdlog::get("logger")->info("[Injection] The initial transverse Gaussian distribution of {} beam-{} bunch-{} has been genetated successfully.",
								beam_name, beamId, bunchId);
}

void Injection::generate_transverse_uniform_distribution()
{
	spdlog::get("logger")->info("[Injection] The initial transverse uniform distribution of {} beam-{} bunch-{} is begin generated ...", beam_name,
								beamId, bunchId);

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

	// [-1��, 1��] = 0.6826894921370859, [-4��, 4��] = 0.9999366575163338
	// [-2��, 2��] = 0.9544997361036416, [-5��, 5��] = 0.9999994266968562
	// [-3��, 3��] = 0.9973002039367398, [-6��, 6��] = 0.9999999980268246
	double x_max = 4 * sigmax;
	double x_min = -4 * sigmax;
	double y_max = 4 * sigmay;
	double y_min = -4 * sigmay;

	int i = bunchId;
	int beam_label = beamId;

	std::uniform_real_distribution<> u1(1e-15, 1.0 - 1e-15);

	Particle host_particle;
	host_particle.mem_allocate_cpu(Np_inj_curTurn);

	for (int j = 0; j < Np_inj_curTurn; ++j)
	{
		double m = 1;
		double x, px, y, py;
		double Xm, thetaXm, Ym, thetaYm;
		double Xl, Xlpx, Yl, Ylpy;
		double a_x, a_y, u_x, u_y, v_x, v_y;
		double alpha_x, alpha_y, chi_x, chi_y;
		double pi = PassConstant::PI;

		double random_s1_x = u1(e1);  // range (0,1)
		double random_s2_x = u1(e1);  // range (0,1)
		double random_s1_y = u1(e1);  // range (0,1)
		double random_s2_y = u1(e1);  // range (0,1)

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
		// printf("Xm = %f, thetaXm = %f, Xl = %f, Xlpx = %f, alphaX = %f, ux = %f, vx = %f, x = %f, px = %f\n", Xm, thetaXm, Xl, Xlpx, alpha_x, u_x,
		// v_x, x, px);

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
	}

	int start_index = Np_inj_total;
	particle_copy(dev_particle, host_particle, Np_inj_curTurn, cudaMemcpyHostToDevice, "dist", start_index, 0);

	host_particle.mem_free_cpu();

	spdlog::get("logger")->info("[Injection] The initial transverse uniform distribution of {} beam-{} bunch-{} has been genetated successfully.",
								beam_name, beamId, bunchId);
}

void Injection::generate_longitudinal_Gaussian_distribution()
{
	//	Generate particle's z position and momentum.
	//	Here we think the correlation coefficient of 2D Gaussian distribution rho = 0

	spdlog::get("logger")->info("[Injection] The initial longitudinal Gaussian distribution of {} beam-{} bunch-{} is begin generated ...", beam_name,
								beamId, bunchId);

	int i = bunchId;
	int beam_label = beamId;

	double sigma_z = sigmaz;
	double sigma_pz = dp;
	// double rho = 0;
	double tmp_z, tmp_pz;

	std::normal_distribution<> n1(0, sigma_z);
	std::normal_distribution<> n2(0, sigma_pz);

	Particle host_particle;
	host_particle.mem_allocate_cpu(Np_inj_curTurn);

	int start_index = Np_inj_total;
	particle_copy(host_particle, dev_particle, Np_inj_curTurn, cudaMemcpyDeviceToHost, "dist", 0, start_index);

	for (int j = 0; j < Np_inj_curTurn; ++j)
	{
		tmp_z = n1(e1);
		tmp_pz = n2(e1);

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

	particle_copy(dev_particle, host_particle, Np_inj_curTurn, cudaMemcpyHostToDevice, "dist", start_index, 0);

	host_particle.mem_free_cpu();
	spdlog::get("logger")->info("[Injection] The initial longitudinal Gaussian distribution of {} beam-{} bunch-{} has been genetated successfully.",
								beam_name, beamId, bunchId);
}

void Injection::generate_longitudinal_coasting_distribution()
{
	//	Generate particle's z position and momentum.
	//	Here we think z follows a uniform distribution and pz follows a Gaussian distribution.
	//	The uniform distribution of z ranges from -sigmaz/2 to sigmaz/2.

	spdlog::get("logger")->info("[Injection] The initial longitudinal coasting distribution of {} beam-{} bunch-{} is begin generated ...", beam_name,
								beamId, bunchId);

	int i = bunchId;
	int beam_label = beamId;

	double sigma_z = sigmaz;  // Acctually, this is the range of uniform distribution, not RMS value
	double sigma_pz = dp;
	double tmp_z, tmp_pz;

	std::uniform_real_distribution<> n1(-0.5 * sigma_z, 0.5 * sigma_z);
	std::normal_distribution<> n2(0, sigma_pz);

	Particle host_particle;
	host_particle.mem_allocate_cpu(Np_inj_curTurn);

	int start_index = Np_inj_total;
	particle_copy(host_particle, dev_particle, Np_inj_curTurn, cudaMemcpyDeviceToHost, "dist", 0, start_index);

	for (int j = 0; j < Np_inj_curTurn; ++j)
	{
		tmp_z = n1(e1);
		tmp_pz = n2(e1);

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

	particle_copy(dev_particle, host_particle, Np_inj_curTurn, cudaMemcpyHostToDevice, "dist", start_index, 0);

	host_particle.mem_free_cpu();
	spdlog::get("logger")->info("[Injection] The initial longitudinal coasting distribution of {} beam-{} bunch-{} has been genetated successfully.",
								beam_name, beamId, bunchId);
}

void Injection::generate_longitudinal_matchZ_distribution()
{
	//	Generate particle's z position and momentum.
	//	Use the method in PyHEADTAIL.

	spdlog::get("logger")->info("[Injection] The initial longitudinal z-matched distribution of {} beam-{} bunch-{} is begin generated ...",
								beam_name, beamId, bunchId);

	int i = bunchId;
	int beam_label = beamId;

	double sigma_z = sigmaz;
	double sigma_pz = dp;
	double tmp_z, tmp_pz;

	double zmax = getZMax();
	double zmin = getZMin();
	double dp = getDeltaPMax();
	double Hmax = getHamiltonianPhi(getUFPPhi(), 0);
	double H0 = 0.0;

	// Check the sigmaz whether the sigmaz > sigma_max
	spdlog::get("logger")->info("Match sigma z, waiting ...");
	double sigma_max = getSigmaZ(zmax);
	double sig = sigma_z;
	if (sig > sigma_max)
	{
		spdlog::get("logger")->warn("Sigma z is larger than the maximal!");
		spdlog::get("logger")->warn("Use the 0.99*sigma_max = {}!", 0.99 * sigma_max);
		sig = 0.99 * sigma_max;	 // if sigmaz > sigma_max, use sigmaz = 0.99 * sigma_max
	}

	// Solve the matched H0
	auto func = [=](double x) -> double { return getSigmaZ(x) - sig; };
	double x2 = sig;
	double x1;
	if (func(x2) < 0)
		x1 = sig * 10.0;
	else
		x1 = sig / 10.0;
	double root = brent(func, x1, x2);
	H0 = H0FromZ(root);
	spdlog::get("logger")->info("Match sigma z, finished!");

	std::uniform_real_distribution<> n1(0, 1);

	Particle host_particle;
	host_particle.mem_allocate_cpu(Np_inj_curTurn);

	int start_index = Np_inj_total;
	particle_copy(host_particle, dev_particle, Np_inj_curTurn, cudaMemcpyDeviceToHost, "dist", 0, start_index);

	for (int j = 0; j < Np_inj_curTurn; ++j)
	{
		double u = 0.0;
		double v = 0.0;
		double s = 0.0;

		do
		{
			u = n1(e1) * (zmax - zmin) + zmin;
			v = n1(e1) * 2.0 * dp - dp;
			s = n1(e1);
		} while (s > psi(u, v, H0, Hmax) ||
				 fabs(getHamiltonianZ(u, v)) > 0.9 * fabs(Hmax));  // for stability, limit particles in the 0.9 times bucket

		tmp_z = u;
		tmp_pz = v;

		if (tmp_z >= (-0.5 * -4 * sigma_z) && tmp_z <= (0.5 * 4 * sigma_z))
		{
			host_particle.z[j] = tmp_z;
			host_particle.pz[j] = tmp_pz;
		}
		else
		{
			--j;
		}
	}

	particle_copy(dev_particle, host_particle, Np_inj_curTurn, cudaMemcpyHostToDevice, "dist", start_index, 0);

	host_particle.mem_free_cpu();
	spdlog::get("logger")->info("[Injection] The initial longitudinal uniform distribution of {} beam-{} bunch-{} has been genetated successfully.",
								beam_name, beamId, bunchId);
}

void Injection::generate_longitudinal_matchDp_distribution()
{
	//	Generate particle's z position and momentum.
	//	Use the method in PyHEADTAIL.

	spdlog::get("logger")->info("[Injection] The initial longitudinal Dp-matched distribution of {} beam-{} bunch-{} is begin generated ...",
								beam_name, beamId, bunchId);

	int i = bunchId;
	int beam_label = beamId;

	double sigma_z = sigmaz;
	double sigma_pz = dp;
	double tmp_z, tmp_pz;

	double zmax = getZMax();
	double zmin = getZMin();
	double dp = getDeltaPMax();
	double Hmax = getHamiltonianPhi(getUFPPhi(), 0);
	double H0 = 0.0;

	// Check the sigmaz whether the sigmadp > sigma_max
	spdlog::get("logger")->info("Match sigma dp, waiting ...");
	double sigma_max = getSigmaDp(dp);
	double sig = sigma_pz;
	if (sig > sigma_max)
	{
		spdlog::get("logger")->warn("Sigma dp is larger than the maximal!");
		spdlog::get("logger")->warn("Use the 0.99*sigma_max = {}!", 0.99 * sigma_max);
		sig = 0.99 * sigma_max;	 // if sigmaz > sigma_max, use sigmaz = 0.99 * sigma_max
	}

	// Solve the matched H0
	auto func = [=](double x) -> double { return getSigmaDp(x) - sig; };
	double x2 = sig;
	double x1;
	if (func(x2) < 0)
		x1 = sig * 10.0;
	else
		x1 = sig / 10.0;
	double root = brent(func, x1, x2);
	H0 = H0FromDeltaP(root);
	spdlog::get("logger")->info("Match dp, finished!");

	std::uniform_real_distribution<> n1(0, 1);

	Particle host_particle;
	host_particle.mem_allocate_cpu(Np_inj_curTurn);

	int start_index = Np_inj_total;
	particle_copy(host_particle, dev_particle, Np_inj_curTurn, cudaMemcpyDeviceToHost, "dist", 0, start_index);

	for (int j = 0; j < Np_inj_curTurn; ++j)
	{
		double u = 0.0;
		double v = 0.0;
		double s = 0.0;

		do
		{
			u = n1(e1) * (zmax - zmin) + zmin;
			v = n1(e1) * 2.0 * dp - dp;
			s = n1(e1);
		} while (s > psi(u, v, H0, Hmax) ||
				 fabs(getHamiltonianZ(u, v)) > 0.9 * fabs(Hmax));  // for stability, limit particles in the 0.9 times bucket

		tmp_z = u;
		tmp_pz = v;

		if (tmp_z >= (-0.5 * -4 * sigma_z) && tmp_z <= (0.5 * 4 * sigma_z))
		{
			host_particle.z[j] = tmp_z;
			host_particle.pz[j] = tmp_pz;
		}
		else
		{
			--j;
		}
	}

	particle_copy(dev_particle, host_particle, Np_inj_curTurn, cudaMemcpyHostToDevice, "dist", start_index, 0);

	host_particle.mem_free_cpu();
	spdlog::get("logger")->info("[Injection] The initial longitudinal uniform distribution of {} beam-{} bunch-{} has been genetated successfully.",
								beam_name, beamId, bunchId);
}

void Injection::save_initial_distribution()
{
	Particle host_particle;
	host_particle.mem_allocate_cpu(Np);

	particle_copy(host_particle, dev_particle, Np, cudaMemcpyDeviceToHost, "dist");

	std::filesystem::path path_tmp = dir_save_distribution / (hourMinSec + "_beam" + std::to_string(beamId) + "_" + beam_name + "_bunch" +
															  std::to_string(bunchId) + "_" + std::to_string(Np) + "_hor_" + dist_transverse +
															  "_longi_" + dist_longitudinal + "_Dx_" + std::to_string(Dx) + "_injection.csv");
	std::ofstream file(path_tmp);

	file << "x" << "," << "px" << "," << "y" << "," << "py" << "," << "z" << "," << "pz" << ","
		 << "tag" << "," << "sliceId" << "," << "lostTurn" << "," << "lostPos" << std::endl;

	for (int j = 0; j < Np; j++)
	{
		file << std::setprecision(10) << host_particle.x[j] << "," << host_particle.px[j] << "," << host_particle.y[j] << "," << host_particle.py[j]
			 << "," << host_particle.z[j] << "," << host_particle.pz[j] << "," << host_particle.tag[j] << "," << host_particle.sliceId[j] << ","
			 << host_particle.lostTurn[j] << "," << host_particle.lostPos[j] << "\n";
	}
	file.close();
	host_particle.mem_free_cpu();

	spdlog::get("logger")->info("[Injection] Initial {} distribution of {} beam-{} bunch-{} has been saved to {}.", dist_transverse, beam_name,
								beamId, bunchId, path_tmp.string());
}

void Injection::add_offset(int turn, double t0)
{
	if (!is_offset_x && !is_offset_y && !is_offset_z)
	{
		return;
	}

	spdlog::get("logger")->info("[Injection] Offset of {} beam-{} bunch-{} is begin added ...", beam_name, beamId, bunchId);

	Particle host_particle;
	host_particle.mem_allocate_cpu(Np);

	particle_copy(host_particle, dev_particle, Np, cudaMemcpyDeviceToHost, "dist");

	double cur_offset_x = 0.0, cur_offset_px = 0.0;
	double cur_offset_y = 0.0, cur_offset_py = 0.0;
	double cur_offset_pz = 0.0;

	if (is_offset_x && is_offset_x_fromFile)
	{
		std::vector<std::vector<double>> var_y = {offset_x, offset_px};

		if (to_lower(offset_x_timekind) == "turn")
		{
			auto result = linearInterpolate(offset_time_x, var_y, turn);
			cur_offset_x = result[0];
			cur_offset_px = result[1];
		}
		else if (to_lower(offset_x_timekind) == "time")
		{
			auto result = linearInterpolate(offset_time_x, var_y, t0);
			cur_offset_x = result[0];
			cur_offset_px = result[1];
		}
		else
		{
			std::string error_msg =
				"[Injection] The time kind of offset x is " + offset_x_timekind + ", which is invalid. It should be 'turn' or 'time'.";
			throw std::invalid_argument(error_msg);
		}
	}
	else if (is_offset_x && !is_offset_x_fromFile)
	{
		cur_offset_x = offset_x[0];
		cur_offset_px = offset_px[0];
	}

	if (is_offset_y && is_offset_y_fromFile)
	{
		std::vector<std::vector<double>> var_y = {offset_y, offset_py};

		if (offset_y_timekind == "turn")
		{
			auto result = linearInterpolate(offset_time_y, var_y, turn);
			cur_offset_y = result[0];
			cur_offset_py = result[1];
		}
		else if (offset_y_timekind == "time")
		{
			auto result = linearInterpolate(offset_time_y, var_y, t0);
			cur_offset_y = result[0];
			cur_offset_py = result[1];
		}
		else
		{
			std::string error_msg =
				"[Injection] The time kind of offset y is " + offset_y_timekind + ", which is invalid. It should be 'turn' or 'time'.";
			throw std::invalid_argument(error_msg);
		}
	}
	else if (is_offset_y && !is_offset_y_fromFile)
	{
		cur_offset_y = offset_y[0];
		cur_offset_py = offset_py[0];
	}

	for (int j = 0; j < Np; ++j)
	{
		double cur_pz = host_particle.pz[j] + cur_offset_pz;
		host_particle.pz[j] = cur_pz;
		host_particle.z[j] -= (1 / gammat / gammat - 1 / gamma / gamma) * (circum - rf_delta_dist) * cur_pz;

		host_particle.x[j] += (cur_offset_x + cur_pz * Dx);
		host_particle.px[j] += (cur_offset_px + cur_pz * Dpx);
		host_particle.y[j] += cur_offset_y;
		host_particle.py[j] += cur_offset_py;
	}

	particle_copy(dev_particle, host_particle, Np, cudaMemcpyHostToDevice, "dist");

	host_particle.mem_free_cpu();
	spdlog::get("logger")->info("[Injection] Offset of {} beam-{} bunch-{} has been added successfully.", beam_name, beamId, bunchId);
}

void ::Injection::insert_particle()
{
	if (is_set_specified_coordinate)
	{
		spdlog::get("logger")->info(
			"[Injection] Replacing the coordinates of the first '{}' particles with the inserted values of {} beam-{} bunch-{} ...",
			specified_coordinate.size(), beam_name, beamId, bunchId);

		if (specified_coordinate.size() > Np)
		{
			spdlog::get("logger")->warn(
				"[Injection] The number of specified coordinates is {} larger than the number of particles, so we only use the first '{}' specified "
				"coordinates.",
				specified_coordinate.size(), Np, Np);
			specified_coordinate.resize(Np);
		}

		Particle host_particle;
		host_particle.mem_allocate_cpu(Np);

		particle_copy(host_particle, dev_particle, Np, cudaMemcpyDeviceToHost, "dist");

		for (size_t i = 0; i < specified_coordinate.size(); i++)
		{
			host_particle.x[i] = specified_coordinate[i][0];
			host_particle.px[i] = specified_coordinate[i][1];
			host_particle.y[i] = specified_coordinate[i][2];
			host_particle.py[i] = specified_coordinate[i][3];
			host_particle.z[i] = specified_coordinate[i][4];
			host_particle.pz[i] = specified_coordinate[i][5];
		}

		particle_copy(dev_particle, host_particle, Np, cudaMemcpyHostToDevice, "dist");

		host_particle.mem_free_cpu();
		spdlog::get("logger")->info(
			"[Injection] The coordinates of the first '{}' particles with the inserted values of {} beam-{} bunch-{} has been genetated "
			"successfully.",
			specified_coordinate.size(), beam_name, beamId, bunchId);
	}
}

void Injection::print_config() {}

const double Injection::phiFromZ(const double z) { return rf_phi - harmonic_num * z / rho; }

const double Injection::zFromPhi(const double phi) { return rho * (rf_phi - phi) / harmonic_num; }

const double Injection::getInitEta()
{
	return 1.0 / gammat / gammat - 1.0 / gamma / gamma;	 // eta = 1/gamma_t^2 - 1/gamma^2
}

const double Injection::getPhiSeparatrix(const double phi)
{
	double E = Ek + m0;
	double eta = getInitEta();
	double PI = PassConstant::PI;
	double temp =
		-1 * qm_ratio * rf_voltage / PI / beta / beta / E / harmonic_num / eta * (cos(phi) + cos(rf_phi) - (PI - phi - rf_phi) * sin(rf_phi));
	if (temp < 0) temp = 0;
	return sqrt(temp);	// dp_sp = sqrt(-q/m*V/pi/beta^2/E/h/eta*(cos(phi)+cos(phi_s)-(pi-phi-phi_s)*sin(phi_s)))
}

const double Injection::getZSeparatrix(const double z) { return getPhiSeparatrix(phiFromZ(z)); }

const double Injection::getUFPPhi() { return PassConstant::PI - rf_phi; }

const double Injection::getDeltaPMax() { return getPhiSeparatrix(rf_phi); }

const double Injection::getPhiMax()
{
	double PI = PassConstant::PI;
	if (getInitEta() < 0)
	{
		return PI - rf_phi;
	}
	else
	{
		double phi_syn = rf_phi;
		auto f = [phi_syn, PI](double x) -> double { return cos(x) + x * sin(phi_syn) + cos(phi_syn) - (PI - phi_syn) * sin(phi_syn); };
		double root = brent(f, phi_syn, 2 * PI);
		return root;
	}
}

const double Injection::getPhiMin()
{
	double PI = PassConstant::PI;
	if (getInitEta() > 0)
	{
		return PI - rf_phi;
	}
	else
	{
		double phi_syn = rf_phi;
		auto f = [phi_syn, PI](double x) -> double { return cos(x) + x * sin(phi_syn) + cos(phi_syn) - (PI - phi_syn) * sin(phi_syn); };
		double root = brent(f, -1 * PI, phi_syn);
		return root;
	}
}

const double Injection::getZMax() { return zFromPhi(getPhiMin()); }

const double Injection::getZMin() { return zFromPhi(getPhiMax()); }

const double Injection::getQs()
{
	double E = Ek + m0;
	double eta = getInitEta();
	double PI = PassConstant::PI;
	return sqrt(-1 * qm_ratio * harmonic_num * rf_voltage * eta * cos(rf_phi) / 2.0 / PI / beta / beta /
				E);	 // vs = sqrt(-h*q/m*V*eta*cos(phi_s)/2/pi/beta^2/E)
}

const double Injection::H0FromZ(const double z)
{
	double E = Ek + m0;
	double eta = getInitEta();
	double Qs = getQs();
	double f0_now = beta * PassConstant::c / circum;
	double PI = PassConstant::PI;
	return -harmonic_num * 2.0 * PI * f0_now * eta * (Qs * z / eta / rho) * (Qs * z / eta / rho);  // H0 = -h*2*pi*f0*eta*(vs*z/eta/rho)^2
}

const double Injection::H0FromDeltaP(const double dp_c)
{
	double E = Ek + m0;
	double eta = getInitEta();
	double Qs = getQs();
	double f0_now = beta * PassConstant::c / circum;
	double PI = PassConstant::PI;
	return -harmonic_num * 2.0 * PI * f0_now * eta * dp_c * dp_c;  // H0 = -h*2*pi*f0*eta*dp^2
}

// H = 1/2*h*omega_0*eta*dp^2+omega_0*q*V/2/pi/beta^2/E*(cos(phi)-cos(phi_s)+(phi-phi_s)*sin(phi_s))
const double Injection::getHamiltonianPhi(const double phi, const double deltap)
{
	double E = Ek + m0;
	double eta = getInitEta();
	double f0_now = beta * PassConstant::c / circum;
	double PI = PassConstant::PI;
	return (1.0 / 2.0 * harmonic_num * 2.0 * PI * f0_now * eta * deltap * deltap) +
		   (2.0 * PI * f0_now * qm_ratio * rf_voltage / 2.0 / PI / beta / beta / E * (cos(phi) - cos(rf_phi) + (phi - rf_phi) * sin(rf_phi)));
}

const double Injection::getHamiltonianZ(const double z, const double deltap) { return getHamiltonianPhi(phiFromZ(z), deltap); }

const double Injection::psi(const double z, const double dp, const double H0, const double Hmax)
{
	// Use the generating function: 1-(exp(H/H0)-1)/(exp(Hmax/H0)-1).
	return 1 - (exp(getHamiltonianZ(z, dp) / H0) - 1) / (exp(Hmax / H0) - 1);
}

const double Injection::getSigmaZ(const double z_c)
{
	double zmax = getZMax();
	double zmin = getZMin();

	// Get the separatrix of the buncket
	auto dp1 = [=](double z) -> double { return -getZSeparatrix(z); };
	auto dp2 = [=](double z) -> double { return getZSeparatrix(z); };

	// Get the H0 and Hmax used in generating function.
	double H0 = H0FromZ(z_c);
	double Hmax = getHamiltonianPhi(getUFPPhi(), 0.0);

	// Get the integral of generating function in the bucket.
	auto psi_q = [=](double z, double dp) -> double { return psi(z, dp, H0, Hmax); };
	double Q = trapz2d(psi_q, dp1, dp2, zmin, zmax);

	// Get the mean value of generating function in the bucket.
	auto psi_m = [=](double z, double dp) -> double { return z * psi(z, dp, H0, Hmax); };
	double M = trapz2d(psi_m, dp1, dp2, zmin, zmax) / Q;

	// Get the standard deviation of generating function in the bucket.
	auto psi_v = [=](double z, double dp) -> double { return (z - M) * (z - M) * psi(z, dp, H0, Hmax); };
	double V = trapz2d(psi_v, dp1, dp2, zmin, zmax) / Q;
	return sqrt(V);
}

const double Injection::getSigmaDp(const double dp_c)
{
	double zmax = getZMax();
	double zmin = getZMin();

	// Get the separatrix of the buncket
	auto dp1 = [=](double z) -> double { return -getZSeparatrix(z); };
	auto dp2 = [=](double z) -> double { return getZSeparatrix(z); };

	// Get the H0 and Hmax used in generating function.
	double H0 = H0FromDeltaP(dp_c);
	double Hmax = getHamiltonianPhi(getUFPPhi(), 0.0);

	// Get the integral of generating function in the bucket.
	auto psi_q = [=](double z, double dp) -> double { return psi(z, dp, H0, Hmax); };
	double Q = trapz2d(psi_q, dp1, dp2, zmin, zmax);

	// Get the mean value of generating function in the bucket.
	auto psi_m = [=](double z, double dp) -> double { return dp * psi(z, dp, H0, Hmax); };
	double M = trapz2d(psi_m, dp1, dp2, zmin, zmax) / Q;

	// Get the standard deviation of generating function in the bucket.
	auto psi_v = [=](double z, double dp) -> double { return (dp - M) * (dp - M) * psi(z, dp, H0, Hmax); };
	double V = trapz2d(psi_v, dp1, dp2, zmin, zmax) / Q;
	return sqrt(V);
}