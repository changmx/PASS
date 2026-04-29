#pragma once

#include "command.h"
#include "particle.h"
#include "parameter.h"

#include <random>

class Injection
{
public:
	Injection(const Parameter& para, int input_beamId, Bunch& Bunch, std::string obj_name);

	double s = -1;
	std::string commandType = "Injection";
	std::string name = "InjectionObj";

	void execute(int turn);

	void load_distribution();

	void generate_transverse_KV_distribution();
	void generate_transverse_Gaussian_distribution();
	void generate_transverse_uniform_distribution();

	void generate_longitudinal_Gaussian_distribution();
	void generate_longitudinal_coasting_distribution();

	void save_initial_distribution();

	void add_Dx();
	void add_offset(int turn, double t0);
	void insert_particle();

	void print_config();

private:
	Particle dev_particle;
	const Bunch& bunchRef;

	int Np = 0;
	int Np_inj_curTurn = 0;
	int Np_inj_total = 0;

	int startTurn = 0;
	int endTurn = 0;
	int intervalTurn = 0;
	std::vector<int> inject_turns;

	double alphax = 0;
	double alphay = 0;
	double betax = 0;
	double betay = 0;
	double gammax = 0;
	double gammay = 0;

	double emitx = 0;
	double emity = 0;
	double emitx_norm = 0;
	double emity_norm = 0;

	double Dx = 0;
	double Dpx = 0;

	double sigmax = 0, sigmay = 0, sigmaz = 0;	// RMS value of horizontal, vertical bunch size and bunch length (m)
	double sigmapx = 0, sigmapy = 0, dp = 0;	// RMS value of horizontal, vertical divergence (rad) and deltap/p

	int beamId = 0;
	int bunchId = 0;

	std::string beam_name;

	std::string dist_transverse;
	std::string dist_longitudinal;

	bool is_load_dist = false;
	std::string filename_load_dist;

	bool is_offset_x = false;
	bool is_offset_y = false;
	bool is_offset_x_fromFile = false;
	bool is_offset_y_fromFile = false;
	std::vector<double> offset_time_x = {};
	std::vector<double> offset_time_y = {};
	std::vector<double> offset_x = {};
	std::vector<double> offset_y = {};
	std::vector<double> offset_px = {};
	std::vector<double> offset_py = {};
	std::string offset_x_filepath;
	std::string offset_y_filepath;
	std::string offset_x_timekind;
	std::string offset_y_timekind;

	bool is_save_initial_dist = false;

	std::filesystem::path dir_load_distribution;
	std::filesystem::path dir_save_distribution;
	std::string hourMinSec;

	std::default_random_engine e1;

	bool is_set_specified_coordinate = false;
	std::vector<std::vector<double>> specified_coordinate;	// Each row is a particle, columns are x, px, y, py, z, dp/p
};

