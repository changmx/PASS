#pragma once

#include <random>

#include "bunch.h"
#include "command.h"
#include "parameter.h"
#include "particle.h"

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
	void generate_longitudinal_matchZ_distribution();
	void generate_longitudinal_matchDp_distribution();

	void save_initial_distribution();

	void add_offset(int turn, double t0);
	void insert_particle();

	void print_config();

	const double phiFromZ(const double z);
	const double zFromPhi(const double phi);
	const double getInitEta();
	const double getPhiSeparatrix(const double phi);
	const double getZSeparatrix(const double z);
	const double getUFPPhi();
	const double getDeltaPMax();
	const double getPhiMax();
	const double getPhiMin();
	const double getZMax();
	const double getZMin();
	const double getQs();
	const double H0FromZ(const double z);
	const double H0FromDeltaP(const double dp_c);
	const double getHamiltonianPhi(const double phi, const double deltap);
	const double getHamiltonianZ(const double z, const double deltap);
	const double psi(const double z, const double dp, const double H0, const double Hmax);
	const double getSigmaZ(const double z_c);
	const double getSigmaDp(const double dp_c);

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
	std::string load_dist_filepath;

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

	bool is_offset_z = false;
	double offset_pz = 0.0;

	double rf_voltage = 0.0;
	double rf_phi = 0.0;
	int harmonic_num = 0;
	int harmonic_id = 0;
	double rf_delta_dist = 0.0;
	double rho = 0.0;
	double gammat = 0.0;
	double gamma = 0.0;
	double beta = 0.0;
	double t0 = 0.0;
	double Ek = 0.0;
	double m0 = 0.0;
	int qm_ratio = 0;
	double circum = 0.0;
	double z_shift = 0.0;

	bool is_save_initial_dist = false;

	std::filesystem::path dir_load_distribution;
	std::filesystem::path dir_save_distribution;
	std::string hourMinSec;

	std::default_random_engine e1;

	bool is_set_specified_coordinate = false;
	std::vector<std::vector<double>> specified_coordinate;	// Each row is a particle, columns are x, px, y, py, z, dp/p
};
