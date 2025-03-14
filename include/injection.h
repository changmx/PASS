#pragma once

#include "command.h"
#include "particle.h"
#include "parameter.h"

class Injection
{
public:
	Injection(const Parameter& Para, int beamId, const Bunch& Bunch);

	double s = -1;
	std::string name = "Injection";

	void run(int turn);

	void load_distribution();

	void generate_transverse_KV_distribution();
	void generate_transverse_Gaussian_distribution();
	void generate_transverse_uniform_distribution();

	void generate_logitudinal_Gaussian_distribution();
	void generate_logitudinal_uniform_distribution();

	void save_initial_distribution();

	void print_config();

private:
	Particle* dev_bunch = NULL;

	int Np = 0;

	int startTurn = 0;
	int endTurn = 0;

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

	double sigmax = 0, sigmay = 0, sigmaz = 0;	// RMS value of horizontal, vertical bunch size and bunch length (m)
	double sigmapx = 0, sigmapy = 0, dp = 0;	// RMS value of horizontal, vertical divergence (rad) and deltap/p

	int beamId = 0;
	int bunchId = 0;

	std::string beam_name;

	std::string injection_mode;
	std::string dist_transverse;
	std::string dist_logitudinal;

	bool is_load_dist = false;
	std::string filename_load_dist;

	bool is_offset_x = false;
	bool is_offset_y = false;
	double offset_x = 0;
	double offset_y = 0;

	bool is_save_initial_dist = false;

	std::vector<int> inject_turns;
	std::filesystem::path dir_load_distribution;
	std::filesystem::path dir_save_distribution;
	std::string hourMinSec;
};

class InjectionCommand : public Command
{
public:
	~InjectionCommand() {};

	InjectionCommand(Injection* inj) {
		injection = inj;
		s = inj->s;
		name = inj->name;
	}

	void execute(int turn) override {
		injection->run(turn);
	}

private:
	Injection* injection;
};


