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

	void execute();

	void load_distribution();
	
	void generate_transverse_KV_distribution();
	void generate_transverse_Gaussian_distribution();
	void generate_transverse_uniform_distribution();

	void generate_logitudinal_Gaussian_distribution();
	void generate_logitudinal_uniform_distribution();

	void save_initial_distribution();

private:
	Particle* dev_bunch = NULL;

	int Np = 0;

	double alphax = 0;
	double alphay = 0;
	double betax = 0;
	double betay = 0;
	double gammax = 0;
	double gammay = 0;

	double emitx = 0;
	double emity = 0;

	double sigmaz = 0;
	double dp = 0;

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

	void execute() override {
		injection->execute();
	}

private:
	Injection* injection;
};


