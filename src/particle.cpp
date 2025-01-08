#include <iostream>
#include <fstream>

#include "particle.h"
#include "parameter.h"
#include "constant.h"
#include "command.h"

Bunch::Bunch(const Parameter& para, int input_beamId, int input_bunchId) {
	using json = nlohmann::json;

	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	bunchId = input_bunchId;
	std::string key_bunch = "bunch" + std::to_string(bunchId);

	try
	{
		Nrp = data.at("Sequence").at("Injection").at(key_bunch).at("Number of real particles per bunch");
		Np = data.at("Sequence").at("Injection").at(key_bunch).at("Number of macro particles per bunch");
		ratio = Nrp / Np;

		Nproton = data.at("Number of protons per particle");
		Nneutron = data.at("Number of neutrons per particle");
		Ncharge = data.at("Number of charges per particle");

		if (Nproton == 0 && Nneutron == 0)
			m0 = PassConstant::me;
		else if (Nproton == 1 && Nneutron == 0)
			m0 = PassConstant::mp;
		else
			m0 = PassConstant::mu * (Nproton + Nneutron);

		Ek = data.at("Sequence").at("Injection").at(key_bunch).at("Kinetic energy per particle (eV)");
		gamma = Ek / m0 + 1;
		beta = sqrt(1 - 1 / (gamma * gamma));
		p0 = gamma * m0 * beta;
		p0_kg = gamma * (m0 * PassConstant::e) / (PassConstant::c * PassConstant::c) * beta * PassConstant::c;

		Brho = p0 / (abs(Ncharge) * PassConstant::c);

		emitx = data.at("Sequence").at("Injection").at(key_bunch).at("Emittance x (m'rad)");
		emity = data.at("Sequence").at("Injection").at(key_bunch).at("Emittance y (m'rad)");
		emitx_norm = emitx * gamma * beta;
		emity_norm = emity * gamma * beta;

		alphax = data.at("Sequence").at("Injection").at(key_bunch).at("Alpha x");
		alphay = data.at("Sequence").at("Injection").at(key_bunch).at("Alpha y");

		betax = data.at("Sequence").at("Injection").at(key_bunch).at("Beta x (m)");
		betay = data.at("Sequence").at("Injection").at(key_bunch).at("Beta y (m)");

		gammax = (1 + alphax * alphax) / betax;
		gammay = (1 + alphay * alphay) / betay;

		sigmax = sqrt(betax * emitx);
		sigmay = sqrt(betay * emity);

		sigmapx = sqrt(gammax * emitx);
		sigmapy = sqrt(gammay * emity);

		sigmaz = data.at("Sequence").at("Injection").at(key_bunch).at("Sigma z (m)");
		dp = data.at("Sequence").at("Injection").at(key_bunch).at("DeltaP/P");

		Qx = data.at("Qx");
		Qy = data.at("Qy");
		Qz = data.at("Qz");
		chromx = data.at("Chromaticity x");
		chromy = data.at("Chromaticity y");
		gammat = data.at("GammaT");


	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}
}

void Bunch::init_gpu_memory() {
	callCuda(cudaMalloc(&dev_particle, Np * sizeof(double)), spdlog::get("logger"));
	callCuda(cudaMemset(dev_particle, 0, Np * sizeof(double)), spdlog::get("logger"));
}

void Bunch::free_gpu_memory() {
	callCuda(cudaFree(dev_particle), spdlog::get("logger"));
}