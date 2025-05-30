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
		Np_sur = Np;
		ratio = Nrp / Np;

		Nproton = data.at("Number of protons per particle");
		Nneutron = data.at("Number of neutrons per particle");
		Ncharge = data.at("Number of charges per particle");

		if (Nproton == 0 && Nneutron == 0)
			m0 = PassConstant::me;
		else if (Nproton == 1 && Nneutron == 0)
			m0 = PassConstant::mp;
		else
			m0 = PassConstant::mu;

		Ek = data.at("Sequence").at("Injection").at(key_bunch).at("Kinetic energy per nucleon (eV/u)");
		gamma = Ek / m0 + 1;
		beta = sqrt(1 - 1 / (gamma * gamma));
		p0 = gamma * m0 * beta;
		p0_kg = gamma * (m0 * PassConstant::e) / (PassConstant::c * PassConstant::c) * beta * PassConstant::c;

		if (Nproton == 0 && Nneutron == 0)
			Brho = p0 / (1 * PassConstant::c);
		else if (Nproton == 1 && Nneutron == 0)
			Brho = p0 / (1 * PassConstant::c);
		else
			Brho = p0 * (Nproton + Nneutron) / (abs(Ncharge) * PassConstant::c);

		gammat = data.at("GammaT");

		//emitx = data.at("Sequence").at("Injection").at(key_bunch).at("Emittance x (m'rad)");
		//emity = data.at("Sequence").at("Injection").at(key_bunch).at("Emittance y (m'rad)");
		//emitx_norm = emitx * gamma * beta;
		//emity_norm = emity * gamma * beta;

		//alphax = data.at("Sequence").at("Injection").at(key_bunch).at("Alpha x");
		//alphay = data.at("Sequence").at("Injection").at(key_bunch).at("Alpha y");

		//betax = data.at("Sequence").at("Injection").at(key_bunch).at("Beta x (m)");
		//betay = data.at("Sequence").at("Injection").at(key_bunch).at("Beta y (m)");

		//gammax = (1 + alphax * alphax) / betax;
		//gammay = (1 + alphay * alphay) / betay;

		//sigmax = sqrt(betax * emitx);
		//sigmay = sqrt(betay * emity);

		//sigmapx = sqrt(gammax * emitx);
		//sigmapy = sqrt(gammay * emity);

		//sigmaz = data.at("Sequence").at("Injection").at(key_bunch).at("Sigma z (m)");
		//dp = data.at("Sequence").at("Injection").at(key_bunch).at("DeltaP/P");

		//Qx = data.at("Qx");
		//Qy = data.at("Qy");
		//Qz = data.at("Qz");
		//chromx = data.at("Chromaticity x");
		//chromy = data.at("Chromaticity y");



	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}
}

void Bunch::init_memory() {
	//std::cout << "pointer 0 " << std::hex << dev_bunch << std::endl;

	callCuda(cudaMalloc(&dev_bunch, Np * sizeof(Particle)));
	callCuda(cudaMalloc(&dev_bunch_tmp, Np * sizeof(Particle)));

	callCuda(cudaMemset(dev_bunch, 0, Np * sizeof(Particle)));
	callCuda(cudaMemset(dev_bunch_tmp, 0, Np * sizeof(Particle)));

	//std::cout << "pointer 1 " << std::hex << dev_bunch << std::endl;

	if (is_slice_for_sc || is_slice_for_bb) {
		callCuda(cudaMalloc(&dev_sort_z, Np * sizeof(double)));
		callCuda(cudaMalloc(&dev_sort_index, Np * sizeof(int)));
	}

	if (is_slice_for_sc) {
		callCuda(cudaMalloc(&dev_slice_sc, Nslice_sc * sizeof(Slice)));
	}
	if (is_slice_for_bb) {
		callCuda(cudaMalloc(&dev_slice_bb, Nslice_bb * sizeof(Slice)));
	}

}

void Bunch::free_memory() {
	callCuda(cudaFree(dev_bunch));
	callCuda(cudaFree(dev_bunch_tmp));

	if (is_slice_for_sc || is_slice_for_bb) {
		callCuda(cudaFree(dev_sort_z));
		callCuda(cudaFree(dev_sort_index));
	}

	if (is_slice_for_sc) {
		callCuda(cudaFree(dev_slice_sc));
	}
	if (is_slice_for_bb) {
		callCuda(cudaFree(dev_slice_bb));
	}

}