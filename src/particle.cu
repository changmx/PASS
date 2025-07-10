#include <iostream>
#include <fstream>
#include <cub/cub.cuh>

#include "particle.h"
#include "pic.h"
#include "cutSlice.h"
#include "parameter.h"
#include "constant.h"
#include "command.h"

#include "amgx_c.h"

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

		if (Nproton == 0 && Nneutron == 0) {
			m0 = PassConstant::me;
			mass = ratio * m0;
			charge = ratio * Ncharge * PassConstant::e;
		}

		else if (Nproton == 1 && Nneutron == 0)
		{
			m0 = PassConstant::mp;
			mass = ratio * m0;
			charge = ratio * Ncharge * PassConstant::e;
		}
		else
		{
			m0 = PassConstant::mu;
			mass = ratio * (Nproton + Nneutron) * m0;
			charge = ratio * Ncharge * PassConstant::e;
		}

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

		if (data.contains("Space-charge simulation parameters"))
		{
			is_slice_for_sc = true;
			Nslice_sc = data.at("Space-charge simulation parameters").at("Number of slices");
		}
		else
		{
			is_slice_for_sc = false;
			Nslice_sc = 0;
		}
		if (data.contains("Beam-beam simulation parameters"))
		{
			is_slice_for_bb = true;
			Nslice_bb = data.at("Beam-beam simulation parameters").at("Number of slices");
		}
		else
		{
			is_slice_for_bb = false;
			Nslice_bb = 0;
		}

		if (data.contains("Particle Monitor parameters"))
		{
			is_enableParticleMonitor = data.at("Particle Monitor parameters").at("Is enable particle monitor");
			Np_PM = data.at("Particle Monitor parameters").at("Number of particles to save");

			int startTurn = data.at("Particle Monitor parameters").at("Save turn range")[0];
			int endTurn = data.at("Particle Monitor parameters").at("Save turn range")[1];
			int stepTurn = data.at("Particle Monitor parameters").at("Save turn range")[2];
			if (endTurn > para.Nturn)
			{
				spdlog::get("logger")->warn("Particle Monitor end turn '{}' exceeds total number of turns '{}'. Adjusting to total turns.", endTurn, para.Nturn);
				endTurn = para.Nturn;
			}

			saveTurn_PM.push_back(CycleRange(startTurn, endTurn, stepTurn));
			Nturn_PM = saveTurn_PM[0].totalPoints;

			Nobs_PM = data.at("Particle Monitor parameters").at("Observer position S (m)").size();
		}

		if (data.contains("Space-charge simulation parameters"))
		{
			is_enable_spaceCharge = data.at("Space-charge simulation parameters").at("Is enable space charge");
			fieldSolver_sc = data.at("Space-charge simulation parameters").at("Field solver");
		}
	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}


	// Start to allocate memory //
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

	callCuda(cudaMalloc(&dev_survive_flags, Np * sizeof(int)));
	callCuda(cudaMalloc(&dev_survive_prefix, Np * sizeof(int)));
	cub::DeviceScan::ExclusiveSum(dev_cub_temp, cub_temp_bytes,
		dev_survive_flags, dev_survive_prefix, Np);
	callCuda(cudaMalloc(&dev_cub_temp, cub_temp_bytes));

	if (is_enableParticleMonitor)
	{
		callCuda(cudaMalloc((void**)&dev_PM, Np_PM * Nobs_PM * Nturn_PM * sizeof(Particle)));
	}

	if (is_enable_spaceCharge && "PIC_FD_AMGX" == fieldSolver_sc)
	{
		AMGX_initialize();
	}

}


Bunch::~Bunch() {

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

	callCuda(cudaFree(dev_survive_flags));
	callCuda(cudaFree(dev_survive_prefix));
	callCuda(cudaFree(dev_cub_temp));

	if (is_enableParticleMonitor)
	{
		callCuda(cudaFree(dev_PM));
	}

	if (is_enable_spaceCharge && "PIC_FD_AMGX" == fieldSolver_sc)
	{
		AMGX_finalize();
	}
}