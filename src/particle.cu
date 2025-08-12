#include <iostream>
#include <fstream>
#include <cub/cub.cuh>

#include "particle.h"
#include "pic.h"
#include "cutSlice.h"
#include "parameter.h"
#include "constant.h"
#include "command.h"

//#include "amgx_c.h"


__host__ void Particle::mem_allocate_gpu(size_t n)
{
	// 分配设备内存
	callCuda(cudaMalloc((void**)&x, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&px, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&y, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&py, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&z, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&pz, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&lostPos, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&tag, n * sizeof(int)));
	callCuda(cudaMalloc((void**)&lostTurn, n * sizeof(int)));
	callCuda(cudaMalloc((void**)&sliceId, n * sizeof(int)));

	callCuda(cudaMemset(x, 0, n * sizeof(double)));
	callCuda(cudaMemset(px, 0, n * sizeof(double)));
	callCuda(cudaMemset(y, 0, n * sizeof(double)));
	callCuda(cudaMemset(py, 0, n * sizeof(double)));
	callCuda(cudaMemset(z, 0, n * sizeof(double)));
	callCuda(cudaMemset(pz, 0, n * sizeof(double)));
	callCuda(cudaMemset(lostPos, -1, n * sizeof(double)));	// Initialize lostPos to -1
	callCuda(cudaMemset(tag, 0, n * sizeof(int)));	// Initialize tag to 0
	callCuda(cudaMemset(lostTurn, -1, n * sizeof(int)));	// Initialize lostTurn to -1
	callCuda(cudaMemset(sliceId, 0, n * sizeof(int)));	// Initialize sliceId to 0

#ifdef PASS_CAL_PHASE
	callCuda(cudaMalloc((void**)&last_x, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&last_y, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&last_px, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&last_py, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&phase_x, n * sizeof(double)));
	callCuda(cudaMalloc((void**)&phase_y, n * sizeof(double)));

	callCuda(cudaMemset(last_x, 0, n * sizeof(double)));
	callCuda(cudaMemset(last_y, 0, n * sizeof(double)));
	callCuda(cudaMemset(last_px, 0, n * sizeof(double)));
	callCuda(cudaMemset(last_py, 0, n * sizeof(double)));
	callCuda(cudaMemset(phase_x, 0, n * sizeof(double)));
	callCuda(cudaMemset(phase_y, 0, n * sizeof(double)));
#endif
}


__host__ void Particle::mem_allocate_cpu(size_t n)
{
	// 分配设备内存
	x = new double[n];
	px = new double[n];
	y = new double[n];
	py = new double[n];
	z = new double[n];
	pz = new double[n];
	lostPos = new double[n];
	tag = new int[n];
	lostTurn = new int[n];
	sliceId = new int[n];

#ifdef PASS_CAL_PHASE
	last_x = new double[n];
	last_y = new double[n];
	last_px = new double[n];
	last_py = new double[n];
	phase_x = new double[n];
	phase_y = new double[n];
#endif
}


__host__ void Particle::mem_free_gpu()
{
	// Release device memory
	callCuda(cudaFree(x));
	callCuda(cudaFree(px));
	callCuda(cudaFree(y));
	callCuda(cudaFree(py));
	callCuda(cudaFree(z));
	callCuda(cudaFree(pz));
	callCuda(cudaFree(lostPos));
	callCuda(cudaFree(tag));
	callCuda(cudaFree(lostTurn));
	callCuda(cudaFree(sliceId));

#ifdef PASS_CAL_PHASE
	callCuda(cudaFree(last_x));
	callCuda(cudaFree(last_y));
	callCuda(cudaFree(last_px));
	callCuda(cudaFree(last_py));
	callCuda(cudaFree(phase_x));
	callCuda(cudaFree(phase_y));
#endif
}


__host__ void Particle::mem_free_cpu()
{
	// Release device memory
	delete[] x;
	delete[] px;
	delete[] y;
	delete[] py;
	delete[] z;
	delete[] pz;
	delete[] lostPos;
	delete[] tag;
	delete[] lostTurn;
	delete[] sliceId;

#ifdef PASS_CAL_PHASE
	delete[] last_x;
	delete[] last_y;
	delete[] last_px;
	delete[] last_py;
	delete[] phase_x;
	delete[] phase_y;
#endif
}


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
		}

		else if (Nproton == 1 && Nneutron == 0)
		{
			m0 = PassConstant::mp;
		}
		else
		{
			m0 = PassConstant::mu;
		}

		Ek = data.at("Sequence").at("Injection").at(key_bunch).at("Kinetic energy per nucleon (eV/u)");
		gamma = Ek / m0 + 1;
		beta = sqrt(1 - 1 / (gamma * gamma));
		p0 = gamma * m0 * beta;	// m0/c/c is in unit of eV, so gamma*m0/c/c*beta*c [unit: eV] = gamma*m0*beta/c [unit: eV] = gamma*m0*beta [unit: eV/c], so no need to multiply c again
		p0_kg = gamma * (m0 * PassConstant::e / (PassConstant::c * PassConstant::c)) * beta * PassConstant::c;

		if (Nproton == 0 && Nneutron == 0)
			qm_ratio = 1;
		else
			qm_ratio = abs(Ncharge) / static_cast<double>(Nproton + Nneutron);

		Brho = p0_kg / (qm_ratio * PassConstant::e);

		gammat = data.at("GammaT");

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
		}
	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}


	// Start to allocate memory //
	dev_particle.mem_allocate_gpu(Np);
	dev_particle_tmp.mem_allocate_gpu(Np);

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
		dev_PM.mem_allocate_gpu(Np_PM * Nobs_PM * Nturn_PM);
	}


}


Bunch::~Bunch() {

	dev_particle.mem_free_gpu();
	dev_particle_tmp.mem_free_gpu();

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
		dev_PM.mem_free_gpu();
	}


}


void particle_copy(Particle dst, Particle src, size_t n, cudaMemcpyKind kind, std::string type) {

	if ("all" == type)
	{
		callCuda(cudaMemcpy(dst.x, src.x, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.px, src.px, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.y, src.y, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.py, src.py, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.z, src.z, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.pz, src.pz, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.lostPos, src.lostPos, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.tag, src.tag, n * sizeof(int), kind));
		callCuda(cudaMemcpy(dst.lostTurn, src.lostTurn, n * sizeof(int), kind));
		callCuda(cudaMemcpy(dst.sliceId, src.sliceId, n * sizeof(int), kind));
#ifdef PASS_CAL_PHASE
		callCuda(cudaMemcpy(dst.last_x, src.last_x, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.last_y, src.last_y, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.last_px, src.last_px, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.last_py, src.last_py, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.phase_x, src.phase_x, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.phase_y, src.phase_y, n * sizeof(double), kind));
#endif
	}
	else if ("dist" == type)
	{
		callCuda(cudaMemcpy(dst.x, src.x, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.px, src.px, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.y, src.y, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.py, src.py, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.z, src.z, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.pz, src.pz, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.lostPos, src.lostPos, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.tag, src.tag, n * sizeof(int), kind));
		callCuda(cudaMemcpy(dst.lostTurn, src.lostTurn, n * sizeof(int), kind));
		callCuda(cudaMemcpy(dst.sliceId, src.sliceId, n * sizeof(int), kind));
	}
	else if ("phase" == type)
	{
		callCuda(cudaMemcpy(dst.tag, src.tag, n * sizeof(int), kind));
#ifdef PASS_CAL_PHASE
		callCuda(cudaMemcpy(dst.phase_x, src.phase_x, n * sizeof(double), kind));
		callCuda(cudaMemcpy(dst.phase_y, src.phase_y, n * sizeof(double), kind));
#endif
	}
	else
	{
		spdlog::get("logger")->error("[particle_copy] Unknown particle copy type: {}, we support all, dist and phase now.", type);
		std::exit(EXIT_FAILURE);
	}
}