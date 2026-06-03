#include "bunch.h"

Bunch::Bunch(const Parameter& para, int input_beamId, int input_bunchId)
{
	using json = nlohmann::json;

	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	bunchId = input_bunchId;
	std::string key_bunch = "bunch" + std::to_string(bunchId);

	try
	{
		Nrp = data.at("Sequence").at("Injection").at(key_bunch).at("Number of Real Particles");
		Np = data.at("Sequence").at("Injection").at(key_bunch).at("Number of Macro Particles");
		Np_sur = Np;
		ratio = Nrp / Np;

		Nproton = data.at("Number of Protons");
		Nneutron = data.at("Number of Neutrons");
		Ncharge = data.at("Number of Electrons");

		if (Nproton == 0 && Nneutron == 0)
		{
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

		Ek = data.at("Sequence").at("Injection").at(key_bunch).at("Kinetic Energy per Nucleon (eV/u)");
		gamma = Ek / m0 + 1;
		beta = sqrt(1 - 1 / (gamma * gamma));
		p0 = gamma * m0 * beta;	 // m0/c/c is in unit of eV, so gamma*m0/c/c*beta*c [unit: eV] = gamma*m0*beta/c [unit: eV] = gamma*m0*beta [unit:
								 // eV/c], so no need to multiply c again
		p0_kg = gamma * (m0 * PassConstant::e / (PassConstant::c * PassConstant::c)) * beta * PassConstant::c;

		if (Nproton == 0 && Nneutron == 0)
			qm_ratio = 1;
		else
			qm_ratio = abs(Ncharge) / static_cast<double>(Nproton + Nneutron);

		Brho = p0_kg / (qm_ratio * PassConstant::e);

		gammat = data.at("Transition Gamma");

		if (data.contains("Space-charge simulation parameters"))
		{
			is_slice_for_sc = true;
			Nslice_sc = data.at("Space-charge simulation parameters").at("Number of slices");
			is_enable_spaceCharge = data.at("Space-charge simulation parameters").at("Is enable space charge");
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
				spdlog::get("logger")->warn("Particle Monitor end turn '{}' exceeds total number of turns '{}'. Adjusting to total turns.", endTurn,
											para.Nturn);
				endTurn = para.Nturn;
			}

			saveTurn_PM.push_back(CycleRange(startTurn, endTurn, stepTurn));
			Nturn_PM = saveTurn_PM[0].totalPoints;

			Nobs_PM = data.at("Particle Monitor parameters").at("Observer position S (m)").size();
		}
	}
	catch (json::exception e)
	{
		// std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}

	// Start to allocate memory //
	dev_particle.mem_allocate_gpu(Np);
	dev_particle_tmp.mem_allocate_gpu(Np);

	// std::cout << "pointer 1 " << std::hex << dev_bunch << std::endl;

	if (is_slice_for_sc || is_slice_for_bb)
	{
		callCuda(cudaMalloc(&dev_sort_z, Np * sizeof(double)));
		callCuda(cudaMalloc(&dev_sort_index, Np * sizeof(int)));
	}

	if (is_slice_for_sc)
	{
		callCuda(cudaMalloc(&dev_slice_sc, Nslice_sc * sizeof(Slice)));
	}
	if (is_slice_for_bb)
	{
		callCuda(cudaMalloc(&dev_slice_bb, Nslice_bb * sizeof(Slice)));
	}

	callCuda(cudaMalloc(&dev_survive_flags, Np * sizeof(int)));
	callCuda(cudaMalloc(&dev_survive_prefix, Np * sizeof(int)));
	cub::DeviceScan::ExclusiveSum(dev_cub_temp, cub_temp_bytes, dev_survive_flags, dev_survive_prefix, Np);
	callCuda(cudaMalloc(&dev_cub_temp, cub_temp_bytes));

	if (is_enableParticleMonitor)
	{
		dev_PM.mem_allocate_gpu(Np_PM * Nobs_PM * Nturn_PM);
	}
}

Bunch::~Bunch()
{
	dev_particle.mem_free_gpu();
	dev_particle_tmp.mem_free_gpu();

	if (is_slice_for_sc || is_slice_for_bb)
	{
		callCuda(cudaFree(dev_sort_z));
		callCuda(cudaFree(dev_sort_index));
	}

	if (is_slice_for_sc)
	{
		callCuda(cudaFree(dev_slice_sc));
	}
	if (is_slice_for_bb)
	{
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