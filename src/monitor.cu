#include "monitor.h"
#include "constant.h"

#include <fstream>


DistMonitor::DistMonitor(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name, TimeEvent& timeevent) :simTime(timeevent) {
	commandType = "DistMonitor";
	name = obj_name;

	dev_particle = Bunch.dev_particle;
	Np = Bunch.Np;

	host_particle.mem_allocate_cpu(Np);

	bunchId = Bunch.bunchId;

	saveDir = para.dir_output_distribution;
	saveName_part = para.hourMinSec + "_beam" + std::to_string(input_beamId) + "_" + para.beam_name[input_beamId] + "_bunch" + std::to_string(bunchId)
		+ "_" + std::to_string(Np) + "_hor_" + Bunch.dist_transverse + "_longi_" + Bunch.dist_longitudinal + "_" + name;

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");

		for (size_t Nset = 0; Nset < data.at("Sequence").at(obj_name).at("Save turns").size(); Nset++)
		{
			if (data.at("Sequence").at(obj_name).at("Save turns")[Nset].size() == 1)
			{
				int startTurn = data.at("Sequence").at(obj_name).at("Save turns")[Nset][0];
				int endTurn = data.at("Sequence").at(obj_name).at("Save turns")[Nset][0];
				int stepTurn = 1;

				saveTurn.push_back(CycleRange(startTurn, endTurn, stepTurn));
			}
			else if (data.at("Sequence").at(obj_name).at("Save turns")[Nset].size() == 3)
			{
				int startTurn = data.at("Sequence").at(obj_name).at("Save turns")[Nset][0];
				int endTurn = data.at("Sequence").at(obj_name).at("Save turns")[Nset][1];
				int stepTurn = data.at("Sequence").at(obj_name).at("Save turns")[Nset][2];

				saveTurn.push_back(CycleRange(startTurn, endTurn, stepTurn));
			}
			else
			{
				spdlog::get("logger")->error("[DistMonitor] Error: The size of turn array to save should be 1 or 3, but now is {}.",
					data.at("Sequence").at(obj_name).at("Save turns")[Nset].size());
				std::exit(EXIT_FAILURE);
			}
		}
	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}

	print_saveTurn();
}

void DistMonitor::execute(int turn) {

	if (is_value_in_turn_ranges(turn, saveTurn))
	{
		callCuda(cudaEventRecord(simTime.start, 0));
		float time_tmp = 0;

		//spdlog::getlogger->debug("[DistMonitor] run: " + name);

		particle_copy(host_particle, dev_particle, Np, cudaMemcpyDeviceToHost, "dist");

		std::filesystem::path saveName_full = saveDir / (saveName_part + "_turn_" + std::to_string(turn) + ".csv");
		std::ofstream file(saveName_full);

		file << "x" << "," << "px" << "," << "y" << "," << "py" << "," << "z" << "," << "pz" << ","
			<< "tag" << "," << "sliceId" << "," << "lostTurn" << "," << "lostPos" << std::endl;

		for (int j = 0; j < Np; j++) {
			file << std::setprecision(10)
				<< host_particle.x[j] << ","
				<< host_particle.px[j] << ","
				<< host_particle.y[j] << ","
				<< host_particle.py[j] << ","
				<< host_particle.z[j] << ","
				<< host_particle.pz[j] << ","
				<< host_particle.tag[j] << ","
				<< host_particle.sliceId[j] << ","
				<< host_particle.lostTurn[j] << ","
				<< host_particle.lostPos[j] << "\n";
		}
		file.close();

		callCuda(cudaEventRecord(simTime.stop, 0));
		callCuda(cudaEventSynchronize(simTime.stop));
		callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
		simTime.saveBunch += time_tmp;
	}
}

void DistMonitor::print_saveTurn() {
	std::string saveTurn_string;
	for (size_t i = 0; i < saveTurn.size(); i++)
	{
		if (i != 0 && (i % 10) == 0)
		{
			if (saveTurn[i].start == saveTurn[i].end)
			{
				saveTurn_string += std::to_string(saveTurn[i].start) + "\n";
			}
			else
			{
				saveTurn_string += std::to_string(saveTurn[i].start) + "-" + std::to_string(saveTurn[i].end) + "-" + std::to_string(saveTurn[i].step) + "\n";
			}

		}
		else
		{
			if (saveTurn[i].start == saveTurn[i].end)
			{
				saveTurn_string += std::to_string(saveTurn[i].start) + ", ";
			}
			else
			{
				saveTurn_string += std::to_string(saveTurn[i].start) + "-" + std::to_string(saveTurn[i].end) + "-" + std::to_string(saveTurn[i].step) + ", ";
			}
		}
	}
	spdlog::get("logger")->info("[DistMonitor] save turns ({}): {}", name, saveTurn_string);
}


StatMonitor::StatMonitor(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name, const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent) {
	commandType = "StatMonitor";
	name = obj_name;

	dev_particle = Bunch.dev_particle;
	Np = Bunch.Np;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	bunchId = Bunch.bunchId;

	saveDir = para.dir_output_statistic;
	saveName_part = para.hourMinSec + "_beam" + std::to_string(input_beamId) + "_" + para.beam_name[input_beamId] + "_bunch" + std::to_string(bunchId)
		+ "_" + std::to_string(Np) + "_stat_" + name;

	callCuda(cudaHostAlloc((void**)&host_statistic, Nstat * sizeof(double), cudaHostAllocMapped));
	callCuda(cudaMallocPitch((void**)&dev_statistic, &pitch_statistic, block_x * sizeof(double), Nstat));
	callCuda(cudaMemset2D(dev_statistic, pitch_statistic, 0, block_x * sizeof(double), Nstat));

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}

}


void StatMonitor::execute(int turn) {

	callCuda(cudaEventRecord(simTime.start, 0));
	float time_tmp = 0;

	// host_statistic[22]
	// 0:x, 1:x^2, 2:x*px, 3:px^2
	// 4:y, 5:y^2, 6:y*py, 7:py^2
	// 8:beam loss
	// 9:z^2, 10:pz^2
	// 11:z, 12:pz
	// 13:x', 14:y'
	// 15:xz, 16:xy, 17:yz
	// 18:x^3, 19:x^4, 20:y^3, 21:y^4

	callCuda(cudaMemset2D(dev_statistic, pitch_statistic, 0, block_x * sizeof(double), Nstat));

	callKernel(cal_statistic_perblock << <block_x, thread_x, 0, 0 >> > (dev_particle, dev_statistic, pitch_statistic, Np));

	// 使用统一虚拟寻址 (UVA) 和固定内存（Mapped Pinned Memory）
	// 通过 cudaHostAlloc 分配固定且映射到设备地址空间的主机内存，使得设备可以直接修改主机内存，省去显式的 cudaMemcpy
	double* host_dev_statistic = nullptr;
	callCuda(cudaHostGetDevicePointer((void**)&host_dev_statistic, host_statistic, 0));

	cal_statistic_allblock_2 << <1, thread_x, 0, 0 >> > (dev_statistic, pitch_statistic, host_dev_statistic, block_x, Np);

	//for (size_t i = 0; i < Nstat; i++)
	//{
	//	callCuda(cudaMemcpy(host_statistic + i, (double*)((char*)dev_statistic + i * pitch_statistic), 1 * sizeof(double), cudaMemcpyDeviceToHost));
	//}

	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.statistic += time_tmp;

	clock_t start_tmp, end_tmp;
	start_tmp = clock();

	save_bunchInfo_statistic(host_statistic, Np, saveDir, saveName_part, turn);

	end_tmp = clock();
	simTime.saveStatistic += (float)(end_tmp - start_tmp) / CLOCKS_PER_SEC * 1000;
}


ParticleMonitor::ParticleMonitor(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent) {

	commandType = "ParticleMonitor";
	name = obj_name;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	dev_particle = Bunch.dev_particle;
	Np = Bunch.Np;

	bunchId = Bunch.bunchId;

	saveDir = para.dir_output_particle;
	saveName_part = para.hourMinSec + "_beam" + std::to_string(input_beamId) + "_" + para.beam_name[input_beamId] + "_bunch" + std::to_string(bunchId);

	is_enableParticleMonitor = Bunch.is_enableParticleMonitor;
	saveTurn = Bunch.saveTurn_PM;
	Np_PM = Bunch.Np_PM;
	Nobs_PM = Bunch.Nobs_PM;
	Nturn_PM = Bunch.Nturn_PM;
	dev_particleMonitor = Bunch.dev_PM;

	saveTurn_step = saveTurn[0].step;

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		obsId = data.at("Sequence").at(obj_name).at("Observer Id");
	}
	catch (json::exception e)
	{
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}

	print();
}


void ParticleMonitor::execute(int turn) {

	if (!is_enableParticleMonitor)
		return;

	if (is_value_in_turn_ranges(turn, saveTurn))
	{
		callCuda(cudaEventRecord(simTime.start, 0));
		float time_tmp = 0;

		callKernel(get_particle_specified_tag << <block_x, thread_x, 0, 0 >> > (dev_particle, dev_particleMonitor, Np, Np_PM, obsId, Nobs_PM, Nturn_PM, turn, saveTurn_step));

		if (saveTurn[0].isLastPoint(turn))
		{
			Particle host_particleMonitor;
			host_particleMonitor.mem_allocate_cpu(Np_PM * Nobs_PM * Nturn_PM);

			particle_copy(host_particleMonitor, dev_particleMonitor, Np_PM * Nobs_PM * Nturn_PM * sizeof(Particle), cudaMemcpyDeviceToHost, "dist");

			for (size_t i = 0; i < Np_PM; i++)
			{
				int j = obsId;
				std::filesystem::path saveName_full = saveDir / (saveName_part + "_particle_tag_" + std::to_string(i + 1) + "_obs_" + std::to_string(j) + ".csv");
				std::ofstream file(saveName_full);

				file << "turn" << ","
					<< "x" << "," << "px" << "," << "y" << "," << "py" << "," << "z" << "," << "pz" << ","
					<< "tag" << "," << "sliceId" << "," << "lostTurn" << "," << "lostPos" << std::endl;

				for (size_t k = 0; k < Nturn_PM; k++)
				{
					int index = i * Nobs_PM * Nturn_PM
						+ j * Nturn_PM
						+ k;
					file << std::setprecision(10)
						<< saveTurn_step * k + 1 << ","
						<< host_particleMonitor.x[index] << ","
						<< host_particleMonitor.px[index] << ","
						<< host_particleMonitor.y[index] << ","
						<< host_particleMonitor.py[index] << ","
						<< host_particleMonitor.z[index] << ","
						<< host_particleMonitor.pz[index] << ","
						<< host_particleMonitor.tag[index] << ","
						<< host_particleMonitor.sliceId[index] << ","
						<< host_particleMonitor.lostTurn[index] << ","
						<< host_particleMonitor.lostPos[index] << "\n";
				}
				file.close();

			}

			host_particleMonitor.mem_free_cpu();
		}

		callCuda(cudaEventRecord(simTime.stop, 0));
		callCuda(cudaEventSynchronize(simTime.stop));
		callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
		simTime.saveBunch += time_tmp;
	}
}


PhaseMonitor::PhaseMonitor(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

	commandType = "PhaseMonitor";
	name = obj_name;
	dev_particle = Bunch.dev_particle;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	bunchId = Bunch.bunchId;

	saveDir = para.dir_output_tuneSpread;
	saveName_part = para.hourMinSec + "_phase_beam" + std::to_string(input_beamId) + "_" + para.beam_name[input_beamId] + "_bunch" + std::to_string(bunchId);

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		is_enablePhaseMonitor = data.at("Sequence").at(obj_name).at("Is enable phase monitor");
		betax = data.at("Sequence").at(obj_name).at("Beta x (m)");
		betay = data.at("Sequence").at(obj_name).at("Beta y (m)");
		alfx = data.at("Sequence").at(obj_name).at("Alpha x");
		alfy = data.at("Sequence").at(obj_name).at("Alpha y");

		for (size_t Nset = 0; Nset < data.at("Sequence").at(obj_name).at("Save turns").size(); Nset++)
		{
			if (data.at("Sequence").at(obj_name).at("Save turns")[Nset].size() == 2) {
				int startTurn = data.at("Sequence").at(obj_name).at("Save turns")[Nset][0];
				int endTurn = data.at("Sequence").at(obj_name).at("Save turns")[Nset][1];

				saveTurn.push_back(CycleRange(startTurn, endTurn, 1));
			}
			else if (data.at("Sequence").at(obj_name).at("Save turns")[Nset].size() == 4)
			{
				int startTurn = data.at("Sequence").at(obj_name).at("Save turns")[Nset][0];
				int endTurn = data.at("Sequence").at(obj_name).at("Save turns")[Nset][1];
				int stepTurn = data.at("Sequence").at(obj_name).at("Save turns")[Nset][2];
				int numSaveTurn = data.at("Sequence").at(obj_name).at("Save turns")[Nset][3];

				if (startTurn < 1)
				{
					spdlog::get("logger")->error("[PhaseMonitor] Error: ({}) Start turn should >= 1, because the first turn is 1 (not 0), now start turn = {}.",
						startTurn);
					std::exit(EXIT_FAILURE);
				}
				if (endTurn > para.Nturn)
				{
					spdlog::get("logger")->warn("[PhaseMonitor] End turn '{}' exceeds total number of turns '{}'. Adjusting to total turns.", endTurn, para.Nturn);
					endTurn = para.Nturn;
				}
				if (stepTurn < numSaveTurn)
				{
					spdlog::get("logger")->error("[PhaseMonitor] Error: ({}) Turn step '{}' can't be less than the number of turns to be saved '{}'.",
						name, stepTurn, numSaveTurn);
					std::exit(EXIT_FAILURE);
				}

				for (size_t i = startTurn; i < endTurn; i += stepTurn)
				{
					if ((i + numSaveTurn) <= endTurn)
					{
						saveTurn.push_back(CycleRange(i, i + numSaveTurn, 1));
					}
					else
					{
						saveTurn.push_back(CycleRange(i, endTurn, 1));
					}

				}
			}
			else
			{
				spdlog::get("logger")->error("[PhaseMonitor] Error: The size of turn array to save should be 2 or 4, but now is {}.",
					data.at("Sequence").at(obj_name).at("Save turns")[Nset].size());
				std::exit(EXIT_FAILURE);
			}
		}
	}
	catch (json::exception e)
	{
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}

	print();
}


void PhaseMonitor::execute(int turn) {

	if (!is_enablePhaseMonitor)
	{
		return;
	}

	bool is_in_ranges = false;
	int index = 0;

	is_in_ranges = is_value_in_turn_ranges(turn, saveTurn, index);

	if (is_in_ranges)
	{
		callCuda(cudaEventRecord(simTime.start, 0));
		float time_tmp = 0;

		int Np_sur = bunchRef.Np_sur;

		int startTurn = saveTurn[index].start;
		int endTurn = saveTurn[index].end;

		if (turn == startTurn)
		{
			callKernel(record_init_value << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur));
		}
		else
		{
			double sqrtBetax = sqrt(betax);
			double sqrtBetay = sqrt(betay);

			callKernel(cal_accumulatePhaseChange << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, sqrtBetax, sqrtBetay, alfx, alfy));

			if (turn == endTurn)
			{
				int totalTurn = endTurn - startTurn;
				callKernel(cal_averagePhaseChange << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, totalTurn));
			}
		}

		callCuda(cudaEventRecord(simTime.stop, 0));
		callCuda(cudaEventSynchronize(simTime.stop));
		callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
		simTime.calPhase += time_tmp;

		if (turn == endTurn) {

			clock_t start_tmp, end_tmp;
			start_tmp = clock();

			std::filesystem::path saveName_full = saveDir / (saveName_part + "_turn_" + std::to_string(startTurn) + "_" + std::to_string(endTurn) + ".csv");
			save_phase(dev_particle, bunchRef.Np, saveName_full);

			end_tmp = clock();
			simTime.savePhase += (float)(end_tmp - start_tmp) / CLOCKS_PER_SEC * 1000;
		}
	}
}


__device__ void warpReduce(volatile double* data, int tid) {

	data[tid] += data[tid + 32];
	data[tid] += data[tid + 16];
	data[tid] += data[tid + 8];
	data[tid] += data[tid + 4];
	data[tid] += data[tid + 2];
	data[tid] += data[tid + 1];
}


__global__ void cal_statistic_perblock(Particle dev_particle, double* dev_statistic, size_t pitch_statistic, int NpPerBunch) {

	// Count the information about the particles in each block

	int i = blockIdx.x;
	int j = threadIdx.x;
	int tid = j + i * blockDim.x;

	const int ThreadsPerBlock = 256;

	// 0:x, 1:x^2, 2:x*px, 3:px^2
	// 4:y, 5:y^2, 6:y*py, 7:py^2
	// 8:beam loss
	// 9:z^2, 10:pz^2
	// 11:z, 12:pz
	// 13:x', 14:y'
	// 15: xz, 16:xy, 17:yz
	// 18:x^3, 19:x^4, 20:y^3, 21:y^4

	__shared__ double x_cache[ThreadsPerBlock];			// 0
	__shared__ double xSquare_cache[ThreadsPerBlock];	// 1
	__shared__ double xpx_cache[ThreadsPerBlock];		// 2
	__shared__ double pxSquare_cache[ThreadsPerBlock];	// 3

	__shared__ double y_cache[ThreadsPerBlock];			// 4
	__shared__ double ySquare_cache[ThreadsPerBlock];	// 5
	__shared__ double ypy_cache[ThreadsPerBlock];		// 6
	__shared__ double pySquare_cache[ThreadsPerBlock];	// 7

	__shared__ double beamLoss_cache[ThreadsPerBlock];	// 8

	__shared__ double zSquare_cache[ThreadsPerBlock];	// 9
	__shared__ double pzSquare_cache[ThreadsPerBlock];	// 10

	__shared__ double z_cache[ThreadsPerBlock];			// 11
	__shared__ double pz_cache[ThreadsPerBlock];		// 12

	__shared__ double px_cache[ThreadsPerBlock];		// 13
	__shared__ double py_cache[ThreadsPerBlock];		// 14

	__shared__ double xz_cache[ThreadsPerBlock];		// 15
	__shared__ double xy_cache[ThreadsPerBlock];		// 16
	__shared__ double yz_cache[ThreadsPerBlock];		// 17

	__shared__ double x_cube_cache[ThreadsPerBlock];	// 18
	__shared__ double x_quad_cache[ThreadsPerBlock];	// 19
	__shared__ double y_cube_cache[ThreadsPerBlock];	// 20
	__shared__ double y_quad_cache[ThreadsPerBlock];	// 21

	int cacheIdx = threadIdx.x;

	// The __shared__ cache value must be initialized explicitly, otherwise we will get the wrong result.
	x_cache[cacheIdx] = 0;
	xSquare_cache[cacheIdx] = 0;
	xpx_cache[cacheIdx] = 0;
	pxSquare_cache[cacheIdx] = 0;

	y_cache[cacheIdx] = 0;
	ySquare_cache[cacheIdx] = 0;
	ypy_cache[cacheIdx] = 0;
	pySquare_cache[cacheIdx] = 0;

	beamLoss_cache[cacheIdx] = 0;

	zSquare_cache[cacheIdx] = 0;
	pzSquare_cache[cacheIdx] = 0;

	z_cache[cacheIdx] = 0;
	pz_cache[cacheIdx] = 0;

	px_cache[cacheIdx] = 0;
	py_cache[cacheIdx] = 0;

	xz_cache[cacheIdx] = 0;
	xy_cache[cacheIdx] = 0;
	yz_cache[cacheIdx] = 0;

	x_cube_cache[cacheIdx] = 0;
	x_quad_cache[cacheIdx] = 0;
	y_cube_cache[cacheIdx] = 0;
	y_quad_cache[cacheIdx] = 0;

	double* dev_x = (double*)((char*)dev_statistic + 0 * pitch_statistic);
	double* dev_xSquare = (double*)((char*)dev_statistic + 1 * pitch_statistic);
	double* dev_xpx = (double*)((char*)dev_statistic + 2 * pitch_statistic);
	double* dev_pxSquare = (double*)((char*)dev_statistic + 3 * pitch_statistic);

	double* dev_y = (double*)((char*)dev_statistic + 4 * pitch_statistic);
	double* dev_ySquare = (double*)((char*)dev_statistic + 5 * pitch_statistic);
	double* dev_ypy = (double*)((char*)dev_statistic + 6 * pitch_statistic);
	double* dev_pySquare = (double*)((char*)dev_statistic + 7 * pitch_statistic);

	double* dev_beamLoss = (double*)((char*)dev_statistic + 8 * pitch_statistic);

	double* dev_zSquare = (double*)((char*)dev_statistic + 9 * pitch_statistic);
	double* dev_pzSquare = (double*)((char*)dev_statistic + 10 * pitch_statistic);
	double* dev_z = (double*)((char*)dev_statistic + 11 * pitch_statistic);
	double* dev_pz = (double*)((char*)dev_statistic + 12 * pitch_statistic);

	double* dev_px = (double*)((char*)dev_statistic + 13 * pitch_statistic);
	double* dev_py = (double*)((char*)dev_statistic + 14 * pitch_statistic);

	double* dev_xz = (double*)((char*)dev_statistic + 15 * pitch_statistic);
	double* dev_xy = (double*)((char*)dev_statistic + 16 * pitch_statistic);
	double* dev_yz = (double*)((char*)dev_statistic + 17 * pitch_statistic);

	double* dev_x_cube = (double*)((char*)dev_statistic + 18 * pitch_statistic);
	double* dev_x_quad = (double*)((char*)dev_statistic + 19 * pitch_statistic);
	double* dev_y_cube = (double*)((char*)dev_statistic + 20 * pitch_statistic);
	double* dev_y_quad = (double*)((char*)dev_statistic + 21 * pitch_statistic);

	for (; tid < NpPerBunch; tid += blockDim.x * gridDim.x)
	{
		if (dev_particle.tag[tid] > 0)
		{
			double x = dev_particle.x[tid];
			double px = dev_particle.px[tid];
			double y = dev_particle.y[tid];
			double py = dev_particle.py[tid];
			double z = dev_particle.z[tid];
			double pz = dev_particle.pz[tid];

			x_cache[cacheIdx] += x;
			xSquare_cache[cacheIdx] += x * x;
			xpx_cache[cacheIdx] += x * px;
			pxSquare_cache[cacheIdx] += px * px;

			y_cache[cacheIdx] += y;
			ySquare_cache[cacheIdx] += y * y;
			ypy_cache[cacheIdx] += y * py;
			pySquare_cache[cacheIdx] += py * py;
			//beamLoss_cache[cacheIdx] += 0;

			zSquare_cache[cacheIdx] += z * z;
			pzSquare_cache[cacheIdx] += pz * pz;
			z_cache[cacheIdx] += z;
			pz_cache[cacheIdx] += pz;

			px_cache[cacheIdx] += px;
			py_cache[cacheIdx] += py;

			xz_cache[cacheIdx] += x * z;
			xy_cache[cacheIdx] += x * y;
			yz_cache[cacheIdx] += y * z;

			x_cube_cache[cacheIdx] += pow(x, 3);
			x_quad_cache[cacheIdx] += pow(x, 4);
			y_cube_cache[cacheIdx] += pow(y, 3);
			y_quad_cache[cacheIdx] += pow(y, 4);

		}
		else
		{
			beamLoss_cache[cacheIdx] += 1;
		}
	}
	__syncthreads();

	for (int m = blockDim.x / 2; m > 32; m >>= 1)
	{
		if (cacheIdx < m) {
			x_cache[cacheIdx] += x_cache[cacheIdx + m];
			xSquare_cache[cacheIdx] += xSquare_cache[cacheIdx + m];
			xpx_cache[cacheIdx] += xpx_cache[cacheIdx + m];
			pxSquare_cache[cacheIdx] += pxSquare_cache[cacheIdx + m];

			y_cache[cacheIdx] += y_cache[cacheIdx + m];
			ySquare_cache[cacheIdx] += ySquare_cache[cacheIdx + m];
			ypy_cache[cacheIdx] += ypy_cache[cacheIdx + m];
			pySquare_cache[cacheIdx] += pySquare_cache[cacheIdx + m];

			beamLoss_cache[cacheIdx] += beamLoss_cache[cacheIdx + m];

			zSquare_cache[cacheIdx] += zSquare_cache[cacheIdx + m];
			pzSquare_cache[cacheIdx] += pzSquare_cache[cacheIdx + m];
			z_cache[cacheIdx] += z_cache[cacheIdx + m];
			pz_cache[cacheIdx] += pz_cache[cacheIdx + m];

			px_cache[cacheIdx] += px_cache[cacheIdx + m];
			py_cache[cacheIdx] += py_cache[cacheIdx + m];

			xz_cache[cacheIdx] += xz_cache[cacheIdx + m];
			xy_cache[cacheIdx] += xy_cache[cacheIdx + m];
			yz_cache[cacheIdx] += yz_cache[cacheIdx + m];

			x_cube_cache[cacheIdx] += x_cube_cache[cacheIdx + m];
			x_quad_cache[cacheIdx] += x_quad_cache[cacheIdx + m];
			y_cube_cache[cacheIdx] += y_cube_cache[cacheIdx + m];
			y_quad_cache[cacheIdx] += y_quad_cache[cacheIdx + m];
		}
		__syncthreads();
	}

	if (cacheIdx < 32)
	{
		warpReduce(x_cache, cacheIdx);
		warpReduce(xSquare_cache, cacheIdx);
		warpReduce(xpx_cache, cacheIdx);
		warpReduce(pxSquare_cache, cacheIdx);

		warpReduce(y_cache, cacheIdx);
		warpReduce(ySquare_cache, cacheIdx);
		warpReduce(ypy_cache, cacheIdx);
		warpReduce(pySquare_cache, cacheIdx);

		warpReduce(beamLoss_cache, cacheIdx);

		warpReduce(zSquare_cache, cacheIdx);
		warpReduce(pzSquare_cache, cacheIdx);
		warpReduce(z_cache, cacheIdx);
		warpReduce(pz_cache, cacheIdx);

		warpReduce(px_cache, cacheIdx);
		warpReduce(py_cache, cacheIdx);

		warpReduce(xz_cache, cacheIdx);
		warpReduce(xy_cache, cacheIdx);
		warpReduce(yz_cache, cacheIdx);

		warpReduce(x_cube_cache, cacheIdx);
		warpReduce(x_quad_cache, cacheIdx);
		warpReduce(y_cube_cache, cacheIdx);
		warpReduce(y_quad_cache, cacheIdx);

	}
	if (0 == cacheIdx)
	{
		dev_x[i] = x_cache[0];
		dev_xSquare[i] = xSquare_cache[0];
		dev_xpx[i] = xpx_cache[0];
		dev_pxSquare[i] = pxSquare_cache[0];

		dev_y[i] = y_cache[0];
		dev_ySquare[i] = ySquare_cache[0];
		dev_ypy[i] = ypy_cache[0];
		dev_pySquare[i] = pySquare_cache[0];

		dev_beamLoss[i] = beamLoss_cache[0];

		dev_zSquare[i] = zSquare_cache[0];
		dev_pzSquare[i] = pzSquare_cache[0];
		dev_z[i] = z_cache[0];
		dev_pz[i] = pz_cache[0];

		dev_px[i] = px_cache[0];
		dev_py[i] = py_cache[0];

		dev_xz[i] = xz_cache[0];
		dev_xy[i] = xy_cache[0];
		dev_yz[i] = yz_cache[0];

		dev_x_cube[i] = x_cube_cache[0];
		dev_x_quad[i] = x_quad_cache[0];
		dev_y_cube[i] = y_cube_cache[0];
		dev_y_quad[i] = y_quad_cache[0];

	}

}


__global__ void cal_statistic_allblock_2(double* dev_statistic, size_t pitch_statistic, double* host_dev_statistic, int gridDimX, int NpInit) {

	// Summarize data in all grids.

	// 0:x, 1:x^2, 2:x*px, 3:px^2
	// 4:y, 5:y^2, 6:y*py, 7:py^2
	// 8:beam loss
	// 9:z^2, 10:pz^2
	// 11:z, 12:pz
	// 13:x', 14:y'
	// 15: xz, 16:xy, 17:yz
	// 18:x^3, 19:x^4, 20:y^3, 21:y^4

	int tid = threadIdx.x;

	const int ThreadsPerBlock = 256;

	__shared__ double x_cache[ThreadsPerBlock];			// 0
	__shared__ double xSquare_cache[ThreadsPerBlock];	// 1
	__shared__ double xpx_cache[ThreadsPerBlock];		// 2
	__shared__ double pxSquare_cache[ThreadsPerBlock];	// 3

	__shared__ double y_cache[ThreadsPerBlock];			// 4
	__shared__ double ySquare_cache[ThreadsPerBlock];	// 5
	__shared__ double ypy_cache[ThreadsPerBlock];		// 6
	__shared__ double pySquare_cache[ThreadsPerBlock];	// 7

	__shared__ double beamLoss_cache[ThreadsPerBlock];	// 8

	__shared__ double zSquare_cache[ThreadsPerBlock];	// 9
	__shared__ double pzSquare_cache[ThreadsPerBlock];	// 10

	__shared__ double z_cache[ThreadsPerBlock];			// 11
	__shared__ double pz_cache[ThreadsPerBlock];		// 12

	__shared__ double px_cache[ThreadsPerBlock];		// 13
	__shared__ double py_cache[ThreadsPerBlock];		// 14

	__shared__ double xz_cache[ThreadsPerBlock];		// 15
	__shared__ double xy_cache[ThreadsPerBlock];		// 16
	__shared__ double yz_cache[ThreadsPerBlock];		// 17

	__shared__ double x_cube_cache[ThreadsPerBlock];	// 18
	__shared__ double x_quad_cache[ThreadsPerBlock];	// 19
	__shared__ double y_cube_cache[ThreadsPerBlock];	// 20
	__shared__ double y_quad_cache[ThreadsPerBlock];	// 21

	int cacheIdx = threadIdx.x;

	x_cache[cacheIdx] = 0;
	xSquare_cache[cacheIdx] = 0;
	xpx_cache[cacheIdx] = 0;
	pxSquare_cache[cacheIdx] = 0;

	y_cache[cacheIdx] = 0;
	ySquare_cache[cacheIdx] = 0;
	ypy_cache[cacheIdx] = 0;
	pySquare_cache[cacheIdx] = 0;

	beamLoss_cache[cacheIdx] = 0;

	zSquare_cache[cacheIdx] = 0;
	pzSquare_cache[cacheIdx] = 0;

	z_cache[cacheIdx] = 0;
	pz_cache[cacheIdx] = 0;

	px_cache[cacheIdx] = 0;
	py_cache[cacheIdx] = 0;

	xz_cache[cacheIdx] = 0;
	xy_cache[cacheIdx] = 0;
	yz_cache[cacheIdx] = 0;

	x_cube_cache[cacheIdx] = 0;
	x_quad_cache[cacheIdx] = 0;
	y_cube_cache[cacheIdx] = 0;
	y_quad_cache[cacheIdx] = 0;

	double* dev_x = (double*)((char*)dev_statistic + 0 * pitch_statistic);
	double* dev_xSquare = (double*)((char*)dev_statistic + 1 * pitch_statistic);
	double* dev_xpx = (double*)((char*)dev_statistic + 2 * pitch_statistic);
	double* dev_pxSquare = (double*)((char*)dev_statistic + 3 * pitch_statistic);

	double* dev_y = (double*)((char*)dev_statistic + 4 * pitch_statistic);
	double* dev_ySquare = (double*)((char*)dev_statistic + 5 * pitch_statistic);
	double* dev_ypy = (double*)((char*)dev_statistic + 6 * pitch_statistic);
	double* dev_pySquare = (double*)((char*)dev_statistic + 7 * pitch_statistic);

	double* dev_beamLoss = (double*)((char*)dev_statistic + 8 * pitch_statistic);

	double* dev_zSquare = (double*)((char*)dev_statistic + 9 * pitch_statistic);
	double* dev_pzSquare = (double*)((char*)dev_statistic + 10 * pitch_statistic);
	double* dev_z = (double*)((char*)dev_statistic + 11 * pitch_statistic);
	double* dev_pz = (double*)((char*)dev_statistic + 12 * pitch_statistic);

	double* dev_px = (double*)((char*)dev_statistic + 13 * pitch_statistic);
	double* dev_py = (double*)((char*)dev_statistic + 14 * pitch_statistic);

	double* dev_xz = (double*)((char*)dev_statistic + 15 * pitch_statistic);
	double* dev_xy = (double*)((char*)dev_statistic + 16 * pitch_statistic);
	double* dev_yz = (double*)((char*)dev_statistic + 17 * pitch_statistic);

	double* dev_x_cube = (double*)((char*)dev_statistic + 18 * pitch_statistic);
	double* dev_x_quad = (double*)((char*)dev_statistic + 19 * pitch_statistic);
	double* dev_y_cube = (double*)((char*)dev_statistic + 20 * pitch_statistic);
	double* dev_y_quad = (double*)((char*)dev_statistic + 21 * pitch_statistic);

	for (; tid < gridDimX; tid += blockDim.x)
	{
		x_cache[cacheIdx] += dev_x[tid];
		xSquare_cache[cacheIdx] += dev_xSquare[tid];
		xpx_cache[cacheIdx] += dev_xpx[tid];
		pxSquare_cache[cacheIdx] += dev_pxSquare[tid];

		y_cache[cacheIdx] += dev_y[tid];
		ySquare_cache[cacheIdx] += dev_ySquare[tid];
		ypy_cache[cacheIdx] += dev_ypy[tid];
		pySquare_cache[cacheIdx] += dev_pySquare[tid];

		beamLoss_cache[cacheIdx] += dev_beamLoss[tid];

		zSquare_cache[cacheIdx] += dev_zSquare[tid];
		pzSquare_cache[cacheIdx] += dev_pzSquare[tid];
		z_cache[cacheIdx] += dev_z[tid];
		pz_cache[cacheIdx] += dev_pz[tid];

		px_cache[cacheIdx] += dev_px[tid];
		py_cache[cacheIdx] += dev_py[tid];

		xz_cache[cacheIdx] += dev_xz[tid];
		xy_cache[cacheIdx] += dev_xy[tid];
		yz_cache[cacheIdx] += dev_yz[tid];

		x_cube_cache[cacheIdx] += dev_x_cube[tid];
		x_quad_cache[cacheIdx] += dev_x_quad[tid];
		y_cube_cache[cacheIdx] += dev_y_cube[tid];
		y_quad_cache[cacheIdx] += dev_y_quad[tid];

	}
	__syncthreads();

	for (int m = blockDim.x / 2; m > 32; m >>= 1)
	{
		if (cacheIdx < m) {
			x_cache[cacheIdx] += x_cache[cacheIdx + m];
			xSquare_cache[cacheIdx] += xSquare_cache[cacheIdx + m];
			xpx_cache[cacheIdx] += xpx_cache[cacheIdx + m];
			pxSquare_cache[cacheIdx] += pxSquare_cache[cacheIdx + m];

			y_cache[cacheIdx] += y_cache[cacheIdx + m];
			ySquare_cache[cacheIdx] += ySquare_cache[cacheIdx + m];
			ypy_cache[cacheIdx] += ypy_cache[cacheIdx + m];
			pySquare_cache[cacheIdx] += pySquare_cache[cacheIdx + m];

			beamLoss_cache[cacheIdx] += beamLoss_cache[cacheIdx + m];

			zSquare_cache[cacheIdx] += zSquare_cache[cacheIdx + m];
			pzSquare_cache[cacheIdx] += pzSquare_cache[cacheIdx + m];
			z_cache[cacheIdx] += z_cache[cacheIdx + m];
			pz_cache[cacheIdx] += pz_cache[cacheIdx + m];

			px_cache[cacheIdx] += px_cache[cacheIdx + m];
			py_cache[cacheIdx] += py_cache[cacheIdx + m];

			xz_cache[cacheIdx] += xz_cache[cacheIdx + m];
			xy_cache[cacheIdx] += xy_cache[cacheIdx + m];
			yz_cache[cacheIdx] += yz_cache[cacheIdx + m];

			x_cube_cache[cacheIdx] += x_cube_cache[cacheIdx + m];
			x_quad_cache[cacheIdx] += x_quad_cache[cacheIdx + m];
			y_cube_cache[cacheIdx] += y_cube_cache[cacheIdx + m];
			y_quad_cache[cacheIdx] += y_quad_cache[cacheIdx + m];

		}
		__syncthreads();
	}

	if (cacheIdx < 32)
	{
		warpReduce(x_cache, cacheIdx);
		warpReduce(xSquare_cache, cacheIdx);
		warpReduce(xpx_cache, cacheIdx);
		warpReduce(pxSquare_cache, cacheIdx);

		warpReduce(y_cache, cacheIdx);
		warpReduce(ySquare_cache, cacheIdx);
		warpReduce(ypy_cache, cacheIdx);
		warpReduce(pySquare_cache, cacheIdx);

		warpReduce(beamLoss_cache, cacheIdx);

		warpReduce(zSquare_cache, cacheIdx);
		warpReduce(pzSquare_cache, cacheIdx);
		warpReduce(z_cache, cacheIdx);
		warpReduce(pz_cache, cacheIdx);

		warpReduce(px_cache, cacheIdx);
		warpReduce(py_cache, cacheIdx);

		warpReduce(xz_cache, cacheIdx);
		warpReduce(xy_cache, cacheIdx);
		warpReduce(yz_cache, cacheIdx);

		warpReduce(x_cube_cache, cacheIdx);
		warpReduce(x_quad_cache, cacheIdx);
		warpReduce(y_cube_cache, cacheIdx);
		warpReduce(y_quad_cache, cacheIdx);

	}

	if (0 == cacheIdx)
	{
		dev_x[0] = x_cache[0];
		dev_xSquare[0] = xSquare_cache[0];
		dev_xpx[0] = xpx_cache[0];
		dev_pxSquare[0] = pxSquare_cache[0];

		dev_y[0] = y_cache[0];
		dev_ySquare[0] = ySquare_cache[0];
		dev_ypy[0] = ypy_cache[0];
		dev_pySquare[0] = pySquare_cache[0];

		dev_beamLoss[0] = beamLoss_cache[0];

		dev_zSquare[0] = zSquare_cache[0];
		dev_pzSquare[0] = pzSquare_cache[0];
		dev_z[0] = z_cache[0];
		dev_pz[0] = pz_cache[0];

		dev_px[0] = px_cache[0];
		dev_py[0] = py_cache[0];

		dev_xz[0] = xz_cache[0];
		dev_xy[0] = xy_cache[0];
		dev_yz[0] = yz_cache[0];

		dev_x_cube[0] = x_cube_cache[0];
		dev_x_quad[0] = x_quad_cache[0];
		dev_y_cube[0] = y_cube_cache[0];
		dev_y_quad[0] = y_quad_cache[0];

		int NpNotLoss = NpInit - dev_beamLoss[0];

		dev_x[0] /= NpNotLoss;
		dev_xSquare[0] /= NpNotLoss;
		dev_xpx[0] /= NpNotLoss;
		dev_pxSquare[0] /= NpNotLoss;

		dev_y[0] /= NpNotLoss;
		dev_ySquare[0] /= NpNotLoss;
		dev_ypy[0] /= NpNotLoss;
		dev_pySquare[0] /= NpNotLoss;

		// Beamloss doesn't need to be averaged.

		dev_zSquare[0] /= NpNotLoss;
		dev_pzSquare[0] /= NpNotLoss;
		dev_z[0] /= NpNotLoss;
		dev_pz[0] /= NpNotLoss;

		dev_px[0] /= NpNotLoss;
		dev_py[0] /= NpNotLoss;

		dev_xz[0] /= NpNotLoss;
		dev_xy[0] /= NpNotLoss;
		dev_yz[0] /= NpNotLoss;

		dev_x_cube[0] /= NpNotLoss;
		dev_x_quad[0] /= NpNotLoss;
		dev_y_cube[0] /= NpNotLoss;
		dev_y_quad[0] /= NpNotLoss;

	}

	host_dev_statistic[0] = dev_x[0];
	host_dev_statistic[1] = dev_xSquare[0];
	host_dev_statistic[2] = dev_xpx[0];
	host_dev_statistic[3] = dev_pxSquare[0];

	host_dev_statistic[4] = dev_y[0];
	host_dev_statistic[5] = dev_ySquare[0];
	host_dev_statistic[6] = dev_ypy[0];
	host_dev_statistic[7] = dev_pySquare[0];

	host_dev_statistic[8] = dev_beamLoss[0];

	host_dev_statistic[9] = dev_zSquare[0];
	host_dev_statistic[10] = dev_pzSquare[0];
	host_dev_statistic[11] = dev_z[0];
	host_dev_statistic[12] = dev_pz[0];

	host_dev_statistic[13] = dev_px[0];
	host_dev_statistic[14] = dev_py[0];

	host_dev_statistic[15] = dev_xz[0];
	host_dev_statistic[16] = dev_xy[0];
	host_dev_statistic[17] = dev_yz[0];

	host_dev_statistic[18] = dev_x_cube[0];
	host_dev_statistic[19] = dev_x_quad[0];
	host_dev_statistic[20] = dev_y_cube[0];
	host_dev_statistic[21] = dev_y_quad[0];
}


void save_bunchInfo_statistic(double* host_statistic, int Np, std::filesystem::path saveDir, std::string saveName_part, int turn) {

	int stat_turn = turn;
	double stat_beamloss = host_statistic[8];
	double stat_lossPercent = host_statistic[8] / Np * 100;

	double stat_xAverage = host_statistic[0];
	double stat_yAverage = host_statistic[4];
	double stat_zAverage = host_statistic[11];

	double stat_pxAverage = host_statistic[13];
	double stat_pyAverage = host_statistic[14];
	double stat_pzAverage = host_statistic[12];

	double xsquareAverage = host_statistic[1];
	double ysquareAverage = host_statistic[5];
	double zsquareAverage = host_statistic[9];

	double pxsquareAverage = host_statistic[3];
	double pysquareAverage = host_statistic[7];
	double pzsquareAverage = host_statistic[10];

	double xpxAverage = host_statistic[2];
	double ypyAverage = host_statistic[6];

	double xcubeAverage = host_statistic[18];
	double xquadAverage = host_statistic[19];
	double ycubeAverage = host_statistic[20];
	double yquadAverage = host_statistic[21];

	double stat_sigmaX = sqrt(xsquareAverage - stat_xAverage * stat_xAverage);
	double stat_sigmaY = sqrt(ysquareAverage - stat_yAverage * stat_yAverage);
	double stat_sigmaZ = sqrt(zsquareAverage - stat_zAverage * stat_zAverage);

	double stat_sigmaPx = sqrt(pxsquareAverage - stat_pxAverage * stat_pxAverage);
	double stat_sigmaPy = sqrt(pysquareAverage - stat_pyAverage * stat_pyAverage);
	double stat_sigmaPz = sqrt(pzsquareAverage - stat_pzAverage * stat_pzAverage);

	double sigmaxpx = xpxAverage - stat_xAverage * stat_pxAverage;
	double sigmaypy = ypyAverage - stat_yAverage * stat_pyAverage;

	double stat_emitX = sqrt(stat_sigmaX * stat_sigmaX * stat_sigmaPx * stat_sigmaPx - sigmaxpx * sigmaxpx);
	double stat_emitY = sqrt(stat_sigmaY * stat_sigmaY * stat_sigmaPy * stat_sigmaPy - sigmaypy * sigmaypy);

	double stat_xzAverage = host_statistic[15];
	double stat_xyAverage = host_statistic[16];
	double stat_yzAverage = host_statistic[17];
	double stat_xzDevideSigmaxSigmaz = host_statistic[15] / (stat_sigmaX * stat_sigmaZ);

	double stat_betax = stat_sigmaX * stat_sigmaX / stat_emitX;
	double stat_betay = stat_sigmaY * stat_sigmaY / stat_emitY;
	double stat_alphax = -1 * sigmaxpx / stat_emitX;
	double stat_alphay = -1 * sigmaypy / stat_emitY;
	double stat_gammax = stat_sigmaPx * stat_sigmaPx / stat_emitX;
	double stat_gammay = stat_sigmaPy * stat_sigmaPy / stat_emitY;
	double stat_invariantx = stat_gammax * stat_betax - stat_alphax * stat_alphax;
	double stat_invarianty = stat_gammay * stat_betay - stat_alphay * stat_alphay;

	double stat_xSkewness = (xcubeAverage - 3 * stat_xAverage * pow(stat_sigmaX, 2) - pow(stat_xAverage, 3))
		/ (pow(stat_sigmaX, 3));
	double stat_xKurtosis = (xquadAverage - 4 * stat_xAverage * xcubeAverage + 2 * pow(stat_xAverage, 2) * xsquareAverage
		+ 4 * pow(stat_xAverage, 2) * pow(stat_sigmaX, 2) + pow(stat_xAverage, 4)) / (pow(stat_sigmaX, 4));
	double stat_ySkewness = (ycubeAverage - 3 * stat_yAverage * pow(stat_sigmaY, 2) - pow(stat_yAverage, 3))
		/ (pow(stat_sigmaY, 3));
	double stat_yKurtosis = (yquadAverage - 4 * stat_yAverage * ycubeAverage + 2 * pow(stat_yAverage, 2) * ysquareAverage
		+ 4 * pow(stat_yAverage, 2) * pow(stat_sigmaY, 2) + pow(stat_yAverage, 4)) / (pow(stat_sigmaY, 4));

	//stat_print(__FILE__);

	std::filesystem::path saveName_full = saveDir / (saveName_part + ".csv");
	std::ofstream file(saveName_full, std::ofstream::out | std::ofstream::app);

	if (1 == turn)
	{
		file << "turn" << ","
			<< "xAverage" << "," << "pxAverage" << "," << "sigmaX" << "," << "sigmaPx" << ","
			<< "yAverage" << "," << "pyAverage" << "," << "sigmaY" << "," << "sigmaPy" << ","
			<< "zAverage" << "," << "pzAverage" << "," << "sigmaZ" << "," << "sigmaPz" << ","
			<< "xEmittance" << "," << "yEmittance" << ","
			<< "betax" << "," << "betay" << "," << "alphax" << "," << "alphay" << "," << "gammax" << "," << "gammay" << ","
			<< "invariantx twiss" << "," << "invarianty twiss" << ","
			<< "xzAverage" << "," << "xyAverage" << "," << "yzAverage" << "," << "xzDevideSigmaxSigmaz" << ","
			<< "beamLossTotal" << "," << "lossPercent" << ","
			<< "xSkewness" << "," << "xKurtosis" << "," << "ySkewness" << "," << "yKurtosis"
			<< std::endl;
	}
	file << stat_turn << ","
		<< stat_xAverage << "," << stat_pxAverage << "," << stat_sigmaX << "," << stat_sigmaPx << ","
		<< stat_yAverage << "," << stat_pyAverage << "," << stat_sigmaY << "," << stat_sigmaPy << ","
		<< stat_zAverage << "," << stat_pzAverage << "," << stat_sigmaZ << "," << stat_sigmaPz << ","
		<< stat_emitX << "," << stat_emitY << ","
		<< stat_betax << "," << stat_betay << "," << stat_alphax << "," << stat_alphay << "," << stat_gammax << "," << stat_gammay << ","
		<< stat_invariantx << "," << stat_invarianty << ","
		<< stat_xzAverage << "," << stat_xyAverage << "," << stat_yzAverage << "," << stat_xzDevideSigmaxSigmaz << ","
		<< stat_beamloss << "," << stat_lossPercent << ","
		<< stat_xSkewness << "," << stat_xKurtosis << "," << stat_ySkewness << "," << stat_yKurtosis
		<< std::endl;

	file.close();

	//printf("stat s : % f, ms : % f\n", (float)(end_tmp - start_tmp) / CLOCKS_PER_SEC, (float)(end_tmp - start_tmp) / CLOCKS_PER_SEC * 1000);
	//std::cout << "start: " << start_tmp << ", end: " << end_tmp << std::endl;
}


__global__ void get_particle_specified_tag(Particle dev_particle, Particle dev_particleMonitor, int Np, int Np_PM,
	int obsId, int Nobs_PM, int Nturn_PM, int current_turn, int saveTurn_step) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	int particleId = 0;
	int index = 0;
	int tag = 0;

	while (tid < Np)
	{
		tag = abs(dev_particle.tag[tid]);
		if (tag >= 1 && tag <= Np_PM)
		{
			particleId = tag - 1;	// tag ranges from 1 not 0
			index = particleId * Nobs_PM * Nturn_PM
				+ obsId * Nturn_PM
				+ (current_turn - 1) / saveTurn_step;	// turn ranges from 1 not 0, and turn may have a step

			dev_particleMonitor.x[index] = dev_particle.x[tid];
			dev_particleMonitor.px[index] = dev_particle.px[tid];
			dev_particleMonitor.y[index] = dev_particle.y[tid];
			dev_particleMonitor.py[index] = dev_particle.py[tid];
			dev_particleMonitor.z[index] = dev_particle.z[tid];
			dev_particleMonitor.pz[index] = dev_particle.pz[tid];
			dev_particleMonitor.lostPos[index] = dev_particle.lostPos[tid];
			dev_particleMonitor.tag[index] = dev_particle.tag[tid];
			dev_particleMonitor.lostTurn[index] = dev_particle.lostTurn[tid];
			dev_particleMonitor.sliceId[index] = dev_particle.sliceId[tid];

#ifdef PASS_CAL_PHASE
			dev_particleMonitor.last_x[index] = dev_particle.last_x[tid];
			dev_particleMonitor.last_y[index] = dev_particle.last_y[tid];
			dev_particleMonitor.last_px[index] = dev_particle.last_px[tid];
			dev_particleMonitor.last_py[index] = dev_particle.last_py[tid];
			dev_particleMonitor.phase_x[index] = dev_particle.phase_x[tid];
			dev_particleMonitor.phase_y[index] = dev_particle.phase_y[tid];
#endif
		}

		tid += stride;
	}
}


__forceinline__ __device__ void physical2normalize(double& x, double& px, double sqrtBetaX, double alphaX) {

	// Transformation from physical coordinates to the normalized coordinates.

	double x1 = x / sqrtBetaX;
	double px1 = alphaX / sqrtBetaX * x + sqrtBetaX * px;

	x = x1;
	px = px1;
}


__forceinline__ __device__ void normalize2physical(double& x, double& px, double sqrtBetaX, double alphaX) {

	// Transformation from normalized coordinates to the physical coordinates.

	double x1 = x * sqrtBetaX;
	double px1 = -1 * alphaX / sqrtBetaX * x + px / sqrtBetaX;

	x = x1;
	px = px1;
}


__forceinline__ __device__ double phaseChange(double& x0, double& px0, double& x1, double& px1) {

	double angle0 = atan2(px0, x0);	// angle = [-pi,pi]
	int mask0 = (angle0 < 0);
	angle0 += (mask0 * 2 * PassConstant::PI);	// convert angle to [0, 2pi]

	double angle1 = atan2(px1, x1);
	int mask1 = (angle1 < 0);
	angle1 += (mask1 * 2 * PassConstant::PI);

	double phase = angle0 - angle1;
	int mask_phase = (phase < 0);
	phase += (mask_phase * 2 * PassConstant::PI);

	return phase;
}


__global__ void record_init_value(Particle dev_particle, int Np_sur) {

	// Recording coordinates for the next turn calculation.

#ifdef PASS_CAL_PHASE

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < Np_sur)
	{
		dev_particle.last_x[tid] = dev_particle.x[tid];
		dev_particle.last_px[tid] = dev_particle.px[tid];
		dev_particle.last_y[tid] = dev_particle.y[tid];
		dev_particle.last_py[tid] = dev_particle.py[tid];

		dev_particle.phase_x[tid] = 0;
		dev_particle.phase_y[tid] = 0;

		tid += stride;
	}

#endif
}


__global__ void cal_accumulatePhaseChange(Particle dev_particle, int Np_sur, double sqrtBetaX, double sqrtBetaY, double alphaX, double alphaY) {

#ifdef PASS_CAL_PHASE

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < Np_sur)
	{
		double x0 = dev_particle.last_x[tid];
		double px0 = dev_particle.last_px[tid];
		double y0 = dev_particle.last_y[tid];
		double py0 = dev_particle.last_py[tid];

		double x1 = dev_particle.x[tid];
		double px1 = dev_particle.px[tid];
		double y1 = dev_particle.y[tid];
		double py1 = dev_particle.py[tid];

		physical2normalize(x0, px0, sqrtBetaX, alphaX);
		physical2normalize(y0, py0, sqrtBetaY, alphaY);
		physical2normalize(x1, px1, sqrtBetaX, alphaX);
		physical2normalize(y1, py1, sqrtBetaY, alphaY);

		double delta_phase_x = phaseChange(x0, px0, x1, px1);
		double delta_phase_y = phaseChange(y0, py0, y1, py1);

		dev_particle.phase_x[tid] += delta_phase_x;
		dev_particle.phase_y[tid] += delta_phase_y;

		dev_particle.last_x[tid] = dev_particle.x[tid];
		dev_particle.last_px[tid] = dev_particle.px[tid];
		dev_particle.last_y[tid] = dev_particle.y[tid];
		dev_particle.last_py[tid] = dev_particle.py[tid];

		tid += stride;
	}
#endif
}


__global__ void cal_averagePhaseChange(Particle dev_particle, int Np_sur, int totalTurn) {

#ifdef PASS_CAL_PHASE

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < Np_sur)
	{
		int alive = (dev_particle.tag[tid] > 0);

		dev_particle.phase_x[tid] /= (totalTurn * alive);
		dev_particle.phase_y[tid] /= (totalTurn * alive);

		tid += stride;
	}
#endif
}


void save_phase(Particle dev_particle, int Np, std::filesystem::path saveName) {

#ifdef PASS_CAL_PHASE

	Particle host_particle;
	host_particle.mem_allocate_cpu(Np);

	particle_copy(host_particle, dev_particle, Np, cudaMemcpyDeviceToHost, "phase");

	std::ofstream file(saveName);

	file << "tag" << "," << "phaseX" << "," << "phaseY" << "," << "nuX" << "," << "nuY" << std::endl;

	for (int i = 0; i < Np; i++)
	{
		file << std::setprecision(10)
			<< host_particle.tag[i] << ","
			<< host_particle.phase_x[i] << ","
			<< host_particle.phase_y[i] << ","
			<< host_particle.phase_x[i] / (2 * PassConstant::PI) << ","
			<< host_particle.phase_y[i] / (2 * PassConstant::PI) << "\n";
	}

	file.close();

	host_particle.mem_free_cpu();

#endif
}