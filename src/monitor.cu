#include "monitor.h"

#include <fstream>


DistMonitor::DistMonitor(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name, TimeEvent& timeevent) :simTime(timeevent) {
	commandType = "DistMonitor";
	name = obj_name;

	dev_bunch = Bunch.dev_bunch;
	Np = Bunch.Np;

	host_bunch = new Particle[Np];

	bunchId = Bunch.bunchId;

	saveDir = para.dir_output_distribution;
	saveName_part = para.hourMinSec + "_beam" + std::to_string(input_beamId) + "_" + para.beam_name[input_beamId] + "_bunch" + std::to_string(bunchId)
		+ "_" + std::to_string(Np) + "_hor_" + Bunch.dist_transverse + "_longi_" + Bunch.dist_logitudinal + "_" + name;

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

void DistMonitor::execute(int turn) {

	callCuda(cudaEventRecord(simTime.start, 0));
	float time_tmp = 0;

	auto logger = spdlog::get("logger");
	logger->debug("[DistMonitor] run: " + name);

	callCuda(cudaMemcpy(host_bunch, dev_bunch, Np * sizeof(Particle), cudaMemcpyDeviceToHost));

	std::filesystem::path saveName_full = saveDir / (saveName_part + "_turn_" + std::to_string(turn) + ".csv");
	std::ofstream file(saveName_full);

	file << "x" << "," << "px" << "," << "y" << "," << "py" << "," << "z" << "," << "pz" << "," << "tag" << "," << "lostTurn" << std::endl;

	for (int j = 0; j < Np; j++) {
		file << std::setprecision(10)
			<< (host_bunch + j)->x << ","
			<< (host_bunch + j)->px << ","
			<< (host_bunch + j)->y << ","
			<< (host_bunch + j)->py << ","
			<< (host_bunch + j)->z << ","
			<< (host_bunch + j)->pz << ","
			<< (host_bunch + j)->tag << ","
			<< (host_bunch + j)->lostTurn << "\n";
	}
	file.close();

	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.saveBunch += time_tmp;

}
