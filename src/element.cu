#include "element.h"

#include <fstream>


MarkerElement::MarkerElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name, const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

	name = obj_name;
	dev_bunch = Bunch.dev_bunch;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	Np = Bunch.Np;
	circumference = para.circumference;

	gamma = Bunch.gamma;
	beta = Bunch.beta;

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);
	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		s_previous = data.at("Sequence").at(obj_name).at("S previous (m)");
	}
	catch (json::exception e)
	{
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}

}

void MarkerElement::execute(int turn) {

	callCuda(cudaEventRecord(simTime.start, 0));
	float time_tmp = 0;

	//auto logger = spdlog::get("logger");
	//logger->info("[Marker Element] run: " + name);

	gamma = bunchRef.gamma;
	beta = bunchRef.beta;

	double l = s - s_previous;

	m12 = l;
	m34 = l;
	m56 = l / (beta * beta * gamma * gamma);

	transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np, m12, m34, m56, beta);

	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.transferElement += time_tmp;
}


SBendElement::SBendElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name) {
	commandType = "SBendElement";
	name = obj_name;
	dev_bunch = Bunch.dev_bunch;

	Np = Bunch.Np;

	//gamma = Bunch.gamma;
	//gammat = Bunch.gammat;
	//sigmaz = Bunch.sigmaz;
	//dp = Bunch.dp;

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		l = data.at("Sequence").at(obj_name).at("l (m)");
		angle = data.at("Sequence").at(obj_name).at("angle (rad)");
		e1 = data.at("Sequence").at(obj_name).at("e1 (rad)");
		e2 = data.at("Sequence").at(obj_name).at("e2 (rad)");
		hgap = data.at("Sequence").at(obj_name).at("hgap (m)");
		fint = data.at("Sequence").at(obj_name).at("fint");
		fintx = data.at("Sequence").at(obj_name).at("fintx");

	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}
}

void SBendElement::execute(int turn) {
	auto logger = spdlog::get("logger");
	logger->info("[SBend Element] run: " + name);
}


RBendElement::RBendElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name) {
	commandType = "RBendElement";
	name = obj_name;
	dev_bunch = Bunch.dev_bunch;

	Np = Bunch.Np;

	//gamma = Bunch.gamma;
	//gammat = Bunch.gammat;
	//sigmaz = Bunch.sigmaz;
	//dp = Bunch.dp;

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		l = data.at("Sequence").at(obj_name).at("l (m)");
		angle = data.at("Sequence").at(obj_name).at("angle (rad)");
		e1 = data.at("Sequence").at(obj_name).at("e1 (rad)");
		e2 = data.at("Sequence").at(obj_name).at("e2 (rad)");
		hgap = data.at("Sequence").at(obj_name).at("hgap (m)");
		fint = data.at("Sequence").at(obj_name).at("fint");
		fintx = data.at("Sequence").at(obj_name).at("fintx");

	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}
}

void RBendElement::execute(int turn) {
	auto logger = spdlog::get("logger");
	logger->info("[RBend Element] run: " + name);
}


QuadrupoleElement::QuadrupoleElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name) {
	commandType = "QuadrupoleElement";
	name = obj_name;
	dev_bunch = Bunch.dev_bunch;

	Np = Bunch.Np;

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		l = data.at("Sequence").at(obj_name).at("l (m)");
		k1 = data.at("Sequence").at(obj_name).at("k1 (m^-2)");

	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}
}

void QuadrupoleElement::execute(int turn) {
	auto logger = spdlog::get("logger");
	logger->info("[Quadrupole Element] run: " + name);
}


SextupoleElement::SextupoleElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name) {
	commandType = "SextupoleElement";
	name = obj_name;
	dev_bunch = Bunch.dev_bunch;

	Np = Bunch.Np;

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		l = data.at("Sequence").at(obj_name).at("l (m)");
		k2 = data.at("Sequence").at(obj_name).at("k2 (m^-3)");

	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}
}

void SextupoleElement::execute(int turn) {
	auto logger = spdlog::get("logger");
	logger->info("[Sextupole Element] run: " + name);
}


__global__ void transfer_drift(Particle* dev_bunch, int Np,
	double m12, double m34, double m56, double beta) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double px1 = 0, py1 = 0, pz1 = 0;
	double pt1;	// pt = DeltaE/(P0*c) = beta*DeltaP/P0

	while (tid < Np) {

		px1 = dev_bunch[tid].px;
		py1 = dev_bunch[tid].py;
		pz1 = dev_bunch[tid].pz;
		pt1 = pz1 * beta;

		dev_bunch[tid].x += m12 * px1;
		dev_bunch[tid].y += m34 * py1;
		dev_bunch[tid].z += m56 * pt1;

		tid += stride;
	}
}