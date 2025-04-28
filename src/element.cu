#include "element.h"

#include <fstream>

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