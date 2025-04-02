#include "lattice.h"
#include "parameter.h"

#include <fstream>

Twiss::Twiss(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name) {

	name = obj_name;
	dev_bunch = Bunch.dev_bunch;

	Np = Bunch.Np;

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	//std::string key_bunch = "bunch" + std::to_string(Bunch.bunchId);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");

		alphax = data.at("Sequence").at(obj_name).at("Alpha x");
		alphay = data.at("Sequence").at(obj_name).at("Alpha y");
		alphax_previous = data.at("Sequence").at(obj_name).at("Alpha x previous");
		alphay_previous = data.at("Sequence").at(obj_name).at("Alpha y previous");

		betax = data.at("Sequence").at(obj_name).at("Beta x (m)");
		betay = data.at("Sequence").at(obj_name).at("Beta y (m)");
		betax_previous = data.at("Sequence").at(obj_name).at("Beta x previous (m)");
		betay_previous = data.at("Sequence").at(obj_name).at("Beta y previous (m)");

		mux = data.at("Sequence").at(obj_name).at("Mu x");
		muy = data.at("Sequence").at(obj_name).at("Mu y");
		mux_previous = data.at("Sequence").at(obj_name).at("Mu x previous");
		muy_previous = data.at("Sequence").at(obj_name).at("Mu y previous");

		Dx = data.at("Sequence").at(obj_name).at("Dx (m)");

		if (data.at("Sequence").at(obj_name).contains("Mu z"))
		{
			muz = data.at("Sequence").at(obj_name).at("Mu z");
			alphaz = data.at("Sequence").at(obj_name).at("Alpha z");
			betaz = data.at("Sequence").at(obj_name).at("Beta z (m)");
		}
		else
		{
			muz = 0;
			alphaz = 0;
			betaz = 0;
		}
	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}
}

void Twiss::print() {
	auto logger = spdlog::get("logger");

	logger->info("[Twiss] name is: {}, s = {}", name, s);
	logger->info("[Twiss] Alpha x = {}, Alpha y = {}", alphax, alphay);
	logger->info("[Twiss] Beta  x = {}, Beta  y = {}", betax, betay);
	logger->info("[Twiss] Mu    x = {}, Mu    y = {}", mux, muy);
	logger->info("[Twiss] Alpha x previous = {}, Alpha y previous = {}", alphax, alphay);
	logger->info("[Twiss] Beta  x previous = {}, Beta  y previous = {}", betax, betay);
	logger->info("[Twiss] Mu    x previous = {}, Mu    y previous = {}", mux, muy);
	logger->info("[Twiss] Dx      = {}", Dx);

	if (muz > 0)
	{
		logger->info("[Twiss] Alpha z = {}", alphaz);
		logger->info("[Twiss] Beta  z = {}", betaz);
		logger->info("[Twiss] Mu    z = {}", muz);
	}
	logger->info("");
}


void Twiss::run(int turn) {
	auto logger = spdlog::get("logger");

	//logger->debug("[Twiss] turn = {}, start running of : {}, s = {}", turn, name, s);
}