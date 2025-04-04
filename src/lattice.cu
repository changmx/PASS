#include "lattice.h"
#include "parameter.h"
#include "constant.h"

#include <fstream>

Twiss::Twiss(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name) {

	name = obj_name;
	dev_bunch = Bunch.dev_bunch;

	Np = Bunch.Np;

	gamma = Bunch.gamma;
	gammat = Bunch.gammat;
	sigmaz = Bunch.sigmaz;
	dp = Bunch.dp;

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	//std::string key_bunch = "bunch" + std::to_string(Bunch.bunchId);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		s_previous = data.at("Sequence").at(obj_name).at("S previous (m)");

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

		//Dx = data.at("Sequence").at(obj_name).at("Dx (m)");

		if (data.at("Sequence").at(obj_name).contains("Mu z"))
		{
			muz = data.at("Sequence").at(obj_name).at("Mu z");
		}
		else
		{
			muz = 0;
		}

		logitudinal_transfer = data.at("Sequence").at(obj_name).at("Logitudinal transfer");
	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}

	double phi_x = (mux - mux_previous) * 2 * PassConstant::PI;
	double phi_y = (muy - muy_previous) * 2 * PassConstant::PI;
	double phi_z = muz * 2 * PassConstant::PI;

	m11_x = sqrt(betax / betax_previous) * (cos(phi_x) + alphax_previous * sin(phi_x));
	m12_x = sqrt(betax * betax_previous);
	m21_x = -1 * (1 + alphax * alphax_previous) / sqrt(betax * betax_previous) * sin(phi_x) + (alphax_previous - alphax) / sqrt(betax * betax_previous) * cos(phi_x);
	m22_x = sqrt(betax_previous / betax) * (cos(phi_x) - alphax * sin(phi_x));

	m11_y = sqrt(betay / betay_previous) * (cos(phi_y) + alphay_previous * sin(phi_y));
	m12_y = sqrt(betay * betay_previous);
	m21_y = -1 * (1 + alphay * alphay_previous) / sqrt(betay * betay_previous) * sin(phi_y) + (alphay_previous - alphay) / sqrt(betay * betay_previous) * cos(phi_y);
	m22_y = sqrt(betay_previous / betay) * (cos(phi_y) - alphay * sin(phi_y));

	if ("drift" == logitudinal_transfer)
	{
		m11_z = 1;
		m12_z = -1 * (1 / (gammat * gammat) - 1 / (gamma * gamma)) * (s - s_previous);
		m21_z = 0;
		m22_z = 1;
	}
	else if ("matrix" == logitudinal_transfer)
	{
		m11_z = cos(phi_z);
		m12_z = -1 * sigmaz / dp * sin(phi_z);
		m21_z = dp / sigmaz * sin(phi_z);
		m22_z = cos(phi_z);
	}
	else
	{
		m11_z = 1;
		m12_z = 0;
		m21_z = 0;
		m22_z = 1;
	}
}

void Twiss::print() {
	auto logger = spdlog::get("logger");

	logger->info("[Twiss] name = {}, s = {}", name, s);
	logger->info("[Twiss] Alpha x = {}, Alpha y = {}", alphax, alphay);
	logger->info("[Twiss] Beta  x = {}, Beta  y = {}", betax, betay);
	logger->info("[Twiss] Mu    x = {}, Mu    y = {}", mux, muy);
	logger->info("[Twiss] Alpha x previous = {}, Alpha y previous = {}", alphax, alphay);
	logger->info("[Twiss] Beta  x previous = {}, Beta  y previous = {}", betax, betay);
	logger->info("[Twiss] Mu    x previous = {}, Mu    y previous = {}", mux, muy);
	//logger->info("[Twiss] Dx      = {}", Dx);

	logger->info("[Twiss] Logitunial transfer = {}", logitudinal_transfer);
	logger->info("[Twiss] Mu   z = {}", muz);
	logger->info("[Twiss] gamma  = {}, gammat = {}", gamma, gammat);
	logger->info("[Twiss] sigmaz = {}, dp     = {}", sigmaz, dp);
}


void Twiss::run(int turn) {
	auto logger = spdlog::get("logger");

	logger->debug("[Twiss] turn = {}, start running of : {}, s = {}", turn, name, s);
}