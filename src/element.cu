#include "element.h"
#include "constant.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <general.h>


MarkerElement::MarkerElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

	commandType = "MarkerElement";
	name = obj_name;
	dev_bunch = Bunch.dev_bunch;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	Np_sur = Bunch.Np_sur;
	circumference = para.circumference;

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		drift_length = data.at("Sequence").at(obj_name).at("Drift length (m)");

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

	Np_sur = bunchRef.Np_sur;
	double gamma = bunchRef.gamma;
	double beta = bunchRef.beta;

	double drift = drift_length;

	transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference, gamma, drift);

	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.transferElement += time_tmp;
}


SBendElement::SBendElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

	commandType = "SBendElement";
	name = obj_name;
	dev_bunch = Bunch.dev_bunch;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	Np_sur = Bunch.Np_sur;
	circumference = para.circumference;

	callCuda(cudaMalloc(&dev_kn, max_error_order * sizeof(double)));
	callCuda(cudaMalloc(&dev_ks, max_error_order * sizeof(double)));

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		l = data.at("Sequence").at(obj_name).at("L (m)");
		drift_length = data.at("Sequence").at(obj_name).at("Drift length (m)");
		angle = data.at("Sequence").at(obj_name).at("angle (rad)");
		e1 = data.at("Sequence").at(obj_name).at("e1 (rad)");
		e2 = data.at("Sequence").at(obj_name).at("e2 (rad)");
		hgap = data.at("Sequence").at(obj_name).at("hgap (m)");
		fint = data.at("Sequence").at(obj_name).at("fint");
		fintx = data.at("Sequence").at(obj_name).at("fintx");

		isFieldError = data.at("Sequence").at(obj_name).at("isFieldError");

		spdlog::get("logger")->debug("[SBend Element] run: {}, s = {}, l = {}, drift_length = {}, angle = {}, e1 = {}, e2 = {}, hgap = {}, fint = {}, fintx = {}, isFieldError = {}",
			name, s, l, drift_length, angle, e1, e2, hgap, fint, fintx, isFieldError);
	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}
}

void SBendElement::execute(int turn) {

	callCuda(cudaEventRecord(simTime.start, 0));
	float time_tmp = 0;

	//auto logger = spdlog::get("logger");
	//logger->debug("[SBend Element] run: " + name);

	Np_sur = bunchRef.Np_sur;
	double gamma = bunchRef.gamma;
	double beta = bunchRef.beta;

	double drift = drift_length;

	double l_used = 0;
	double angle_used = 0;

	if (isFieldError)
	{
		l_used = l / 2;
		angle_used = angle / 2;
	}
	else
	{
		l_used = l;
		angle_used = angle;
	}

	double rho = l_used / angle_used;
	double h = 1 / rho;

	double cx = cos(h * l_used);
	double sx = sin(h * l_used);

	double r11 = cx;
	double r12 = sx / h;
	double r16 = (1 - cx) / (h * beta);
	double r21 = -h * sx;
	double r22 = cx;
	double r26 = sx / beta;
	double r34 = l_used;
	double r51 = -sx / beta;
	double r52 = -(1 - cx) / (h * beta);
	double r56 = l_used / (beta * beta * gamma * gamma) - (h * l_used - sx) / (h * beta * beta);

	double psi1 = e1;
	double psi2 = e2;
	double fint1 = fint;
	double fint2 = fintx;
	double psip1 = psi1 - 2.0 * hgap * h * fint1 / cos(psi1) * (1.0 + pow(sin(psi1), 2));
	double psip2 = psi2 - 2.0 * hgap * h * fint2 / cos(psi2) * (1.0 + pow(sin(psi2), 2));

	double fl21i = h * tan(psi1);
	double fl43i = -h * tan(psip1);
	double fr21i = h * tan(psi2);
	double fr43i = -h * tan(psip2);

	callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference, gamma, drift));

	if (isFieldError)
	{
		callKernel(transfer_dipole_half_left << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference,
			r11, r12, r16, r21, r22, r26, r34, r51, r52, r56, fl21i, fl43i, fr21i, fr43i));

		callKernel(transfer_multipole_kicker << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, max_error_order, dev_kn, dev_ks, l));

		callKernel(transfer_dipole_half_right << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference,
			r11, r12, r16, r21, r22, r26, r34, r51, r52, r56, fl21i, fl43i, fr21i, fr43i));
	}
	else
	{
		callKernel(transfer_dipole_full << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference,
			r11, r12, r16, r21, r22, r26, r34, r51, r52, r56, fl21i, fl43i, fr21i, fr43i));
	}

	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.transferElement += time_tmp;
}


RBendElement::RBendElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

	commandType = "RBendElement";
	name = obj_name;
	dev_bunch = Bunch.dev_bunch;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	Np_sur = Bunch.Np_sur;
	circumference = para.circumference;

	callCuda(cudaMalloc(&dev_kn, max_error_order * sizeof(double)));
	callCuda(cudaMalloc(&dev_ks, max_error_order * sizeof(double)));

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		l = data.at("Sequence").at(obj_name).at("L (m)");
		drift_length = data.at("Sequence").at(obj_name).at("Drift length (m)");
		angle = data.at("Sequence").at(obj_name).at("angle (rad)");
		e1 = data.at("Sequence").at(obj_name).at("e1 (rad)");
		e2 = data.at("Sequence").at(obj_name).at("e2 (rad)");
		hgap = data.at("Sequence").at(obj_name).at("hgap (m)");
		fint = data.at("Sequence").at(obj_name).at("fint");
		fintx = data.at("Sequence").at(obj_name).at("fintx");

		isFieldError = data.at("Sequence").at(obj_name).at("isFieldError");
	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}
}


void RBendElement::execute(int turn) {

	callCuda(cudaEventRecord(simTime.start, 0));
	float time_tmp = 0;

	//auto logger = spdlog::get("logger");
	//logger->debug("[RBend Element] run: " + name);

	Np_sur = bunchRef.Np_sur;
	double gamma = bunchRef.gamma;
	double beta = bunchRef.beta;

	double drift = drift_length;

	double l_used = 0;
	double angle_used = 0;

	if (isFieldError)
	{
		l_used = l / 2;
		angle_used = angle / 2;
	}
	else
	{
		l_used = l;
		angle_used = angle;
	}

	double rho = l_used / angle_used;
	double h = 1 / rho;

	double cx = cos(h * l_used);
	double sx = sin(h * l_used);

	double r11 = cx;
	double r12 = sx / h;
	double r16 = (1 - cx) / (h * beta);
	double r21 = -h * sx;
	double r22 = cx;
	double r26 = sx / beta;
	double r34 = l_used;
	double r51 = -sx / beta;
	double r52 = -(1 - cx) / (h * beta);
	double r56 = l_used / (beta * beta * gamma * gamma) - (h * l_used - sx) / (h * beta * beta);

	double psi1 = e1;
	double psi2 = e2;
	double fint1 = fint;
	double fint2 = fintx;
	double psip1 = psi1 - 2.0 * hgap * h * fint1 / cos(psi1) * (1.0 + pow(sin(psi1), 2));
	double psip2 = psi2 - 2.0 * hgap * h * fint2 / cos(psi2) * (1.0 + pow(sin(psi2), 2));

	double fl21i = h * tan(psi1);
	double fl43i = -h * tan(psip1);
	double fr21i = h * tan(psi2);
	double fr43i = -h * tan(psip2);

	transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference, gamma, drift);

	if (isFieldError)
	{
		transfer_dipole_half_left << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference,
			r11, r12, r16, r21, r22, r26, r34, r51, r52, r56, fl21i, fl43i, fr21i, fr43i);

		callKernel(transfer_multipole_kicker << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, max_error_order, dev_kn, dev_ks, l));

		transfer_dipole_half_right << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference,
			r11, r12, r16, r21, r22, r26, r34, r51, r52, r56, fl21i, fl43i, fr21i, fr43i);
	}
	else
	{
		transfer_dipole_full << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference,
			r11, r12, r16, r21, r22, r26, r34, r51, r52, r56, fl21i, fl43i, fr21i, fr43i);
	}

	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.transferElement += time_tmp;

}


QuadrupoleElement::QuadrupoleElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

	commandType = "QuadrupoleElement";
	name = obj_name;
	dev_bunch = Bunch.dev_bunch;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	Np_sur = Bunch.Np_sur;
	circumference = para.circumference;

	callCuda(cudaMalloc(&dev_kn, max_error_order * sizeof(double)));
	callCuda(cudaMalloc(&dev_ks, max_error_order * sizeof(double)));

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		l = data.at("Sequence").at(obj_name).at("L (m)");
		drift_length = data.at("Sequence").at(obj_name).at("Drift length (m)");
		k1 = data.at("Sequence").at(obj_name).at("k1 (m^-2)");
		k1s = data.at("Sequence").at(obj_name).at("k1s (m^-2)");

		isFieldError = data.at("Sequence").at(obj_name).at("isFieldError");

		spdlog::get("logger")->debug("[Quadrupole Element] run: {}, s = {}, l = {}, drift_length = {}, k1 = {}, k1s = {}, isFieldError = {}",
			name, s, l, drift_length, k1, k1s, isFieldError);
	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}
}

void QuadrupoleElement::execute(int turn) {
	callCuda(cudaEventRecord(simTime.start, 0));
	float time_tmp = 0;

	//auto logger = spdlog::get("logger");
	//logger->debug("[Quadrupole Element] run: " + name);

	Np_sur = bunchRef.Np_sur;
	double gamma = bunchRef.gamma;
	double beta = bunchRef.beta;

	double drift = drift_length;

	double EPSILON = 1e-9;

	callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference, gamma, drift));

	if (isFieldError)
	{

		if (fabs(k1) > EPSILON && fabs(k1s) < EPSILON)	// k1 != 0 && k1s == 0
		{
			callKernel(transfer_quadrupole_norm << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference, k1, l / 2));
			callKernel(transfer_multipole_kicker << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, max_error_order, dev_kn, dev_ks, l));
			callKernel(transfer_quadrupole_norm << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference, k1, l / 2));
		}
		else if (fabs(k1) < EPSILON && fabs(k1s) > EPSILON)	// k1 == 0 && k1s != 0
		{
			callKernel(transfer_quadrupole_skew << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference, k1s, l / 2));
			callKernel(transfer_multipole_kicker << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, max_error_order, dev_kn, dev_ks, l));
			callKernel(transfer_quadrupole_skew << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference, k1s, l / 2));
		}
		else if (fabs(k1) < EPSILON && fabs(k1s) < EPSILON)	// k1 == 0 && k1s == 0
		{
			callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference, gamma, l / 2));
			callKernel(transfer_multipole_kicker << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, max_error_order, dev_kn, dev_ks, l));
			callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference, gamma, l / 2));
		}
		else
		{
			spdlog::get("logger")->error("[Quadrupole Element] {}: k1 = {}, k1s = {}, there should be and only 1 variable equal to 0",
				name, k1, k1s);
			std::exit(EXIT_FAILURE);
		}
	}
	else
	{
		if (fabs(k1) > EPSILON && fabs(k1s) < EPSILON)
		{
			callKernel(transfer_quadrupole_norm << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference, k1, l));
		}
		else if (fabs(k1) < EPSILON && fabs(k1s) > EPSILON)
		{
			callKernel(transfer_quadrupole_skew << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference, k1s, l));
		}
		else if (fabs(k1) < EPSILON && fabs(k1s) < EPSILON)	// k1 == 0 && k1s == 0
		{
			callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference, gamma, l));
		}
		else
		{
			spdlog::get("logger")->error("[Quadrupole Element] {}: k1 = {}, k1s = {}, there should be and only 1 variable equal to 0",
				name, k1, k1s);
			std::exit(EXIT_FAILURE);
		}

	}

	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.transferElement += time_tmp;
}


SextupoleElement::SextupoleElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

	commandType = "SextupoleElement";
	name = obj_name;
	dev_bunch = Bunch.dev_bunch;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	Np_sur = Bunch.Np_sur;
	circumference = para.circumference;

	callCuda(cudaMalloc(&dev_kn, max_error_order * sizeof(double)));
	callCuda(cudaMalloc(&dev_ks, max_error_order * sizeof(double)));

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		l = data.at("Sequence").at(obj_name).at("L (m)");
		drift_length = data.at("Sequence").at(obj_name).at("Drift length (m)");
		k2 = data.at("Sequence").at(obj_name).at("k2 (m^-3)");
		k2s = data.at("Sequence").at(obj_name).at("k2s (m^-3)");

		isFieldError = data.at("Sequence").at(obj_name).at("isFieldError");
	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}
}

void SextupoleElement::execute(int turn) {
	callCuda(cudaEventRecord(simTime.start, 0));
	float time_tmp = 0;

	//auto logger = spdlog::get("logger");
	//logger->debug("[Sextupole Element] run: " + name);

	Np_sur = bunchRef.Np_sur;
	double gamma = bunchRef.gamma;
	double beta = bunchRef.beta;

	double drift = drift_length;

	double EPSILON = 1e-9;

	transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference, gamma, drift + l / 2);

	if (fabs(k2) > EPSILON && fabs(k2s) < EPSILON)	// k2 != 0 && k2s == 0
	{
		transfer_sextupole_norm << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, k2, l);

	}
	else if (fabs(k2) < EPSILON && fabs(k2s) > EPSILON)	// k2 == 0 && k2s != 0
	{
		transfer_sextupole_skew << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, k2s, l);
	}
	else if (fabs(k2) < EPSILON && fabs(k2s) < EPSILON)	// k2 == 0 && k2s == 0
	{
		// Thin lens approximation, do nothing
	}
	else
	{
		spdlog::get("logger")->error("[Sextupole Element] {}: k2 = {}, k2s = {}, there should be and only 1 variable equal to 0",
			name, k2, k2s);
		std::exit(EXIT_FAILURE);
	}

	if (isFieldError)
	{
		callKernel(transfer_multipole_kicker << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, max_error_order, dev_kn, dev_ks, l));
	}

	transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference, gamma, l / 2);

	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.transferElement += time_tmp;

}


OctupoleElement::OctupoleElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

	commandType = "OctupoleElement";
	name = obj_name;
	dev_bunch = Bunch.dev_bunch;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	Np_sur = Bunch.Np_sur;
	circumference = para.circumference;

	callCuda(cudaMalloc(&dev_kn, max_error_order * sizeof(double)));
	callCuda(cudaMalloc(&dev_ks, max_error_order * sizeof(double)));

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		l = data.at("Sequence").at(obj_name).at("L (m)");
		drift_length = data.at("Sequence").at(obj_name).at("Drift length (m)");
		k3 = data.at("Sequence").at(obj_name).at("k3 (m^-4)");
		k3s = data.at("Sequence").at(obj_name).at("k3s (m^-4)");

		isFieldError = data.at("Sequence").at(obj_name).at("isFieldError");
	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}
}


void OctupoleElement::execute(int turn) {
	callCuda(cudaEventRecord(simTime.start, 0));
	float time_tmp = 0;

	//auto logger = spdlog::get("logger");
	//logger->debug("[Octupole Element] run: " + name);

	Np_sur = bunchRef.Np_sur;
	double gamma = bunchRef.gamma;
	double beta = bunchRef.beta;

	double drift = drift_length;

	double EPSILON = 1e-9;

	transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference, gamma, drift + l / 2);

	if (fabs(k3) > EPSILON && fabs(k3s) < EPSILON)	// k3 != 0 && k3s == 0
	{
		transfer_octupole_norm << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, k3, l);

	}
	else if (fabs(k3) < EPSILON && fabs(k3s) > EPSILON)	// k3 == 0 && k3s != 0
	{
		transfer_octupole_skew << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, k3s, l);
	}
	else if (fabs(k3) < EPSILON && fabs(k3s) < EPSILON)	// k3 == 0 && k3s == 0
	{
		// do nothing
	}
	else
	{
		spdlog::get("logger")->error("[Octupole Element] {}: k3 = {}, k3s = {}, there should be and only 1 variable equal to 0",
			name, k3, k3s);
		std::exit(EXIT_FAILURE);
	}

	if (isFieldError)
	{
		callKernel(transfer_multipole_kicker << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, max_error_order, dev_kn, dev_ks, l));
	}

	transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference, gamma, l / 2);

	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.transferElement += time_tmp;

}


HKickerElement::HKickerElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

	commandType = "HKickerElement";
	name = obj_name;
	dev_bunch = Bunch.dev_bunch;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	Np_sur = Bunch.Np_sur;
	circumference = para.circumference;

	callCuda(cudaMalloc(&dev_kn, max_error_order * sizeof(double)));
	callCuda(cudaMalloc(&dev_ks, max_error_order * sizeof(double)));

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		l = data.at("Sequence").at(obj_name).at("L (m)");
		drift_length = data.at("Sequence").at(obj_name).at("Drift length (m)");
		kick = data.at("Sequence").at(obj_name).at("kick");

		isFieldError = data.at("Sequence").at(obj_name).at("isFieldError");
	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}
}


void HKickerElement::execute(int turn) {
	callCuda(cudaEventRecord(simTime.start, 0));
	float time_tmp = 0;

	//auto logger = spdlog::get("logger");
	//logger->debug("[HKicker Element] run: " + name);

	Np_sur = bunchRef.Np_sur;
	double gamma = bunchRef.gamma;
	double beta = bunchRef.beta;

	double drift = drift_length;

	transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference, gamma, drift + l / 2);

	transfer_hkicker << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, kick);

	if (isFieldError)
	{
		callKernel(transfer_multipole_kicker << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, max_error_order, dev_kn, dev_ks, l));
	}

	transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference, gamma, l / 2);

	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.transferElement += time_tmp;

}


VKickerElement::VKickerElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

	commandType = "VKickerElement";
	name = obj_name;
	dev_bunch = Bunch.dev_bunch;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	Np_sur = Bunch.Np_sur;
	circumference = para.circumference;

	callCuda(cudaMalloc(&dev_kn, max_error_order * sizeof(double)));
	callCuda(cudaMalloc(&dev_ks, max_error_order * sizeof(double)));

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		l = data.at("Sequence").at(obj_name).at("L (m)");
		drift_length = data.at("Sequence").at(obj_name).at("Drift length (m)");
		kick = data.at("Sequence").at(obj_name).at("kick");

		isFieldError = data.at("Sequence").at(obj_name).at("isFieldError");
	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}
}


void VKickerElement::execute(int turn) {
	callCuda(cudaEventRecord(simTime.start, 0));
	float time_tmp = 0;

	//auto logger = spdlog::get("logger");
	//logger->debug("[VKicker Element] run: " + name);

	Np_sur = bunchRef.Np_sur;
	double gamma = bunchRef.gamma;
	double beta = bunchRef.beta;

	double drift = drift_length;

	transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference, gamma, drift + l / 2);

	transfer_vkicker << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, kick);

	if (isFieldError)
	{
		callKernel(transfer_multipole_kicker << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, max_error_order, dev_kn, dev_ks, l));
	}

	transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, circumference, gamma, l / 2);

	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.transferElement += time_tmp;

}


RFElement::RFElement(const Parameter& para, int input_beamId, Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

	commandType = "RFElement";
	name = obj_name;
	dev_bunch = Bunch.dev_bunch;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	Np_sur = Bunch.Np_sur;
	circumference = para.circumference;

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		//l = data.at("Sequence").at(obj_name).at("L (m)");
		//drift_length = data.at("Sequence").at(obj_name).at("Drift length (m)");

		for (size_t Nset = 0; Nset < data.at("Sequence").at(obj_name).at("RF Data files").size(); Nset++) {
			filenames.push_back(data.at("Sequence").at(obj_name).at("RF Data files")[Nset]);
		}

	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}

	if (Bunch.Nproton == 0 && Bunch.Nneutron == 0)
		ratio = 1.0;
	else if (Bunch.Nproton == 1 && Bunch.Nneutron == 0)
		ratio = 1.0;
	else
		ratio = (double)Bunch.Ncharge / (Bunch.Nproton + Bunch.Nneutron);

	radius = circumference / (2 * PassConstant::PI);

	Nrf = filenames.size();

	std::vector<RFData> fileData_tmp = readRFDataFromCSV(filenames[0]);
	Nturn_rf = fileData_tmp.size();

	callCuda(cudaMallocPitch((void**)&dev_rf_data, &pitch_rf, Nturn_rf * sizeof(RFData), Nrf));
	callCuda(cudaMemset2D(dev_rf_data, pitch_rf, 0, Nturn_rf * sizeof(RFData), Nrf));

	for (int i = 0; i < Nrf; i++) {
		std::vector<RFData> hostData = readRFDataFromCSV(filenames[i]);

		host_rf_data.push_back(hostData);

		RFData* dev_rf_data_row = (RFData*)((char*)dev_rf_data + i * pitch_rf);
		callCuda(cudaMemcpy(dev_rf_data_row, hostData.data(), Nturn_rf * sizeof(RFData), cudaMemcpyHostToDevice));
	}

}


void RFElement::execute(int turn) {

	callCuda(cudaEventRecord(simTime.start, 0));
	float time_tmp = 0;

	//auto logger = spdlog::get("logger");
	//logger->debug("[RF Element] run: " + name);

	Np_sur = bunchRef.Np_sur;

	double m0 = bunchRef.m0;
	double gammat = bunchRef.gammat;

	double Ek0 = bunchRef.Ek;
	double gamma0 = bunchRef.gamma;
	double beta0 = bunchRef.beta;
	double eta0 = 1 / (gammat * gammat) - 1 / (gamma0 * gamma0);

	//double drift = drift_length;

	double dE_syn = 0.0;

	for (int i = 0; i < Nrf; i++)
	{
		dE_syn += ratio * host_rf_data[i][turn - 1].voltage * sin(host_rf_data[i][turn - 1].phis);
	}

	double Ek1 = Ek0 + dE_syn;
	double gamma1 = Ek1 / m0 + 1;
	double beta1 = sqrt(1 - 1 / (gamma1 * gamma1));
	double eta1 = 1 / (gammat * gammat) - 1 / (gamma1 * gamma1);

	bunchRef.Ek = Ek1;
	bunchRef.gamma = gamma1;
	bunchRef.beta = beta1;

	//transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, beta, gamma, drift + l);

	transfer_rf << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np_sur, turn, beta0, beta1, gamma0, gamma1,
		dev_rf_data, pitch_rf, Nrf, Nturn_rf,
		radius, ratio, dE_syn, eta1, Ek1 + m0);

	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.transferElement += time_tmp;

}


__global__ void transfer_drift(Particle* dev_bunch, int Np_sur, double beta, double circumference,
	double gamma, double drift_length) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double r12 = drift_length;
	double r34 = drift_length;
	double r56 = drift_length / (beta * beta * gamma * gamma);

	double tau0 = 0, tau1 = 0;	// tau = z/beta - ct(=0) = z/beta
	double pt0 = 0;	// pt = DeltaE/(P0*c) = beta*DeltaP/P0

	double c_half = 0;
	int over = 0, under = 0;

	while (tid < Np_sur) {

		tau0 = dev_bunch[tid].z / beta;
		pt0 = dev_bunch[tid].pz * beta;

		dev_bunch[tid].x += r12 * dev_bunch[tid].px;
		dev_bunch[tid].y += r34 * dev_bunch[tid].py;

		tau1 = tau0 + r56 * pt0;
		dev_bunch[tid].z = tau1 * beta;

		c_half = circumference * 0.5;
		over = (dev_bunch[tid].z > c_half);
		under = (dev_bunch[tid].z < -c_half);
		dev_bunch[tid].z += (under - over) * circumference;
		//if (dev_bunch[tid].tag == 1)
		//{
		//	printf("dev_bunch[1] pz0 = %f, z0 = %f, z1 = %f, z2 = %f\n",
		//		pt0 / beta, tau0 * beta, tau1 * beta, dev_bunch[tid].z);
		//}
		tid += stride;
	}
}

__global__ void transfer_dipole_full(Particle* dev_bunch, int Np_sur, double beta, double circumference,
	double r11, double r12, double r16, double r21, double r22, double r26,
	double r34, double r51, double r52, double r56,
	double fl21i, double fl43i, double fr21i, double fr43i) {

	// I would like to express my gratitude to Dr.Ren Hang(renhang@impcas.ac.cn)
	// for providing the code for dipole transfer 

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double x1 = 0, px1 = 0, y1 = 0, py1 = 0;
	double x2 = 0, px2 = 0, y2 = 0, py2 = 0;
	double x3 = 0, px3 = 0, y3 = 0, py3 = 0;
	double tau0 = 0, tau1 = 0, tau2 = 0, tau3 = 0;	// tau = z/beta - ct(=0) = z/beta
	double pt0 = 0, pt1 = 0, pt2 = 0, pt3 = 0;	// pt = DeltaE/(P0*c) = beta*DeltaP/P0

	double fl21 = 0, fl43 = 0, fr21 = 0, fr43 = 0;

	double d = 0;	// about power error

	double c_half = 0;
	int over = 0, under = 0;

	while (tid < Np_sur) {

		tau0 = dev_bunch[tid].z / beta;
		pt0 = dev_bunch[tid].pz * beta;

		fl21 = fl21i / (1 + pt0 / beta);
		fl43 = fl43i / (1 + pt0 / beta);
		fr21 = fr21i / (1 + pt0 / beta);
		fr43 = fr43i / (1 + pt0 / beta);

		// apply the influence of left fringe field
		x1 = dev_bunch[tid].x;
		px1 = dev_bunch[tid].px + fl21 * x1;
		y1 = dev_bunch[tid].y;
		py1 = dev_bunch[tid].py + fl43 * y1;
		tau1 = tau0;
		pt1 = pt0 + d;

		// apply the influence of dipole
		x2 = r11 * x1 + r12 * px1 + r16 * pt1;
		px2 = r21 * x1 + r22 * px1 + r26 * pt1;
		y2 = y1 + r34 * py1;
		py2 = py1;
		tau2 = r51 * x1 + r52 * px1 + tau1 + r56 * pt1;
		pt2 = pt1;

		// apply the influece of right fringe field
		x3 = x2;
		px3 = px2 + fr21 * x2;
		y3 = y2;
		py3 = py2 + fr43 * y2;
		tau3 = tau2;
		pt3 = pt2 - d;

		dev_bunch[tid].x = x3;
		dev_bunch[tid].px = px3;
		dev_bunch[tid].y = y3;
		dev_bunch[tid].py = py3;
		dev_bunch[tid].z = tau3 * beta;
		dev_bunch[tid].pz = pt3 / beta;

		c_half = circumference * 0.5;
		over = (dev_bunch[tid].z > c_half);
		under = (dev_bunch[tid].z < -c_half);
		dev_bunch[tid].z += (under - over) * circumference;
		//if (dev_bunch[tid].tag == 1)
		//{
		//	printf("dev_bunch[1] z0 = %f, z1 = %f, z2 = %f, circumference = %f\n",
		//		tau0 * beta, tau1 * beta, dev_bunch[tid].z, circumference);
		//}
		tid += stride;
	}

}


__global__ void transfer_dipole_half_left(Particle* dev_bunch, int Np_sur, double beta, double circumference,
	double r11, double r12, double r16, double r21, double r22, double r26,
	double r34, double r51, double r52, double r56,
	double fl21i, double fl43i, double fr21i, double fr43i) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double x1 = 0, px1 = 0, y1 = 0, py1 = 0;
	double x2 = 0, px2 = 0, y2 = 0, py2 = 0;
	double x3 = 0, px3 = 0, y3 = 0, py3 = 0;
	double tau0 = 0, tau1 = 0, tau2 = 0, tau3 = 0;	// tau = z/beta - ct(=0) = z/beta
	double pt0 = 0, pt1 = 0, pt2 = 0, pt3 = 0;	// pt = DeltaE/(P0*c) = beta*DeltaP/P0

	double fl21 = 0, fl43 = 0;

	double d = 0;	// about power error

	double c_half = 0;
	int over = 0, under = 0;

	while (tid < Np_sur) {

		tau0 = dev_bunch[tid].z / beta;
		pt0 = dev_bunch[tid].pz * beta;

		fl21 = fl21i / (1 + pt0 / beta);
		fl43 = fl43i / (1 + pt0 / beta);

		// apply the influence of left fringe field
		x1 = dev_bunch[tid].x;
		px1 = dev_bunch[tid].px + fl21 * x1;
		y1 = dev_bunch[tid].y;
		py1 = dev_bunch[tid].py + fl43 * y1;
		tau1 = tau0;
		pt1 = pt0 + d;

		// apply the influence of dipole
		x2 = r11 * x1 + r12 * px1 + r16 * pt1;
		px2 = r21 * x1 + r22 * px1 + r26 * pt1;
		y2 = y1 + r34 * py1;
		py2 = py1;
		tau2 = r51 * x1 + r52 * px1 + tau1 + r56 * pt1;
		pt2 = pt1;

		// no influece of right fringe field
		x3 = x2;
		px3 = px2;
		y3 = y2;
		py3 = py2;
		tau3 = tau2;
		pt3 = pt2 - d;

		dev_bunch[tid].x = x3;
		dev_bunch[tid].px = px3;
		dev_bunch[tid].y = y3;
		dev_bunch[tid].py = py3;
		dev_bunch[tid].z = tau3 * beta;
		dev_bunch[tid].pz = pt3 / beta;

		c_half = circumference * 0.5;
		over = (dev_bunch[tid].z > c_half);
		under = (dev_bunch[tid].z < -c_half);
		dev_bunch[tid].z += (under - over) * circumference;
		//if (dev_bunch[tid].tag == 1)
		//{
		//	printf("dev_bunch[1] z0 = %f, z1 = %f, z2 = %f, circumference = %f\n",
		//		tau0 * beta, tau1 * beta, dev_bunch[tid].z, circumference);
		//}
		tid += stride;
	}

}


__global__ void transfer_dipole_half_right(Particle* dev_bunch, int Np_sur, double beta, double circumference,
	double r11, double r12, double r16, double r21, double r22, double r26,
	double r34, double r51, double r52, double r56,
	double fl21i, double fl43i, double fr21i, double fr43i) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double x1 = 0, px1 = 0, y1 = 0, py1 = 0;
	double x2 = 0, px2 = 0, y2 = 0, py2 = 0;
	double x3 = 0, px3 = 0, y3 = 0, py3 = 0;
	double tau0 = 0, tau1 = 0, tau2 = 0, tau3 = 0;	// tau = z/beta - ct(=0) = z/beta
	double pt0 = 0, pt1 = 0, pt2 = 0, pt3 = 0;	// pt = DeltaE/(P0*c) = beta*DeltaP/P0

	double fr21 = 0, fr43 = 0;

	double d = 0;	// about power error

	double c_half = 0;
	int over = 0, under = 0;

	while (tid < Np_sur) {

		tau0 = dev_bunch[tid].z / beta;
		pt0 = dev_bunch[tid].pz * beta;

		fr21 = fr21i / (1 + pt0 / beta);
		fr43 = fr43i / (1 + pt0 / beta);

		// no influence of left fringe field
		x1 = dev_bunch[tid].x;
		px1 = dev_bunch[tid].px;
		y1 = dev_bunch[tid].y;
		py1 = dev_bunch[tid].py;
		tau1 = tau0;
		pt1 = pt0 + d;

		// apply the influence of dipole
		x2 = r11 * x1 + r12 * px1 + r16 * pt1;
		px2 = r21 * x1 + r22 * px1 + r26 * pt1;
		y2 = y1 + r34 * py1;
		py2 = py1;
		tau2 = r51 * x1 + r52 * px1 + tau1 + r56 * pt1;
		pt2 = pt1;

		// apply the influece of right fringe field
		x3 = x2;
		px3 = px2 + fr21 * x2;
		y3 = y2;
		py3 = py2 + fr43 * y2;
		tau3 = tau2;
		pt3 = pt2 - d;

		dev_bunch[tid].x = x3;
		dev_bunch[tid].px = px3;
		dev_bunch[tid].y = y3;
		dev_bunch[tid].py = py3;
		dev_bunch[tid].z = tau3 * beta;
		dev_bunch[tid].pz = pt3 / beta;

		c_half = circumference * 0.5;
		over = (dev_bunch[tid].z > c_half);
		under = (dev_bunch[tid].z < -c_half);
		dev_bunch[tid].z += (under - over) * circumference;
		//if (dev_bunch[tid].tag == 1)
		//{
		//	printf("dev_bunch[1] z0 = %f, z1 = %f, z2 = %f, circumference = %f\n",
		//		tau0 * beta, tau1 * beta, dev_bunch[tid].z, circumference);
		//}
		tid += stride;
	}

}


__global__ void transfer_quadrupole_norm(Particle* dev_bunch, int Np_sur, double beta, double circumference,
	double k1, double l) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double x0 = 0, px0 = 0, y0 = 0, py0 = 0;
	double x1 = 0, px1 = 0, y1 = 0, py1 = 0;

	double tau0 = 0, tau1 = 0;	// tau = z/beta - ct(=0) = z/beta
	double pt0 = 0;	// pt = DeltaE/(P0*c) = beta*DeltaP/P0

	double k1_chrom = 0, omega = 0;
	double cx = 0, sx = 0, chx = 0, shx = 0;
	double r11 = 0, r12 = 0, r21 = 0, r22 = 0, r33 = 0, r34 = 0, r43 = 0, r44 = 0, r56 = 0;

	double c_half = 0;
	int over = 0, under = 0;

	while (tid < Np_sur) {

		x0 = dev_bunch[tid].x;
		px0 = dev_bunch[tid].px;
		y0 = dev_bunch[tid].y;
		py0 = dev_bunch[tid].py;
		tau0 = dev_bunch[tid].z / beta;
		pt0 = dev_bunch[tid].pz * beta;

		k1_chrom = k1 / (1 + pt0 / beta);
		omega = sqrt(fabs(k1_chrom));

		cx = cos(omega * l);
		sx = sin(omega * l);
		chx = cosh(omega * l);
		shx = sinh(omega * l);

		//printf("Quadrupole: k1_chrom = %f, pz = %f, beta = %f, pt = %f, cx = %f, sx = %f, chx =%f, shx = %f\n",
		//	k1_chrom, dev_bunch[tid].pz, beta, pt0, cx, sx, chx, shx);

		if (k1_chrom > 0) {
			r11 = cx;
			r12 = sx / omega;
			r21 = -omega * sx;
			r22 = cx;
			r33 = chx;
			r34 = shx / omega;
			r43 = omega * shx;
			r44 = chx;
			r56 = l * (1 / (beta * beta) - 1);
		}
		else {
			r11 = chx;
			r12 = shx / omega;
			r21 = omega * shx;
			r22 = chx;
			r33 = cx;
			r34 = sx / omega;
			r43 = -omega * sx;
			r44 = cx;
			r56 = l * (1 / (beta * beta) - 1);
		}

		x1 = r11 * x0 + r12 * px0;
		px1 = r21 * x0 + r22 * px0;
		y1 = r33 * y0 + r34 * py0;
		py1 = r43 * y0 + r44 * py0;
		tau1 = tau0 + r56 * pt0;

		dev_bunch[tid].x = x1;
		dev_bunch[tid].px = px1;
		dev_bunch[tid].y = y1;
		dev_bunch[tid].py = py1;
		dev_bunch[tid].z = tau1 * beta;

		c_half = circumference * 0.5;
		over = (dev_bunch[tid].z > c_half);
		under = (dev_bunch[tid].z < -c_half);
		dev_bunch[tid].z += (under - over) * circumference;
		//if (dev_bunch[tid].tag == 1)
		//{
		//	printf("dev_bunch[1] z0 = %f, z1 = %f, z2 = %f, circumference = %f\n",
		//		tau0 * beta, tau1 * beta, dev_bunch[tid].z, circumference);
		//}
		tid += stride;
	}
}


__global__ void transfer_quadrupole_skew(Particle* dev_bunch, int Np_sur, double beta, double circumference,
	double k1s, double l) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double x0 = 0, px0 = 0, y0 = 0, py0 = 0;
	double x1 = 0, px1 = 0, y1 = 0, py1 = 0;

	double tau0 = 0, tau1 = 0;	// tau = z/beta - ct(=0) = z/beta
	double pt0 = 0;	// pt = DeltaE/(P0*c) = beta*DeltaP/P0

	double k1s_chrom = 0, omega = 0;
	double cx = 0, sx = 0, chx = 0, shx = 0;
	double cp = 0, cm = 0, sp = 0, sm = 0;
	double r11 = 0, r12 = 0, r13 = 0, r14 = 0, r21 = 0, r22 = 0, r23 = 0, r24 = 0;
	double r31 = 0, r32 = 0, r33 = 0, r34 = 0, r41 = 0, r42 = 0, r43 = 0, r44 = 0;
	double r56 = 0;

	double c_half = 0;
	int over = 0, under = 0;

	while (tid < Np_sur) {

		x0 = dev_bunch[tid].x;
		px0 = dev_bunch[tid].px;
		y0 = dev_bunch[tid].y;
		py0 = dev_bunch[tid].py;
		tau0 = dev_bunch[tid].z / beta;
		pt0 = dev_bunch[tid].pz * beta;

		k1s_chrom = k1s / (1 + pt0 / beta);
		omega = sqrt(fabs(k1s_chrom));

		cx = cos(omega * l);
		sx = sin(omega * l);
		chx = cosh(omega * l);
		shx = sinh(omega * l);

		cp = (cx + chx) / 2;
		cm = (cx - chx) / 2;
		sp = (sx + shx) / 2;
		sm = (sx - shx) / 2;


		if (k1s_chrom > 0) {
			r11 = cp;
			r12 = sp / omega;
			r13 = cm;
			r14 = sm / omega;
			r21 = -omega * sm;
			r22 = cp;
			r23 = -omega * sp;
			r24 = cm;
			r31 = cm;
			r32 = sm / omega;
			r33 = cp;
			r34 = sp / omega;
			r41 = -omega * sp;
			r42 = cm;
			r43 = -omega * sm;
			r44 = cp;
			r56 = l * (1 / (beta * beta) - 1);
		}
		else {
			r11 = cp;
			r12 = sp / omega;
			r13 = -cm;
			r14 = -sm / omega;
			r21 = -omega * sm;
			r22 = cp;
			r23 = omega * sp;
			r24 = -cm;
			r31 = -cm;
			r32 = -sm / omega;
			r33 = cp;
			r34 = sp / omega;
			r41 = omega * sp;
			r42 = -cm;
			r43 = -omega * sm;
			r44 = cp;
			r56 = l * (1 / (beta * beta) - 1);
		}

		x1 = r11 * x0 + r12 * px0 + r13 * y0 + r14 * py0;
		px1 = r21 * x0 + r22 * px0 + r23 * y0 + r24 * py0;
		y1 = r31 * x0 + r32 * px0 + r33 * y0 + r34 * py0;
		py1 = r41 * x0 + r42 * px0 + r43 * y0 + r44 * py0;
		tau1 = tau0 + r56 * pt0;

		dev_bunch[tid].x = x1;
		dev_bunch[tid].px = px1;
		dev_bunch[tid].y = y1;
		dev_bunch[tid].py = py1;
		dev_bunch[tid].z = tau1 * beta;

		c_half = circumference * 0.5;
		over = (dev_bunch[tid].z > c_half);
		under = (dev_bunch[tid].z < -c_half);
		dev_bunch[tid].z += (under - over) * circumference;
		//if (dev_bunch[tid].tag == 1)
		//{
		//	printf("dev_bunch[1] z0 = %f, z1 = %f, z2 = %f, circumference = %f\n",
		//		tau0 * beta, tau1 * beta, dev_bunch[tid].z, circumference);
		//}
		tid += stride;
	}
}


__global__ void transfer_sextupole_norm(Particle* dev_bunch, int Np_sur, double beta,
	double k2, double l) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double x0 = 0, y0 = 0;
	double k2_chrom = 0;

	while (tid < Np_sur) {

		x0 = dev_bunch[tid].x;
		y0 = dev_bunch[tid].y;

		k2_chrom = k2 / (1 + dev_bunch[tid].pz);

		dev_bunch[tid].px += -0.5 * k2_chrom * l * (x0 * x0 - y0 * y0);
		dev_bunch[tid].py += k2_chrom * l * x0 * y0;

		tid += stride;
	}
}


__global__ void transfer_sextupole_skew(Particle* dev_bunch, int Np_sur, double beta,
	double k2s, double l) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double x0 = 0, y0 = 0;
	double k2s_chrom = 0;

	while (tid < Np_sur) {

		x0 = dev_bunch[tid].x;
		y0 = dev_bunch[tid].y;

		k2s_chrom = k2s / (1 + dev_bunch[tid].pz);

		dev_bunch[tid].px += k2s_chrom * l * x0 * y0;
		dev_bunch[tid].py += 0.5 * k2s_chrom * l * (x0 * x0 - y0 * y0);

		tid += stride;
	}
}


__global__ void transfer_octupole_norm(Particle* dev_bunch, int Np_sur, double beta,
	double k3, double l) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double x0 = 0, y0 = 0;
	double k3_chrom = 0;

	while (tid < Np_sur) {

		x0 = dev_bunch[tid].x;
		y0 = dev_bunch[tid].y;

		k3_chrom = k3 / (1 + dev_bunch[tid].pz);

		dev_bunch[tid].px += -1.0 / 6 * k3_chrom * l * (pow(x0, 3) - 3 * x0 * pow(y0, 2));
		dev_bunch[tid].py += 1.0 / 6 * k3_chrom * l * (3 * pow(x0, 2) * y0 - pow(y0, 3));

		tid += stride;
	}
}


__global__ void transfer_octupole_skew(Particle* dev_bunch, int Np_sur, double beta,
	double k3s, double l) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double x0 = 0, y0 = 0;
	double k3s_chrom = 0;

	while (tid < Np_sur) {

		x0 = dev_bunch[tid].x;
		y0 = dev_bunch[tid].y;

		k3s_chrom = k3s / (1 + dev_bunch[tid].pz);

		dev_bunch[tid].px += 1.0 / 6 * k3s_chrom * l * (3 * pow(x0, 2) * y0 - pow(y0, 3));
		dev_bunch[tid].py += 1.0 / 6 * k3s_chrom * l * (pow(x0, 3) - 3 * x0 * pow(y0, 2));

		tid += stride;
	}
}


__global__ void transfer_hkicker(Particle* dev_bunch, int Np_sur, double beta,
	double kick) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double kick_chrom = 0;

	while (tid < Np_sur) {

		kick_chrom = kick / (1 + dev_bunch[tid].pz);

		dev_bunch[tid].px += kick_chrom;

		tid += stride;
	}
}


__global__ void transfer_vkicker(Particle* dev_bunch, int Np_sur, double beta,
	double kick) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double kick_chrom = 0;

	while (tid < Np_sur) {

		kick_chrom = kick / (1 + dev_bunch[tid].pz);

		dev_bunch[tid].py += kick_chrom;

		tid += stride;
	}
}


__global__ void transfer_rf(Particle* dev_bunch, int Np_sur, int turn, double beta0, double beta1, double gamma0, double gamma1,
	RFData* dev_rf_data, size_t  pitch_rf, int Nrf, size_t Nturn_rf,
	double radius, double ratio, double dE_syn, double eta1, double E_total1) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double z0 = 0, pz0 = 0, theta0 = 0, dE0 = 0;
	double z1 = 0, pz1 = 0, theta1 = 0, dE1 = 0;

	double dE_non_syn = 0;

	double voltage = 0, harmonic = 0, phis = 0, phi_offset = 0;

	double pi = PassConstant::PI;
	double trans_scale = beta0 * gamma0 / (beta1 * gamma1);	// px0*beta0*gamma0 = px1*beta1*gamma1

	while (tid < Np_sur) {

		z0 = dev_bunch[tid].z;
		pz0 = dev_bunch[tid].pz;

		convert_z_dp_to_theta_dE(z0, pz0, theta0, dE0, radius, beta0);

		for (int i = 0; i < Nrf; i++)
		{
			RFData* dev_rf_data_row = (RFData*)((char*)dev_rf_data + i * pitch_rf);

			voltage = dev_rf_data_row[turn - 1].voltage;
			harmonic = dev_rf_data_row[turn - 1].harmonic;
			phis = dev_rf_data_row[turn - 1].phis;
			phi_offset = dev_rf_data_row[turn - 1].phi_offset;

			dE_non_syn += ratio * voltage * sin(phis - harmonic * theta0 + phi_offset);
		}

		dE1 = (beta1 / beta0) * (dE0 + dE_non_syn - dE_syn);
		theta1 = fmod((theta0 - 2 * pi * eta1 * dE1 / (E_total1 * beta1 * beta1) + pi), (2 * pi)) - pi;

		convert_theta_dE_to_z_dp(z1, pz1, theta1, dE1, radius, beta1);

		dev_bunch[tid].z = z1;
		dev_bunch[tid].pz = pz1;

		dev_bunch[tid].px *= trans_scale;
		dev_bunch[tid].py *= trans_scale;

		tid += stride;
	}
}


__device__ void convert_z_dp_to_theta_dE(double z, double dp, double& theta, double& dE, double radius, double beta) {
	theta = z / radius;
	dE = dp * beta * beta;
}


__device__ void convert_theta_dE_to_z_dp(double& z, double& dp, double theta, double dE, double radius, double beta) {
	z = theta * radius;
	dp = dE / (beta * beta);
}


std::vector<RFData> readRFDataFromCSV(const std::string& filename) {
	// Read RF data from file

	auto logger = spdlog::get("logger");

	std::vector<RFData> data;
	std::ifstream file(filename);

	if (!file.is_open()) {
		logger->error("[Read RF Data] Open file failed: {}", filename);
		std::exit(EXIT_FAILURE);
	}

	std::string line;

	// Skip the first line
	std::getline(file, line);

	while (std::getline(file, line)) {
		std::stringstream ss(line);
		std::string cell;
		std::vector<double> values;

		// split data
		while (std::getline(ss, cell, ',')) {
			try {
				values.push_back(std::stod(cell));
			}
			catch (const std::exception& e) {
				logger->error("[Read RF Data] Error parsing value in line: {}", line);
				values.clear();
				break;
			}
		}

		if (values.size() == 4) {
			data.push_back({ values[0], values[1], values[2], values[3] });
		}
		else {
			logger->error("[Read RF Data] Invalid data format in line: {}", line);
			std::exit(EXIT_FAILURE);
		}
	}

	return data;
}


__global__ void transfer_multipole_kicker(Particle* dev_bunch, int Np_sur, int order, const double* dev_kn, const double* dev_ks, double l) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	if (order < 0) return;

	const int max_order = 20;
	if (order > max_order)
		order = max_order;

	double x = 0, y = 0;
	double dpx = 0, dpy = 0;

	double real = 1.0, imag = 0.0;
	double real_temp = 0.0, imag_temp = 0.0;

	double inv_factorial = 0;
	constexpr double factorial_table[max_order + 1] = { 1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0,
		3628800.0, 39916800.0, 479001600.0, 6227020800.0, 87178291200.0, 1307674368000.0,20922789888000.0,
		355687428096000.0, 6402373705728000.0, 121645100408832000.0 ,2432902008176640000.0 };

	while (tid < Np_sur)
	{
		x = dev_bunch[tid].x;
		y = dev_bunch[tid].y;

		dpx = 0.0;
		dpy = 0.0;
		real = 1.0;
		imag = 0.0;

		if (order >= 0)
		{
			double kn0l = dev_kn[0] * l;
			double ks0l = dev_ks[0] * l;
			dpx += -kn0l * real;
			dpy += ks0l * real;
		}

		for (int iorder = 1; iorder <= order; iorder++)
		{
			real_temp = real * x - imag * y;
			imag_temp = real * y + imag * x;
			real = real_temp;
			imag = imag_temp;

			inv_factorial = 1.0 / factorial_table[iorder];

			double knl = dev_kn[iorder] * l;
			double ksl = dev_ks[iorder] * l;

			dpx += -knl * (inv_factorial * real) + ksl * (inv_factorial * imag);
			dpy += knl * (inv_factorial * imag) + ksl * (inv_factorial * real);
		}

		dev_bunch[tid].px += dpx;
		dev_bunch[tid].py += dpy;

		tid += stride;
	}
}