#include "element.h"
#include "constant.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <general.h>

#include <cuda_runtime.h>

namespace fs = std::filesystem;

MarkerElement::MarkerElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

	commandType = "MarkerElement";
	name = obj_name;
	dev_particle = Bunch.dev_particle;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	circumference = para.circumference;

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
	}
	catch (json::exception e)
	{
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}

}

void MarkerElement::execute(int turn) {

	// do nothing
}


DriftElement::DriftElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

	commandType = "DriftElement";
	name = obj_name;
	dev_particle = Bunch.dev_particle;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	circumference = para.circumference;

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		l = data.at("Sequence").at(obj_name).at("L (m)");
	}
	catch (json::exception e)
	{
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}

}

void DriftElement::execute(int turn) {

	double drift = l;

	if (drift < 1e-10) return;

	callCuda(cudaEventRecord(simTime.start, 0));
	float time_tmp = 0;

	//auto logger = spdlog::get("logger");
	//logger->debug("[Drift Element] run: " + name);

	int Np_sur = bunchRef.Np_sur;
	double gamma = bunchRef.gamma;
	double beta = bunchRef.beta;

	callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, drift));

	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.transferElement += time_tmp;

}


SBendElement::SBendElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

	commandType = "SBendElement";
	name = obj_name;
	dev_particle = Bunch.dev_particle;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	circumference = para.circumference;

	callCuda(cudaMalloc(&dev_knl, max_error_order * sizeof(double)));
	callCuda(cudaMalloc(&dev_ksl, max_error_order * sizeof(double)));

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		l = data.at("Sequence").at(obj_name).at("L (m)");
		angle = data.at("Sequence").at(obj_name).at("Angle (rad)");
		e1 = data.at("Sequence").at(obj_name).at("E1 (rad)");
		e2 = data.at("Sequence").at(obj_name).at("E2 (rad)");
		hgap = data.at("Sequence").at(obj_name).at("Hgap (m)");
		fint = data.at("Sequence").at(obj_name).at("Fint");
		fintx = data.at("Sequence").at(obj_name).at("Fintx");

		is_field_error = data.at("Sequence").at(obj_name).at("Is field error");
		std::vector<double>knl, ksl;
		if (is_field_error)
		{
			int size_knl = data.at("Sequence").at(obj_name).at("Field error KNL").size();
			int size_ksl = data.at("Sequence").at(obj_name).at("Field error KSL").size();

			for (int i = 0; i < size_knl; i++)
			{
				knl.push_back(data.at("Sequence").at(obj_name).at("Field error KNL")[i]);
			}
			for (int i = 0; i < size_ksl; i++)
			{
				ksl.push_back(data.at("Sequence").at(obj_name).at("Field error KSL")[i]);
			}

			cur_error_order = std::max(size_knl, size_ksl) - 1;
			if (cur_error_order > max_error_order)
			{
				spdlog::get("logger")->warn("[SBend Element] {}: current error_order = {}, since it exceeds the max_error_order = {}, the excess values will be ignored.",
					name, cur_error_order, max_error_order);
				cur_error_order = max_error_order;
			}

			knl.resize(cur_error_order + 1, 0.0);
			ksl.resize(cur_error_order + 1, 0.0);


			callCuda(cudaMemcpy(dev_knl, knl.data(), (cur_error_order + 1) * sizeof(double), cudaMemcpyHostToDevice));
			callCuda(cudaMemcpy(dev_ksl, ksl.data(), (cur_error_order + 1) * sizeof(double), cudaMemcpyHostToDevice));
		}

		if (data.at("Sequence").at(obj_name).contains("Is ramping"))
		{
			is_ramping = data.at("Sequence").at(obj_name).at("Is ramping");

			if (data.at("Sequence").at(obj_name).contains("K0L ramping file") && is_ramping)
			{
				std::string ramping_file_path = data.at("Sequence").at(obj_name).at("K0L ramping file");
				if (!ramping_file_path.empty() && ramping_file_path.find_first_not_of(" \t\r\n") != std::string::npos)
				{
					ramping_file_path.erase(0, ramping_file_path.find_first_not_of(" \t\r\n"));
					ramping_file_path.erase(ramping_file_path.find_last_not_of(" \t\r\n") + 1);
					fs::path fs_path(ramping_file_path);

					if (fs::is_regular_file(fs_path))
					{
						std::string fs_filename = fs_path.filename().string();
						fs::copy(ramping_file_path, para.dir_output_parameter / (para.hourMinSec + "_beam" + std::to_string(input_beamId) + "_" + name + "_" + fs_filename), std::filesystem::copy_options::overwrite_existing);

						std::vector<std::vector<double>> ramping_data = loadtxt(ramping_file_path, ',', 1, { 0,1 });
						ramping_k0l = ramping_data[1];
					}
					else
					{
						spdlog::get("logger")->error("[SBend Element] {}: path: {} is not regular file", name, ramping_file_path);
						std::exit(EXIT_FAILURE);
					}
				}
			}
		}

		spdlog::get("logger")->debug("[SBend Element] run: {}, s = {}, l = {}, angle = {}, e1 = {}, e2 = {}, hgap = {}, fint = {}, fintx = {}, is_field_error = {}",
			name, s, l, angle, e1, e2, hgap, fint, fintx, is_field_error);
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

	int Np_sur = bunchRef.Np_sur;
	double gamma = bunchRef.gamma;
	double beta = bunchRef.beta;

	double l_used = 0;
	double angle_used = 0;

	if (is_field_error)
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

	if (is_field_error)
	{
		callKernel(transfer_dipole_half_left << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference,
			r11, r12, r16, r21, r22, r26, r34, r51, r52, r56, fl21i, fl43i, fr21i, fr43i));

		callKernel(transfer_multipole_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, cur_error_order, dev_knl, dev_ksl));

		callKernel(transfer_dipole_half_right << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference,
			r11, r12, r16, r21, r22, r26, r34, r51, r52, r56, fl21i, fl43i, fr21i, fr43i));
	}
	else
	{
		callKernel(transfer_dipole_full << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference,
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
	dev_particle = Bunch.dev_particle;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	circumference = para.circumference;

	callCuda(cudaMalloc(&dev_knl, max_error_order * sizeof(double)));
	callCuda(cudaMalloc(&dev_ksl, max_error_order * sizeof(double)));

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		l = data.at("Sequence").at(obj_name).at("L (m)");
		angle = data.at("Sequence").at(obj_name).at("Angle (rad)");
		e1 = data.at("Sequence").at(obj_name).at("E1 (rad)");
		e2 = data.at("Sequence").at(obj_name).at("E2 (rad)");
		hgap = data.at("Sequence").at(obj_name).at("Hgap (m)");
		fint = data.at("Sequence").at(obj_name).at("Fint");
		fintx = data.at("Sequence").at(obj_name).at("Fintx");

		is_field_error = data.at("Sequence").at(obj_name).at("Is field error");
		std::vector<double>knl, ksl;
		if (is_field_error)
		{
			int size_knl = data.at("Sequence").at(obj_name).at("Field error KNL").size();
			int size_ksl = data.at("Sequence").at(obj_name).at("Field error KSL").size();

			for (int i = 0; i < size_knl; i++)
			{
				knl.push_back(data.at("Sequence").at(obj_name).at("Field error KNL")[i]);
			}
			for (int i = 0; i < size_ksl; i++)
			{
				ksl.push_back(data.at("Sequence").at(obj_name).at("Field error KSL")[i]);
			}

			cur_error_order = std::max(size_knl, size_ksl) - 1;
			if (cur_error_order > max_error_order)
			{
				spdlog::get("logger")->warn("[RBend Element] {}: current error_order = {}, since it exceeds the max_error_order = {}, the excess values will be ignored.",
					name, cur_error_order, max_error_order);
				cur_error_order = max_error_order;
			}

			knl.resize(cur_error_order + 1, 0.0);
			ksl.resize(cur_error_order + 1, 0.0);


			callCuda(cudaMemcpy(dev_knl, knl.data(), (cur_error_order + 1) * sizeof(double), cudaMemcpyHostToDevice));
			callCuda(cudaMemcpy(dev_ksl, ksl.data(), (cur_error_order + 1) * sizeof(double), cudaMemcpyHostToDevice));
		}

		if (data.at("Sequence").at(obj_name).contains("Is ramping"))
		{
			is_ramping = data.at("Sequence").at(obj_name).at("Is ramping");

			if (data.at("Sequence").at(obj_name).contains("K0L ramping file") && is_ramping)
			{
				std::string ramping_file_path = data.at("Sequence").at(obj_name).at("K0L ramping file");
				if (!ramping_file_path.empty() && ramping_file_path.find_first_not_of(" \t\r\n") != std::string::npos)
				{
					ramping_file_path.erase(0, ramping_file_path.find_first_not_of(" \t\r\n"));
					ramping_file_path.erase(ramping_file_path.find_last_not_of(" \t\r\n") + 1);
					fs::path fs_path(ramping_file_path);

					if (fs::is_regular_file(fs_path))
					{
						std::string fs_filename = fs_path.filename().string();
						fs::copy(ramping_file_path, para.dir_output_parameter / (para.hourMinSec + "_beam" + std::to_string(input_beamId) + "_" + name + "_" + fs_filename), std::filesystem::copy_options::overwrite_existing);

						std::vector<std::vector<double>> ramping_data = loadtxt(ramping_file_path, ',', 1, { 0,1 });
						ramping_k0l = ramping_data[1];
					}
					else
					{
						spdlog::get("logger")->error("[RBend Element] {}: path: {} is not regular file", name, ramping_file_path);
						std::exit(EXIT_FAILURE);
					}
				}
			}
		}

		spdlog::get("logger")->debug("[RBend Element] run: {}, s = {}, l = {}, angle = {}, e1 = {}, e2 = {}, hgap = {}, fint = {}, fintx = {}, is_field_error = {}",
			name, s, l, angle, e1, e2, hgap, fint, fintx, is_field_error);

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

	int Np_sur = bunchRef.Np_sur;
	double gamma = bunchRef.gamma;
	double beta = bunchRef.beta;

	double l_used = 0;
	double angle_used = 0;

	if (is_field_error)
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

	if (is_field_error)
	{
		callKernel(transfer_dipole_half_left << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference,
			r11, r12, r16, r21, r22, r26, r34, r51, r52, r56, fl21i, fl43i, fr21i, fr43i));

		callKernel(transfer_multipole_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, cur_error_order, dev_knl, dev_ksl));

		callKernel(transfer_dipole_half_right << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference,
			r11, r12, r16, r21, r22, r26, r34, r51, r52, r56, fl21i, fl43i, fr21i, fr43i));
	}
	else
	{
		callKernel(transfer_dipole_full << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference,
			r11, r12, r16, r21, r22, r26, r34, r51, r52, r56, fl21i, fl43i, fr21i, fr43i));
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
	dev_particle = Bunch.dev_particle;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	circumference = para.circumference;

	callCuda(cudaMalloc(&dev_knl, max_error_order * sizeof(double)));
	callCuda(cudaMalloc(&dev_ksl, max_error_order * sizeof(double)));

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		l = data.at("Sequence").at(obj_name).at("L (m)");
		k1l = data.at("Sequence").at(obj_name).at("K1L (m^-1)");
		k1sl = data.at("Sequence").at(obj_name).at("K1SL (m^-1)");

		is_field_error = data.at("Sequence").at(obj_name).at("Is field error");
		std::vector<double>knl, ksl;
		if (is_field_error)
		{
			int size_knl = data.at("Sequence").at(obj_name).at("Field error KNL").size();
			int size_ksl = data.at("Sequence").at(obj_name).at("Field error KSL").size();

			for (int i = 0; i < size_knl; i++)
			{
				knl.push_back(data.at("Sequence").at(obj_name).at("Field error KNL")[i]);
			}
			for (int i = 0; i < size_ksl; i++)
			{
				ksl.push_back(data.at("Sequence").at(obj_name).at("Field error KSL")[i]);
			}

			cur_error_order = std::max(size_knl, size_ksl) - 1;
			if (cur_error_order > max_error_order)
			{
				spdlog::get("logger")->warn("[Quadrupole Element] {}: current error_order = {}, since it exceeds the max_error_order = {}, the excess values will be ignored.",
					name, cur_error_order, max_error_order);
				cur_error_order = max_error_order;
			}

			knl.resize(cur_error_order + 1, 0.0);
			ksl.resize(cur_error_order + 1, 0.0);


			callCuda(cudaMemcpy(dev_knl, knl.data(), (cur_error_order + 1) * sizeof(double), cudaMemcpyHostToDevice));
			callCuda(cudaMemcpy(dev_ksl, ksl.data(), (cur_error_order + 1) * sizeof(double), cudaMemcpyHostToDevice));
		}

		if (data.at("Sequence").at(obj_name).contains("Is ramping"))
		{
			is_ramping = data.at("Sequence").at(obj_name).at("Is ramping");

			if (data.at("Sequence").at(obj_name).contains("K1L ramping file") && is_ramping)
			{
				std::string ramping_file_path = data.at("Sequence").at(obj_name).at("K1L ramping file");
				if (!ramping_file_path.empty() && ramping_file_path.find_first_not_of(" \t\r\n") != std::string::npos)
				{
					ramping_file_path.erase(0, ramping_file_path.find_first_not_of(" \t\r\n"));
					ramping_file_path.erase(ramping_file_path.find_last_not_of(" \t\r\n") + 1);
					fs::path fs_path(ramping_file_path);

					if (fs::is_regular_file(fs_path))
					{
						std::string fs_filename = fs_path.filename().string();
						fs::copy(ramping_file_path, para.dir_output_parameter / (para.hourMinSec + "_beam" + std::to_string(input_beamId) + "_" + name + "_" + fs_filename), std::filesystem::copy_options::overwrite_existing);

						std::vector<std::vector<double>> ramping_data = loadtxt(ramping_file_path, ',', 1, { 0,1 });
						ramping_k1l = ramping_data[1];
					}
					else
					{
						spdlog::get("logger")->error("[Quadrupole Element] {}: path: {} is not regular file", name, ramping_file_path);
						std::exit(EXIT_FAILURE);
					}
				}
			}

			if (data.at("Sequence").at(obj_name).contains("K1SL ramping file") && is_ramping)
			{
				std::string ramping_file_path = data.at("Sequence").at(obj_name).at("K1SL ramping file");
				if (!ramping_file_path.empty() && ramping_file_path.find_first_not_of(" \t\r\n") != std::string::npos)
				{
					ramping_file_path.erase(0, ramping_file_path.find_first_not_of(" \t\r\n"));
					ramping_file_path.erase(ramping_file_path.find_last_not_of(" \t\r\n") + 1);
					fs::path fs_path(ramping_file_path);

					if (fs::is_regular_file(fs_path))
					{
						std::string fs_filename = fs_path.filename().string();
						fs::copy(ramping_file_path, para.dir_output_parameter / (para.hourMinSec + "_beam" + std::to_string(input_beamId) + "_" + name + "_" + fs_filename), std::filesystem::copy_options::overwrite_existing);

						std::vector<std::vector<double>> ramping_data = loadtxt(ramping_file_path, ',', 1, { 0,1 });
						ramping_k1sl = ramping_data[1];
					}
					else
					{
						spdlog::get("logger")->error("[Quadrupole Element] {}: path: {} is not regular file", name, ramping_file_path);
						std::exit(EXIT_FAILURE);
					}
				}
			}
		}

		if (is_ramping)
		{
			std::cout << __FILE__ << ": " << __LINE__ << std::endl;
			if (ramping_k1l.size() > 0 && ramping_k1sl.size() == 0)
			{
				quad_type = "normal";
			}
			else if (ramping_k1l.size() == 0 && ramping_k1sl.size() > 0)
			{
				quad_type = "skew";
			}
			else if (ramping_k1l.size() == 0 && ramping_k1sl.size() == 0)
			{
				quad_type = "drift";
			}
			else
			{
				spdlog::get("logger")->error("[Quadrupole Element] {}: both the normal and skew files have been provided. But only 1 file is needed", name);
				std::exit(EXIT_FAILURE);
			}
		}
		else
		{
			std::cout << __FILE__ << ": " << __LINE__ << std::endl;
			if (fabs(k1l) > 1e-10 && fabs(k1sl) < 1e-10)
			{
				quad_type = "normal";
			}
			else if (fabs(k1l) < 1e-10 && fabs(k1sl) > 1e-10)
			{
				quad_type = "skew";
			}
			else if (fabs(k1l) < 1e-10 && fabs(k1sl) < 1e-10)
			{
				quad_type = "drift";
			}
			else
			{
				spdlog::get("logger")->error("[Quadrupole Element] {}: k1l = {}, k1sl = {}, there should be and only 1 variable not equal to 0",
					name, k1l, k1sl);
				std::exit(EXIT_FAILURE);
			}
		}
		std::cout << __FILE__ << ": " << __LINE__ << std::endl;
		spdlog::get("logger")->debug("[Quadrupole Element] run: {}, s = {}, l = {}, quad_type = {}, k1l = {}, k1sl = {}, is_field_error = {}, isRamping = {}",
			name, s, l, quad_type, k1l, k1sl, is_field_error, is_ramping);
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

	int Np_sur = bunchRef.Np_sur;
	double gamma = bunchRef.gamma;
	double beta = bunchRef.beta;

	double EPSILON = 1e-10;

	if (quad_type == "normal")
	{
		double k1l_current = 0;

		if (is_ramping)
		{
			if (turn > ramping_k1l.size())
			{
				spdlog::get("logger")->warn("[Quadrupole Element] {}, ramping data size ({}) is less than turn ({}), so k1l is set to the last value ({})", name, ramping_k1l.size(), turn, ramping_k1l.back());
				k1l_current = ramping_k1l.back();
			}
			else
			{
				k1l_current = ramping_k1l[turn - 1];
			}
		}
		else
		{
			k1l_current = k1l;
		}

		if (fabs(l) > EPSILON)
		{
			if (is_field_error)
			{
				callKernel(transfer_quadrupole_thicklens_norm << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, k1l_current / l, l / 2));
				callKernel(transfer_multipole_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, cur_error_order, dev_knl, dev_ksl));
				callKernel(transfer_quadrupole_thicklens_norm << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, k1l_current / l, l / 2));
			}
			else
			{
				callKernel(transfer_quadrupole_thicklens_norm << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, k1l_current / l, l));
			}
		}
		else
		{
			if (is_field_error)
			{
				callKernel(transfer_multipole_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, cur_error_order, dev_knl, dev_ksl));
			}
			callKernel(transfer_quadrupole_thinlens_norm << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, k1l_current));
		}
	}

	else if (quad_type == "skew")
	{
		double k1sl_current = 0;

		if (is_ramping)
		{
			if (turn > ramping_k1sl.size())
			{
				spdlog::get("logger")->warn("[Quadrupole Element] {}, ramping data size ({}) is less than turn ({}), so k1sl is set to the last value ({})", name, ramping_k1sl.size(), turn, ramping_k1sl.back());
				k1sl_current = ramping_k1sl.back();
			}
			else
			{
				k1sl_current = ramping_k1sl[turn - 1];
			}
		}
		else
		{
			k1sl_current = k1sl;
		}

		if (fabs(l) > EPSILON)
		{
			if (is_field_error)
			{
				callKernel(transfer_quadrupole_thicklens_skew << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, k1sl_current / l, l / 2));
				callKernel(transfer_multipole_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, cur_error_order, dev_knl, dev_ksl));
				callKernel(transfer_quadrupole_thicklens_skew << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, k1sl_current / l, l / 2));
			}
			else
			{
				callKernel(transfer_quadrupole_thicklens_skew << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, k1sl_current / l, l));
			}
		}
		else
		{
			if (is_field_error)
			{
				callKernel(transfer_multipole_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, cur_error_order, dev_knl, dev_ksl));
			}
			callKernel(transfer_quadrupole_thinlens_skew << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, k1sl_current));
		}
	}

	else if (quad_type == "drift")
	{
		if (fabs(l) > EPSILON)
		{
			if (is_field_error)
			{
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
				callKernel(transfer_multipole_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, cur_error_order, dev_knl, dev_ksl));
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
			}
			else
			{
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l));
			}
		}
		else
		{
			;
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
	dev_particle = Bunch.dev_particle;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	circumference = para.circumference;

	callCuda(cudaMalloc(&dev_knl, max_error_order * sizeof(double)));
	callCuda(cudaMalloc(&dev_ksl, max_error_order * sizeof(double)));

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		l = data.at("Sequence").at(obj_name).at("L (m)");
		k2l = data.at("Sequence").at(obj_name).at("K2L (m^-2)");
		k2sl = data.at("Sequence").at(obj_name).at("K2SL (m^-2)");

		is_field_error = data.at("Sequence").at(obj_name).at("Is field error");
		std::vector<double>knl, ksl;
		if (is_field_error)
		{
			int size_knl = data.at("Sequence").at(obj_name).at("Field error KNL").size();
			int size_ksl = data.at("Sequence").at(obj_name).at("Field error KSL").size();

			for (int i = 0; i < size_knl; i++)
			{
				knl.push_back(data.at("Sequence").at(obj_name).at("Field error KNL")[i]);
			}
			for (int i = 0; i < size_ksl; i++)
			{
				ksl.push_back(data.at("Sequence").at(obj_name).at("Field error KSL")[i]);
			}

			cur_error_order = std::max(size_knl, size_ksl) - 1;
			if (cur_error_order > max_error_order)
			{
				spdlog::get("logger")->warn("[Sextupole Element] {}: current error_order = {}, since it exceeds the max_error_order = {}, the excess values will be ignored.",
					name, cur_error_order, max_error_order);
				cur_error_order = max_error_order;
			}

			knl.resize(cur_error_order + 1, 0.0);
			ksl.resize(cur_error_order + 1, 0.0);


			callCuda(cudaMemcpy(dev_knl, knl.data(), (cur_error_order + 1) * sizeof(double), cudaMemcpyHostToDevice));
			callCuda(cudaMemcpy(dev_ksl, ksl.data(), (cur_error_order + 1) * sizeof(double), cudaMemcpyHostToDevice));
		}

		if (data.at("Sequence").at(obj_name).contains("Is ramping"))
		{
			is_ramping = data.at("Sequence").at(obj_name).at("Is ramping");

			if (data.at("Sequence").at(obj_name).contains("K2L ramping file") && is_ramping)
			{
				std::string ramping_file_path = data.at("Sequence").at(obj_name).at("K2L ramping file");
				if (!ramping_file_path.empty() && ramping_file_path.find_first_not_of(" \t\r\n") != std::string::npos)
				{
					ramping_file_path.erase(0, ramping_file_path.find_first_not_of(" \t\r\n"));
					ramping_file_path.erase(ramping_file_path.find_last_not_of(" \t\r\n") + 1);
					fs::path fs_path(ramping_file_path);

					if (fs::is_regular_file(fs_path))
					{
						std::string fs_filename = fs_path.filename().string();
						fs::copy(ramping_file_path, para.dir_output_parameter / (para.hourMinSec + "_beam" + std::to_string(input_beamId) + "_" + name + "_" + fs_filename), std::filesystem::copy_options::overwrite_existing);

						std::vector<std::vector<double>> ramping_data = loadtxt(ramping_file_path, ',', 1, { 0,1 });
						ramping_k2l = ramping_data[1];
					}
					else
					{
						spdlog::get("logger")->error("[Sextupole Element] {}: path: {} is not regular file", name, ramping_file_path);
						std::exit(EXIT_FAILURE);
					}
				}
			}

			if (data.at("Sequence").at(obj_name).contains("K2SL ramping file") && is_ramping)
			{
				std::string ramping_file_path = data.at("Sequence").at(obj_name).at("K2SL ramping file");
				if (!ramping_file_path.empty() && ramping_file_path.find_first_not_of(" \t\r\n") != std::string::npos)
				{
					ramping_file_path.erase(0, ramping_file_path.find_first_not_of(" \t\r\n"));
					ramping_file_path.erase(ramping_file_path.find_last_not_of(" \t\r\n") + 1);
					fs::path fs_path(ramping_file_path);

					if (fs::is_regular_file(fs_path))
					{
						std::string fs_filename = fs_path.filename().string();
						fs::copy(ramping_file_path, para.dir_output_parameter / (para.hourMinSec + "_beam" + std::to_string(input_beamId) + "_" + name + "_" + fs_filename), std::filesystem::copy_options::overwrite_existing);

						std::vector<std::vector<double>> ramping_data = loadtxt(ramping_file_path, ',', 1, { 0,1 });
						ramping_k2sl = ramping_data[1];
					}
					else
					{
						spdlog::get("logger")->error("[Sextupole Element] {}: path: {} is not regular file", name, ramping_file_path);
						std::exit(EXIT_FAILURE);
					}
				}
			}
		}

		if (is_ramping)
		{
			if (ramping_k2l.size() > 0 && ramping_k2sl.size() == 0)
			{
				sext_type = "normal";
			}
			else if (ramping_k2l.size() == 0 && ramping_k2sl.size() > 0)
			{
				sext_type = "skew";
			}
			else if (ramping_k2l.size() == 0 && ramping_k2sl.size() == 0)
			{
				sext_type = "drift";
			}
			else
			{
				spdlog::get("logger")->error("[Sextupole Element] {}: both the normal and skew files have been provided. But only 1 file is needed", name);
				std::exit(EXIT_FAILURE);
			}
		}
		else
		{
			if (fabs(k2l) > 1e-10 && fabs(k2sl) < 1e-10)
			{
				sext_type = "normal";
			}
			else if (fabs(k2l) < 1e-10 && fabs(k2sl) > 1e-10)
			{
				sext_type = "skew";
			}
			else if (fabs(k2l) < 1e-10 && fabs(k2sl) < 1e-10)
			{
				sext_type = "drift";
			}
			else
			{
				spdlog::get("logger")->error("[Sextupole Element] {}: k2l = {}, k2sl = {}, there should be and only 1 variable not equal to 0",
					name, k2l, k2sl);
				std::exit(EXIT_FAILURE);
			}
		}

		spdlog::get("logger")->debug("[Sextupole Element] run: {}, s = {}, l = {}, sext_type = {}, k2l = {}, k2sl = {}, is_field_error = {}, isRamping = {}",
			name, s, l, sext_type, k2l, k2sl, is_field_error, is_ramping);
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

	int Np_sur = bunchRef.Np_sur;
	double gamma = bunchRef.gamma;
	double beta = bunchRef.beta;

	double EPSILON = 1e-10;

	if (sext_type == "normal")
	{
		double k2l_current = 0;

		if (is_ramping)
		{
			if (turn > ramping_k2l.size())
			{
				spdlog::get("logger")->warn("[Sextupole Element] {}, ramping data size ({}) is less than turn ({}), so k2l is set to the last value ({})", name, ramping_k2l.size(), turn, ramping_k2l.back());
				k2l_current = ramping_k2l.back();
			}
			else
			{
				k2l_current = ramping_k2l[turn - 1];
			}
		}
		else
		{
			k2l_current = k2l;
		}

		if (fabs(l) > EPSILON)
		{
			if (is_field_error)
			{
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
				callKernel(transfer_sextupole_thinlens_norm << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, k2l_current));
				callKernel(transfer_multipole_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, cur_error_order, dev_knl, dev_ksl));
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
			}
			else
			{
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
				callKernel(transfer_sextupole_thinlens_norm << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, k2l_current));
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
			}
		}
		else
		{
			callKernel(transfer_sextupole_thinlens_norm << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, k2l_current));

			if (is_field_error)
			{
				callKernel(transfer_multipole_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, cur_error_order, dev_knl, dev_ksl));
			}
		}
	}

	else if (sext_type == "skew")
	{
		double k2sl_current = 0;

		if (is_ramping)
		{
			if (turn > ramping_k2sl.size())
			{
				spdlog::get("logger")->warn("[Sextupole Element] {}, ramping data size ({}) is less than turn ({}), so k2sl is set to the last value ({})", name, ramping_k2sl.size(), turn, ramping_k2sl.back());
				k2sl_current = ramping_k2sl.back();
			}
			else
			{
				k2sl_current = ramping_k2sl[turn - 1];
			}
		}
		else
		{
			k2sl_current = k2sl;
		}

		if (fabs(l) > EPSILON)
		{
			if (is_field_error)
			{
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
				callKernel(transfer_sextupole_thinlens_skew << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, k2sl_current));
				callKernel(transfer_multipole_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, cur_error_order, dev_knl, dev_ksl));
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
			}
			else
			{
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
				callKernel(transfer_sextupole_thinlens_skew << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, k2sl_current));
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
			}
		}
		else
		{
			callKernel(transfer_sextupole_thinlens_skew << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, k2sl_current));

			if (is_field_error)
			{
				callKernel(transfer_multipole_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, cur_error_order, dev_knl, dev_ksl));
			}
		}
	}

	else if (sext_type == "drift")
	{
		if (fabs(l) > EPSILON)
		{
			if (is_field_error)
			{
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
				callKernel(transfer_multipole_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, cur_error_order, dev_knl, dev_ksl));
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
			}
			else
			{
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l));
			}
		}
		else
		{
			;
		}
	}

	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.transferElement += time_tmp;

}


OctupoleElement::OctupoleElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

	commandType = "OctupoleElement";
	name = obj_name;
	dev_particle = Bunch.dev_particle;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	circumference = para.circumference;

	callCuda(cudaMalloc(&dev_knl, max_error_order * sizeof(double)));
	callCuda(cudaMalloc(&dev_ksl, max_error_order * sizeof(double)));

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		l = data.at("Sequence").at(obj_name).at("L (m)");
		k3l = data.at("Sequence").at(obj_name).at("K3L (m^-3)");
		k3sl = data.at("Sequence").at(obj_name).at("K3SL (m^-3)");

		is_field_error = data.at("Sequence").at(obj_name).at("Is field error");
		std::vector<double>knl, ksl;
		if (is_field_error)
		{
			int size_knl = data.at("Sequence").at(obj_name).at("Field error KNL").size();
			int size_ksl = data.at("Sequence").at(obj_name).at("Field error KSL").size();

			for (int i = 0; i < size_knl; i++)
			{
				knl.push_back(data.at("Sequence").at(obj_name).at("Field error KNL")[i]);
			}
			for (int i = 0; i < size_ksl; i++)
			{
				ksl.push_back(data.at("Sequence").at(obj_name).at("Field error KSL")[i]);
			}

			cur_error_order = std::max(size_knl, size_ksl) - 1;
			if (cur_error_order > max_error_order)
			{
				spdlog::get("logger")->warn("[Octupole Element] {}: current error_order = {}, since it exceeds the max_error_order = {}, the excess values will be ignored.",
					name, cur_error_order, max_error_order);
				cur_error_order = max_error_order;
			}

			knl.resize(cur_error_order + 1, 0.0);
			ksl.resize(cur_error_order + 1, 0.0);

			callCuda(cudaMemcpy(dev_knl, knl.data(), (cur_error_order + 1) * sizeof(double), cudaMemcpyHostToDevice));
			callCuda(cudaMemcpy(dev_ksl, ksl.data(), (cur_error_order + 1) * sizeof(double), cudaMemcpyHostToDevice));
		}

		if (data.at("Sequence").at(obj_name).contains("Is ramping"))
		{
			is_ramping = data.at("Sequence").at(obj_name).at("Is ramping");

			if (data.at("Sequence").at(obj_name).contains("K3L ramping file") && is_ramping)
			{
				std::string ramping_file_path = data.at("Sequence").at(obj_name).at("K3L ramping file");
				if (!ramping_file_path.empty() && ramping_file_path.find_first_not_of(" \t\r\n") != std::string::npos)
				{
					ramping_file_path.erase(0, ramping_file_path.find_first_not_of(" \t\r\n"));
					ramping_file_path.erase(ramping_file_path.find_last_not_of(" \t\r\n") + 1);
					fs::path fs_path(ramping_file_path);

					if (fs::is_regular_file(fs_path))
					{
						std::string fs_filename = fs_path.filename().string();
						fs::copy(ramping_file_path, para.dir_output_parameter / (para.hourMinSec + "_beam" + std::to_string(input_beamId) + "_" + name + "_" + fs_filename), std::filesystem::copy_options::overwrite_existing);

						std::vector<std::vector<double>> ramping_data = loadtxt(ramping_file_path, ',', 1, { 0,1 });
						ramping_k3l = ramping_data[1];
					}
					else
					{
						spdlog::get("logger")->error("[Octupole Element] {}: path: {} is not regular file", name, ramping_file_path);
						std::exit(EXIT_FAILURE);
					}
				}
			}

			if (data.at("Sequence").at(obj_name).contains("K3SL ramping file") && is_ramping)
			{
				std::string ramping_file_path = data.at("Sequence").at(obj_name).at("K3SL ramping file");
				if (!ramping_file_path.empty() && ramping_file_path.find_first_not_of(" \t\r\n") != std::string::npos)
				{
					ramping_file_path.erase(0, ramping_file_path.find_first_not_of(" \t\r\n"));
					ramping_file_path.erase(ramping_file_path.find_last_not_of(" \t\r\n") + 1);
					fs::path fs_path(ramping_file_path);

					if (fs::is_regular_file(fs_path))
					{
						std::string fs_filename = fs_path.filename().string();
						fs::copy(ramping_file_path, para.dir_output_parameter / (para.hourMinSec + "_beam" + std::to_string(input_beamId) + "_" + name + "_" + fs_filename), std::filesystem::copy_options::overwrite_existing);

						std::vector<std::vector<double>> ramping_data = loadtxt(ramping_file_path, ',', 1, { 0,1 });
						ramping_k3sl = ramping_data[1];
					}
					else
					{
						spdlog::get("logger")->error("[Octupole Element] {}: path: {} is not regular file", name, ramping_file_path);
						std::exit(EXIT_FAILURE);
					}
				}
			}
		}

		if (is_ramping)
		{
			if (ramping_k3l.size() > 0 && ramping_k3sl.size() == 0)
			{
				oct_type = "normal";
			}
			else if (ramping_k3l.size() == 0 && ramping_k3sl.size() > 0)
			{
				oct_type = "skew";
			}
			else if (ramping_k3l.size() == 0 && ramping_k3sl.size() == 0)
			{
				oct_type = "drift";
			}
			else
			{
				spdlog::get("logger")->error("[Octupole Element] {}: both the normal and skew files have been provided. But only 1 file is needed", name);
				std::exit(EXIT_FAILURE);
			}
		}
		else
		{
			if (fabs(k3l) > 1e-10 && fabs(k3sl) < 1e-10)
			{
				oct_type = "normal";
			}
			else if (fabs(k3l) < 1e-10 && fabs(k3sl) > 1e-10)
			{
				oct_type = "skew";
			}
			else if (fabs(k3l) < 1e-10 && fabs(k3sl) < 1e-10)
			{
				oct_type = "drift";
			}
			else
			{
				spdlog::get("logger")->error("[Octupole Element] {}: k3l = {}, k3sl = {}, there should be and only 1 variable not equal to 0",
					name, k3l, k3sl);
				std::exit(EXIT_FAILURE);
			}
		}

		spdlog::get("logger")->debug("[Octupole Element] run: {}, s = {}, l = {}, oct_type = {}, k3l = {}, k3sl = {}, is_field_error = {}, isRamping = {}",
			name, s, l, oct_type, k3l, k3sl, is_field_error, is_ramping);
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

	int Np_sur = bunchRef.Np_sur;
	double gamma = bunchRef.gamma;
	double beta = bunchRef.beta;

	double EPSILON = 1e-10;

	if (oct_type == "normal")
	{
		double k3l_current = 0;

		if (is_ramping)
		{
			if (turn > ramping_k3l.size())
			{
				spdlog::get("logger")->warn("[Octupole Element] {}, ramping data size ({}) is less than turn ({}), so k2l is set to the last value ({})", name, ramping_k3l.size(), turn, ramping_k3l.back());
				k3l_current = ramping_k3l.back();
			}
			else
			{
				k3l_current = ramping_k3l[turn - 1];
			}
		}
		else
		{
			k3l_current = k3l;
		}

		if (fabs(l) > EPSILON)
		{
			if (is_field_error)
			{
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
				callKernel(transfer_octupole_thinlens_norm << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, k3l_current));
				callKernel(transfer_multipole_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, cur_error_order, dev_knl, dev_ksl));
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
			}
			else
			{
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
				callKernel(transfer_octupole_thinlens_norm << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, k3l_current));
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
			}
		}
		else
		{
			callKernel(transfer_octupole_thinlens_norm << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, k3l_current));

			if (is_field_error)
			{
				callKernel(transfer_multipole_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, cur_error_order, dev_knl, dev_ksl));
			}
		}
	}

	else if (oct_type == "skew")
	{
		double k3sl_current = 0;

		if (is_ramping)
		{
			if (turn > ramping_k3sl.size())
			{
				spdlog::get("logger")->warn("[Octupole Element] {}, ramping data size ({}) is less than turn ({}), so k2l is set to the last value ({})", name, ramping_k3sl.size(), turn, ramping_k3sl.back());
				k3sl_current = ramping_k3sl.back();
			}
			else
			{
				k3sl_current = ramping_k3sl[turn - 1];
			}
		}
		else
		{
			k3sl_current = k3sl;
		}

		if (fabs(l) > EPSILON)
		{
			if (is_field_error)
			{
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
				callKernel(transfer_octupole_thinlens_norm << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, k3sl_current));
				callKernel(transfer_multipole_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, cur_error_order, dev_knl, dev_ksl));
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
			}
			else
			{
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
				callKernel(transfer_octupole_thinlens_skew << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, k3sl_current));
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
			}
		}
		else
		{
			callKernel(transfer_octupole_thinlens_norm << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, k3sl_current));

			if (is_field_error)
			{
				callKernel(transfer_multipole_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, cur_error_order, dev_knl, dev_ksl));
			}
		}
	}

	else if (oct_type == "drift")
	{
		if (fabs(l) > EPSILON)
		{
			if (is_field_error)
			{
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
				callKernel(transfer_multipole_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, cur_error_order, dev_knl, dev_ksl));
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
			}
			else
			{
				callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l));
			}
		}
		else
		{
			;
		}
	}

	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.transferElement += time_tmp;

}


MultipoleElement::MultipoleElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

	commandType = "MultipoleElement";
	name = obj_name;
	dev_particle = Bunch.dev_particle;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	circumference = para.circumference;

	callCuda(cudaMalloc(&dev_knl, max_order * sizeof(double)));
	callCuda(cudaMalloc(&dev_ksl, max_order * sizeof(double)));

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		l = data.at("Sequence").at(obj_name).at("L (m)");

		std::vector<double>knl, ksl;

		int size_knl = data.at("Sequence").at(obj_name).at("KiL").size();
		int size_ksl = data.at("Sequence").at(obj_name).at("KiSL").size();

		for (int i = 0; i < size_knl; i++)
		{
			knl.push_back(data.at("Sequence").at(obj_name).at("KiL")[i]);
		}
		for (int i = 0; i < size_ksl; i++)
		{
			ksl.push_back(data.at("Sequence").at(obj_name).at("KiSL")[i]);
		}

		cur_order = std::max(size_knl, size_ksl) - 1;
		if (cur_order > max_order)
		{
			spdlog::get("logger")->warn("[Multipole Element] {}: current order = {}, since it exceeds the max_order = {}, the excess values will be ignored.",
				name, cur_order, max_order);
			cur_order = max_order;
		}

		knl.resize(cur_order + 1, 0.0);
		ksl.resize(cur_order + 1, 0.0);

		callCuda(cudaMemcpy(dev_knl, knl.data(), (cur_order + 1) * sizeof(double), cudaMemcpyHostToDevice));
		callCuda(cudaMemcpy(dev_ksl, ksl.data(), (cur_order + 1) * sizeof(double), cudaMemcpyHostToDevice));


		if (data.at("Sequence").at(obj_name).contains("Is ramping"))
		{
			is_ramping = data.at("Sequence").at(obj_name).at("Is ramping");

			if (data.at("Sequence").at(obj_name).contains("KL ramping file") && is_ramping)
			{
				std::string ramping_file_path = data.at("Sequence").at(obj_name).at("KL ramping file");
				if (!ramping_file_path.empty() && ramping_file_path.find_first_not_of(" \t\r\n") != std::string::npos)
				{
					ramping_file_path.erase(0, ramping_file_path.find_first_not_of(" \t\r\n"));
					ramping_file_path.erase(ramping_file_path.find_last_not_of(" \t\r\n") + 1);
					fs::path fs_path(ramping_file_path);

					if (fs::is_regular_file(fs_path))
					{
						std::string fs_filename = fs_path.filename().string();
						fs::copy(ramping_file_path, para.dir_output_parameter / (para.hourMinSec + "_beam" + std::to_string(input_beamId) + "_" + name + "_" + fs_filename), std::filesystem::copy_options::overwrite_existing);

						std::vector<std::vector<double>> ramping_data = loadtxt(ramping_file_path, ',', 1);

						cur_order = (ramping_data.size() - 1) / 2 - 1;
						if (cur_order > max_order)
						{
							spdlog::get("logger")->warn("[Multipole Element] {}: current order = {}, since it exceeds the max_order = {}, the excess values will be ignored.",
								name, cur_order, max_order);
							cur_order = max_order;
						}

						for (size_t icolumn = 1; icolumn < ramping_data.size(); icolumn += 2)	// icolumn = 0 corresponding to turn
						{
							ramping_knl.push_back(ramping_data[icolumn]);
						}
						for (size_t icolumn = 2; icolumn < ramping_data.size(); icolumn += 2)
						{
							ramping_ksl.push_back(ramping_data[icolumn]);
						}
					}
					else
					{
						spdlog::get("logger")->error("[Multipole Element] {}: path: {} is not regular file", name, ramping_file_path);
						std::exit(EXIT_FAILURE);
					}
				}
			}
		}

		spdlog::get("logger")->debug("[Multipole Element] run: {}, s = {}, l = {}, isRamping = {}",
			name, s, l, is_ramping);

	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}
}


void MultipoleElement::execute(int turn) {
	callCuda(cudaEventRecord(simTime.start, 0));
	float time_tmp = 0;

	//auto logger = spdlog::get("logger");
	//logger->debug("[Multipole Element] run: " + name);

	int Np_sur = bunchRef.Np_sur;
	double gamma = bunchRef.gamma;
	double beta = bunchRef.beta;

	if (is_ramping)
	{
		std::vector<double>knl(cur_order + 1);
		std::vector<double>ksl(cur_order + 1);

		if (turn > ramping_knl[0].size())
		{
			spdlog::get("logger")->warn("[Multipole Element] {}, ramping data size ({}) is less than turn ({}), so kl is set to the last value", name, ramping_knl[0].size(), turn);

			for (size_t i = 0; i < ramping_knl.size(); i++)
			{
				knl.push_back(ramping_knl[i].back());
			}
			for (size_t i = 0; i < ramping_ksl.size(); i++)
			{
				ksl.push_back(ramping_ksl[i].back());
			}
		}
		else
		{
			for (size_t i = 0; i < ramping_knl.size(); i++)
			{
				knl.push_back(ramping_knl[turn - 1].back());
			}
			for (size_t i = 0; i < ramping_ksl.size(); i++)
			{
				ksl.push_back(ramping_ksl[turn - 1].back());
			}
		}

		callCuda(cudaMemcpy(dev_knl, knl.data(), (cur_order + 1) * sizeof(double), cudaMemcpyHostToDevice));
		callCuda(cudaMemcpy(dev_ksl, ksl.data(), (cur_order + 1) * sizeof(double), cudaMemcpyHostToDevice));
	}
	else
	{
		// use original dev_knl and dev_ksl
	}


	if (fabs(l) > 1e-10)
	{
		callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
		callKernel(transfer_multipole_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, cur_order, dev_knl, dev_ksl));
		callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
	}
	else
	{
		callKernel(transfer_multipole_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, cur_order, dev_knl, dev_ksl));
	}


	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.transferElement += time_tmp;

}


KickerElement::KickerElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

	commandType = "KickerElement";
	name = obj_name;
	dev_particle = Bunch.dev_particle;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	circumference = para.circumference;

	callCuda(cudaMalloc(&dev_knl, max_error_order * sizeof(double)));
	callCuda(cudaMalloc(&dev_ksl, max_error_order * sizeof(double)));

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		l = data.at("Sequence").at(obj_name).at("L (m)");
		hkick = data.at("Sequence").at(obj_name).at("Hkick (rad)");
		vkick = data.at("Sequence").at(obj_name).at("Vkick (rad)");

		is_field_error = data.at("Sequence").at(obj_name).at("Is field error");
		std::vector<double>knl, ksl;
		if (is_field_error)
		{
			int size_knl = data.at("Sequence").at(obj_name).at("Field error KNL").size();
			int size_ksl = data.at("Sequence").at(obj_name).at("Field error KSL").size();

			for (int i = 0; i < size_knl; i++)
			{
				knl.push_back(data.at("Sequence").at(obj_name).at("Field error KNL")[i]);
			}
			for (int i = 0; i < size_ksl; i++)
			{
				ksl.push_back(data.at("Sequence").at(obj_name).at("Field error KSL")[i]);
			}

			cur_error_order = std::max(size_knl, size_ksl) - 1;
			if (cur_error_order > max_error_order)
			{
				spdlog::get("logger")->warn("[Kicker Element] {}: current error_order = {}, since it exceeds the max_error_order = {}, the excess values will be ignored.",
					name, cur_error_order, max_error_order);
				cur_error_order = max_error_order;
			}

			knl.resize(cur_error_order + 1, 0.0);
			ksl.resize(cur_error_order + 1, 0.0);

			callCuda(cudaMemcpy(dev_knl, knl.data(), (cur_error_order + 1) * sizeof(double), cudaMemcpyHostToDevice));
			callCuda(cudaMemcpy(dev_ksl, ksl.data(), (cur_error_order + 1) * sizeof(double), cudaMemcpyHostToDevice));
		}

		if (data.at("Sequence").at(obj_name).contains("Is ramping"))
		{
			is_ramping = data.at("Sequence").at(obj_name).at("Is ramping");

			if (data.at("Sequence").at(obj_name).contains("Kick ramping file") && is_ramping)
			{
				std::string ramping_file_path = data.at("Sequence").at(obj_name).at("Kick ramping file");
				if (!ramping_file_path.empty() && ramping_file_path.find_first_not_of(" \t\r\n") != std::string::npos)
				{
					ramping_file_path.erase(0, ramping_file_path.find_first_not_of(" \t\r\n"));
					ramping_file_path.erase(ramping_file_path.find_last_not_of(" \t\r\n") + 1);
					fs::path fs_path(ramping_file_path);

					if (fs::is_regular_file(fs_path))
					{
						std::string fs_filename = fs_path.filename().string();
						fs::copy(ramping_file_path, para.dir_output_parameter / (para.hourMinSec + "_beam" + std::to_string(input_beamId) + "_" + name + "_" + fs_filename), std::filesystem::copy_options::overwrite_existing);

						std::vector<std::vector<double>> ramping_data = loadtxt(ramping_file_path, ',', 1, { 0, 1, 2 });
						ramping_hkick = ramping_data[1];
						ramping_vkick = ramping_data[2];
					}
					else
					{
						spdlog::get("logger")->error("[Kicker Element] {}: path: {} is not regular file", name, ramping_file_path);
						std::exit(EXIT_FAILURE);
					}
				}
			}
		}

		spdlog::get("logger")->debug("[Kicker Element] run: {}, s = {}, l = {}, hkick = {}, Vkick = {}, is_field_error = {}, isRamping = {}",
			name, s, l, hkick, vkick, is_field_error, is_ramping);

	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}
}


void KickerElement::execute(int turn) {
	callCuda(cudaEventRecord(simTime.start, 0));
	float time_tmp = 0;

	//auto logger = spdlog::get("logger");
	//logger->debug("[Kicker Element] run: " + name);

	int Np_sur = bunchRef.Np_sur;
	double gamma = bunchRef.gamma;
	double beta = bunchRef.beta;

	double hkick_current = 0;
	double vkick_current = 0;

	if (is_ramping)
	{
		if (turn > ramping_hkick.size())
		{
			spdlog::get("logger")->warn("[Kicker Element] {}, ramping data size {} is less than turn {}, so hkick is set to the last value {}", name, ramping_hkick.size(), turn, ramping_hkick.back());
			hkick_current = ramping_hkick.back();
		}
		else
		{
			hkick_current = ramping_hkick[turn - 1];
		}

		if (turn > ramping_vkick.size())
		{
			spdlog::get("logger")->warn("[Kicker Element] {}, ramping data size {} is less than turn {}, so vkick is set to the last value {}", name, ramping_vkick.size(), turn, ramping_vkick.back());
			vkick_current = ramping_vkick.back();
		}
		else
		{
			vkick_current = ramping_vkick[turn - 1];
		}
	}
	else
	{
		hkick_current = hkick;
		vkick_current = vkick;
	}

	if (fabs(l) > 1e-10)
	{
		if (is_field_error)
		{
			callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
			callKernel(transfer_kicker_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, hkick_current, vkick_current));
			callKernel(transfer_multipole_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, cur_error_order, dev_knl, dev_ksl));
			callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
		}
		else
		{
			callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
			callKernel(transfer_kicker_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, hkick_current, vkick_current));
			callKernel(transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, circumference, gamma, l / 2));
		}
	}
	else
	{
		callKernel(transfer_kicker_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, hkick_current, vkick_current));

		if (is_field_error)
		{
			callKernel(transfer_multipole_thinlens << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, cur_error_order, dev_knl, dev_ksl));
		}
	}

	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.transferElement += time_tmp;

}


RFElement::RFElement(const Parameter& para, int input_beamId, Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

	commandType = "RFElement";
	name = obj_name;
	dev_particle = Bunch.dev_particle;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	circumference = para.circumference;

	qm_ratio = Bunch.qm_ratio;

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		l = data.at("Sequence").at(obj_name).at("L (m)");
		pz_aperture_lower = data.at("Sequence").at(obj_name).at("DeltaP/P aperture")[0];
		pz_aperture_upper = data.at("Sequence").at(obj_name).at("DeltaP/P aperture")[1];
		//z_aperture_lower = data.at("Sequence").at(obj_name).at("Z aperture")[0];
		//z_aperture_upper = data.at("Sequence").at(obj_name).at("Z aperture")[1];

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

	if (turn > host_rf_data[0].size())
	{
		return;
	}

	callCuda(cudaEventRecord(simTime.start, 0));
	float time_tmp = 0;

	//auto logger = spdlog::get("logger");
	//logger->debug("[RF Element] run: " + name);

	int Np_sur = bunchRef.Np_sur;

	double m0 = bunchRef.m0;
	double gammat = bunchRef.gammat;

	double Ek0 = bunchRef.Ek;
	double gamma0 = bunchRef.gamma;
	double beta0 = bunchRef.beta;

	//double eta0 = 1 / (gammat * gammat) - 1 / (gamma0 * gamma0);

	//double drift = drift_length;

	double dE_syn = 0.0;

	for (int i = 0; i < Nrf; i++)
	{
		dE_syn += qm_ratio * host_rf_data[i][turn - 1].voltage * sin(host_rf_data[i][turn - 1].phis);
	}

	double Ek1 = Ek0 + dE_syn;
	double gamma1 = Ek1 / m0 + 1;
	double beta1 = sqrt(1 - 1 / (gamma1 * gamma1));
	double p1 = gamma1 * m0 * beta1;	// In unit of eV/c, so no need to multiply by c
	double p0_kg1 = gamma1 * (m0 * PassConstant::e / (PassConstant::c * PassConstant::c)) * beta1 * PassConstant::c;

	double eta1 = 1 / (gammat * gammat) - 1 / (gamma1 * gamma1);

	bunchRef.Ek = Ek1;
	bunchRef.gamma = gamma1;
	bunchRef.beta = beta1;
	bunchRef.p0 = p1;
	bunchRef.p0_kg = p0_kg1;

	bunchRef.Brho = bunchRef.p0_kg / (qm_ratio * PassConstant::e);

	//transfer_drift << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, beta, gamma, drift + l);
	callKernel(transfer_rf << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, turn, s, beta0, beta1, gamma0, gamma1,
		dev_rf_data, pitch_rf, Nrf, Nturn_rf,
		radius, qm_ratio, dE_syn, eta1, Ek1 + m0,
		pz_aperture_lower, pz_aperture_upper));

	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.transferElement += time_tmp;

}


ElSeparatorElement::ElSeparatorElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

	commandType = "ElSeparatorElement";
	name = obj_name;
	dev_particle = Bunch.dev_particle;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	Np = Bunch.Np;
	Nturn = para.Nturn;
	circumference = para.circumference;

	bunchId = Bunch.bunchId;

	saveDir = para.dir_output_slowExt_particle;
	saveName_part = para.hourMinSec + "_beam" + std::to_string(input_beamId) + "_" + para.beam_name[input_beamId] + "_bunch" + std::to_string(bunchId);

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		l = data.at("Sequence").at(obj_name).at("L (m)");
		//drift_length = data.at("Sequence").at(obj_name).at("Drift length (m)");

		// Todo
		//Ex = data.at("Sequence").at(obj_name).at("Ex (kV/cm)");
		//Ey = data.at("Sequence").at(obj_name).at("Ey (kV/cm)");

		ES_hor_position = data.at("Sequence").at(obj_name).at("ES Horizontal position (m)");
	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}

	dev_particle_Es.mem_allocate_gpu(Np);

	callCuda(cudaMalloc((void**)&dev_counter, 1 * sizeof(int)));
	callCuda(cudaMemset(dev_counter, 0, 1 * sizeof(int)));

}


void ElSeparatorElement::execute(int turn) {
	callCuda(cudaEventRecord(simTime.start, 0));
	float time_tmp = 0;

	//auto logger = spdlog::get("logger");
	//logger->debug("[VKicker Element] run: " + name);

	//int Np_sur = bunchRef.Np_sur;
	//double gamma = bunchRef.gamma;
	//double beta = bunchRef.beta;

	//double drift = drift_length;

	callKernel(check_particle_in_ElSeparator << <block_x, thread_x, 0, 0 >> > (dev_particle, Np, dev_particle_Es, ES_hor_position, dev_counter, s, turn));

	if (turn == Nturn)
	{
		Particle host_particle_Es;
		host_particle_Es.mem_allocate_cpu(Np);

		//callCuda(cudaMemcpy(host_particle_Es, dev_particle_Es, Np * sizeof(Particle), cudaMemcpyDeviceToHost));
		particle_copy(host_particle_Es, dev_particle_Es, Np, cudaMemcpyDeviceToHost, "dist");

		std::filesystem::path saveName_full = saveDir / (saveName_part + "_slowExtraction_ES_" + std::to_string(ES_hor_position) + "_.csv");
		std::ofstream file(saveName_full);

		file << "x" << "," << "px" << "," << "y" << "," << "py" << "," << "z" << "," << "pz" << ","
			<< "tag" << "," << "lostTurn" << "," << "lostPos" << std::endl;

		for (size_t i = 0; i < Np; i++)
		{
			if (host_particle_Es.tag[i] < 0)	// tag != 0, means there is a particle at this index
			{
				file << std::setprecision(10)
					<< host_particle_Es.x[i] << ","
					<< host_particle_Es.px[i] << ","
					<< host_particle_Es.y[i] << ","
					<< host_particle_Es.py[i] << ","
					<< host_particle_Es.z[i] << ","
					<< host_particle_Es.pz[i] << ","
					<< host_particle_Es.tag[i] << ","
					<< host_particle_Es.lostTurn[i] << ","
					<< host_particle_Es.lostPos[i] << "\n";
			}
		}

		file.close();
		host_particle_Es.mem_free_cpu();
	}

	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.transferElement += time_tmp;

}


TuneExciterElement::TuneExciterElement(const Parameter& para, int input_beamId, Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

	commandType = "TuneExciterElement";
	name = obj_name;
	dev_particle = Bunch.dev_particle;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	circumference = para.circumference;

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		kick_angle = data.at("Sequence").at(obj_name).at("Kick angle (rad)");
		freq_center = data.at("Sequence").at(obj_name).at("Frequency center (Hz)");
		scan_period = data.at("Sequence").at(obj_name).at("Scan period (s)");
		scan_freq_range = data.at("Sequence").at(obj_name).at("Scan frequency range (Hz)");
		kick_direction = data.at("Sequence").at(obj_name).at("Kick direction");
		turn_start = data.at("Sequence").at(obj_name).at("Turns")[0];
		turn_end = data.at("Sequence").at(obj_name).at("Turns")[1];

		if (kick_direction != "x" || kick_direction != "X" || kick_direction != "y" || kick_direction != "Y")
		{
			spdlog::get("logger")->warn("[TuneExciter Element] {}: Kick direction = {}, which should be x/X or y/Y", name, kick_direction);
		}

		/*spdlog::get("logger")->info("[TuneExciter Element] {}: kick_angle = {} rad, freq_center = {} Hz, scan_period = {} s, scan_freq_range = {} Hz, kick_direction = {}, turn_start = {}, turn_end = {}",
			name, kick_angle, freq_center, scan_period, scan_freq_range, kick_direction, turn_start, turn_end);*/

	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}

}


void TuneExciterElement::execute(int turn) {

	if (turn < turn_start || turn > turn_end)
	{
		return;
	}

	callCuda(cudaEventRecord(simTime.start, 0));
	float time_tmp = 0;

	//auto logger = spdlog::get("logger");
	//logger->debug("[TuneExciter Element] run: " + name);

	int Np_sur = bunchRef.Np_sur;
	double t0 = bunchRef.t0;
	double beta = bunchRef.beta;

	spdlog::get("logger")->debug("[TuneExciter Element] {}: turn = {}, t0 = {}, beta = {}", name, turn, t0, beta);

	if (kick_direction == "x" || kick_direction == "X")
	{
		callKernel(transfer_tuneExciter_x << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, kick_angle, t0, beta,
			freq_center, scan_period, scan_freq_range));
	}
	if (kick_direction == "y" || kick_direction == "Y")
	{
		callKernel(transfer_tuneExciter_y << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, kick_angle, t0, beta,
			freq_center, scan_period, scan_freq_range));
	}

	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.transferElement += time_tmp;

}


__global__ void transfer_drift(Particle dev_particle, int Np_sur, double beta, double circumference,
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

		tau0 = dev_particle.z[tid] / beta;
		pt0 = dev_particle.pz[tid] * beta;

		int mask = (dev_particle.tag[tid] > 0);

		dev_particle.x[tid] += (r12 * dev_particle.px[tid]) * mask;
		dev_particle.y[tid] += (r34 * dev_particle.py[tid]) * mask;

		tau1 = tau0 + (r56 * pt0) * mask;
		dev_particle.z[tid] = tau1 * beta;

		c_half = circumference * 0.5;
		over = (dev_particle.z[tid] > c_half);
		under = (dev_particle.z[tid] < -c_half);
		dev_particle.z[tid] += (under - over) * circumference;

		tid += stride;
	}
}

__global__ void transfer_dipole_full(Particle dev_particle, int Np_sur, double beta, double circumference,
	double r11, double r12, double r16, double r21, double r22, double r26,
	double r34, double r51, double r52, double r56,
	double fl21i, double fl43i, double fr21i, double fr43i) {

	// I would like to express my gratitude to Dr.Ren Hang(renhang@impcas.ac.cn)
	// for providing the code for dipole transfer 

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double d = 0;	// about power error

	while (tid < Np_sur) {

		int tag = dev_particle.tag[tid];

		if (tag > 0)
		{
			double tau0 = dev_particle.z[tid] / beta;	// tau = z/beta - ct(=0) = z/beta
			double pt0 = dev_particle.pz[tid] * beta;	// pt = DeltaE/(P0*c) = beta*DeltaP/P0

			double fl21 = fl21i / (1 + pt0 / beta);
			double fl43 = fl43i / (1 + pt0 / beta);
			double fr21 = fr21i / (1 + pt0 / beta);
			double fr43 = fr43i / (1 + pt0 / beta);

			// apply the influence of left fringe field
			double x1 = dev_particle.x[tid];
			double px1 = dev_particle.px[tid] + fl21 * x1;
			double y1 = dev_particle.y[tid];
			double py1 = dev_particle.py[tid] + fl43 * y1;
			double tau1 = tau0;
			double pt1 = pt0 + d;

			// apply the influence of dipole
			double x2 = r11 * x1 + r12 * px1 + r16 * pt1;
			double px2 = r21 * x1 + r22 * px1 + r26 * pt1;
			double y2 = y1 + r34 * py1;
			double py2 = py1;
			double tau2 = r51 * x1 + r52 * px1 + tau1 + r56 * pt1;
			double pt2 = pt1;

			// apply the influece of right fringe field
			double x3 = x2;
			double px3 = px2 + fr21 * x2;
			double y3 = y2;
			double py3 = py2 + fr43 * y2;
			double tau3 = tau2;
			double pt3 = pt2 - d;

			dev_particle.z[tid] = tau3 * beta;
			dev_particle.pz[tid] = pt3 / beta;
			dev_particle.x[tid] = x3;
			dev_particle.px[tid] = px3;
			dev_particle.y[tid] = y3;
			dev_particle.py[tid] = py3;

			double c_half = circumference * 0.5;
			int over = (dev_particle.z[tid] > c_half);
			int under = (dev_particle.z[tid] < -c_half);
			dev_particle.z[tid] += (under - over) * circumference;

			tid += stride;
		}

	}

}


__global__ void transfer_dipole_half_left(Particle dev_particle, int Np_sur, double beta, double circumference,
	double r11, double r12, double r16, double r21, double r22, double r26,
	double r34, double r51, double r52, double r56,
	double fl21i, double fl43i, double fr21i, double fr43i) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double d = 0;	// about power error

	while (tid < Np_sur) {

		int tag = dev_particle.tag[tid];

		if (tag > 0)
		{
			double tau0 = dev_particle.z[tid] / beta;	// tau = z/beta - ct(=0) = z/beta
			double pt0 = dev_particle.pz[tid] * beta;	// pt = DeltaE/(P0*c) = beta*DeltaP/P0

			double fl21 = fl21i / (1 + pt0 / beta);
			double fl43 = fl43i / (1 + pt0 / beta);

			// apply the influence of left fringe field
			double x1 = dev_particle.x[tid];
			double px1 = dev_particle.px[tid] + fl21 * x1;
			double y1 = dev_particle.y[tid];
			double py1 = dev_particle.py[tid] + fl43 * y1;
			double tau1 = tau0;
			double pt1 = pt0 + d;

			// apply the influence of dipole
			double x2 = r11 * x1 + r12 * px1 + r16 * pt1;
			double px2 = r21 * x1 + r22 * px1 + r26 * pt1;
			double y2 = y1 + r34 * py1;
			double py2 = py1;
			double tau2 = r51 * x1 + r52 * px1 + tau1 + r56 * pt1;
			double pt2 = pt1;

			// no influece of right fringe field
			double x3 = x2;
			double px3 = px2;
			double y3 = y2;
			double py3 = py2;
			double tau3 = tau2;
			double pt3 = pt2 - d;

			dev_particle.z[tid] = tau3 * beta;
			dev_particle.pz[tid] = pt3 / beta;
			dev_particle.x[tid] = x3;
			dev_particle.px[tid] = px3;
			dev_particle.y[tid] = y3;
			dev_particle.py[tid] = py3;

			double c_half = circumference * 0.5;
			int over = (dev_particle.z[tid] > c_half);
			int under = (dev_particle.z[tid] < -c_half);
			dev_particle.z[tid] += (under - over) * circumference;

			tid += stride;
		}


	}

}


__global__ void transfer_dipole_half_right(Particle dev_particle, int Np_sur, double beta, double circumference,
	double r11, double r12, double r16, double r21, double r22, double r26,
	double r34, double r51, double r52, double r56,
	double fl21i, double fl43i, double fr21i, double fr43i) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double d = 0;	// about power error

	while (tid < Np_sur) {

		int tag = dev_particle.tag[tid];

		if (tag > 0)
		{

			double tau0 = dev_particle.z[tid] / beta;	// tau = z/beta - ct(=0) = z/beta
			double pt0 = dev_particle.pz[tid] * beta;	// pt = DeltaE/(P0*c) = beta*DeltaP/P0

			double fr21 = fr21i / (1 + pt0 / beta);
			double fr43 = fr43i / (1 + pt0 / beta);

			// no influence of left fringe field
			double x1 = dev_particle.x[tid];
			double px1 = dev_particle.px[tid];
			double y1 = dev_particle.y[tid];
			double py1 = dev_particle.py[tid];
			double tau1 = tau0;
			double pt1 = pt0 + d;

			// apply the influence of dipole
			double x2 = r11 * x1 + r12 * px1 + r16 * pt1;
			double px2 = r21 * x1 + r22 * px1 + r26 * pt1;
			double y2 = y1 + r34 * py1;
			double py2 = py1;
			double tau2 = r51 * x1 + r52 * px1 + tau1 + r56 * pt1;
			double pt2 = pt1;

			// apply the influece of right fringe field
			double x3 = x2;
			double px3 = px2 + fr21 * x2;
			double y3 = y2;
			double py3 = py2 + fr43 * y2;
			double tau3 = tau2;
			double pt3 = pt2 - d;

			dev_particle.z[tid] = tau3 * beta;
			dev_particle.pz[tid] = pt3 / beta;
			dev_particle.x[tid] = x3;
			dev_particle.px[tid] = px3;
			dev_particle.y[tid] = y3;
			dev_particle.py[tid] = py3;

			double c_half = circumference * 0.5;
			int over = (dev_particle.z[tid] > c_half);
			int under = (dev_particle.z[tid] < -c_half);
			dev_particle.z[tid] += (under - over) * circumference;

			tid += stride;
		}
	}

}


__global__ void transfer_quadrupole_thicklens_norm(Particle dev_particle, int Np_sur, double beta, double circumference,
	double k1, double l) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double r11 = 0, r12 = 0, r21 = 0, r22 = 0, r33 = 0, r34 = 0, r43 = 0, r44 = 0, r56 = 0;

	while (tid < Np_sur) {

		if (dev_particle.tag[tid] > 0)
		{
			double x0 = dev_particle.x[tid];
			double px0 = dev_particle.px[tid];
			double y0 = dev_particle.y[tid];
			double py0 = dev_particle.py[tid];
			double tau0 = dev_particle.z[tid] / beta;
			double pt0 = dev_particle.pz[tid] * beta;

			double k1_chrom = k1 / (1 + pt0 / beta);
			double omega = sqrt(fabs(k1_chrom));

			double cx = cos(omega * l);
			double sx = sin(omega * l);
			double chx = cosh(omega * l);
			double shx = sinh(omega * l);

			//printf("Quadrupole: k1_chrom = %f, pz = %f, beta = %f, pt = %f, cx = %f, sx = %f, chx =%f, shx = %f\n",
			//	k1_chrom, dev_particle.pz, beta, pt0, cx, sx, chx, shx);

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

			double x1 = r11 * x0 + r12 * px0;
			double px1 = r21 * x0 + r22 * px0;
			double y1 = r33 * y0 + r34 * py0;
			double py1 = r43 * y0 + r44 * py0;
			double tau1 = tau0 + r56 * pt0;

			dev_particle.z[tid] = tau1 * beta;
			dev_particle.x[tid] = x1;
			dev_particle.px[tid] = px1;
			dev_particle.y[tid] = y1;
			dev_particle.py[tid] = py1;

			double c_half = circumference * 0.5;
			int over = (dev_particle.z[tid] > c_half);
			int under = (dev_particle.z[tid] < -c_half);
			dev_particle.z[tid] += (under - over) * circumference;

			tid += stride;
		}

	}
}


__global__ void transfer_quadrupole_thicklens_skew(Particle dev_particle, int Np_sur, double beta, double circumference,
	double k1s, double l) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double r11 = 0, r12 = 0, r13 = 0, r14 = 0, r21 = 0, r22 = 0, r23 = 0, r24 = 0;
	double r31 = 0, r32 = 0, r33 = 0, r34 = 0, r41 = 0, r42 = 0, r43 = 0, r44 = 0;
	double r56 = 0;

	while (tid < Np_sur) {

		if (dev_particle.tag[tid] > 0)
		{
			double x0 = dev_particle.x[tid];
			double px0 = dev_particle.px[tid];
			double y0 = dev_particle.y[tid];
			double py0 = dev_particle.py[tid];
			double tau0 = dev_particle.z[tid] / beta;
			double pt0 = dev_particle.pz[tid] * beta;

			double k1s_chrom = k1s / (1 + pt0 / beta);
			double omega = sqrt(fabs(k1s_chrom));

			double cx = cos(omega * l);
			double sx = sin(omega * l);
			double chx = cosh(omega * l);
			double shx = sinh(omega * l);

			double cp = (cx + chx) / 2;
			double cm = (cx - chx) / 2;
			double sp = (sx + shx) / 2;
			double sm = (sx - shx) / 2;


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

			double x1 = r11 * x0 + r12 * px0 + r13 * y0 + r14 * py0;
			double px1 = r21 * x0 + r22 * px0 + r23 * y0 + r24 * py0;
			double y1 = r31 * x0 + r32 * px0 + r33 * y0 + r34 * py0;
			double py1 = r41 * x0 + r42 * px0 + r43 * y0 + r44 * py0;
			double tau1 = tau0 + r56 * pt0;

			dev_particle.z[tid] = tau1 * beta;
			dev_particle.x[tid] = x1;
			dev_particle.px[tid] = px1;
			dev_particle.y[tid] = y1;
			dev_particle.py[tid] = py1;

			double c_half = circumference * 0.5;
			int over = (dev_particle.z[tid] > c_half);
			int under = (dev_particle.z[tid] < -c_half);
			dev_particle.z[tid] += (under - over) * circumference;

			tid += stride;
		}
	}
}


__global__ void transfer_quadrupole_thinlens_norm(Particle dev_particle, int Np_sur, double k1l) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double r21 = 0, r43 = 0;

	while (tid < Np_sur) {


		double x0 = dev_particle.x[tid];
		double y0 = dev_particle.y[tid];
		double pz0 = dev_particle.pz[tid];

		double k1l_chrom = k1l / (1 + pz0);

		r21 = -k1l_chrom;
		r43 = k1l_chrom;

		double delta_px = r21 * x0;
		double delta_py = r43 * y0;

		int mask = (dev_particle.tag[tid] > 0);

		dev_particle.px[tid] += delta_px * mask;
		dev_particle.py[tid] += delta_py * mask;

		tid += stride;

	}
}


__global__ void transfer_quadrupole_thinlens_skew(Particle dev_particle, int Np_sur, double k1sl) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double r23 = 0, r41 = 0;

	while (tid < Np_sur) {

		double x0 = dev_particle.x[tid];
		double y0 = dev_particle.y[tid];
		double pz0 = dev_particle.pz[tid];

		double k1sl_chrom = k1sl / (1 + pz0);

		r23 = -k1sl_chrom;
		r41 = -k1sl_chrom;

		double delta_px = r23 * y0;
		double delta_py = r41 * x0;

		int mask = (dev_particle.tag[tid] > 0);

		dev_particle.px[tid] += delta_px * mask;
		dev_particle.py[tid] += delta_py * mask;

		tid += stride;

	}
}


__global__ void transfer_sextupole_thinlens_norm(Particle dev_particle, int Np_sur, double k2l) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double x0 = 0, y0 = 0;
	double k2l_chrom = 0;

	while (tid < Np_sur) {

		x0 = dev_particle.x[tid];
		y0 = dev_particle.y[tid];

		k2l_chrom = k2l / (1 + dev_particle.pz[tid]);

		int mask = (dev_particle.tag[tid] > 0);

		dev_particle.px[tid] += (-0.5 * k2l_chrom * (x0 * x0 - y0 * y0)) * mask;
		dev_particle.py[tid] += (k2l_chrom * x0 * y0) * mask;

		tid += stride;
	}
}


__global__ void transfer_sextupole_thinlens_skew(Particle dev_particle, int Np_sur, double k2sl) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double x0 = 0, y0 = 0;
	double k2sl_chrom = 0;

	while (tid < Np_sur) {

		x0 = dev_particle.x[tid];
		y0 = dev_particle.y[tid];

		k2sl_chrom = k2sl / (1 + dev_particle.pz[tid]);

		int mask = (dev_particle.tag[tid] > 0);

		dev_particle.px[tid] += (k2sl_chrom * x0 * y0) * mask;
		dev_particle.py[tid] += (0.5 * k2sl_chrom * (x0 * x0 - y0 * y0)) * mask;

		tid += stride;
	}
}


__global__ void transfer_octupole_thinlens_norm(Particle dev_particle, int Np_sur, double k3l) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double x0 = 0, y0 = 0;
	double k3l_chrom = 0;

	while (tid < Np_sur) {

		x0 = dev_particle.x[tid];
		y0 = dev_particle.y[tid];

		k3l_chrom = k3l / (1 + dev_particle.pz[tid]);

		int mask = (dev_particle.tag[tid] > 0);

		dev_particle.px[tid] += (-1.0 / 6 * k3l_chrom * (pow(x0, 3) - 3 * x0 * pow(y0, 2))) * mask;
		dev_particle.py[tid] += (1.0 / 6 * k3l_chrom * (3 * pow(x0, 2) * y0 - pow(y0, 3))) * mask;

		tid += stride;
	}
}


__global__ void transfer_octupole_thinlens_skew(Particle dev_particle, int Np_sur, double k3sl) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double x0 = 0, y0 = 0;
	double k3sl_chrom = 0;

	while (tid < Np_sur) {

		x0 = dev_particle.x[tid];
		y0 = dev_particle.y[tid];

		k3sl_chrom = k3sl / (1 + dev_particle.pz[tid]);

		int mask = (dev_particle.tag[tid] > 0);

		dev_particle.px[tid] += (1.0 / 6 * k3sl_chrom * (3 * pow(x0, 2) * y0 - pow(y0, 3))) * mask;
		dev_particle.py[tid] += (1.0 / 6 * k3sl_chrom * (pow(x0, 3) - 3 * x0 * pow(y0, 2))) * mask;

		tid += stride;
	}
}


__global__ void transfer_multipole_thinlens(Particle dev_particle, int Np_sur, int order, const double* dev_knl, const double* dev_ksl) {

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
	constexpr double factorial_table[max_order + 1] = {
		1.0,
		1.0,
		2.0,
		6.0,
		24.0,
		120.0,
		720.0,
		5040.0,
		40320.0,
		362880.0,
		3628800.0,
		39916800.0,
		479001600.0,
		6227020800.0,
		87178291200.0,
		1307674368000.0,
		20922789888000.0,
		355687428096000.0,
		6402373705728000.0,
		121645100408832000.0 ,
		2432902008176640000.0 };

	while (tid < Np_sur)
	{
		x = dev_particle.x[tid];
		y = dev_particle.y[tid];

		dpx = 0.0;
		dpy = 0.0;
		real = 1.0;
		imag = 0.0;

		if (order >= 0)
		{
			double kn0l = dev_knl[0];
			double ks0l = dev_ksl[0];
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

			double knl = dev_knl[iorder];
			double ksl = dev_ksl[iorder];

			dpx += -knl * (inv_factorial * real) + ksl * (inv_factorial * imag);
			dpy += knl * (inv_factorial * imag) + ksl * (inv_factorial * real);
		}

		int mask = (dev_particle.tag[tid] > 0);

		dev_particle.px[tid] += dpx * mask;
		dev_particle.py[tid] += dpy * mask;

		tid += stride;
	}
}


__global__ void transfer_kicker_thinlens(Particle dev_particle, int Np_sur, double hkick, double vkick) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double hkick_chrom = 0;
	double vkick_chrom = 0;

	while (tid < Np_sur) {

		hkick_chrom = hkick / (1 + dev_particle.pz[tid]);
		vkick_chrom = vkick / (1 + dev_particle.pz[tid]);

		int mask = (dev_particle.tag[tid] > 0);

		dev_particle.px[tid] += hkick_chrom * mask;
		dev_particle.py[tid] += vkick_chrom * mask;

		tid += stride;
	}
}


__global__ void transfer_rf(Particle dev_particle, int Np_sur, int turn, double s, double beta0, double beta1, double gamma0, double gamma1,
	RFData* dev_rf_data, size_t  pitch_rf, int Nrf, size_t Nturn_rf,
	double radius, double ratio, double dE_syn, double eta1, double E_total1,
	double pz_min, double pz_max) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	//const double circumference = 2 * PassConstant::PI * radius;
	const double trans_scale = beta0 * gamma0 / (beta1 * gamma1);	// px0*beta0*gamma0 = px1*beta1*gamma1

	const double E_total0 = E_total1 - dE_syn;
	while (tid < Np_sur) {

		if (dev_particle.tag[tid] > 0)
		{

			double z0 = 0, pz0 = 0, theta0 = 0, dE0 = 0;
			double z1 = 0, pz1 = 0, theta1 = 0, dE1 = 0;

			double dE_non_syn = 0;

			double voltage = 0, harmonic = 0, phis = 0, phi_offset = 0;

			z0 = dev_particle.z[tid];
			pz0 = dev_particle.pz[tid];

			convert_z_dp_to_theta_dE(z0, pz0, theta0, dE0, radius, beta0, E_total0);

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
			theta1 = theta0;

			convert_theta_dE_to_z_dp(z1, pz1, theta1, dE1, radius, beta1, E_total1);

			dev_particle.pz[tid] = pz1;

			dev_particle.px[tid] = dev_particle.px[tid] * trans_scale;
			dev_particle.py[tid] = dev_particle.py[tid] * trans_scale;

			if (pz1 < pz_min || pz1 > pz_max)
			{
				dev_particle.tag[tid] *= -1;
				dev_particle.lostTurn[tid] = turn;
				dev_particle.lostPos[tid] = s;
			}

		}
		tid += stride;
	}
}


__device__ void convert_z_dp_to_theta_dE(double z, double dp, double& theta, double& dE, double radius, double beta, double Es) {
	theta = z / radius;
	dE = dp * beta * beta * Es;
}


__device__ void convert_theta_dE_to_z_dp(double& z, double& dp, double theta, double dE, double radius, double beta, double Es) {
	z = theta * radius;
	dp = dE / (beta * beta * Es);
}


__global__ void check_particle_in_ElSeparator(Particle dev_particle, int Np_sur, Particle dev_particle_ES, double ES_position, int* global_counter, double s, int turn) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < Np_sur) {

		if (dev_particle.x[tid] >= ES_position && dev_particle.tag[tid] > 0) {

			dev_particle.tag[tid] *= -1;
			dev_particle.lostPos[tid] = s;
			dev_particle.lostTurn[tid] = turn;

			// 每次排序后粒子坐标会变，这里通过原子加法，确保在不同圈数中，如果多个粒子下标对应同一个tid值时，数据不会被覆盖
			int write_index = atomicAdd(global_counter, 1);

			//dev_particle_ES[write_index] = dev_particle[tid]; // Copy the particle to the ElSeparator array
			dev_particle_ES.x[write_index] = dev_particle.x[tid];
			dev_particle_ES.px[write_index] = dev_particle.px[tid];
			dev_particle_ES.y[write_index] = dev_particle.y[tid];
			dev_particle_ES.py[write_index] = dev_particle.py[tid];
			dev_particle_ES.z[write_index] = dev_particle.z[tid];
			dev_particle_ES.pz[write_index] = dev_particle.pz[tid];
			dev_particle_ES.lostPos[write_index] = dev_particle.lostPos[tid];
			dev_particle_ES.tag[write_index] = dev_particle.tag[tid];
			dev_particle_ES.lostTurn[write_index] = dev_particle.lostTurn[tid];
			dev_particle_ES.sliceId[write_index] = dev_particle.sliceId[tid];

#ifdef PASS_CAL_PHASE
			dev_particle_ES.last_x[write_index] = dev_particle.last_x[tid];
			dev_particle_ES.last_y[write_index] = dev_particle.last_y[tid];
			dev_particle_ES.last_px[write_index] = dev_particle.last_px[tid];
			dev_particle_ES.last_py[write_index] = dev_particle.last_py[tid];
			dev_particle_ES.phase_x[write_index] = dev_particle.phase_x[tid];
			dev_particle_ES.phase_y[write_index] = dev_particle.phase_y[tid];
#endif
		}

		// 同步块内线程
		__syncthreads();

		tid += stride;
	}
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
	file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

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

	//logger->debug("Number of RF data: {}", data.size());
	//for (size_t i = 0; i < data.size(); i++)
	//{
	//	logger->debug("{} {} {} {}", data[i].harmonic, data[i].voltage, data[i].phis, data[i].phi_offset);
	//}

	return data;
}


std::vector<std::pair<double, double>> readSextRampingDataFromCSV(const std::string& filename) {
	// Read Sextupole ramping data from file

	auto logger = spdlog::get("logger");

	std::vector<std::pair<double, double>> data;
	std::ifstream file(filename);

	if (!file.is_open()) {
		logger->error("[Read Sextupole Ramping Data] Open file failed: {}", filename);
		std::exit(EXIT_FAILURE);
	}

	std::string line;

	// Skip the first line
	//std::getline(file, line);

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
				logger->error("[Read Sextupole Ramping Data] Error parsing value in line: {}", line);
				values.clear();
				break;
			}
		}

		if (values.size() == 2) {
			data.push_back(std::pair(values[0], values[1]));
		}
		else {
			logger->error("[Read Sextupole Ramping Data] Invalid data format in line: {}", line);
			std::exit(EXIT_FAILURE);
		}
	}

	return data;

}


std::vector<TuneExciterData> readTuneExciterDataFromCSV(const std::string& filename) {

	// Read TuneExciter data from file

	auto logger = spdlog::get("logger");

	std::vector<TuneExciterData> data;
	std::ifstream file(filename);

	if (!file.is_open()) {
		logger->error("[Read TuneExciter Data] Open file failed: {}", filename);
		std::exit(EXIT_FAILURE);
	}

	std::string line;

	// Skip the first line
	file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

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
				logger->error("[Read TuneExciter Data] Error parsing value in line: {}", line);
				values.clear();
				break;
			}
		}

		if (values.size() == 3) {
			data.push_back({ values[1], values[2] });	// values[0] is turn number, not needed here
		}
		else {
			logger->error("[Read TuneExciter Data] Invalid data format in line: {}", line);
			std::exit(EXIT_FAILURE);
		}
	}

	return data;
}


__global__ void transfer_tuneExciter_x(Particle dev_particle, int Np_sur, double kick_angle, double t0, double beta,
	double freq_center, double scan_period, double scan_freq_range) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < Np_sur) {
		if (dev_particle.tag[tid] > 0)
		{
			double t = -1 * dev_particle.z[tid] / (beta * PassConstant::c) + t0; // time in seconds
			double f = freq_center + (t / scan_period - floor(t / scan_period) - 0.5) * scan_freq_range;
			double kick = kick_angle * sin(2 * PassConstant::PI * f * t);
			dev_particle.px[tid] += kick;
			if (tid == 0 || tid == 1 || tid == 2) {
				printf("z = %e, t = %e, f = %e, kick = %e\n", dev_particle.z[tid], t, f, kick);
			}
		}
		tid += stride;
	}
}


__global__ void transfer_tuneExciter_y(Particle dev_particle, int Np_sur, double kick_angle, double t0, double beta,
	double freq_center, double scan_period, double scan_freq_range) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < Np_sur) {
		if (dev_particle.tag[tid] > 0)
		{
			double t = -1 * dev_particle.z[tid] / (beta * PassConstant::c) + t0; // time in seconds
			double f = freq_center + (t / scan_period - floor(t / scan_period) - 0.5) * scan_freq_range;
			double kick = kick_angle * sin(2 * PassConstant::PI * f * t);
			dev_particle.py[tid] += kick;
		}
		tid += stride;
	}
}
