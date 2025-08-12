#include "twiss.h"
#include "parameter.h"
#include "constant.h"

#include <fstream>
#include <cuda_runtime.h>

Twiss::Twiss(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name, const ParallelPlan1d& plan1d, TimeEvent& timeevent)
	:simTime(timeevent), bunchRef(Bunch) {

	name = obj_name;
	dev_particle = Bunch.dev_particle;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	Np = Bunch.Np;
	circumference = para.circumference;

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

		Dx = data.at("Sequence").at(obj_name).at("Dx (m)");
		Dx_previous = data.at("Sequence").at(obj_name).at("Dx (m) previous");

		Dpx = data.at("Sequence").at(obj_name).at("Dpx");
		Dpx_previous = data.at("Sequence").at(obj_name).at("Dpx previous");

		DQx = data.at("Sequence").at(obj_name).at("DQx");
		DQy = data.at("Sequence").at(obj_name).at("DQy");

		longitudinal_transfer = data.at("Sequence").at(obj_name).at("Longitudinal transfer");

		// when ¦Ã > ¦Ãt (¦Ç > 0), muz (input value) should be > 0
		// when ¦Ã < ¦Ãt (¦Ç < 0), muz (input value) should be < 0
		muz = data.at("Sequence").at(obj_name).at("Mu z");
		muz_previous = data.at("Sequence").at(obj_name).at("Mu z previous");

	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}

	phi_x = (mux - mux_previous) * 2 * PassConstant::PI;
	phi_y = (muy - muy_previous) * 2 * PassConstant::PI;
	phi_z = (muz - muz_previous) * 2 * PassConstant::PI;

	if ("drift" == longitudinal_transfer)
	{
		m11_z = 1;
		m12_z = -1 * (1 / (gammat * gammat) - 1 / (gamma * gamma)) * (s - s_previous);
		m21_z = 0;
		m22_z = 1;
	}
	else if ("matrix" == longitudinal_transfer)
	{
		m11_z = cos(phi_z);
		m12_z = sigmaz / dp * sin(phi_z);
		m21_z = -1 * dp / sigmaz * sin(phi_z);
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

	logger->info("[Twiss] name    = {}, s       = {}", name, s);
	logger->info("[Twiss] Alpha x = {}, Alpha y = {}", alphax, alphay);
	logger->info("[Twiss] Beta  x = {}, Beta  y = {}", betax, betay);
	logger->info("[Twiss] Mu    x = {}, Mu    y = {}", mux, muy);
	logger->info("[Twiss] Alpha x previous = {}, Alpha y previous = {}", alphax_previous, alphay_previous);
	logger->info("[Twiss] Beta  x previous = {}, Beta  y previous = {}", betax_previous, betay_previous);
	logger->info("[Twiss] Mu    x previous = {}, Mu    y previous = {}", mux_previous, muy_previous);

	logger->info("[Twiss] Dx               = {}, Dx previous      = {}", Dx, Dx_previous);
	logger->info("[Twiss] Dpx              = {}, Dpx previous     = {}", Dpx, Dpx_previous);
	logger->info("[Twiss] DQx              = {}, DQy              = {}", DQx, DQy);

	logger->info("[Twiss] Longitudinal transfer = {}", longitudinal_transfer);
	logger->info("[Twiss] Mu   z          = {}", muz);
	logger->info("[Twiss] Mu   z previous = {}", muz);
	logger->info("[Twiss] gamma  = {}, gammat = {}", gamma, gammat);
	logger->info("[Twiss] sigmaz = {}, dp     = {}", sigmaz, dp);
}


void Twiss::execute(int turn) {

	callCuda(cudaEventRecord(simTime.start, 0));
	float time_tmp = 0;

	//auto logger = spdlog::get("logger");
	//logger->debug("[Twiss] turn = {}, start running of : {}, s = {}, 6D (logi = {})", turn, name, s, longitudinal_transfer);

	int Np_sur = bunchRef.Np_sur;

	callKernel(
		transfer_matrix_6D << <block_x, thread_x, 0, 0 >> > (dev_particle, Np_sur, circumference,
			betax, betax_previous, alphax, alphax_previous,
			betay, betay_previous, alphay, alphay_previous,
			phi_x, phi_y, DQx * 2 * PassConstant::PI, DQy * 2 * PassConstant::PI,
			Dx, Dx_previous, Dpx, Dpx_previous,
			m11_z, m12_z, m21_z, m22_z)
	);

	//callCuda(cudaDeviceSynchronize());
	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.twiss += time_tmp;
}


__global__ void transfer_matrix_6D(Particle dev_particle, int Np_sur, double circumference,
	double betax, double betax_previous, double alphax, double alphax_previous,
	double betay, double betay_previous, double alphay, double alphay_previous,
	double phix, double phiy, double DQx, double DQy,
	double Dx, double Dx_previous, double Dpx, double Dpx_previous,
	double m11_z, double m12_z, double m21_z, double m22_z) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	const double c_half = circumference * 0.5;

	const double sqrt_betax_betaxprev = sqrt(betax * betax_previous);
	const double sqrt_betax_de_betaxprev = sqrt(betax / betax_previous);
	const double sqrt_betaxprev_de_betax = sqrt(betax_previous / betax);

	const double sqrt_betay_betayprev = sqrt(betay * betay_previous);
	const double sqrt_betay_de_betayprev = sqrt(betay / betay_previous);
	const double sqrt_betayprev_de_betay = sqrt(betay_previous / betay);

	while (tid < Np_sur)
	{
		int tag = dev_particle.tag[tid];

		if (tag > 0)
		{
			double z1 = dev_particle.z[tid];
			double pz1 = dev_particle.pz[tid];

			double x1 = dev_particle.x[tid] - Dx_previous * pz1;
			double px1 = dev_particle.px[tid] - Dpx_previous * pz1;

			double y1 = dev_particle.y[tid];
			double py1 = dev_particle.py[tid];

			double phi_x = phix + pz1 * DQx;
			double phi_y = phiy + pz1 * DQy;

			double cx = 0, sx = 0, cy = 0, sy = 0;

			sincos(phi_x, &sx, &cx);	// calculate sin(phix) and cos(phix) simultaneously
			sincos(phi_y, &sy, &cy);

			double m11_x = sqrt_betax_de_betaxprev * (cx + alphax_previous * sx);
			double m12_x = sqrt_betax_betaxprev * sx;
			double m21_x = -1 * (1 + alphax * alphax_previous) / sqrt_betax_betaxprev * sx + (alphax_previous - alphax) / sqrt_betax_betaxprev * cx;
			double m22_x = sqrt_betaxprev_de_betax * (cx - alphax * sx);

			double m11_y = sqrt_betay_de_betayprev * (cy + alphay_previous * sy);
			double m12_y = sqrt_betay_betayprev * sy;
			double m21_y = -1 * (1 + alphay * alphay_previous) / sqrt_betay_betayprev * sy + (alphay_previous - alphay) / sqrt_betay_betayprev * cy;
			double m22_y = sqrt_betayprev_de_betay * (cy - alphay * sy);

			double z2 = z1 * m11_z + pz1 * m12_z;
			double pz2 = z1 * m21_z + pz1 * m22_z;

			double x2 = x1 * m11_x + px1 * m12_x + Dx * pz2;
			double px2 = x1 * m21_x + px1 * m22_x + Dpx * pz2;

			double y2 = y1 * m11_y + py1 * m12_y;
			double py2 = y1 * m21_y + py1 * m22_y;

			int over = (z2 > c_half);
			int under = (z2 < -c_half);
			z2 += (under - over) * circumference;

			dev_particle.z[tid] = z2;
			dev_particle.pz[tid] = pz2;
			dev_particle.x[tid] = x2;
			dev_particle.px[tid] = px2;
			dev_particle.y[tid] = y2;
			dev_particle.py[tid] = py2;

			tid += stride;
		}

	}
}
