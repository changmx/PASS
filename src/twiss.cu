#include "twiss.h"
#include "parameter.h"
#include "constant.h"

#include <fstream>

Twiss::Twiss(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name, const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent) {

	name = obj_name;
	dev_bunch = Bunch.dev_bunch;

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

		//Dx = data.at("Sequence").at(obj_name).at("Dx (m)");

		logitudinal_transfer = data.at("Sequence").at(obj_name).at("Logitudinal transfer");

		if ("matrix" == logitudinal_transfer)
		{
			// when ¦Ã > ¦Ãt (¦Ç > 0), muz (input value) should be > 0
			// when ¦Ã < ¦Ãt (¦Ç < 0), muz (input value) should be < 0
			muz = data.at("Sequence").at(obj_name).at("Mu z");
			muz_previous = data.at("Sequence").at(obj_name).at("Mu z previous");
		}
		else
		{
			muz = 0;
			muz_previous = 0;
		}
	}
	catch (json::exception e)
	{
		//std::cout << e.what() << std::endl;
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}

	double phi_x = (mux - mux_previous) * 2 * PassConstant::PI;
	double phi_y = (muy - muy_previous) * 2 * PassConstant::PI;
	double phi_z = (muz - muz_previous) * 2 * PassConstant::PI;

	m11_x = sqrt(betax / betax_previous) * (cos(phi_x) + alphax_previous * sin(phi_x));
	m12_x = sqrt(betax * betax_previous) * sin(phi_x);
	m21_x = -1 * (1 + alphax * alphax_previous) / sqrt(betax * betax_previous) * sin(phi_x) + (alphax_previous - alphax) / sqrt(betax * betax_previous) * cos(phi_x);
	m22_x = sqrt(betax_previous / betax) * (cos(phi_x) - alphax * sin(phi_x));

	m11_y = sqrt(betay / betay_previous) * (cos(phi_y) + alphay_previous * sin(phi_y));
	m12_y = sqrt(betay * betay_previous) * sin(phi_y);
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
	logger->info("[Twiss] Mu   z previous = {}", muz);
	logger->info("[Twiss] gamma  = {}, gammat = {}", gamma, gammat);
	logger->info("[Twiss] sigmaz = {}, dp     = {}", sigmaz, dp);
}


void Twiss::execute(int turn) {

	callCuda(cudaEventRecord(simTime.start, 0));
	float time_tmp = 0;

	auto logger = spdlog::get("logger");

	if ("drift" == logitudinal_transfer || "matrix" == logitudinal_transfer) {

		//logger->debug("[Twiss] turn = {}, start running of : {}, s = {}, 6D (logi = {})", turn, name, s, logitudinal_transfer);

		transfer_matrix_6D << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np, circumference,
			m11_x, m12_x, m21_x, m22_x,
			m11_y, m12_y, m21_y, m22_y,
			m11_z, m12_z, m21_z, m22_z);
	}
	else
	{
		//logger->debug("[Twiss] turn = {}, start running of : {}, s = {}, 4D (logi = {})", turn, name, s, logitudinal_transfer);

		transfer_matrix_4D << <block_x, thread_x, 0, 0 >> > (dev_bunch, Np,
			m11_x, m12_x, m21_x, m22_x,
			m11_y, m12_y, m21_y, m22_y);
	}

	//callCuda(cudaDeviceSynchronize());
	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.twiss += time_tmp;
}


__global__ void transfer_matrix_4D(Particle* dev_bunch, int Np,
	double m11_x, double m12_x, double m21_x, double m22_x,
	double m11_y, double m12_y, double m21_y, double m22_y) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double x1 = 0, px1 = 0, y1 = 0, py1 = 0;

	while (tid < Np)
	{
		x1 = dev_bunch[tid].x;
		px1 = dev_bunch[tid].px;

		y1 = dev_bunch[tid].y;
		py1 = dev_bunch[tid].py;

		dev_bunch[tid].x = x1 * m11_x + px1 * m12_x;
		dev_bunch[tid].px = x1 * m21_x + px1 * m22_x;

		dev_bunch[tid].y = y1 * m11_y + py1 * m12_y;
		dev_bunch[tid].py = y1 * m21_y + py1 * m22_y;

		//if (tid == 0)
		//{
		//	printf("dev_bunch[%d]: x = %.10f\n", tid, dev_bunch[tid].x);
		//}

		tid += stride;
	}
}


__global__ void transfer_matrix_6D(Particle* dev_bunch, int Np, double circumference,
	double m11_x, double m12_x, double m21_x, double m22_x,
	double m11_y, double m12_y, double m21_y, double m22_y,
	double m11_z, double m12_z, double m21_z, double m22_z) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double x1 = 0, px1 = 0, y1 = 0, py1 = 0, z1 = 0, pz1 = 0;

	while (tid < Np)
	{
		x1 = dev_bunch[tid].x;
		px1 = dev_bunch[tid].px;

		y1 = dev_bunch[tid].y;
		py1 = dev_bunch[tid].py;

		z1 = dev_bunch[tid].z;
		pz1 = dev_bunch[tid].pz;

		dev_bunch[tid].x = x1 * m11_x + px1 * m12_x;
		dev_bunch[tid].px = x1 * m21_x + px1 * m22_x;

		dev_bunch[tid].y = y1 * m11_y + py1 * m12_y;
		dev_bunch[tid].py = y1 * m21_y + py1 * m22_y;

		dev_bunch[tid].z = z1 * m11_z + pz1 * m12_z;
		dev_bunch[tid].pz = z1 * m21_z + pz1 * m22_z;

		if (dev_bunch[tid].z > circumference / 2) {
			dev_bunch[tid].z -= circumference;
		}
		else if (dev_bunch[tid].z < -circumference / 2)
		{
			dev_bunch[tid].z += circumference;
		}

		//if (tid == 0)
		//{
		//	printf("dev_bunch[%d]: z1 = %.10f, z2 = %.10f, m11_z = %.5f, m12_z = %.5f\n", tid, z1, dev_bunch[tid].z, m11_z, m12_z);
		//}

		tid += stride;
	}
}
