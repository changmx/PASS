#pragma once

#include <typeinfo>

#include "command.h"
#include "particle.h"
#include "parameter.h"
#include "general.h"
#include "parallelPlan.h"


class Element
{
public:

	virtual ~Element() = default;

	double s = -1;
	std::string commandType = "Element";
	std::string name = "ElementObj";

	virtual void execute(int turn) = 0;
	virtual void print() = 0;

};


class MarkerElement :public Element
{
public:
	MarkerElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~MarkerElement() = default;

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[Makrer Element] print");
	}
private:
	Particle* dev_bunch = nullptr;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	double circumference = 0;

	double drift_length = 0;

	int thread_x = 0;
	int block_x = 0;

	bool isAperture = 0;	// Whether to calculate the particle loss caused by the aperture
};


class SBendElement :public Element
{
public:
	SBendElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~SBendElement() {
		callCuda(cudaFree(dev_kn));
		callCuda(cudaFree(dev_ks));
	}

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[SBend Element] print");
	}
private:
	Particle* dev_bunch = nullptr;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool isFieldError = false;
	const int max_error_order = 20;	// k0, k1 ... k20
	double* dev_kn = nullptr;
	double* dev_ks = nullptr;

	double l = 0;
	double drift_length = 0;
	double angle = 0;	// in unit of rad
	double e1 = 0;
	double e2 = 0;
	double hgap = 0;	// in unit of m
	double fint = 0;
	double fintx = 0;

};


class RBendElement :public Element
{
public:
	RBendElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~RBendElement() {
		callCuda(cudaFree(dev_kn));
		callCuda(cudaFree(dev_ks));
	}

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[RBend Element] print");
	}
private:
	Particle* dev_bunch = nullptr;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool isFieldError = false;
	const int max_error_order = 20;	// k0, k1 ... k20
	double* dev_kn = nullptr;
	double* dev_ks = nullptr;

	double l = 0;
	double drift_length = 0;
	double angle = 0;	// in unit of rad
	double e1 = 0;
	double e2 = 0;
	double hgap = 0;	// in unit of m
	double fint = 0;
	double fintx = 0;

};


class QuadrupoleElement :public Element
{
public:
	QuadrupoleElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~QuadrupoleElement() {
		callCuda(cudaFree(dev_kn));
		callCuda(cudaFree(dev_ks));
	}

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[Quadrupole Element] print");
	}
private:
	Particle* dev_bunch = nullptr;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool isFieldError = false;
	const int max_error_order = 20;	// k0, k1 ... k20
	double* dev_kn = nullptr;
	double* dev_ks = nullptr;

	double l = 0;
	double drift_length = 0;

	double k1 = 0;	// in unit of (m^-2)
	double k1s = 0;	// in unit of (m^-2)

};


class SextupoleNormElement :public Element
{
public:
	SextupoleNormElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~SextupoleNormElement() {
		callCuda(cudaFree(dev_kn));
		callCuda(cudaFree(dev_ks));
	}

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[Sextupole Normal Element] print");
	}
private:
	Particle* dev_bunch = nullptr;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool isFieldError = false;
	const int max_error_order = 20;	// k0, k1 ... k20
	double* dev_kn = nullptr;
	double* dev_ks = nullptr;

	double l = 0;
	double drift_length = 0;
	double k2 = 0;	// in unit of (m^-3)

	bool is_ignore_length = false;	// If true, the length of the sextupole is ignored in the transfer map calculation. Mainly used for Twiss transfer and the sextupole is regarded as a thin lens.

	bool is_ramping = false;
	std::vector<std::pair<double, double>> ramping_data;	// (turn, k2) pairs for ramping, ramping_data[i].first = turn, ramping_data[i].second = k2

};


class SextupoleSkewElement :public Element
{
public:
	SextupoleSkewElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~SextupoleSkewElement() {
		callCuda(cudaFree(dev_kn));
		callCuda(cudaFree(dev_ks));
	}

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[Sextupole Skew Element] print");
	}
private:
	Particle* dev_bunch = nullptr;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool isFieldError = false;
	const int max_error_order = 20;	// k0, k1 ... k20
	double* dev_kn = nullptr;
	double* dev_ks = nullptr;

	double l = 0;
	double drift_length = 0;
	double k2s = 0;	// in unit of (m^-3)

	bool is_ignore_length = false;	// If true, the length of the sextupole is ignored in the transfer map calculation. Mainly used for Twiss transfer and the sextupole is regarded as a thin lens.

	bool is_ramping = false;
	std::vector<std::pair<double, double>> ramping_data;	// (turn, k2s) pairs for ramping, ramping_data[i].first = turn, ramping_data[i].second = k2s

};


class OctupoleElement :public Element
{
public:
	OctupoleElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~OctupoleElement() {
		callCuda(cudaFree(dev_kn));
		callCuda(cudaFree(dev_ks));
	}

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[Octupole Element] print");
	}
private:
	Particle* dev_bunch = nullptr;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool isFieldError = false;
	const int max_error_order = 20;	// k0, k1 ... k20
	double* dev_kn = nullptr;
	double* dev_ks = nullptr;

	double l = 0;
	double drift_length = 0;
	double k3 = 0;	// in unit of (m^-4)
	double k3s = 0;	// in unit of (m^-4)

};


class HKickerElement :public Element
{
public:
	HKickerElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~HKickerElement() {
		callCuda(cudaFree(dev_kn));
		callCuda(cudaFree(dev_ks));
	}

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[Hor. Kicker Element] print");
	}
private:
	Particle* dev_bunch = nullptr;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool isFieldError = false;
	const int max_error_order = 20;	// k0, k1 ... k20
	double* dev_kn = nullptr;
	double* dev_ks = nullptr;

	double l = 0;
	double drift_length = 0;
	double kick = 0;

};


class VKickerElement :public Element
{
public:
	VKickerElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~VKickerElement() {
		callCuda(cudaFree(dev_kn));
		callCuda(cudaFree(dev_ks));
	}

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[Ver. Kicker Element] print");
	}
private:
	Particle* dev_bunch = nullptr;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool isFieldError = false;
	const int max_error_order = 20;	// k0, k1 ... k20
	double* dev_kn = nullptr;
	double* dev_ks = nullptr;

	double l = 0;
	double drift_length = 0;
	double kick = 0;

};


struct RFData {
	double harmonic;
	double voltage;
	double phis;
	double phi_offset;
};


class RFElement :public Element
{
public:
	RFElement(const Parameter& para, int input_beamId, Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~RFElement() {
		callCuda(cudaFree(dev_rf_data));
	}

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[RF Element] print");
	}
private:
	Particle* dev_bunch = nullptr;
	TimeEvent& simTime;
	Bunch& bunchRef;

	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	//double l = 0;
	//double drift_length = 0;

	double qm_ratio = 0;	// q/m

	double radius = 0;

	std::vector<std::string> filenames;
	int Nrf = 0;

	std::vector<std::vector<RFData>> host_rf_data;

	RFData* dev_rf_data = nullptr;
	size_t pitch_rf = 0;
	size_t Nturn_rf = 0;


};


class ElSeparatorElement :public Element
{
public:
	ElSeparatorElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~ElSeparatorElement() {
		callCuda(cudaFree(dev_counter));
		callCuda(cudaFree(dev_particle_Es));
	}

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[Electrostatic Separator Element] print");
	}
private:
	Particle* dev_bunch = nullptr;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	int Np = 0;
	int Nturn = 0;
	int bunchId = 0;
	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	double Ex = 0, Ey = 0;

	double l = 0;
	double drift_length = 0;

	int* dev_counter = 0;

	double ES_hor_position = 0;	// The horizontal position of the separator relative to the center of the beam pipe, in unit of m

	Particle* dev_particle_Es = nullptr;	// Device memory for the particles transfered into the electrostatic separator
	
	std::filesystem::path saveDir;
	std::string saveName_part;
};


__global__ void transfer_drift(Particle* dev_bunch, int Np_sur, double beta, double circumference,
	double gamma, double drift_length);

__global__ void transfer_dipole_full(Particle* dev_bunch, int Np_sur, double beta, double circumference,
	double r11, double r12, double r16, double r21, double r22, double r26,
	double r34, double r51, double r52, double r56,
	double fl21i, double fl43i, double fr21i, double fr43i);

__global__ void transfer_dipole_half_left(Particle* dev_bunch, int Np_sur, double beta, double circumference,
	double r11, double r12, double r16, double r21, double r22, double r26,
	double r34, double r51, double r52, double r56,
	double fl21i, double fl43i, double fr21i, double fr43i);

__global__ void transfer_dipole_half_right(Particle* dev_bunch, int Np_sur, double beta, double circumference,
	double r11, double r12, double r16, double r21, double r22, double r26,
	double r34, double r51, double r52, double r56,
	double fl21i, double fl43i, double fr21i, double fr43i);

__global__ void transfer_quadrupole_norm(Particle* dev_bunch, int Np_sur, double beta, double circumference,
	double k1, double l);

__global__ void transfer_quadrupole_skew(Particle* dev_bunch, int Np_sur, double beta, double circumference,
	double k1s, double l);

__global__ void transfer_sextupole_norm(Particle* dev_bunch, int Np_sur, double beta,
	double k2, double l);

__global__ void transfer_sextupole_skew(Particle* dev_bunch, int Np_sur, double beta,
	double k2s, double l);

__global__ void transfer_octupole_norm(Particle* dev_bunch, int Np_sur, double beta,
	double k2, double l);

__global__ void transfer_octupole_skew(Particle* dev_bunch, int Np_sur, double beta,
	double k2s, double l);

__global__ void transfer_hkicker(Particle* dev_bunch, int Np_sur, double beta,
	double kick);

__global__ void transfer_vkicker(Particle* dev_bunch, int Np_sur, double beta,
	double kick);

__global__ void transfer_rf(Particle* dev_bunch, int Np_sur, int turn, double beta0, double beta1, double gamma0, double gamma1,
	RFData* dev_rf_data, size_t  pitch_rf, int Nrf, size_t Nturn_rf,
	double radius, double ratio, double dE_syn, double eta1, double E_total1);

__device__ void convert_z_dp_to_theta_dE(double z, double dp, double& theta, double& dE, double radius, double beta);

__device__ void convert_theta_dE_to_z_dp(double& z, double& dp, double theta, double dE, double radius, double beta);

__global__ void transfer_multipole_kicker(Particle* dev_bunch, int Np_sur, int order, const double* dev_kn, const double* dev_ks, double l);

std::vector<RFData> readRFDataFromCSV(const std::string& filename);

std::vector<std::pair<double, double>> readSextRampingDataFromCSV(const std::string& filename);

__global__ void check_particle_in_ElSeparator(Particle* dev_bunch, int Np_sur, Particle* dev_particle_ES, double ES_hor_position, int* global_counter, double s, int turn);