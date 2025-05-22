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

	int Np = 0;
	double circumference = 0;

	double drift_length = 0;

	int thread_x = 0;
	int block_x = 0;
};


class SBendElement :public Element
{
public:
	SBendElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~SBendElement() = default;

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[SBend Element] print");
	}
private:
	Particle* dev_bunch = nullptr;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	int Np = 0;
	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool isFieldError = false;

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

	~RBendElement() = default;

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[RBend Element] print");
	}
private:
	Particle* dev_bunch = nullptr;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	int Np = 0;
	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool isFieldError = false;

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

	~QuadrupoleElement() = default;

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[Quadrupole Element] print");
	}
private:
	Particle* dev_bunch = nullptr;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	int Np = 0;
	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool isFieldError = false;

	double l = 0;
	double drift_length = 0;

	double k1 = 0;	// in unit of (m^-2)
	double k1s = 0;	// in unit of (m^-2)

};


class SextupoleElement :public Element
{
public:
	SextupoleElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~SextupoleElement() = default;

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[Sextupole Element] print");
	}
private:
	Particle* dev_bunch = nullptr;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	int Np = 0;
	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool isFieldError = false;

	double l = 0;
	double drift_length = 0;
	double k2 = 0;	// in unit of (m^-3)
	double k2s = 0;	// in unit of (m^-3)

};


class OctupoleElement :public Element
{
public:
	OctupoleElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~OctupoleElement() = default;

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[Octupole Element] print");
	}
private:
	Particle* dev_bunch = nullptr;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	int Np = 0;
	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool isFieldError = false;

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

	~HKickerElement() = default;

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[Hor. Kicker Element] print");
	}
private:
	Particle* dev_bunch = nullptr;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	int Np = 0;
	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool isFieldError = false;

	double l = 0;
	double drift_length = 0;
	double kick = 0;

};


class VKickerElement :public Element
{
public:
	VKickerElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~VKickerElement() = default;

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[Ver. Kicker Element] print");
	}
private:
	Particle* dev_bunch = nullptr;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	int Np = 0;
	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool isFieldError = false;

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

	int Np = 0;
	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	//double l = 0;
	//double drift_length = 0;

	double ratio = 0;	// e*q/m

	double radius = 0;

	std::vector<std::string> filenames;
	int Nrf = 0;

	std::vector<std::vector<RFData>> host_rf_data;

	RFData* dev_rf_data = nullptr;
	size_t pitch_rf = 0;
	size_t Nturn_rf = 0;


};

__global__ void transfer_drift(Particle* dev_bunch, int Np,
	double beta, double gamma, double drift_length);

__global__ void transfer_dipole_full(Particle* dev_bunch, int Np, double beta,
	double r11, double r12, double r16, double r21, double r22, double r26,
	double r34, double r51, double r52, double r56,
	double fl21i, double fl43i, double fr21i, double fr43i);

__global__ void transfer_dipole_half_left(Particle* dev_bunch, int Np, double beta,
	double r11, double r12, double r16, double r21, double r22, double r26,
	double r34, double r51, double r52, double r56,
	double fl21i, double fl43i, double fr21i, double fr43i);

__global__ void transfer_dipole_half_right(Particle* dev_bunch, int Np, double beta,
	double r11, double r12, double r16, double r21, double r22, double r26,
	double r34, double r51, double r52, double r56,
	double fl21i, double fl43i, double fr21i, double fr43i);

__global__ void transfer_quadrupole_norm(Particle* dev_bunch, int Np, double beta,
	double k1, double l);

__global__ void transfer_quadrupole_skew(Particle* dev_bunch, int Np, double beta,
	double k1s, double l);

__global__ void transfer_sextupole_norm(Particle* dev_bunch, int Np, double beta,
	double k2, double l);

__global__ void transfer_sextupole_skew(Particle* dev_bunch, int Np, double beta,
	double k2s, double l);

__global__ void transfer_octupole_norm(Particle* dev_bunch, int Np, double beta,
	double k2, double l);

__global__ void transfer_octupole_skew(Particle* dev_bunch, int Np, double beta,
	double k2s, double l);

__global__ void transfer_hkicker(Particle* dev_bunch, int Np, double beta,
	double kick);

__global__ void transfer_vkicker(Particle* dev_bunch, int Np, double beta,
	double kick);

__global__ void transfer_rf(Particle* dev_bunch, int Np, int turn, double beta0, double beta1, double gamma0, double gamma1,
	RFData* dev_rf_data, size_t  pitch_rf, int Nrf, size_t Nturn_rf,
	double radius, double ratio, double dE_syn, double eta1, double E_total1);

__device__ void convert_z_dp_to_theta_dE(double z, double dp, double& theta, double& dE, double radius, double beta);

__device__ void convert_theta_dE_to_z_dp(double& z, double& dp, double theta, double dE, double radius, double beta);

std::vector<RFData> readRFDataFromCSV(const std::string& filename);