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
	Particle dev_particle;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	double circumference = 0;

	double drift_length = 0;	// Drift length between the head of the current element and the tail of the previous element

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
		callCuda(cudaFree(dev_knl));
		callCuda(cudaFree(dev_ksl));
	}

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[SBend Element] print");
	}
private:
	Particle dev_particle;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool isFieldError = false;
	const int max_error_order = 20;	// k0, k1 ... k20
	int cur_error_order = -1;	// -1 means no error. 0 refers to dipole field error, 1 refers to quad. field error , etc.
	double* dev_knl = nullptr;
	double* dev_ksl = nullptr;

	double l = 0;
	double drift_length = 0;	// Drift length between the head of the current element and the tail of the previous element
	double angle = 0;	// in unit of rad, 1/rho=k0, k0l=l/rho=angle
	double e1 = 0;
	double e2 = 0;
	double hgap = 0;	// in unit of m
	double fint = 0;
	double fintx = 0;

	// If the actural turn is outside the range of the turns in ramping data, kl will be set to the last value !!!
	// Therefore, please make sure the ramping data cover all the simulation turns.
	bool is_ramping = false;
	std::vector<double> ramping_k0l;	// for the i-th turn (turn start from 1, not 0), kl = ramping_kl[turn-1]
};


class RBendElement :public Element
{
public:
	RBendElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~RBendElement() {
		callCuda(cudaFree(dev_knl));
		callCuda(cudaFree(dev_ksl));
	}

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[RBend Element] print");
	}
private:
	Particle dev_particle;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool isFieldError = false;
	const int max_error_order = 20;	// k0, k1 ... k20
	int cur_error_order = -1;	// -1 means no error. 0 refers to dipole field error, 1 refers to quad. field error , etc.
	double* dev_knl = nullptr;
	double* dev_ksl = nullptr;

	double l = 0;
	double drift_length = 0;	// Drift length between the head of the current element and the tail of the previous element
	double angle = 0;	// in unit of rad, 1/rho=k0, k0l=l/rho=angle
	double e1 = 0;
	double e2 = 0;
	double hgap = 0;	// in unit of m
	double fint = 0;
	double fintx = 0;

	// If the actural turn is outside the range of the turns in ramping data, kl will be set to the last value !!!
	// Therefore, please make sure the ramping data cover all the simulation turns.
	bool is_ramping = false;
	std::vector<double> ramping_k0l;	// for the i-th turn (turn start from 1, not 0), kl = ramping_kl[turn-1]
};


class QuadrupoleElement :public Element
{
public:
	QuadrupoleElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~QuadrupoleElement() {
		callCuda(cudaFree(dev_knl));
		callCuda(cudaFree(dev_ksl));
	}

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[Quadrupole Element] print");
	}
private:
	Particle dev_particle;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool isFieldError = false;
	const int max_error_order = 20;	// k0, k1 ... k20
	int cur_error_order = -1;	// -1 means no error. 0 refers to dipole field error, 1 refers to quad. field error, etc.
	double* dev_knl = nullptr;
	double* dev_ksl = nullptr;

	double l = 0;
	double drift_length = 0;	// Drift length between the head of the current element and the tail of the previous element

	std::string quad_type = "drift";	// normal or skew or drift

	double k1l = 0;	// in unit of (m^-1)
	double k1sl = 0;	// in unit of (m^-1)

	// If the actural turn is outside the range of the turns in ramping data, kl will be set to the last value !!!
	// Therefore, please make sure the ramping data cover all the simulation turns.
	bool is_ramping = false;
	std::vector<double> ramping_k1l;	// for the i-th turn (turn start from 1, not 0), kl = ramping_kl[turn-1]
	std::vector<double> ramping_k1sl;	// for the i-th turn (turn start from 1, not 0), ksl = ramping_ksl[turn-1]
};


class SextupoleElement :public Element
{
public:
	SextupoleElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~SextupoleElement() {
		callCuda(cudaFree(dev_knl));
		callCuda(cudaFree(dev_ksl));
	}

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[Sextupole Element] print");
	}
private:
	Particle dev_particle;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool isFieldError = false;
	const int max_error_order = 20;	// k0, k1 ... k20
	int cur_error_order = -1;	// -1 means no error. 0 refers to dipole field error, 1 refers to quad. field error , etc.
	double* dev_knl = nullptr;
	double* dev_ksl = nullptr;

	double l = 0;
	double drift_length = 0;	// Drift length between the head of the current element and the tail of the previous element

	std::string sext_type = "drift";	// normal or skew or drift

	double k2l = 0;	// in unit of (m^-2)
	double k2sl = 0;	// in unit of (m^-2)

	// If the actural turn is outside the range of the turns in ramping data, kl will be set to the last value !!!
	// Therefore, please make sure the ramping data cover all the simulation turns.
	bool is_ramping = false;
	std::vector<double> ramping_k2l;	// for the i-th turn (turn start from 1, not 0), kl = ramping_kl[turn-1]
	std::vector<double> ramping_k2sl;	// for the i-th turn (turn start from 1, not 0), kl = ramping_kl[turn-1]
};


class OctupoleElement :public Element
{
public:
	OctupoleElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~OctupoleElement() {
		callCuda(cudaFree(dev_knl));
		callCuda(cudaFree(dev_ksl));
	}

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[Octupole Element] print");
	}
private:
	Particle dev_particle;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool isFieldError = false;
	const int max_error_order = 20;	// k0, k1 ... k20
	int cur_error_order = -1;	// -1 means no error. 0 refers to dipole field error, 1 refers to quad. field error , etc.
	double* dev_knl = nullptr;
	double* dev_ksl = nullptr;

	double l = 0;
	double drift_length = 0;
	double k3 = 0;	// in unit of (m^-4)
	double k3s = 0;	// in unit of (m^-4)

	bool is_thin_lens = false;	// If true, the length of the octupole is ignored in the transfer map calculation. Mainly used for Twiss transfer and the sextupole is regarded as a thin lens.

};


class HKickerElement :public Element
{
public:
	HKickerElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~HKickerElement() {
		callCuda(cudaFree(dev_knl));
		callCuda(cudaFree(dev_ksl));
	}

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[Hor. Kicker Element] print");
	}
private:
	Particle dev_particle;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool isFieldError = false;
	const int max_error_order = 20;	// k0, k1 ... k20
	int cur_error_order = -1;	// -1 means no error. 0 refers to dipole field error, 1 refers to quad. field error , etc.
	double* dev_knl = nullptr;
	double* dev_ksl = nullptr;

	double l = 0;
	double drift_length = 0;
	bool is_thin_lens = false;

	double kick = 0;

};


class VKickerElement :public Element
{
public:
	VKickerElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~VKickerElement() {
		callCuda(cudaFree(dev_knl));
		callCuda(cudaFree(dev_ksl));
	}

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[Ver. Kicker Element] print");
	}
private:
	Particle dev_particle;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	bool isFieldError = false;
	const int max_error_order = 20;	// k0, k1 ... k20
	int cur_error_order = -1;	// -1 means no error. 0 refers to dipole field error, 1 refers to quad. field error , etc.
	double* dev_knl = nullptr;
	double* dev_ksl = nullptr;

	double l = 0;
	double drift_length = 0;
	bool is_thin_lens = false;

	double kick = 0;

};


class MultipoleElement :public Element
{
public:
	MultipoleElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~MultipoleElement() {
		callCuda(cudaFree(dev_knl));
		callCuda(cudaFree(dev_ksl));
	}

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[Multipole Element] print");
	}
private:
	Particle dev_particle;
	TimeEvent& simTime;
	const Bunch& bunchRef;

	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	const int max_error_order = 20;	// k0, k1 ... k20
	int cur_error_order = -1;	// -1 means no multipole filed. 0 refers to dipole field, 1 refers to quad. field, etc.
	double* dev_knl = nullptr;
	double* dev_ksl = nullptr;

	double l = 0;
	double drift_length = 0;
	bool is_thin_lens = false;
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
	Particle dev_particle;
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

	double pz_aperture_lower = -1.0;
	double pz_aperture_upper = +1.0;

};


class ElSeparatorElement :public Element
{
public:
	ElSeparatorElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~ElSeparatorElement() {
		callCuda(cudaFree(dev_counter));
		dev_particle_Es.mem_free_gpu();
	}

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[Electrostatic Separator Element] print");
	}
private:
	Particle dev_particle;
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

	Particle dev_particle_Es;	// Device memory for the particles transfered into the electrostatic separator

	std::filesystem::path saveDir;
	std::string saveName_part;
};


struct TuneExciterData {
	double frequency;
	double voltage;
};


class TuneExciterElement :public Element
{
public:
	TuneExciterElement(const Parameter& para, int input_beamId, Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~TuneExciterElement() {
		//callCuda(cudaFree(dev_tuneExciter_data));
	}

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[TuneExciter Element] print");
	}
private:
	Particle dev_particle;
	TimeEvent& simTime;
	Bunch& bunchRef;

	double circumference = 0;

	int thread_x = 0;
	int block_x = 0;

	double kick_angle = 0;
	double freq_center = 0;
	double scan_period = 0;
	double scan_freq_range = 0;
	int turn_start = 0;
	int turn_end = 0;
	std::string kick_direction = "empty";	// "x" means horizontal, "y" means vertical

	//double l = 0;
	//double drift_length = 0;

	//double exciter_length = 0;
	//double exciter_gap = 0;
	//int exciter_status_x = 0;	// 0 means off, 1 means on
	//int exciter_status_y = 0;	// 0 means off, 1 means on

	//std::string filename;
	//size_t Nturn_tuneExciter = 0;

	//std::vector<TuneExciterData> host_tuneExciter_data;

	//TuneExciterData* dev_tuneExciter_data = nullptr;

};


__global__ void transfer_drift(Particle dev_particle, int Np_sur, double beta, double circumference,
	double gamma, double drift_length);

__global__ void transfer_dipole_full(Particle dev_particle, int Np_sur, double beta, double circumference,
	double r11, double r12, double r16, double r21, double r22, double r26,
	double r34, double r51, double r52, double r56,
	double fl21i, double fl43i, double fr21i, double fr43i);

__global__ void transfer_dipole_half_left(Particle dev_particle, int Np_sur, double beta, double circumference,
	double r11, double r12, double r16, double r21, double r22, double r26,
	double r34, double r51, double r52, double r56,
	double fl21i, double fl43i, double fr21i, double fr43i);

__global__ void transfer_dipole_half_right(Particle dev_particle, int Np_sur, double beta, double circumference,
	double r11, double r12, double r16, double r21, double r22, double r26,
	double r34, double r51, double r52, double r56,
	double fl21i, double fl43i, double fr21i, double fr43i);

__global__ void transfer_quadrupole_thicklens_norm(Particle dev_particle, int Np_sur, double beta, double circumference,
	double k1, double l);

__global__ void transfer_quadrupole_thicklens_skew(Particle dev_particle, int Np_sur, double beta, double circumference,
	double k1s, double l);

__global__ void transfer_quadrupole_thinlens_norm(Particle dev_particle, int Np_sur, double k1l);

__global__ void transfer_quadrupole_thinlens_skew(Particle dev_particle, int Np_sur, double k1sl);

__global__ void transfer_sextupole_thinlens_norm(Particle dev_particle, int Np_sur, double k2l);

__global__ void transfer_sextupole_thinlens_skew(Particle dev_particle, int Np_sur, double k2sl);

__global__ void transfer_octupole_norm(Particle dev_particle, int Np_sur, double beta,
	double k2, double l);

__global__ void transfer_octupole_skew(Particle dev_particle, int Np_sur, double beta,
	double k2s, double l);

__global__ void transfer_hkicker(Particle dev_particle, int Np_sur, double beta,
	double kick);

__global__ void transfer_vkicker(Particle dev_particle, int Np_sur, double beta,
	double kick);

__global__ void transfer_rf(Particle dev_particle, int Np_sur, int turn, double s, double beta0, double beta1, double gamma0, double gamma1,
	RFData* dev_rf_data, size_t  pitch_rf, int Nrf, size_t Nturn_rf,
	double radius, double ratio, double dE_syn, double eta1, double E_total1,
	double pz_min, double pz_max);

__device__ void convert_z_dp_to_theta_dE(double z, double dp, double& theta, double& dE, double radius, double beta, double Es);

__device__ void convert_theta_dE_to_z_dp(double& z, double& dp, double theta, double dE, double radius, double beta, double Es);

__global__ void transfer_multipole_kicker(Particle dev_particle, int Np_sur, int order, const double* dev_knl, const double* dev_ksl);

std::vector<RFData> readRFDataFromCSV(const std::string& filename);

std::vector<std::pair<double, double>> readSextRampingDataFromCSV(const std::string& filename);

std::vector<double> readRampingDataFromCSV(const std::string& filename);

__global__ void check_particle_in_ElSeparator(Particle dev_particle, int Np_sur, Particle dev_particle_ES, double ES_hor_position, int* global_counter, double s, int turn);

std::vector<TuneExciterData> readTuneExciterDataFromCSV(const std::string& filename);

__global__ void transfer_tuneExciter_x(Particle dev_particle, int Np_sur, double kick_angle, double t0, double beta,
	double freq_center, double scan_period, double scan_freq_range);

__global__ void transfer_tuneExciter_y(Particle dev_particle, int Np_sur, double kick_angle, double t0, double beta,
	double freq_center, double scan_period, double scan_freq_range);