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
	MarkerElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name, const ParallelPlan1d& plan1d, TimeEvent& timeevent);

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

	double s_previous = 0;

	int Np = 0;
	double circumference = 0;

	double gamma = 0;
	double beta = 0;

	double muz = 0;
	double muz_previous = 0;

	double m12 = 0, m34 = 0, m56 = 0;	// transfer matrix elements;

	int thread_x = 0;
	int block_x = 0;

};


class SBendElement :public Element
{
public:
	SBendElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name);

	~SBendElement() = default;

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[SBend Element] print");
	}
private:
	Particle* dev_bunch = nullptr;

	int Np = 0;

	double l = 0;
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
	RBendElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name);

	~RBendElement() = default;

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[RBend Element] print");
	}
private:
	Particle* dev_bunch = nullptr;

	int Np = 0;

	double l = 0;
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
	QuadrupoleElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name);

	~QuadrupoleElement() = default;

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[Quadrupole Element] print");
	}
private:
	Particle* dev_bunch = nullptr;

	int Np = 0;

	double l = 0;
	double k1 = 0;	// in unit of (m^-2)

};


class SextupoleElement :public Element
{
public:
	SextupoleElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name);

	~SextupoleElement() = default;

	void execute(int turn) override;

	void print() override {
		auto logger = spdlog::get("logger");
		logger->info("[Sextupole Element] print");
	}
private:
	Particle* dev_bunch = nullptr;

	int Np = 0;

	double l = 0;
	double k2 = 0;	// in unit of (m^-3)

};


__global__ void transfer_drift(Particle* dev_bunch, int Np,
	double m12, double m34, double m56, double beta);