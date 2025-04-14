#pragma once

#include "command.h"
#include "particle.h"
#include "parameter.h"

class Element
{
public:

	virtual ~Element() = default;

	double s = -1;
	std::string name = "Element";

	virtual void run(int turn) = 0;
	virtual void print() = 0;

};


template<typename ElementType>
class ElementCommand :public Command {
public:
	~ElementCommand() {};
	explicit ElementCommand(ElementType* ele) {
		element = ele;
		s = ele->s;
		name = ele->name;
	}
	void execute(int turn) override {
		element->run(turn);
	}
private:
	ElementType* element = nullptr;
};


class SBendElement :public Element
{
public:
	SBendElement(const Parameter& para, int input_beamId, const Bunch& Bunch, std::string obj_name);

	~SBendElement() = default;

	void run(int turn) override;

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

	void run(int turn) override;

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

	void run(int turn) override;

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

	void run(int turn) override;

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
