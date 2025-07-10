#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "parameter.h"
#include "constant.h"
#include "command.h"
#include "aperture.h"
#include "parallelPlan.h"
#include "general.h"
#include "cutSlice.h"

#include "amgx_c.h"

class Particle;	// Forward declaration of Particle class
class Bunch;	// Forward declaration of Bunch class


enum class FieldSolverMethod
{
	PIC_FD_AMGX,	// Using finite difference method with AMGX solver, any aperture
	PIC_conv,	// Using Green function with open boundary condition, only rectangle aperture
	PIC_FD_FFT,	// Using finite difference method with FFT solver, only rectangle aperture
	Eq_quasi_static,	// Quasi-static equation solver, using the B-E formula, each calculation is based on the current sigmax, sigmay, any aperture
	Eq_frozen	// Frozen equation solver, using th B-E formula with unchanged sigmax and sigmay, any aperture
};


class PicConfig
{
public:
	PicConfig() = default;
	PicConfig(int input_Nx, int input_Ny, double input_Lx, double input_Ly, int input_Nslice, FieldSolverMethod input_solverMethod, std::unique_ptr<Aperture> input_aperture)
		: Nx(input_Nx), Ny(input_Ny), Lx(input_Lx), Ly(input_Ly), Nslice(input_Nslice), solverMethod(input_solverMethod), aperture(std::move(input_aperture))
	{
		if (!aperture) {
			spdlog::get("logger")->error("Aperture is null in PicConfig constructor, please check your code.");
			exit(EXIT_FAILURE);
		}

		if (input_solverMethod == FieldSolverMethod::PIC_FD_AMGX)
		{
			callCuda(cudaMalloc((void**)&dev_charDensity, Nx * Ny * Nslice * sizeof(double)));
			callCuda(cudaMalloc((void**)&dev_potential, Nx * Ny * Nslice * sizeof(double)));
			callCuda(cudaMalloc((void**)&dev_electicField, Nx * Ny * Nslice * sizeof(double)));

			// AMGX create

			callCuda(cudaMalloc((void**)&dev_meshMask, Nx * Ny * sizeof(double)));
		}
	}

	int Nx = 0;	// Number of grid points in x direction, Nx = Ngrid_x + 1
	int Ny = 0;
	double Lx = 0;	// Length of one grid in x direction (m), mesh size in x direction = Lx * (Nx - 1)
	double Ly = 0;
	int Nslice = 0;	// Number of bunch slices

	FieldSolverMethod solverMethod;	// Field solver method

	std::unique_ptr<Aperture> aperture;	// Aperture for PIC boundary condition and particle loss

	double* dev_charDensity = nullptr;
	double* dev_potential = nullptr;
	double* dev_electicField = nullptr;

	// PIC_FD_AMGX: related parameters, if not using PIC_FD_AMGX, these parameters will be ignored
	// Solve Mx = y
	AMGX_config_handle amgx_config = nullptr;	// AMGX configuration handle
	AMGX_resources_handle amgx_resources = nullptr;	// AMGX resources handle
	AMGX_matrix_handle amgx_matrix = nullptr;	// AMGX matrix handle
	AMGX_vector_handle amgx_vector_x = nullptr;	// AMGX left vector handle
	AMGX_vector_handle amgx_vector_y = nullptr;	// AMGX right vector handle
	AMGX_solver_handle amgx_solver = nullptr;	// AMGX solver handle
	double* dev_meshMask = nullptr;	// Device memory for mesh mask, used to mark the aperture position


	// PIC_FD_FFT: related parameters, if not using PIC_FD_FFT, these parameters will be ignored


	bool operator==(const PicConfig& other) const
	{
		if (Nx != other.Nx || Ny != other.Ny || fabs(Lx - other.Lx) > 1e-10 || fabs(Ly - other.Ly) > 1e-10 || solverMethod != other.solverMethod)
		{
			return false;
		}

		return aperture->is_equal(other.aperture.get());
	}

	bool operator!=(const PicConfig& other) const {
		return !(*this == other);
	}

	~PicConfig() {

		//if (aperture) delete aperture;	// 智能指针自动管理释放

		if (dev_charDensity) callCuda(cudaFree(dev_charDensity));
		if (dev_potential) callCuda(cudaFree(dev_potential));
		if (dev_electicField) callCuda(cudaFree(dev_electicField));

		if (amgx_solver) AMGX_solver_destroy(amgx_solver);
		if (amgx_vector_y) AMGX_vector_destroy(amgx_vector_y);
		if (amgx_vector_x) AMGX_vector_destroy(amgx_vector_x);
		if (amgx_matrix) AMGX_matrix_destroy(amgx_matrix);
		if (amgx_resources) AMGX_resources_destroy(amgx_resources);
		if (amgx_config) AMGX_config_destroy(amgx_config);
		if (dev_meshMask) callCuda(cudaFree(dev_meshMask));

	}

private:

};


class PicSolver_AMGX
{
public:
	PicSolver_AMGX(const Parameter& para, int input_beamId, Bunch& Bunch, std::string obj_name,
		const ParallelPlan1d& plan1d, TimeEvent& timeevent);

	~PicSolver_AMGX() = default;


private:

	Particle* dev_bunch = nullptr;
	TimeEvent& simTime;
	Bunch& bunchRef;

	std::shared_ptr<PicConfig> picConfig;
};


__global__ void allocate2grid_multi_slice(const Particle* dev_bunch, double* dev_charDensity, const Slice* dev_slice, int Np_sur, int Nslice, int Nx, int Ny, double Lx, double Ly, double charge);