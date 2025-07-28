#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cudss.h>

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


enum class FieldSolverType
{
	PIC_Conv,	// Using Green function with open boundary condition, only rectangle aperture
	PIC_FD_CUDSS,	// Using finite difference method with CUDSS solver, any aperture
	PIC_FD_AMGX,	// Using finite difference method with AMGX solver, any aperture
	PIC_FD_FFT,	// Using finite difference method with FFT solver, only rectangle aperture
	Eq_Quasi_Static,	// Quasi-static equation solver, using the B-E formula, each calculation is based on the current sigmax, sigmay, any aperture
	Eq_Frozen	// Frozen equation solver, using th B-E formula with unchanged sigmax and sigmay, any aperture
};


FieldSolverType  string2Enum(const std::string& str);


class MeshMask
{
public:
	MeshMask() = default;
	__host__ __device__ ~MeshMask() {};

	int mask_grid = 0;	// 1: grid is in aperture, 0: grid is out of aperture
	double inv_hx_left = 0, inv_hx_right = 0; // Inverse of distance between the grid point to neighbor grid points, in unit of m
	double inv_hy_bottom = 0, inv_hy_top = 0;
};


class FieldSolver
{
public:
	virtual ~FieldSolver();

	virtual void initialize() = 0;	// Initialize the field solver, allocate memory, etc.
	virtual void update_b_values(Particle* dev_bunch, const Slice* dev_slice, int Np_sur, double charge, int thread_x, int block_x) = 0;
	virtual void solve_x_values() = 0;
	virtual void calculate_electricField() = 0;

	bool operator==(const FieldSolver& other) const;
	bool operator!=(const FieldSolver& other) const;

protected:
	FieldSolver(int input_Nx, int input_Ny, double input_Lx, double input_Ly, int input_Nslice,
		FieldSolverType input_solverType, std::shared_ptr<Aperture> input_aperture);

	int Nx = 0;	// Number of grid points in x direction, Nx = Ngrid_x + 1
	int Ny = 0;
	double Lx = 0;	// Length of one grid in x direction (m), mesh size in x direction = Lx * (Nx - 1)
	double Ly = 0;
	int Nslice = 0;	// Number of bunch slices
	double charge = 0;	// Charge of macro-particle

	FieldSolverType  solverType;	// Field solver method

	std::shared_ptr<Aperture> aperture;	// Aperture for FieldSolver boundary condition and particle loss

	double* dev_charDensity = nullptr;
	double* dev_potential = nullptr;
	double2* dev_electicField = nullptr;
	MeshMask* dev_meshMask = nullptr;	// Device memory for mesh mask, used to mark the aperture position

private:
	// Disable the copy constructor and assignment operator
	FieldSolver(const FieldSolver&) = delete;
	FieldSolver& operator=(const FieldSolver&) = delete;

};


class FieldSolverCUDSS : public FieldSolver
{
	/*
		Using cuDSS APIs for solving a system of linear algebraic equations with a sparse matrix:
			Ax = b,
		where:
			A is the sparse input matrix,
			b is the (dense) right-hand side vector (or a matrix),
			x is the (dense) solution vector (or a matrix).
		website:
			https://docs.nvidia.com/cuda/cudss/getting_started.html
	*/
public:
	FieldSolverCUDSS(int input_Nx, int input_Ny, double input_Lx, double input_Ly, int input_Nslice,
		std::shared_ptr<Aperture> input_aperture);

	~FieldSolverCUDSS() override;

	void initialize() override;
	void update_b_values(Particle* dev_bunch, const Slice* dev_slice, int Np_sur, double charge, int thread_x, int block_x) override;
	void solve_x_values() override;
	void calculate_electricField() override;

private:
	cudssHandle_t cudss_handle = nullptr;
	cudssConfig_t cudss_config = nullptr;
	cudssData_t cudss_data = nullptr;
	cudssMatrix_t cudss_A = nullptr;
	cudssMatrix_t cudss_x = nullptr;
	cudssMatrix_t cudss_b = nullptr;

	int ncol = 0, nrow = 0;	// Number of rows in the grid, excluding the boundary points
	int n = 0;	// Matrix dimension of A, A is a square matrix of size n x n, without condidering the boundary points
	int nnz = 0;	// Number of non-zero value
	int nrhs = 0;	// Number of right-hand side vector

	int* csr_row_ptr_d = nullptr;
	int* csr_col_indices_d = nullptr;
	double* csr_values_d = nullptr;

};


class FieldSolverAMGX :public FieldSolver
{
	/*
		Using AmgX APIs for solving a system of linear algebraic equations with a sparse matrix:
			Ax = b,
		where:
			A is the sparse input matrix,
			b is the (dense) right-hand side vector (or a matrix),
			x is the (dense) solution vector (or a matrix).
		website:
			https://developer.nvidia.com/amgx
	*/
public:
	FieldSolverAMGX(int input_Nx, int input_Ny, double input_Lx, double input_Ly, int input_Nslice,
		std::shared_ptr<Aperture> input_aperture);

	~FieldSolverAMGX() override;

	void initialize() override;
	void update_b_values(Particle* dev_bunch, const Slice* dev_slice, int Np_sur, double charge, int thread_x, int block_x) override;
	void solve_x_values() override;
	void calculate_electricField() override;

private:
	AMGX_config_handle amgx_config = nullptr;	// AMGX configuration handle
	AMGX_resources_handle amgx_resources = nullptr;	// AMGX resources handle
	AMGX_matrix_handle amgx_matrix = nullptr;	// AMGX matrix handle
	AMGX_vector_handle amgx_vector_x = nullptr;	// AMGX left vector handle
	AMGX_vector_handle amgx_vector_y = nullptr;	// AMGX right vector handle
	AMGX_solver_handle amgx_solver = nullptr;	// AMGX solver handle

};


__global__ void allocate2grid_circle_multi_slice(Particle* dev_bunch, double* dev_charDensity, const Slice* dev_slice, const MeshMask* dev_meshMask,
	int Np_sur, int Nslice, int Nx, int Ny, double Lx, double Ly, double charge, double radius_square);

__global__ void allocate2grid_rectangle_multi_slice(Particle* dev_bunch, double* dev_charDensity, const Slice* dev_slice, const MeshMask* dev_meshMask,
	int Np_sur, int Nslice, int Nx, int Ny, double Lx, double Ly, double charge, double half_width, double half_height);

__global__ void allocate2grid_ellipse_multi_slice(Particle* dev_bunch, double* dev_charDensity, const Slice* dev_slice, const MeshMask* dev_meshMask,
	int Np_sur, int Nslice, int Nx, int Ny, double Lx, double Ly, double charge, double hor_semi_axis, double ver_semi_axis);

__global__ void cal_electricField(double* dev_potential, double2* dev_electricField, const MeshMask* dev_meshMask,
	int Nx, int Ny, int Nslice);

void generate_5points_FD_matrix_exclude_boundary(int Nx, int Ny, double Lx, double Ly, double* host_matrix);
void generate_5points_FD_matrix_include_boundary(int Nx, int Ny, double Lx, double Ly, double* host_matrix);

void convert_2d_matrix_to_CSR(int nrow, int ncol, int nnz, double* matrix, int* csr_row_ptr, int* csr_col_indices, double* csr_values);

void generate_meshMask_and_5points_FD_matrix_include_boundary(MeshMask* host_meshMask, double* host_matrix, int Nx, int Ny, double Lx, double Ly, int& nnz,
	const std::shared_ptr<Aperture>& aperture);

void generate_5points_FD_CSR_matrix_and_meshMask_include_boundary(
	std::vector<int>& csr_row_ptr, std::vector<int>& csr_col_indices, std::vector<double>& csr_values,
	MeshMask* host_meshMask, int Nx, int Ny, double Lx, double Ly, int& nnz, const std::shared_ptr<Aperture>& aperture);