#include "pic.h"
#include "particle.h"
#include "cutSlice.h"

#include "amgx_c.h"

#include <fstream>
#include <cudss.h>


FieldSolverType string2Enum(const std::string& str) {

	// 转换为小写（使匹配不区分大小写）
	std::string lowerStr = str;
	std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(),
		[](unsigned char c) { return std::tolower(c); });

	// 建立映射表
	static const std::map<std::string, FieldSolverType> solverMap = {
		{"pic_conv",   FieldSolverType::PIC_Conv},
		{"pic_fd_cudss",FieldSolverType::PIC_FD_CUDSS},
		{"pic_fd_amgx",   FieldSolverType::PIC_FD_AMGX},
		{"pic_fd_fft",   FieldSolverType::PIC_FD_FFT},
		{"eq_quasi_static",   FieldSolverType::Eq_Quasi_Static},
		{"eq_frozen",   FieldSolverType::Eq_Frozen}
	};

	// 查找并返回枚举值
	auto it = solverMap.find(lowerStr);
	if (it != solverMap.end()) {
		return it->second;
	}
	else {
		spdlog::get("logger")->error("[FieldSolver] Invalid solver type: {}", str);
		std::exit(EXIT_FAILURE);
	}
}


FieldSolver::FieldSolver(int input_Nx, int input_Ny, double input_Lx, double input_Ly, int input_Nslice,
	FieldSolverType input_solverType, std::shared_ptr<Aperture> input_aperture)
	: Nx(input_Nx), Ny(input_Ny), Lx(input_Lx), Ly(input_Ly), Nslice(input_Nslice),
	solverType(input_solverType), aperture(input_aperture)
{
	if (!aperture) {
		spdlog::get("logger")->error("[FieldSolver] Aperture is null in FieldSolver constructor");
		exit(EXIT_FAILURE);
	}

	cudaMalloc((void**)&dev_charDensity, Nx * Ny * Nslice * sizeof(double));
	cudaMalloc((void**)&dev_potential, Nx * Ny * Nslice * sizeof(double));
	cudaMalloc((void**)&dev_electicField, Nx * Ny * Nslice * sizeof(double));
	cudaMalloc((void**)&dev_meshMask, Nx * Ny * sizeof(double));
}


FieldSolver::~FieldSolver() {

	if (dev_charDensity) cudaFree(dev_charDensity);
	if (dev_potential) cudaFree(dev_potential);
	if (dev_electicField) cudaFree(dev_electicField);
	if (dev_meshMask) cudaFree(dev_meshMask);
}


bool FieldSolver::operator==(const FieldSolver& other) const
{
	if (Nx != other.Nx || Ny != other.Ny || fabs(Lx - other.Lx) > 1e-10 || fabs(Ly - other.Ly) > 1e-10 || solverType != other.solverType)
	{
		return false;
	}

	return aperture->is_equal(other.aperture.get());
}


bool FieldSolver::operator!=(const FieldSolver& other) const {
	return !(*this == other);
}


FieldSolverCUDSS::FieldSolverCUDSS(int input_Nx, int input_Ny, double input_Lx, double input_Ly, int input_Nslice,
	std::shared_ptr<Aperture> input_aperture)
	: FieldSolver(input_Nx, input_Ny, input_Lx, input_Ly, input_Nslice, FieldSolverType::PIC_FD_CUDSS, input_aperture)
{

}


FieldSolverCUDSS::~FieldSolverCUDSS()
{

}


void FieldSolverCUDSS::initialize() {

	spdlog::get("logger")->info("[FieldSolver] Start initializing CUDSS solver ...");

	//ncol = Nx - 2;	// Number of columns in the grid, excluding the boundary points
	//nrow = Ny - 2;	// Number of rows in the grid, excluding the boundary points
	//n = nrow * ncol;	// Matrix dimension, A is a square matrix of size n x n, without condidering the boundary points
	//nnz											// Number of non-zero elements in the sparse matrix A
	//	= 4 * 3										// 3-point per grid point on the sub-boundary corner
	//	+ (ncol - 2) * 4 * 2 + (nrow - 2) * 4 * 2	// 4-point per grid point on the sub-boundary edge
	//	+ (ncol - 2) * (nrow - 2) * 5;				// 5-point per grid point
	//nrhs = Nslice;

	ncol = Nx;	// Number of columns in the grid, including the boundary points
	nrow = Ny;	// Number of rows in the grid, including the boundary points
	n = nrow * ncol;	// Matrix dimension, A is a square matrix of size n x n, condidering the boundary points
	nnz
		= (2 * (ncol + nrow) - 4) * 1				// 1-point per grid point on all boundary points
		+ 4 * 3										// 3-point per grid point on the sub-boundary corner
		+ (ncol - 4) * 4 * 2 + (nrow - 4) * 4 * 2	// 4-point per grid point on the sub-boundary edge
		+ (ncol - 4) * (nrow - 4) * 5;				// 5-point per grid point
	nrhs = Nslice;

	int* csr_offsets_h = nullptr;
	int* csr_columns_h = nullptr;
	double* csr_values_h = nullptr;
	double* A_values_h = nullptr;

	// Allocate host memory for the sparse input matrix A, right-hand side x and solution b
	csr_offsets_h = (int*)malloc((n + 1) * sizeof(int));
	csr_columns_h = (int*)malloc(nnz * sizeof(int));
	csr_values_h = (double*)malloc(nnz * sizeof(double));
	A_values_h = (double*)malloc(n * n * sizeof(double));

	if (!csr_offsets_h || !csr_columns_h || !csr_values_h || !A_values_h) {
		spdlog::get("logger")->error("[FieldSolver] Memory allocation failed for CSR format.");
		std::exit(EXIT_FAILURE);
	}

	// Allocate device memory for A, x and b
	callCuda(cudaMalloc((void**)&csr_offsets_d, (n + 1) * sizeof(int)));
	callCuda(cudaMalloc((void**)&csr_columns_d, nnz * sizeof(int)));
	callCuda(cudaMalloc((void**)&csr_values_d, nnz * sizeof(double)));

	// Initialize host memory for A
	//generate_5points_FD_matrix_exclude_boundary(Nx, Ny, Lx, Ly, A_values_h);
	generate_5points_FD_matrix_include_boundary(Nx, Ny, Lx, Ly, A_values_h);
	convert_2d_matrix_to_CSR(n, n, nnz, A_values_h, csr_offsets_h, csr_columns_h, csr_values_h);

	// Copy host memory to device for A, and memset x, b to zero
	callCuda(cudaMemcpy(csr_offsets_d, csr_offsets_h, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
	callCuda(cudaMemcpy(csr_columns_d, csr_columns_h, nnz * sizeof(int), cudaMemcpyHostToDevice));
	callCuda(cudaMemcpy(csr_values_d, csr_values_h, nnz * sizeof(double), cudaMemcpyHostToDevice));

	// Create cuDSS configuration and handle
	callCudss(cudssCreate(&cudss_handle));
	callCudss(cudssConfigCreate(&cudss_config));
	callCudss(cudssDataCreate(cudss_handle, &cudss_data));

	cudssMatrixType_t mtype = CUDSS_MTYPE_GENERAL;
	cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;
	cudssIndexBase_t base = CUDSS_BASE_ZERO;

	callCudss(cudssMatrixCreateCsr(&cudss_A, n, n, nnz, csr_offsets_d, NULL, csr_columns_d, csr_values_d, CUDA_R_32I, CUDA_R_64F, mtype, mview, base));

	int ldb = n;	// Leading dimension of b
	int ldx = n;	// Leading dimension of x

	callCudss(cudssMatrixCreateDn(&cudss_x, n, nrhs, ldx, dev_potential, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
	callCudss(cudssMatrixCreateDn(&cudss_b, n, nrhs, ldb, dev_charDensity, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));

	// Symbolic factorization
	callCudss(cudssExecute(cudss_handle, CUDSS_PHASE_ANALYSIS, cudss_config, cudss_data, cudss_A, NULL, NULL));

	// Factorization
	callCudss(cudssExecute(cudss_handle, CUDSS_PHASE_FACTORIZATION, cudss_config, cudss_data, cudss_A, NULL, NULL));

	free(csr_offsets_h);
	free(csr_columns_h);
	free(csr_values_h);
	free(A_values_h);

	spdlog::get("logger")->info("[FieldSolver] CUDSS solver initialized successfully.");

}


void FieldSolverCUDSS::finalize() {

	callCuda(cudaFree(csr_offsets_d));
	callCuda(cudaFree(csr_columns_d));
	callCuda(cudaFree(csr_values_d));

	callCudss(cudssMatrixDestroy(cudss_A));
	callCudss(cudssMatrixDestroy(cudss_x));
	callCudss(cudssMatrixDestroy(cudss_b));
	callCudss(cudssDataDestroy(cudss_handle, cudss_data));
	callCudss(cudssConfigDestroy(cudss_config));
	callCudss(cudssDestroy(cudss_handle));

}


void FieldSolverCUDSS::solve_x_values() {

	callCudss(cudssExecute(cudss_handle, CUDSS_PHASE_SOLVE, cudss_config, cudss_data, cudss_A, cudss_x, cudss_b));

	// Output x value and check it
	//double* host_potential = (double*)malloc(Nx * Ny * Nslice * sizeof(double));
	//callCuda(cudaMemcpy(host_potential, dev_potential, Nx * Ny * Nslice * sizeof(double), cudaMemcpyDeviceToHost));
	//cudaDeviceSynchronize();

	//std::filesystem::path matrix_savepath = "D:/PASS/test/potential.csv";
	//std::ofstream file(matrix_savepath);

	//for (int i = 0; i < Nx * Ny; i++)
	//{
	//	for (int j = 0; j < Nslice; j++)
	//	{
	//		file << std::setprecision(15)
	//			<< host_potential[i + j * Nx * Ny];
	//		if (j < (Nslice - 1))
	//		{
	//			file << ",";
	//		}
	//	}
	//	file << "\n";
	//}
	//file.close();

	//spdlog::get("logger")->info("[FieldSolver] func(solve_x_values): potential matrix data has been writted to {}", matrix_savepath.string());

	//free(host_potential);
}


void FieldSolverCUDSS::update_b_values(const Particle* dev_bunch, const Slice* dev_slice, int Np_sur, double charge, int thread_x, int block_x) {

	callKernel(allocate2grid_multi_slice << <block_x, thread_x, 0, 0 >> > (dev_bunch, dev_charDensity, dev_slice, Np_sur, Nslice, Nx, Ny, Lx, Ly, charge));

	// Output b value and check it
	//double* host_charDensity = (double*)malloc(Nx * Ny * Nslice * sizeof(double));
	//callCuda(cudaMemcpy(host_charDensity, dev_charDensity, Nx * Ny * Nslice * sizeof(double), cudaMemcpyDeviceToHost));
	//cudaDeviceSynchronize();

	//std::filesystem::path matrix_savepath = "D:/PASS/test/charDensity.csv";
	//std::ofstream file(matrix_savepath);

	//for (int i = 0; i < Nx * Ny; i++)
	//{
	//	for (int j = 0; j < Nslice; j++)
	//	{
	//		file << std::setprecision(15)
	//			<< host_charDensity[i + j * Nx * Ny];
	//		if (j < (Nslice - 1))
	//		{
	//			file << ",";
	//		}
	//	}
	//	file << "\n";
	//}
	//file.close();

	//spdlog::get("logger")->info("[FieldSolver] func(update_b_values): charge density matrix data has been writted to {}", matrix_savepath.string());

	//free(host_charDensity);
}


void FieldSolverCUDSS::calculate_electricField() {


}


FieldSolverAMGX::FieldSolverAMGX(int input_Nx, int input_Ny, double input_Lx, double input_Ly, int input_Nslice,
	std::shared_ptr<Aperture> input_aperture)
	: FieldSolver(input_Nx, input_Ny, input_Lx, input_Ly, input_Nslice, FieldSolverType::PIC_FD_AMGX, input_aperture)
{
	// PIC_FD_AMGX: related parameters, if not using PIC_FD_AMGX, these parameters will be ignored
	// Solve Ax = b
	//AMGX_config_create(&amgx_config, "");
}


FieldSolverAMGX::~FieldSolverAMGX() {
	//if (amgx_solver) AMGX_solver_destroy(amgx_solver);
	//if (amgx_vector_y) AMGX_vector_destroy(amgx_vector_y);
	//if (amgx_vector_x) AMGX_vector_destroy(amgx_vector_x);
	//if (amgx_matrix) AMGX_matrix_destroy(amgx_matrix);
	//if (amgx_resources) AMGX_resources_destroy(amgx_resources);
	//if (amgx_config) AMGX_config_destroy(amgx_config);
}


void FieldSolverAMGX::initialize() {


}


void FieldSolverAMGX::finalize() {


}


void FieldSolverAMGX::solve_x_values() {


}


void FieldSolverAMGX::update_b_values(const Particle* dev_bunch, const Slice* dev_slice, int Np_sur, double charge, int thread_x, int block_x) {


}


void FieldSolverAMGX::calculate_electricField() {


}


__global__ void allocate2grid_multi_slice(const Particle* dev_bunch, double* dev_charDensity, const Slice* dev_slice, int Np_sur, int Nslice, int Nx, int Ny, double Lx, double Ly, double charge) {

	// Allocate charges to grid and calculate the rho/epsilon

	const double epsilon = PassConstant::epsilon0;

	int x_index = 0, y_index = 0;
	double dx = 0, dy = 0;
	double grid_area = Lx * Ly;
	double  LB = 0, RB = 0, LT = 0, RT = 0;		//Left bottom, Right bottom, Left top, Right top

	double factor = charge * 1 / (grid_area * epsilon);

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	const double xmax = (Nx - 1) / 2.0 * Lx;
	const double xmin = -xmax;
	const double ymax = (Ny - 1) / 2.0 * Ly;
	const double ymin = -ymax;

	while (tid < Np_sur)
	{
		double x = dev_bunch[tid].x;
		double y = dev_bunch[tid].y;
		int tag = dev_bunch[tid].tag;

		// x is in [xmin, xmax)，y is in [ymin, ymax). When x=xmax, the index of RD and RT are Nx, which causes the problem of out-of-bounds memory access. So is y.
		int inBoundary = (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax) & (tag > 0);

		unsigned mask = __ballot_sync(0xFFFFFFFF, inBoundary);
		if (!mask) return;

		if (inBoundary)
		{
			x_index = floor((x - xmin) / Lx);	// x index of the left bottom grid point
			y_index = floor((y - ymin) / Ly);	// y index of the left bottom grid point

			dx = x - (xmin + x_index * Lx);	// distance to the left grid line
			dy = y - (ymin + y_index * Ly);	// distance to the bottom grid line

			LB = (1 - dx / Lx) * (1 - dy / Ly) * factor;
			RB = (dx / Lx) * (1 - dy / Ly) * factor;
			LT = (1 - dx / Lx) * (dy / Ly) * factor;
			RT = (dx / Lx) * (dy / Ly) * factor;

			int slice_index = find_slice_index(dev_slice, Nslice, tid);
			int base = slice_index * Nx * Ny;

			int LB_index = base + y_index * Nx + x_index;
			int RB_index = base + y_index * Nx + x_index + 1;
			int LT_index = base + (y_index + 1) * Nx + x_index;
			int RT_index = base + (y_index + 1) * Nx + x_index + 1;

			atomicAdd(&(dev_charDensity[LB_index]), LB);
			atomicAdd(&(dev_charDensity[RB_index]), RB);
			atomicAdd(&(dev_charDensity[LT_index]), LT);
			atomicAdd(&(dev_charDensity[RT_index]), RT);

		}

		tid += stride;
	}
}


void generate_5points_FD_matrix_exclude_boundary(int Nx, int Ny, double Lx, double Ly, double* host_matrix) {

	// Generate a 5-point finite difference matrix for Poisson's equation in 2D

	int nrow = Ny - 2;	// Number of rows in the grid, excluding the boundary points
	int ncol = Nx - 2;	// Number of columns in the grid, excluding the boundary points
	int n = nrow * ncol;	// Matrix dimension, A is a square matrix of size n x n, without condidering the boundary points
	double a = 1.0 / (Lx * Lx);
	double b = 1.0 / (Ly * Ly);
	double c = -2.0 * (a + b); // Coefficient for the diagonal elements

	int row, col;	// Index of the grid point in the mesh, not in the matrix A
	int gridId;		// Serial number of the grid point in the mesh, row-major order
	int index;		// Index of value in the matrix A

	// Initialize the host matrix to zero
	for (int i = 0; i < n * n; i++) {
		host_matrix[i] = 0.0;
	}

	// Fill the matrix with the finite difference coefficients

	// Fill the matrix for the boundary corner points
	row = 0, col = 0;	// Left Bottom vertice
	gridId = row * ncol + col;
	index = gridId * (ncol * nrow) + gridId;
	host_matrix[index] = c;	// Diagonal element
	host_matrix[index + 1] = a;	// Right neighbor
	host_matrix[index + ncol] = b;	// Up neighbor

	row = 0, col = ncol - 1;	// Right Bottom vertice
	gridId = row * ncol + col;
	index = gridId * (ncol * nrow) + gridId;
	host_matrix[index] = c;	// Diagonal element
	host_matrix[index - 1] = a;	// Left neighbor
	host_matrix[index + ncol] = b;	// Up neighbor

	row = nrow - 1, col = 0;	// Left Top vertice
	gridId = row * ncol + col;
	index = gridId * (ncol * nrow) + gridId;
	host_matrix[index] = c;	// Diagonal element
	host_matrix[index + 1] = a;	// Right neighbor
	host_matrix[index - ncol] = b;	// Down neighbor

	row = nrow - 1, col = ncol - 1;	// Right Top vertice
	gridId = row * ncol + col;
	index = gridId * (ncol * nrow) + gridId;
	host_matrix[index] = c;	// Diagonal element
	host_matrix[index - 1] = a;	// Left neighbor
	host_matrix[index - ncol] = b;	// Down neighbor

	// Fill the matrix for the boundary edge points
	std::vector<std::pair<int, int>>boundary_top;
	std::vector<std::pair<int, int>>boundary_bottom;
	std::vector<std::pair<int, int>>boundary_left;
	std::vector<std::pair<int, int>>boundary_right;

	for (int i = 1; i < (ncol - 1); i++) {
		boundary_top.push_back({ nrow - 1, i });	// Top boundary
		boundary_bottom.push_back({ 0, i });	// Bottom boundary
	}
	for (int i = 1; i < (nrow - 1); i++)
	{
		boundary_left.push_back({ i, 0 });	// Left boundary
		boundary_right.push_back({ i, ncol - 1 });	// Right boundary
	}

	for (const auto& point : boundary_top) {
		row = point.first;
		col = point.second;
		gridId = row * ncol + col;
		index = gridId * (ncol * nrow) + gridId;
		host_matrix[index] = c;	// Diagonal element
		host_matrix[index - 1] = a;	// Left neighbor
		host_matrix[index + 1] = a;	// Right neighbor
		host_matrix[index - ncol] = b;	// Down neighbor
	}

	for (const auto& point : boundary_bottom) {
		row = point.first;
		col = point.second;
		gridId = row * ncol + col;
		index = gridId * (ncol * nrow) + gridId;
		host_matrix[index] = c;	// Diagonal element
		host_matrix[index - 1] = a;	// Left neighbor
		host_matrix[index + 1] = a;	// Right neighbor
		host_matrix[index + ncol] = b;	// Up neighbor
	}

	for (const auto& point : boundary_left) {
		row = point.first;
		col = point.second;
		gridId = row * ncol + col;
		index = gridId * (ncol * nrow) + gridId;
		host_matrix[index] = c;	// Diagonal element
		host_matrix[index + 1] = a;	// Right neighbor
		host_matrix[index - ncol] = b;	// Down neighbor
		host_matrix[index + ncol] = b;	// Up neighbor
	}

	for (const auto& point : boundary_right) {
		row = point.first;
		col = point.second;
		gridId = row * ncol + col;
		index = gridId * (ncol * nrow) + gridId;
		host_matrix[index] = c;	// Diagonal element
		host_matrix[index - 1] = a;	// Left neighbor
		host_matrix[index - ncol] = b;	// Down neighbor
		host_matrix[index + ncol] = b;	// Up neighbor
	}

	// Fill the matrix for the inner points
	std::vector<std::pair<int, int>>inner_points;

	for (int i = 1; i < (nrow - 1); i++)
	{
		for (int j = 1; j < (ncol - 1); j++)
		{
			inner_points.push_back({ i, j });	// Inner points
		}
	}

	for (const auto& point : inner_points) {
		row = point.first;
		col = point.second;
		gridId = row * ncol + col;
		index = gridId * (ncol * nrow) + gridId;
		host_matrix[index] = c;	// Diagonal element
		host_matrix[index - 1] = a;	// Left neighbor
		host_matrix[index + 1] = a;	// Right neighbor
		host_matrix[index - ncol] = b;	// Down neighbor
		host_matrix[index + ncol] = b;	// Up neighbor
	}

	// Output matrix value and check it
	std::filesystem::path matrix_savepath = "D:/PASS/test/fd_matrix.csv";
	std::ofstream file(matrix_savepath);

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			file << host_matrix[i * n + j];
			if (j < (n - 1))
			{
				file << ",";
			}
		}
		file << "\n";
	}
	file.close();

	spdlog::get("logger")->info("[FieldSolver] func(generate_5points_FD_matrix): 5-points FD matrix data has been writted to {}", matrix_savepath.string());

}


void generate_5points_FD_matrix_include_boundary(int Nx, int Ny, double Lx, double Ly, double* host_matrix) {

	// Generate a 5-point finite difference matrix for Poisson's equation in 2D

	int nrow = Ny;	// Number of rows in the grid, including the boundary points
	int ncol = Nx;	// Number of columns in the grid, including the boundary points
	int n = nrow * ncol;	// Matrix dimension, A is a square matrix of size n x n, condidering the boundary points
	double a = 1.0 / (Lx * Lx);
	double b = 1.0 / (Ly * Ly);
	double c = -2.0 * (a + b); // Coefficient for the diagonal elements

	int row, col;	// Index of the grid point in the mesh, not in the matrix A
	int gridId;		// Serial number of the grid point in the mesh, row-major order
	int index;		// Index of value in the matrix A

	// Initialize the host matrix to zero
	for (int i = 0; i < n * n; i++) {
		host_matrix[i] = 0.0;
	}

	// Fill the matrix with the finite difference coefficients

	// Fill the matrix for the boundary corner points
	row = 0, col = 0;	// Left Bottom vertice
	gridId = row * ncol + col;
	index = gridId * (ncol * nrow) + gridId;
	host_matrix[index] = c;	// Diagonal element

	row = 0, col = ncol - 1;	// Right Bottom vertice
	gridId = row * ncol + col;
	index = gridId * (ncol * nrow) + gridId;
	host_matrix[index] = c;	// Diagonal element

	row = nrow - 1, col = 0;	// Left Top vertice
	gridId = row * ncol + col;
	index = gridId * (ncol * nrow) + gridId;
	host_matrix[index] = c;	// Diagonal element

	row = nrow - 1, col = ncol - 1;	// Right Top vertice
	gridId = row * ncol + col;
	index = gridId * (ncol * nrow) + gridId;
	host_matrix[index] = c;	// Diagonal element

	// Fill the matrix for the boundary edge points
	std::vector<std::pair<int, int>>boundary_top;
	std::vector<std::pair<int, int>>boundary_bottom;
	std::vector<std::pair<int, int>>boundary_left;
	std::vector<std::pair<int, int>>boundary_right;

	for (int i = 1; i < (ncol - 1); i++) {
		boundary_top.push_back({ nrow - 1, i });	// Top boundary
		boundary_bottom.push_back({ 0, i });	// Bottom boundary
	}
	for (int i = 1; i < (nrow - 1); i++)
	{
		boundary_left.push_back({ i, 0 });	// Left boundary
		boundary_right.push_back({ i, ncol - 1 });	// Right boundary
	}

	for (const auto& point : boundary_top) {
		row = point.first;
		col = point.second;
		gridId = row * ncol + col;
		index = gridId * (ncol * nrow) + gridId;
		host_matrix[index] = c;	// Diagonal element
	}

	for (const auto& point : boundary_bottom) {
		row = point.first;
		col = point.second;
		gridId = row * ncol + col;
		index = gridId * (ncol * nrow) + gridId;
		host_matrix[index] = c;	// Diagonal element
	}

	for (const auto& point : boundary_left) {
		row = point.first;
		col = point.second;
		gridId = row * ncol + col;
		index = gridId * (ncol * nrow) + gridId;
		host_matrix[index] = c;	// Diagonal element
	}

	for (const auto& point : boundary_right) {
		row = point.first;
		col = point.second;
		gridId = row * ncol + col;
		index = gridId * (ncol * nrow) + gridId;
		host_matrix[index] = c;	// Diagonal element
	}

	// Fill the matrix for the sub-boundary corner points
	row = 1, col = 1;	// Left Bottom vertice
	gridId = row * ncol + col;
	index = gridId * (ncol * nrow) + gridId;
	host_matrix[index] = c;	// Diagonal element
	host_matrix[index + 1] = a;	// Right neighbor
	host_matrix[index + ncol] = b;	// Up neighbor

	row = 1, col = ncol - 2;	// Right Bottom vertice
	gridId = row * ncol + col;
	index = gridId * (ncol * nrow) + gridId;
	host_matrix[index] = c;	// Diagonal element
	host_matrix[index - 1] = a;	// Left neighbor
	host_matrix[index + ncol] = b;	// Up neighbor

	row = nrow - 2, col = 1;	// Left Top vertice
	gridId = row * ncol + col;
	index = gridId * (ncol * nrow) + gridId;
	host_matrix[index] = c;	// Diagonal element
	host_matrix[index + 1] = a;	// Right neighbor
	host_matrix[index - ncol] = b;	// Down neighbor

	row = nrow - 2, col = ncol - 2;	// Right Top vertice
	gridId = row * ncol + col;
	index = gridId * (ncol * nrow) + gridId;
	host_matrix[index] = c;	// Diagonal element
	host_matrix[index - 1] = a;	// Left neighbor
	host_matrix[index - ncol] = b;	// Down neighbor

	// Fill the matrix for the sub-boundary edge points
	std::vector<std::pair<int, int>>sub_boundary_top;
	std::vector<std::pair<int, int>>sub_boundary_bottom;
	std::vector<std::pair<int, int>>sub_boundary_left;
	std::vector<std::pair<int, int>>sub_boundary_right;

	for (int i = 2; i < (ncol - 2); i++) {
		sub_boundary_top.push_back({ nrow - 2, i });	// Top boundary
		sub_boundary_bottom.push_back({ 1, i });	// Bottom boundary
	}
	for (int i = 2; i < (nrow - 2); i++)
	{
		sub_boundary_left.push_back({ i, 1 });	// Left boundary
		sub_boundary_right.push_back({ i, ncol - 2 });	// Right boundary
	}

	for (const auto& point : sub_boundary_top) {
		row = point.first;
		col = point.second;
		gridId = row * ncol + col;
		index = gridId * (ncol * nrow) + gridId;
		host_matrix[index] = c;	// Diagonal element
		host_matrix[index - 1] = a;	// Left neighbor
		host_matrix[index + 1] = a;	// Right neighbor
		host_matrix[index - ncol] = b;	// Down neighbor
	}

	for (const auto& point : sub_boundary_bottom) {
		row = point.first;
		col = point.second;
		gridId = row * ncol + col;
		index = gridId * (ncol * nrow) + gridId;
		host_matrix[index] = c;	// Diagonal element
		host_matrix[index - 1] = a;	// Left neighbor
		host_matrix[index + 1] = a;	// Right neighbor
		host_matrix[index + ncol] = b;	// Up neighbor
	}

	for (const auto& point : sub_boundary_left) {
		row = point.first;
		col = point.second;
		gridId = row * ncol + col;
		index = gridId * (ncol * nrow) + gridId;
		host_matrix[index] = c;	// Diagonal element
		host_matrix[index + 1] = a;	// Right neighbor
		host_matrix[index - ncol] = b;	// Down neighbor
		host_matrix[index + ncol] = b;	// Up neighbor
	}

	for (const auto& point : sub_boundary_right) {
		row = point.first;
		col = point.second;
		gridId = row * ncol + col;
		index = gridId * (ncol * nrow) + gridId;
		host_matrix[index] = c;	// Diagonal element
		host_matrix[index - 1] = a;	// Left neighbor
		host_matrix[index - ncol] = b;	// Down neighbor
		host_matrix[index + ncol] = b;	// Up neighbor
	}

	// Fill the matrix for the inner points
	std::vector<std::pair<int, int>>inner_points;

	for (int i = 2; i < (nrow - 2); i++)
	{
		for (int j = 2; j < (ncol - 2); j++)
		{
			inner_points.push_back({ i, j });	// Inner points
		}
	}

	for (const auto& point : inner_points) {
		row = point.first;
		col = point.second;
		gridId = row * ncol + col;
		index = gridId * (ncol * nrow) + gridId;
		host_matrix[index] = c;	// Diagonal element
		host_matrix[index - 1] = a;	// Left neighbor
		host_matrix[index + 1] = a;	// Right neighbor
		host_matrix[index - ncol] = b;	// Down neighbor
		host_matrix[index + ncol] = b;	// Up neighbor
	}

	// Output matrix value and check it
	//std::filesystem::path matrix_savepath = "D:/PASS/test/fd_matrix.csv";
	//std::ofstream file(matrix_savepath);

	//for (int i = 0; i < n; i++)
	//{
	//	for (int j = 0; j < n; j++)
	//	{
	//		file << host_matrix[i * n + j];
	//		if (j < (n - 1))
	//		{
	//			file << ",";
	//		}
	//	}
	//	file << "\n";
	//}
	//file.close();

	//spdlog::get("logger")->info("[FieldSolver] func(generate_5points_FD_matrix): 5-points FD matrix data has been writted to {}", matrix_savepath.string());

}


void convert_2d_matrix_to_CSR(int nrow, int ncol, int nnz, double* matrix, int* csr_row_ptr, int* csr_col_indices, double* csr_values) {

	// Initialize row_ptr value
	for (int i = 0; i < (nrow + 1); i++)
	{
		csr_row_ptr[i] = 0;
	}

	// Count the number of non-zero elements in each row
	for (int i = 0; i < nrow; i++)
	{
		for (int j = 0; j < ncol; j++)
		{
			double value = matrix[i * ncol + j];

			if (fabs(value) > 1e-10) {
				csr_row_ptr[i + 1]++;
			}
		}
	}

	// Convert the count to cumulative sum
	for (int i = 0; i < nrow; i++)
	{
		csr_row_ptr[i + 1] += csr_row_ptr[i];
	}

	// Fill col_indices and values arrays
	for (int i = 0; i < nrow; i++)
	{
		int start = csr_row_ptr[i];
		int count = 0;

		for (int j = 0; j < ncol; j++)
		{
			double value = matrix[i * ncol + j];

			if (fabs(value) > 1e-10) {
				int pos = start + count;
				csr_col_indices[pos] = j;	// Column index
				csr_values[pos] = value;	// Value at (i, j)
				count++;
			}
		}
	}

	// Output CSR matrix value and check it
	//std::filesystem::path csr_matrix_savepath = "D:/PASS/test/fd_matrix_csr.csv";
	//std::ofstream file(csr_matrix_savepath);

	//for (int i = 0; i < (nrow + 1); i++)
	//{
	//	file << csr_row_ptr[i];
	//	if (i < nrow)
	//	{
	//		file << ",";
	//	}
	//}
	//file << "\n";

	//for (int i = 0; i < nnz; i++)
	//{
	//	file << csr_col_indices[i];
	//	if (i < (nnz - 1))
	//	{
	//		file << ",";
	//	}
	//}
	//file << "\n";

	//for (int i = 0; i < nnz; i++)
	//{
	//	file << csr_values[i];
	//	if (i < (nnz - 1))
	//	{
	//		file << ",";
	//	}
	//}
	//file << "\n";

	//file.close();

	//spdlog::get("logger")->info("[FieldSolver] func(convert_2d_matrix_to_CSR): CSR format matrix data has been writted to {}", csr_matrix_savepath.string());

	//spdlog::get("logger")->info("[FieldSolver] func(convert_2d_denseMatrix_to_CSR): 2D dense matrix has been converted to CSR format.");
}