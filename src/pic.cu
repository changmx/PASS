#include "pic.h"
#include "particle.h"
#include "cutSlice.h"
#include "aperture.h"

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

	callCuda(cudaMalloc((void**)&dev_charDensity, Nx * Ny * Nslice * sizeof(double)));
	callCuda(cudaMalloc((void**)&dev_potential, Nx * Ny * Nslice * sizeof(double)));
	callCuda(cudaMalloc((void**)&dev_electicField, Nx * Ny * Nslice * sizeof(double)));
	callCuda(cudaMalloc((void**)&dev_meshMask, Nx * Ny * sizeof(MeshMask)));
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

	//ncol = Nx;	// Number of columns in the grid, including the boundary points
	//nrow = Ny;	// Number of rows in the grid, including the boundary points
	//n = nrow * ncol;	// Matrix dimension, A is a square matrix of size n x n, condidering the boundary points
	//nnz
	//	= (2 * (ncol + nrow) - 4) * 1				// 1-point per grid point on all boundary points
	//	+ 4 * 3										// 3-point per grid point on the sub-boundary corner
	//	+ (ncol - 4) * 4 * 2 + (nrow - 4) * 4 * 2	// 4-point per grid point on the sub-boundary edge
	//	+ (ncol - 4) * (nrow - 4) * 5;				// 5-point per grid point
	//nrhs = Nslice;

	ncol = Nx;	// Number of columns in the grid, including the boundary points
	nrow = Ny;	// Number of rows in the grid, including the boundary points
	n = nrow * ncol;	// Matrix dimension, A is a square matrix of size n x n, condidering the boundary points
	nrhs = Nslice;

	// Allocate host memory for matrix A and meshMask
	double* A_values_h = nullptr;
	MeshMask* host_meshMask = nullptr;

	A_values_h = (double*)malloc(n * n * sizeof(double));
	host_meshMask = (MeshMask*)malloc(Nx * Ny * sizeof(MeshMask));

	if (!A_values_h || !host_meshMask) {
		spdlog::get("logger")->error("[FieldSolver] Memory allocation failed for matrix.");
		std::exit(EXIT_FAILURE);
	}

	// Initialize host memory for A and meshMask
	generate_meshMask_and_5points_FD_matrix_include_boundary(host_meshMask, A_values_h, Nx, Ny, Lx, Ly, nnz, aperture);

	callCuda(cudaMemcpy(dev_meshMask, host_meshMask, Nx * Ny * sizeof(MeshMask), cudaMemcpyHostToDevice));

	spdlog::get("logger")->info("[FieldSolver] nnz: count = {}, formula = {}", nnz,
		(2 * (ncol + nrow) - 4) * 1 + 4 * 3 + (ncol - 4) * 4 * 2 + (nrow - 4) * 4 * 2 + (ncol - 4) * (nrow - 4) * 5);

	// Allocate host memory for the CSR format of matrix A
	int* csr_offsets_h = nullptr;
	int* csr_columns_h = nullptr;
	double* csr_values_h = nullptr;

	csr_offsets_h = (int*)malloc((n + 1) * sizeof(int));
	csr_columns_h = (int*)malloc(nnz * sizeof(int));
	csr_values_h = (double*)malloc(nnz * sizeof(double));

	if (!csr_offsets_h || !csr_columns_h || !csr_values_h) {
		spdlog::get("logger")->error("[FieldSolver] Memory allocation failed for CSR format.");
		std::exit(EXIT_FAILURE);
	}

	// Convert matrix A to CSR format
	convert_2d_matrix_to_CSR(n, n, nnz, A_values_h, csr_offsets_h, csr_columns_h, csr_values_h);

	// Allocate device memory for A, x and b
	callCuda(cudaMalloc((void**)&csr_offsets_d, (n + 1) * sizeof(int)));
	callCuda(cudaMalloc((void**)&csr_columns_d, nnz * sizeof(int)));
	callCuda(cudaMalloc((void**)&csr_values_d, nnz * sizeof(double)));

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

	free(host_meshMask);
	free(A_values_h);
	free(csr_offsets_h);
	free(csr_columns_h);
	free(csr_values_h);

	spdlog::get("logger")->info("[FieldSolver] CUDSS solver initialized successfully.");

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


void FieldSolverCUDSS::update_b_values(Particle* dev_bunch, const Slice* dev_slice, int Np_sur, double charge, int thread_x, int block_x) {

	if (aperture->type == Aperture::CIRCLE)
	{
		CircleAperture* a = static_cast<CircleAperture*>(aperture.get());
		callKernel(allocate2grid_circle_multi_slice << <block_x, thread_x, 0, 0 >> > (dev_bunch, dev_charDensity, dev_slice, dev_meshMask,
			Np_sur, Nslice, Nx, Ny, Lx, Ly, charge, a->radius_square));
	}
	else if (aperture->type == Aperture::RECTANGLE)
	{
		RectangleAperture* a = static_cast<RectangleAperture*>(aperture.get());
		callKernel(allocate2grid_rectangle_multi_slice << <block_x, thread_x, 0, 0 >> > (dev_bunch, dev_charDensity, dev_slice, dev_meshMask,
			Np_sur, Nslice, Nx, Ny, Lx, Ly, charge, a->half_width, a->half_height));
	}
	else if (aperture->type == Aperture::ELLIPSE)
	{
		EllipseAperture* a = static_cast<EllipseAperture*>(aperture.get());
		callKernel(allocate2grid_ellipse_multi_slice << <block_x, thread_x, 0, 0 >> > (dev_bunch, dev_charDensity, dev_slice, dev_meshMask,
			Np_sur, Nslice, Nx, Ny, Lx, Ly, charge, a->hor_semi_axis, a->ver_semi_axis));
	}

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


void FieldSolverAMGX::solve_x_values() {


}


void FieldSolverAMGX::update_b_values(Particle* dev_bunch, const Slice* dev_slice, int Np_sur, double charge, int thread_x, int block_x) {


}


void FieldSolverAMGX::calculate_electricField() {


}


__global__ void allocate2grid_circle_multi_slice(Particle* dev_bunch, double* dev_charDensity, const Slice* dev_slice, const MeshMask* dev_meshMask,
	int Np_sur, int Nslice, int Nx, int Ny, double Lx, double Ly, double charge, double radius_square) {

	// Allocate charges to grid and calculate the rho/epsilon

	const double epsilon = PassConstant::epsilon0;

	const double xmin = -(Nx - 1) / 2.0 * Lx;
	const double ymin = -(Ny - 1) / 2.0 * Ly;

	const double inv_Lx = 1.0 / Lx;
	const double inv_Ly = 1.0 / Ly;
	const double grid_area = Lx * Ly;
	const double factor = charge * 1 / (grid_area * epsilon);

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < Np_sur)
	{
		Particle* p = &dev_bunch[tid];

		double x = p->x;
		double y = p->y;
		int tag = p->tag;

		const double epsilon = 1.0e-10;

		double dist_square = x * x + y * y;
		double diff = dist_square - radius_square;

		int is_inside = (diff < -epsilon);
		int inAperture = (is_inside << 1) - 1;

		int loss_now = (tag > 0) & (inAperture != 1);
		int flip_factor = 1 - 2 * loss_now;
		int skip = (tag < 0) | (loss_now);
		int alive = 1 - skip;
		p->tag *= flip_factor;

		int x_index = floor((x - xmin) / Lx);	// x index of the left bottom grid point
		int y_index = floor((y - ymin) / Ly);	// y index of the left bottom grid point

		double dx = x - (xmin + x_index * Lx);	// distance to the left grid line
		double dy = y - (ymin + y_index * Ly);	// distance to the bottom grid line

		double dx_ratio = dx * inv_Lx;
		double dy_ratio = dy * inv_Ly;

		//double LB = (1 - dx / Lx) * (1 - dy / Ly) * factor;
		//double RB = (dx / Lx) * (1 - dy / Ly) * factor;
		//double LT = (1 - dx / Lx) * (dy / Ly) * factor;
		//double RT = (dx / Lx) * (dy / Ly) * factor;
		double LB = (1 - dx_ratio) * (1 - dy_ratio) * factor;
		double RB = dx_ratio * (1 - dy_ratio) * factor;
		double LT = (1 - dx_ratio) * dy_ratio * factor;
		double RT = dx_ratio * dy_ratio * factor;

		int slice_index = find_slice_index(dev_slice, Nslice, tid);
		int base = slice_index * Nx * Ny;

		int LB_index = y_index * Nx + x_index;
		int RB_index = LB_index + 1;
		int LT_index = (y_index + 1) * Nx + x_index;
		int RT_index = LT_index + 1;

		atomicAdd(&(dev_charDensity[base + LB_index]), LB * alive * dev_meshMask[LB_index].mask_grid);
		atomicAdd(&(dev_charDensity[base + RB_index]), RB * alive * dev_meshMask[RB_index].mask_grid);
		atomicAdd(&(dev_charDensity[base + LT_index]), LT * alive * dev_meshMask[LT_index].mask_grid);
		atomicAdd(&(dev_charDensity[base + RT_index]), RT * alive * dev_meshMask[RT_index].mask_grid);

		tid += stride;
	}
}


__global__ void allocate2grid_rectangle_multi_slice(Particle* dev_bunch, double* dev_charDensity, const Slice* dev_slice, const MeshMask* dev_meshMask,
	int Np_sur, int Nslice, int Nx, int Ny, double Lx, double Ly, double charge, double half_width, double half_height) {

	// Allocate charges to grid and calculate the rho/epsilon

	const double epsilon = PassConstant::epsilon0;

	const double xmin = -(Nx - 1) / 2.0 * Lx;
	const double ymin = -(Ny - 1) / 2.0 * Ly;

	const double inv_Lx = 1.0 / Lx;
	const double inv_Ly = 1.0 / Ly;
	const double grid_area = Lx * Ly;
	const double factor = charge * 1 / (grid_area * epsilon);

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < Np_sur)
	{
		Particle* p = &dev_bunch[tid];

		double x = p->x;
		double y = p->y;
		int tag = p->tag;

		const double epsilon = 1.0e-10;

		double dist_left = -x - half_width;	// if inside, dist_left < 0
		double dist_right = x - half_width;	// if inside, dist_right < 0
		double dist_bottom = -y - half_height;	// if inside, dist_bottom < 0
		double dist_top = y - half_height;	// if inside, dist_top < 0

		int is_inside = (dist_left < -epsilon) & (dist_right < -epsilon) & (dist_bottom < -epsilon) & (dist_top < -epsilon);
		int inAperture = (is_inside << 1) - 1;

		int loss_now = (tag > 0) & (inAperture != 1);
		int flip_factor = 1 - 2 * loss_now;
		int skip = (tag < 0) | (loss_now);
		int alive = 1 - skip;
		p->tag *= flip_factor;

		int x_index = floor((x - xmin) / Lx);	// x index of the left bottom grid point
		int y_index = floor((y - ymin) / Ly);	// y index of the left bottom grid point

		double dx = x - (xmin + x_index * Lx);	// distance to the left grid line
		double dy = y - (ymin + y_index * Ly);	// distance to the bottom grid line

		double dx_ratio = dx * inv_Lx;
		double dy_ratio = dy * inv_Ly;

		//double LB = (1 - dx / Lx) * (1 - dy / Ly) * factor;
		//double RB = (dx / Lx) * (1 - dy / Ly) * factor;
		//double LT = (1 - dx / Lx) * (dy / Ly) * factor;
		//double RT = (dx / Lx) * (dy / Ly) * factor;
		double LB = (1 - dx_ratio) * (1 - dy_ratio) * factor;
		double RB = dx_ratio * (1 - dy_ratio) * factor;
		double LT = (1 - dx_ratio) * dy_ratio * factor;
		double RT = dx_ratio * dy_ratio * factor;

		int slice_index = find_slice_index(dev_slice, Nslice, tid);
		int base = slice_index * Nx * Ny;

		int LB_index = y_index * Nx + x_index;
		int RB_index = LB_index + 1;
		int LT_index = (y_index + 1) * Nx + x_index;
		int RT_index = LT_index + 1;

		atomicAdd(&(dev_charDensity[base + LB_index]), LB * alive * dev_meshMask[LB_index].mask_grid);
		atomicAdd(&(dev_charDensity[base + RB_index]), RB * alive * dev_meshMask[RB_index].mask_grid);
		atomicAdd(&(dev_charDensity[base + LT_index]), LT * alive * dev_meshMask[LT_index].mask_grid);
		atomicAdd(&(dev_charDensity[base + RT_index]), RT * alive * dev_meshMask[RT_index].mask_grid);

		tid += stride;
	}
}


__global__ void allocate2grid_ellipse_multi_slice(Particle* dev_bunch, double* dev_charDensity, const Slice* dev_slice, const MeshMask* dev_meshMask,
	int Np_sur, int Nslice, int Nx, int Ny, double Lx, double Ly, double charge, double hor_semi_axis, double ver_semi_axis) {

	// Allocate charges to grid and calculate the rho/epsilon

	const double epsilon = PassConstant::epsilon0;

	const double xmin = -(Nx - 1) / 2.0 * Lx;
	const double ymin = -(Ny - 1) / 2.0 * Ly;

	const double inv_Lx = 1.0 / Lx;
	const double inv_Ly = 1.0 / Ly;
	const double grid_area = Lx * Ly;
	const double factor = charge * 1 / (grid_area * epsilon);

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < Np_sur)
	{
		Particle* p = &dev_bunch[tid];

		double x = p->x;
		double y = p->y;
		int tag = p->tag;

		const double epsilon = 1.0e-10;

		double x1 = x / hor_semi_axis;	// x/a
		double y1 = y / ver_semi_axis;	// y/b

		double dist_square = x1 * x1 + y1 * y1;

		int is_inside = (dist_square < (1.0 - epsilon));
		int inAperture = (is_inside << 1) - 1;

		int loss_now = (tag > 0) & (inAperture != 1);
		int flip_factor = 1 - 2 * loss_now;
		int skip = (tag < 0) | (loss_now);
		int alive = 1 - skip;
		p->tag *= flip_factor;

		int x_index = floor((x - xmin) / Lx);	// x index of the left bottom grid point
		int y_index = floor((y - ymin) / Ly);	// y index of the left bottom grid point

		double dx = x - (xmin + x_index * Lx);	// distance to the left grid line
		double dy = y - (ymin + y_index * Ly);	// distance to the bottom grid line

		double dx_ratio = dx * inv_Lx;
		double dy_ratio = dy * inv_Ly;

		//double LB = (1 - dx / Lx) * (1 - dy / Ly) * factor;
		//double RB = (dx / Lx) * (1 - dy / Ly) * factor;
		//double LT = (1 - dx / Lx) * (dy / Ly) * factor;
		//double RT = (dx / Lx) * (dy / Ly) * factor;
		double LB = (1 - dx_ratio) * (1 - dy_ratio) * factor;
		double RB = dx_ratio * (1 - dy_ratio) * factor;
		double LT = (1 - dx_ratio) * dy_ratio * factor;
		double RT = dx_ratio * dy_ratio * factor;

		int slice_index = find_slice_index(dev_slice, Nslice, tid);
		int base = slice_index * Nx * Ny;

		int LB_index = y_index * Nx + x_index;
		int RB_index = LB_index + 1;
		int LT_index = (y_index + 1) * Nx + x_index;
		int RT_index = LT_index + 1;

		atomicAdd(&(dev_charDensity[base + LB_index]), LB * alive * dev_meshMask[LB_index].mask_grid);
		atomicAdd(&(dev_charDensity[base + RB_index]), RB * alive * dev_meshMask[RB_index].mask_grid);
		atomicAdd(&(dev_charDensity[base + LT_index]), LT * alive * dev_meshMask[LT_index].mask_grid);
		atomicAdd(&(dev_charDensity[base + RT_index]), RT * alive * dev_meshMask[RT_index].mask_grid);

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


void generate_meshMask_and_5points_FD_matrix_include_boundary(MeshMask* host_meshMask, double* host_matrix, int Nx, int Ny, double Lx, double Ly, int& nnz,
	const std::shared_ptr<Aperture>& aperture) {

	// Generate the matrix coefficients for solving potential and the coefficients for solving electric field in the five-point difference method

	int nrow = Ny;	// Number of rows in the particle grid, including the boundary points
	int ncol = Nx;	// Number of columns in the particle grid, including the boundary points
	int n = nrow * ncol;	// Matrix dimension, A is a square matrix of size n x n, condidering the boundary points

	double a = 1.0 / (Lx * Lx);
	double b = 1.0 / (Ly * Ly);
	double c = -2.0 * (a + b); // Coefficient for the diagonal elements

	double x0 = 0, y0 = 0;	// coordinate of the grid point
	const double xmin = -(Nx - 1) / 2.0 * Lx;
	const double ymin = -(Ny - 1) / 2.0 * Ly;
	int inAperture;

	int row, col;	// Index of the grid point in the particle grid, not in the matrix A
	int gridId;		// Serial number of the grid point in the particle grid, row-major order
	int index;		// Index of value in the matrix A
	spdlog::get("logger")->debug("FieldSolver] func(generate_meshMask_and_5points_FD_matrix): line = {}", __LINE__);
	// Initialize the host matrix to zero
	for (int i = 0; i < n * n; i++) {
		host_matrix[i] = 0.0;
	}
	spdlog::get("logger")->debug("FieldSolver] func(generate_meshMask_and_5points_FD_matrix): line = {}", __LINE__);
	// For the boundary corner points
	row = 0, col = 0;	// Left Bottom vertice
	gridId = row * ncol + col;
	index = gridId * (ncol * nrow) + gridId;
	host_meshMask[gridId].mask_grid = 0;	// No charge on boundary
	host_matrix[index] = c;	// Diagonal element

	row = 0, col = ncol - 1;	// Right Bottom vertice
	gridId = row * ncol + col;
	index = gridId * (ncol * nrow) + gridId;
	host_meshMask[gridId].mask_grid = 0;	// No charge on boundary
	host_matrix[index] = c;	// Diagonal element

	row = nrow - 1, col = 0;	// Left Top vertice
	gridId = row * ncol + col;
	index = gridId * (ncol * nrow) + gridId;
	host_meshMask[gridId].mask_grid = 0;	// No charge on boundary
	host_matrix[index] = c;	// Diagonal element

	row = nrow - 1, col = ncol - 1;	// Right Top vertice
	gridId = row * ncol + col;
	index = gridId * (ncol * nrow) + gridId;
	host_meshMask[gridId].mask_grid = 0;	// No charge on boundary
	host_matrix[index] = c;	// Diagonal element
	spdlog::get("logger")->debug("FieldSolver] func(generate_meshMask_and_5points_FD_matrix): line = {}", __LINE__);
	// For the boundary edge points
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
	spdlog::get("logger")->debug("FieldSolver] func(generate_meshMask_and_5points_FD_matrix): line = {}", __LINE__);
	for (const auto& point : boundary_top) {
		row = point.first;
		col = point.second;
		gridId = row * ncol + col;
		index = gridId * (ncol * nrow) + gridId;
		host_meshMask[gridId].mask_grid = 0;	// No charge on boundary
		host_matrix[index] = c;	// Diagonal element
	}

	for (const auto& point : boundary_bottom) {
		row = point.first;
		col = point.second;
		gridId = row * ncol + col;
		index = gridId * (ncol * nrow) + gridId;
		host_meshMask[gridId].mask_grid = 0;	// No charge on boundary
		host_matrix[index] = c;	// Diagonal element
	}

	for (const auto& point : boundary_left) {
		row = point.first;
		col = point.second;
		gridId = row * ncol + col;
		index = gridId * (ncol * nrow) + gridId;
		host_meshMask[gridId].mask_grid = 0;	// No charge on boundary
		host_matrix[index] = c;	// Diagonal element
	}

	for (const auto& point : boundary_right) {
		row = point.first;
		col = point.second;
		gridId = row * ncol + col;
		index = gridId * (ncol * nrow) + gridId;
		host_meshMask[gridId].mask_grid = 0;	// No charge on boundary
		host_matrix[index] = c;	// Diagonal element
	}
	spdlog::get("logger")->debug("FieldSolver] func(generate_meshMask_and_5points_FD_matrix): line = {}", __LINE__);
	// For the inner points
	std::vector<std::pair<int, int>>inner_points;

	for (int i = 1; i < (nrow - 1); i++)
	{
		for (int j = 1; j < (ncol - 1); j++)
		{
			inner_points.push_back({ i, j });	// Inner points
		}
	}
	spdlog::get("logger")->debug("FieldSolver] func(generate_meshMask_and_5points_FD_matrix): line = {}", __LINE__);
	for (const auto& point : inner_points)
	{
		row = point.first;
		col = point.second;
		gridId = row * ncol + col;
		index = gridId * (ncol * nrow) + gridId;

		x0 = xmin + col * Lx;
		y0 = ymin + row * Ly;

		inAperture = aperture->get_particle_position(x0, y0);

		if (inAperture == 1)
		{
			host_meshMask[gridId].mask_grid = 1;	// Charge on the grid
		}
		else
		{
			host_meshMask[gridId].mask_grid = 0;	// No charge on the grid
		}
	}
	spdlog::get("logger")->debug("FieldSolver] func(generate_meshMask_and_5points_FD_matrix): line = {}", __LINE__);
	for (const auto& point : inner_points) {
		row = point.first;
		col = point.second;
		gridId = row * ncol + col;
		index = gridId * (ncol * nrow) + gridId;

		x0 = xmin + col * Lx;
		y0 = ymin + row * Ly;

		inAperture = aperture->get_particle_position(x0, y0);

		if (inAperture == 1)
		{
			auto tmp = aperture->get_intersection_points(x0, y0);

			double hx_left = ((x0 - tmp.x_left) >= Lx) ? Lx : x0 - tmp.x_left;
			double hx_right = ((tmp.x_right - x0) >= Lx) ? Lx : tmp.x_right - x0;
			double hy_bottom = ((y0 - tmp.y_bottom) >= Ly) ? Ly : y0 - tmp.y_bottom;
			double hy_top = ((tmp.y_top - y0) >= Ly) ? Ly : tmp.y_top - y0;

			double FD_C = -(2.0 / (hx_left * hx_right) + 2.0 / (hy_bottom * hy_top));
			double FD_L = 2.0 / (hx_left * (hx_left + hx_right));
			double FD_R = 2.0 / (hx_right * (hx_left + hx_right));
			double FD_B = 2.0 / (hy_bottom * (hy_bottom + hy_top));
			double FD_T = 2.0 / (hy_top * (hy_bottom + hy_top));

			host_matrix[index] = FD_C;	// Diagonal element
			host_matrix[index - 1] = FD_L * host_meshMask[gridId - 1].mask_grid;	// Left neighbor
			host_matrix[index + 1] = FD_R * host_meshMask[gridId + 1].mask_grid;	// Right neighbor
			host_matrix[index - ncol] = FD_B * host_meshMask[gridId - ncol].mask_grid;	// Bottom neighbor
			host_matrix[index + ncol] = FD_T * host_meshMask[gridId + ncol].mask_grid;	// Top neighbor

			host_meshMask[gridId].inv_hx_left = 1.0 / hx_left;		// Distance to the left grid line
			host_meshMask[gridId].inv_hx_right = 1.0 / hx_right;	// Distance to the right grid line
			host_meshMask[gridId].inv_hy_bottom = 1.0 / hy_bottom;	// Distance to the bottom grid line
			host_meshMask[gridId].inv_hy_top = 1.0 / hy_top;		// Distance to the top grid line

			spdlog::get("logger")->debug("x0 = {}, x_left = {}, x_right = {}, xmin = {}, Lx = {}, hx_left = {}, hx_right = {}",
				x0, tmp.x_left, tmp.x_right, xmin, Lx, hx_left, hx_right);
			spdlog::get("logger")->debug("y0 = {}, y_bottom = {}, y_top = {}, ymin = {}, Ly = {}, hy_bottom = {}, hy_top = {}",
				y0, tmp.y_bottom, tmp.y_top, ymin, Ly, hy_bottom, hy_top);
			spdlog::get("logger")->debug("FD_C = {}, FD_L = {}, FD_R = {}, FD_B = {}, FD_T = {}",
				FD_C, FD_L, FD_R, FD_B, FD_T);
			spdlog::get("logger")->debug("mask L = {}, R = {}, B = {}, T = {}",
				host_meshMask[gridId - 1].mask_grid, host_meshMask[gridId + 1].mask_grid, host_meshMask[gridId - ncol].mask_grid, host_meshMask[gridId + ncol].mask_grid);
			spdlog::get("logger")->debug("FD_C = {}, FD_L = {}, FD_R = {}, FD_B = {}, FD_T = {}\n",
				host_matrix[index], host_matrix[index - 1], host_matrix[index + 1], host_matrix[index - ncol], host_matrix[index + ncol]);
		}

		else
		{
			host_matrix[index] = c;	// Diagonal element
		}
	}
	spdlog::get("logger")->debug("FieldSolver] func(generate_meshMask_and_5points_FD_matrix): line = {}", __LINE__);
	for (int i = 0; i < n * n; i++) {
		if (fabs(host_matrix[i]) > 1e-10)
		{
			nnz++;
		}
	}
	spdlog::get("logger")->debug("FieldSolver] func(generate_meshMask_and_5points_FD_matrix): line = {}", __LINE__);
	//Output matrix value and check it
	std::filesystem::path matrix_savepath = "D:/PASS/test/fd_matrix.csv";
	std::ofstream file1(matrix_savepath);

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			file1 << host_matrix[i * n + j];
			if (j < (n - 1))
			{
				file1 << ",";
			}
		}
		file1 << "\n";
	}
	file1.close();
	spdlog::get("logger")->info("[FieldSolver] func(generate_meshMask_and_5points_FD_matrix): 5-points FD matrix data has been writted to {}", matrix_savepath.string());

	std::filesystem::path meshMask_savepath = "D:/PASS/test/mesh_mask.csv";
	std::ofstream file2(meshMask_savepath);

	for (int i = 0; i < nrow; i++)
	{
		for (int j = 0; j < ncol; j++)
		{
			file2 << host_meshMask[i * ncol + j].mask_grid;
			if (j < (ncol - 1))
			{
				file2 << ",";
			}
		}
		file2 << "\n";
	}
	file2.close();

	spdlog::get("logger")->info("[FieldSolver] func(generate_meshMask_and_5points_FD_matrix): 5-points FD mesh mask data has been writted to {}", meshMask_savepath.string());

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