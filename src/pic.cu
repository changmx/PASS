#include "pic.h"
#include "particle.h"
#include "cutSlice.h"

#include "amgx_c.h"


PicSolver_AMGX::PicSolver_AMGX(const Parameter& para, int input_beamId, Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {


}


__global__ void allocate2grid_multi_slice(const Particle* dev_bunch, double* dev_charDensity, const Slice* dev_slice, int Np_sur, int Nslice, int Nx, int Ny, double Lx, double Ly, double charge) {

	// Allocate charges to grid and calculate the rho/epsilon

	const double epsilon = PassConstant::epsilon0;

	int x_index = 0, y_index = 0;
	double dx = 0, dy = 0;
	double grid_area = Lx * Ly;
	double  LD = 0, RD = 0, LT = 0, RT = 0;		//Left down, Right down, Left top, Right top

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

		// x is in [xmin, xmax)£¬y is in [ymin, ymax). When x=xmax, the index of RD and RT are Nx, which causes the problem of out-of-bounds memory access. So is y.
		int inBoundary = (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax) & (tag > 0);

		unsigned mask = __ballot_sync(0xFFFFFFFF, inBoundary);
		if (!mask) return;

		if (inBoundary)
		{
			x_index = floor((x - xmin) / Lx);	// x index of the left down grid point
			y_index = floor((y - ymin) / Ly);	// y index of the left down grid point

			dx = x - (xmin + x_index * Lx);	// distance to the left grid line
			dy = y - (ymin + y_index * Ly);	// distance to the down grid line

			LD = (1 - dx / Lx) * (1 - dy / Ly) * factor;
			RD = (dx / Lx) * (1 - dy / Ly) * factor;
			LT = (1 - dx / Lx) * (dy / Ly) * factor;
			RT = (dx / Lx) * (dy / Ly) * factor;

			int slice_index = find_slice_index(dev_slice, Nslice, tid);
			int base = slice_index * Nx * Ny;

			int LD_index = base + y_index * Nx + x_index;
			int RD_index = base + y_index * Nx + x_index + 1;
			int LT_index = base + (y_index + 1) * Nx + x_index;
			int RT_index = base + (y_index + 1) * Nx + x_index + 1;

			atomicAdd(&(dev_charDensity[LD_index]), LD);
			atomicAdd(&(dev_charDensity[RD_index]), RD);
			atomicAdd(&(dev_charDensity[LT_index]), LT);
			atomicAdd(&(dev_charDensity[RT_index]), RT);

		}

		tid += stride;
	}
}