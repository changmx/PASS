#include "spaceCharge.h"

#include <algorithm>

SpaceCharge::SpaceCharge(const Parameter& para, int input_beamId, Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

	name = obj_name;
	dev_bunch = Bunch.dev_bunch;

	circumference = para.circumference;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	dev_slice = Bunch.dev_slice_sc;
	Ncharge = Bunch.Ncharge;
	ratio = Bunch.ratio;
	qm_ratio = Bunch.qm_ratio;

	int Nslice = 0;					// Number of slices
	FieldSolverType solverType;		// Field solver type
	int Ngridx = 0, Ngridy = 0;		// Number of grid in x and y directions
	int Nx = 0, Ny = 0;				// Number of grid points in x and y directions
	double Lx = 0, Ly = 0;			// Length of the grid in x and y directions

	std::shared_ptr<Aperture> aperture;
	std::string apertureType;
	int apertureValueSize;

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		is_enable_spaceCharge = data.at("Space-charge simulation parameters").at("Is enable space charge");
		Nslice = data.at("Space-charge simulation parameters").at("Number of slices");
		solverType = string2Enum(data.at("Space-charge simulation parameters").at("Field solver"));

		s = data.at("Sequence").at(obj_name).at("S (m)");
		sc_length = data.at("Sequence").at(obj_name).at("Length (m)");

		apertureType = data.at("Sequence").at(obj_name).at("Aperture type");
		apertureValueSize = data.at("Sequence").at(obj_name).at("Aperture value").size();

		if ("Circle" == apertureType)
		{
			if (apertureValueSize >= 1)
			{
				double radius = data.at("Sequence").at(obj_name).at("Aperture value")[0];

				aperture = std::make_shared<CircleAperture>(radius);
			}
			else
			{
				spdlog::get("logger")->error("[SpaceCharge] Aperture type: {} need {} value, but now is {}", apertureType, 1, apertureValueSize);
				std::exit(EXIT_FAILURE);
			}
		}
		else if ("Rectangle" == apertureType)
		{
			if (apertureValueSize >= 2)
			{
				double half_width = 0;
				double half_height = 0;
				half_width = data.at("Sequence").at(obj_name).at("Aperture value")[0];
				half_height = data.at("Sequence").at(obj_name).at("Aperture value")[1];

				aperture = std::make_shared<RectangleAperture>(half_width, half_height);
			}
			else
			{
				spdlog::get("logger")->error("[SpaceCharge] Aperture type: {} need {} value, but now is {}", apertureType, 2, apertureValueSize);
				std::exit(EXIT_FAILURE);
			}
		}
		else if ("Ellipse" == apertureType)
		{
			if (apertureValueSize >= 2)
			{
				double hor_semi_axis = data.at("Sequence").at(obj_name).at("Aperture value")[0];
				double ver_semi_axis = data.at("Sequence").at(obj_name).at("Aperture value")[1];

				aperture = std::make_shared<EllipseAperture>(hor_semi_axis, ver_semi_axis);
			}
			else
			{
				spdlog::get("logger")->error("[SpaceCharge] Aperture type: {} need {} value, but now is {}", apertureType, 2, apertureValueSize);
				std::exit(EXIT_FAILURE);
			}
		}
		else
		{
			spdlog::get("logger")->error("[SpaceCharge] We don't support this kind of aperture now: {}", apertureType);
			std::exit(EXIT_FAILURE);
		}

		double xmin = aperture.get()->get_xmin();
		double xmax = aperture.get()->get_xmax();
		double ymin = aperture.get()->get_ymin();
		double ymax = aperture.get()->get_ymax();

		if (solverType == FieldSolverType::PIC_Conv ||
			solverType == FieldSolverType::PIC_FD_CUDSS ||
			solverType == FieldSolverType::PIC_FD_AMGX ||
			solverType == FieldSolverType::PIC_FD_FFT)
		{
			Ngridx = data.at("Sequence").at(obj_name).at("Number of PIC grid x");
			Ngridy = data.at("Sequence").at(obj_name).at("Number of PIC grid y");
			Lx = data.at("Sequence").at(obj_name).at("Grid x length");
			Ly = data.at("Sequence").at(obj_name).at("Grid y length");

			Nx = Ngridx + 1;
			Ny = Ngridy + 1;

			double left_mesh = -Ngridx / 2.0 * Lx;
			if (left_mesh > xmin) {
				double Lx_old = Lx;
				Lx = (2.0 * std::fabs(xmin)) / Ngridx;
				spdlog::get("logger")->warn("[SpaceCharge] Hor. min mesh size: {} can't cover aperture: {}, we have changed Lx from: {} to :{}", left_mesh, xmin, Lx_old, Lx);
			}

			double right_mesh = Ngridx / 2.0 * Lx;
			if (right_mesh < xmax) {
				double Lx_old = Lx;
				Lx = (2.0 * std::fabs(xmax)) / Ngridx;
				spdlog::get("logger")->warn("[SpaceCharge] Hor. max mesh size: {} can't cover aperture: {}, we have changed Lx from: {} to :{}", right_mesh, xmax, Lx_old, Lx);
			}

			double lower_mesh = -Ngridy / 2.0 * Ly;
			if (lower_mesh > ymin) {
				double Ly_old = Ly;
				Ly = (2.0 * std::fabs(ymin)) / Ngridy;
				spdlog::get("logger")->warn("[SpaceCharge] Ver. min mesh size: {} can't cover aperture: {}, we have changed Ly from: {} to :{}", lower_mesh, ymin, Ly_old, Ly);
			}

			double upper_mesh = Ngridy / 2.0 * Ly;
			if (upper_mesh < ymax) {
				double Ly_old = Ly;
				Ly = (2.0 * std::fabs(ymax)) / Ngridy;
				spdlog::get("logger")->warn("[SpaceCharge] Ver. max mesh size: {} can't cover aperture: {}, we have changed Ly from: {} to :{}", upper_mesh, ymax, Ly_old, Ly);
			}
		}
		else if (solverType == FieldSolverType::Eq_Quasi_Static || solverType == FieldSolverType::Eq_Frozen)
		{

		}

	}
	catch (json::exception e)
	{
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}

	if (is_enable_spaceCharge)
	{
		if (solverType == FieldSolverType::PIC_FD_CUDSS)
		{
			try {
				solver = std::make_shared<FieldSolverCUDSS>(Nx, Ny, Lx, Ly, Nslice, aperture);
			}
			catch (const std::exception& e) {
				spdlog::get("logger")->error("[SpaceCharge] Error creating FieldSolver: {} at command = {}, s = {}", e.what(), name, s);
				std::exit(EXIT_FAILURE);
			}

			auto it = std::find_if(Bunch.solver_sc.begin(), Bunch.solver_sc.end(), [&](const std::shared_ptr<FieldSolver>& solver_) {
				return *solver_ == *solver;
				});

			if (it != Bunch.solver_sc.end()) {
				solver = *it;
				size_t solver_index = std::distance(Bunch.solver_sc.begin(), it);
				spdlog::get("logger")->info("[SpaceCharge] Reusing existing FieldSolver instance (index = {}) at command = {}, s = {}", solver_index, name, s);
			}
			else
			{
				Bunch.solver_sc.push_back(solver);
				solver->initialize();
				spdlog::get("logger")->info("[SpaceCharge] Creating new FieldSolver instance (index = {}) at command = {}, s = {}", Bunch.solver_sc.size() - 1, name, s);
			}

		}
		else if (solverType == FieldSolverType::PIC_FD_AMGX)
		{

		}
	}

}


void SpaceCharge::execute(int turn) {

	if (is_enable_spaceCharge)
	{
		// Solve Ax=b for space charge

		int Np_sur = bunchRef.Np_sur;
		if (Np_sur <= 0) {
			spdlog::get("logger")->warn("[SpaceCharge] No particles in the bunch, skipping space charge calculation.");
			return;
		}

		double charge_per_mp = ratio * Ncharge * PassConstant::e;
		double p0_per_mp = ratio * (Ncharge / qm_ratio) * bunchRef.p0_kg;

		// Step 1: Allocate particle to grid
		callCuda(cudaEventRecord(simTime.start, 0));
		float time_tmp1 = 0;

		solver->update_b_values(dev_bunch, dev_slice, Np_sur, charge_per_mp, thread_x, block_x, turn, s);

		callCuda(cudaEventRecord(simTime.stop, 0));
		callCuda(cudaEventSynchronize(simTime.stop));
		callCuda(cudaEventElapsedTime(&time_tmp1, simTime.start, simTime.stop));
		simTime.allocate2gridSC += time_tmp1;

		// Step 2: Solve Ax=b
		callCuda(cudaEventRecord(simTime.start, 0));
		float time_tmp2 = 0;

		solver->solve_x_values();

		callCuda(cudaEventRecord(simTime.stop, 0));
		callCuda(cudaEventSynchronize(simTime.stop));
		callCuda(cudaEventElapsedTime(&time_tmp2, simTime.start, simTime.stop));
		simTime.calPotentialSC += time_tmp2;

		// Step 3: Calculate electric field
		callCuda(cudaEventRecord(simTime.start, 0));
		float time_tmp3 = 0;

		solver->calculate_electricField();

		callCuda(cudaEventRecord(simTime.stop, 0));
		callCuda(cudaEventSynchronize(simTime.stop));
		callCuda(cudaEventElapsedTime(&time_tmp3, simTime.start, simTime.stop));
		simTime.calElectricSC += time_tmp3;

		// Step 4: Apply space-charge kick to particles
		callCuda(cudaEventRecord(simTime.start, 0));
		float time_tmp4 = 0;

		const double2* dev_E = solver->get_electricField();
		double sc_factor = 1 / (bunchRef.gamma * bunchRef.gamma) * charge_per_mp * sc_length / (p0_per_mp * bunchRef.beta * PassConstant::c);
		double Lx = solver->get_Lx();
		double Ly = solver->get_Ly();
		int Nx = solver->get_Nx();
		int Ny = solver->get_Ny();
		int Nslice = solver->get_Nslice();

		callKernel(cal_spaceCharge_kick << <block_x, thread_x, 0, 0 >> > (dev_bunch, dev_E, dev_slice, Np_sur, Nx, Ny, Lx, Ly, Nslice, sc_factor));

		callCuda(cudaEventRecord(simTime.stop, 0));
		callCuda(cudaEventSynchronize(simTime.stop));
		callCuda(cudaEventElapsedTime(&time_tmp4, simTime.start, simTime.stop));
		simTime.calKickSC += time_tmp4;


	}

}


__global__ void cal_spaceCharge_kick(Particle* dev_bunch, const double2* dev_E, const Slice* dev_slice,
	int Np_sur, int Nx, int Ny, double Lx, double Ly, int Nslice, double sc_factor) {

	const double xmin = -(Nx - 1) / 2.0 * Lx;
	const double ymin = -(Ny - 1) / 2.0 * Ly;

	const double inv_Lx = 1.0 / Lx;
	const double inv_Ly = 1.0 / Ly;

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < Np_sur)
	{
		Particle* p = &dev_bunch[tid];

		double x = p->x;
		double y = p->y;
		int alive = (p->tag > 0);

		int x_index = floor((x - xmin) / Lx);	// x index of the left bottom grid point
		int y_index = floor((y - ymin) / Ly);	// y index of the left bottom grid point

		double dx = x - (xmin + x_index * Lx);	// distance to the left grid line
		double dy = y - (ymin + y_index * Ly);	// distance to the bottom grid line

		double dx_ratio = dx * inv_Lx;
		double dy_ratio = dy * inv_Ly;

		//double LB = (1 - dx / Lx) * (1 - dy / Ly);
		//double RB = (dx / Lx) * (1 - dy / Ly);
		//double LT = (1 - dx / Lx) * (dy / Ly) ;
		//double RT = (dx / Lx) * (dy / Ly) ;
		double LB = (1 - dx_ratio) * (1 - dy_ratio);
		double RB = dx_ratio * (1 - dy_ratio);
		double LT = (1 - dx_ratio) * dy_ratio;
		double RT = dx_ratio * dy_ratio;

		//int slice_index = find_slice_index(dev_slice, Nslice, tid);
		int slice_index = p->sliceId;
		int base = slice_index * Nx * Ny;

		int LB_index = y_index * Nx + x_index;
		int RB_index = LB_index + 1;
		int LT_index = (y_index + 1) * Nx + x_index;
		int RT_index = LT_index + 1;

		double Ex_LB = dev_E[base + LB_index].x;
		double Ey_LB = dev_E[base + LB_index].y;
		double Ex_RB = dev_E[base + RB_index].x;
		double Ey_RB = dev_E[base + RB_index].y;
		double Ex_LT = dev_E[base + LT_index].x;
		double Ey_LT = dev_E[base + LT_index].y;
		double Ex_RT = dev_E[base + RT_index].x;
		double Ey_RT = dev_E[base + RT_index].y;

		double Ex = Ex_LB * LB + Ex_RB * RB + Ex_LT * LT + Ex_RT * RT;
		double Ey = Ey_LB * LB + Ey_RB * RB + Ey_LT * LT + Ey_RT * RT;

		double slice_length = dev_slice[slice_index].z_start - dev_slice[slice_index].z_end;

		double delta_px = 1 / slice_length * Ex * sc_factor;
		double delta_py = 1 / slice_length * Ey * sc_factor;
		
		p->px += delta_px * alive;
		p->py += delta_py * alive;

		tid += stride;

	}
}