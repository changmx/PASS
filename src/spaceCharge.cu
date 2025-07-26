#include "spaceCharge.h"

#include <algorithm>

SpaceCharge::SpaceCharge(const Parameter& para, int input_beamId, Bunch& Bunch, std::string obj_name,
	const ParallelPlan1d& plan1d, TimeEvent& timeevent) :simTime(timeevent), bunchRef(Bunch) {

	name = obj_name;
	dev_bunch = Bunch.dev_bunch;

	Np_sur = Bunch.Np_sur;
	circumference = para.circumference;

	thread_x = plan1d.get_threads_per_block();
	block_x = plan1d.get_blocks_x();

	dev_slice = Bunch.dev_slice_sc;
	charge = Bunch.charge;

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
		scLength = data.at("Sequence").at(obj_name).at("Length (m)");

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

		Np_sur = bunchRef.Np_sur;
		if (Np_sur <= 0) {
			spdlog::get("logger")->warn("[SpaceCharge] No particles in the bunch, skipping space charge calculation.");
			return;
		}

		// Allocate particle to grid
		callCuda(cudaEventRecord(simTime.start, 0));
		float time_tmp1 = 0;

		solver->update_b_values(dev_bunch, dev_slice, Np_sur, charge, thread_x, block_x);

		callCuda(cudaEventRecord(simTime.stop, 0));
		callCuda(cudaEventSynchronize(simTime.stop));
		callCuda(cudaEventElapsedTime(&time_tmp1, simTime.start, simTime.stop));
		simTime.allocate2grid += time_tmp1;

		// Solve Ax=b
		callCuda(cudaEventRecord(simTime.start, 0));
		float time_tmp2 = 0;

		solver->solve_x_values();

		callCuda(cudaEventRecord(simTime.stop, 0));
		callCuda(cudaEventSynchronize(simTime.stop));
		callCuda(cudaEventElapsedTime(&time_tmp2, simTime.start, simTime.stop));
		simTime.calPotential += time_tmp2;

		// Calculate electric field

		callCuda(cudaEventRecord(simTime.start, 0));
		float time_tmp3 = 0;

		solver->calculate_electricField();

		callCuda(cudaEventRecord(simTime.stop, 0));
		callCuda(cudaEventSynchronize(simTime.stop));
		callCuda(cudaEventElapsedTime(&time_tmp3, simTime.start, simTime.stop));
		simTime.calElectric += time_tmp3;

		//simTime.spaceCharge += (time_tmp1 + time_tmp2 + time_tmp3);

	}

}