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

	int Nslice = 0;					// Number of slices
	FieldSolverMethod fieldSolver;	// Field solver type
	int Ngridx = 0, Ngridy = 0;		// Number of grid in x and y directions
	int Nx = 0, Ny = 0;				// Number of grid points in x and y directions
	double Lx = 0, Ly = 0;			// Length of the grid in x and y directions

	std::unique_ptr<Aperture> aperture;
	std::string apertureType;
	int apertureValueSize;

	using json = nlohmann::json;
	std::ifstream jsonFile(para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	try
	{
		s = data.at("Sequence").at(obj_name).at("S (m)");
		scLength = data.at("Sequence").at(obj_name).at("Length (m)");

		apertureType = data.at("Sequence").at(obj_name).at("Aperture type");
		apertureValueSize = data.at("Sequence").at(obj_name).at("Aperture value").size();
		if ("Circle" == apertureType)
		{
			if (apertureValueSize >= 1)
			{
				double radius = data.at("Sequence").at(obj_name).at("Aperture value")[0];

				aperture = std::make_unique<CircleAperture>(radius);
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

				aperture = std::make_unique<RectangleAperture>(half_width, half_height);
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

				aperture = std::make_unique<EllipseAperture>(hor_semi_axis, ver_semi_axis);
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


		is_enable_spaceCharge = data.at("Space-charge simulation parameters").at("Is space charge");
		Nslice = data.at("Space-charge simulation parameters").at("Number of slices");
		fieldSolver = data.at("Space-charge simulation parameters").at("Field solver");
		Ngridx = data.at("Space-charge simulation parameters").at("Number of grid x");
		Ngridy = data.at("Space-charge simulation parameters").at("Number of grid y");
		Lx = data.at("Space-charge simulation parameters").at("Grid x length");
		Ly = data.at("Space-charge simulation parameters").at("Grid y length");

		Nx = Ngridx + 1;
		Ny = Ngridy + 1;

		double xmin = aperture.get()->get_xmin();
		double xmax = aperture.get()->get_xmax();
		double ymin = aperture.get()->get_ymin();
		double ymax = aperture.get()->get_ymax();

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
	catch (json::exception e)
	{
		spdlog::get("logger")->error(e.what());
		std::exit(EXIT_FAILURE);
	}

	picConfig = std::make_shared<PicConfig>(Nx, Ny, Lx, Ly, Nslice, fieldSolver, std::move(aperture));

	auto it = std::find_if(Bunch.picConfig_sc.begin(), Bunch.picConfig_sc.end(), [&](const std::shared_ptr<PicConfig>& p) {
		return *p == *picConfig;
		});
	if (it != Bunch.picConfig_sc.end()) {
		picConfig = *it;
	}
	else
	{
		Bunch.picConfig_sc.push_back(picConfig);
	}

}


void SpaceCharge::execute(int turn) {

	callCuda(cudaEventRecord(simTime.start, 0));
	float time_tmp = 0;






	callCuda(cudaEventRecord(simTime.stop, 0));
	callCuda(cudaEventSynchronize(simTime.stop));
	callCuda(cudaEventElapsedTime(&time_tmp, simTime.start, simTime.stop));
	simTime.spaceCharge += time_tmp;
}