#include "general.h"

#include <ctime>
#include <iomanip>
#include <sstream>
#include <fstream>


void print_logo(const Parameter& Para) {

	auto logger = spdlog::get("logger");
	logger->set_pattern("%v");

	logger->info("                                                                                 ");
	logger->info(" PPPPPPPPPPPPPPPPP        AAA                 SSSSSSSSSSSSSSS    SSSSSSSSSSSSSSS ");
	logger->info(" P::::::::::::::::P      A:::A              SS:::::::::::::::S SS:::::::::::::::S");
	logger->info(" P::::::PPPPPP:::::P    A:::::A            S:::::SSSSSS::::::SS:::::SSSSSS::::::S");
	logger->info(" PP:::::P     P:::::P  A:::::::A           S:::::S     SSSSSSSS:::::S     SSSSSSS");
	logger->info("   P::::P     P:::::P A:::::::::A          S:::::S            S:::::S            ");
	logger->info("   P::::P     P:::::PA:::::A:::::A         S:::::S            S:::::S            ");
	logger->info("   P::::PPPPPP:::::PA:::::A A:::::A         S::::SSSS          S::::SSSS         ");
	logger->info("   P:::::::::::::PPA:::::A   A:::::A         SS::::::SSSSS      SS::::::SSSSS    ");
	logger->info("   P::::PPPPPPPPP A:::::A     A:::::A          SSS::::::::SS      SSS::::::::SS  ");
	logger->info("   P::::P        A:::::AAAAAAAAA:::::A            SSSSSS::::S        SSSSSS::::S ");
	logger->info("   P::::P       A:::::::::::::::::::::A                S:::::S            S:::::S");
	logger->info("   P::::P      A:::::AAAAAAAAAAAAA:::::A               S:::::S            S:::::S");
	logger->info(" PP::::::PP   A:::::A             A:::::A  SSSSSSS     S:::::SSSSSSSS     S:::::S");
	logger->info(" P::::::::P  A:::::A               A:::::A S::::::SSSSSS:::::SS::::::SSSSSS:::::S");
	logger->info(" P::::::::P A:::::A                 A:::::AS:::::::::::::::SS S:::::::::::::::SS ");
	logger->info(" PPPPPPPPPPAAAAAAA                   AAAAAAASSSSSSSSSSSSSSS    SSSSSSSSSSSSSSS   ");
	logger->info("                                                                                 ");

	logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

}

void print_copyright(const Parameter& Para) {};

std::string to_scientific_string(const double value, const int precision) {
	std::ostringstream out;
	out << std::scientific << std::setprecision(precision) << value;
	return out.str();
}

void get_cmd_input(int argc, char** argv, std::vector<std::filesystem::path>& path_input_para, std::string& yearMonDay, std::string& hourMinSec) {

	// create a parser
	cmdline::parser a;

	// add specified type of variable.
	// 1st argument is long name
	// 2nd argument is short name (no short name if '\0' specified)
	// 3rd argument is description
	// 4th argument is mandatory (optional. default is false)
	// 5th argument is default value  (optional. it used when mandatory is false)

	a.add<std::string>("beam0", '\0', "paramerter file path of beam0", true, "");
	a.add<std::string>("beam1", '\0', "paramerter file path of beam1", false, "empty");
	a.add<std::string>("ymd", '\0', "starting time: year-month-day [yyyy_mmdd]", false, "empty");
	a.add<std::string>("hms", '\0', "starting time: hour-minute-second [hhmm_ss]", false, "empty");

	a.parse_check(argc, argv);

	if (a.exist("beam0")) {
		std::filesystem::path path_tmp = a.get<std::string>("beam0");
		if (std::filesystem::exists(path_tmp))
		{
			path_input_para.push_back(path_tmp);
		}
		else
		{
			std::cerr << "Error: File \"" << path_tmp << "\" is not exist." << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}
	else
	{
		std::cerr << "Error: need option of \"--beam0=path\"" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	if (a.exist("beam1")) {
		std::filesystem::path path_tmp = a.get<std::string>("beam1");
		if (std::filesystem::exists(path_tmp))
		{
			path_input_para.push_back(path_tmp);
		}
		else
		{
			std::cerr << "Error: File \"" << path_tmp << "\" is not exist." << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}
	else
	{
		//do nothing
	}

	std::time_t currentTime = std::time(nullptr);
	std::tm* localTime = std::localtime(&currentTime);

	int year = 1900 + localTime->tm_year;
	int month = 1 + localTime->tm_mon;
	int day = localTime->tm_mday;
	int hour = localTime->tm_hour;
	int minute = localTime->tm_min;
	int second = localTime->tm_sec;

	std::string syear = std::to_string(year);
	std::string smonth, sday, shour, sminute, ssecond;

	if (month < 10)
		smonth = "0" + std::to_string(month);
	else
		smonth = std::to_string(month);

	if (day < 10)
		sday = "0" + std::to_string(day);
	else
		sday = std::to_string(day);

	if (hour < 10)
		shour = "0" + std::to_string(hour);
	else
		shour = std::to_string(hour);

	if (minute < 10)
		sminute = "0" + std::to_string(minute);
	else
		sminute = std::to_string(minute);

	if (second < 10)
		ssecond = "0" + std::to_string(second);
	else
		ssecond = std::to_string(second);

	std::string yearMonDay_tmp = syear + "_" + smonth + sday;
	std::string hourMinSec_tmp = shour + sminute + "_" + ssecond;

	if (a.exist("ymd")) {
		yearMonDay = a.get<std::string>("ymd");
	}
	else
	{
		yearMonDay = yearMonDay_tmp;
	}

	if (a.exist("hms")) {
		hourMinSec = a.get<std::string>("hms");
	}
	else
	{
		hourMinSec = hourMinSec_tmp;
	}

	//for (auto i : path_input_para)
	//{
	//	std::cout << i << std::endl;
	//}
	//std::cout << yearMonDay << std::endl;
	//std::cout << hourMinSec << std::endl;
}

void print_config_parameter(const Parameter& Para) {

	using namespace tabulate;

	Table Para_table;

	Para_table.add_row({ "Parameter", "Beam0", (1 != Para.Nbeam) ? "Beam1" : "^_^" });
	Para_table.add_row({ "Beam name", Para.beam_name[0], (1 != Para.Nbeam) ? Para.beam_name[1] : "^_^" });
	Para_table.add_row({ "Beam Id", std::to_string(Para.beamId[0]), (1 != Para.Nbeam) ? std::to_string(Para.beamId[1]) : "^_^" });
	Para_table.add_row({ "Number of bunches", std::to_string(Para.Nbunch[0]), (1 != Para.Nbeam) ? std::to_string(Para.Nbunch[1]) : "^_^" });
	Para_table.add_row({ "Number of turns", std::to_string(Para.Nturn), (1 != Para.Nbeam) ? "<--" : "^_^" });
	Para_table.add_row({ "Number of GPU devices", std::to_string(Para.Ngpu), (1 != Para.Nbeam) ? "<--" : "^_^" });

	std::string tmp_gpuId;
	for (auto i : Para.gpuId)
	{
		tmp_gpuId += std::to_string(i);
		tmp_gpuId += ", ";
	}
	Para_table.add_row({ "GPU device Id", tmp_gpuId, (1 != Para.Nbeam) ? "<--" : "^_^" });

	//tables.add_row({ "Beam-beam effect", Para.is_beambeam ? "on" : "off", (1 != Para.Nbeam) ? "<--" : "^_^" });
	//tables.add_row({ "Space charge effect", Para.is_spacecharge ? "on" : "off", (1 != Para.Nbeam) ? "<--" : "^_^" });

	Para_table.add_row({ "Input path", Para.path_input_para[0].string(), (1 != Para.Nbeam) ? Para.path_input_para[1].string() : "^_^" });
	Para_table.add_row({ "Output directory", Para.dir_output.string(), (1 != Para.Nbeam) ? "<--" : "^_^" });
	Para_table.add_row({ "Starting time", Para.yearMonDay + ", " + Para.hourMinSec, (1 != Para.Nbeam) ? "<--" : "^_^" });

	/////////////////////////////////////////// configure table start ///////////////////////////////////////////

	Para_table[0][0].format().font_color(Color::yellow).font_align(FontAlign::center).font_style({ FontStyle::bold });
	Para_table[0][1].format().font_color(Color::yellow).font_align(FontAlign::center).font_style({ FontStyle::bold });
	if (2 == Para.Nbeam)
		Para_table[0][2].format().font_color(Color::yellow).font_align(FontAlign::center).font_style({ FontStyle::bold });

	Para_table.column(0).format().font_align(FontAlign::left);
	Para_table.column(0).format().width(25);
	Para_table.column(1).format().font_align(FontAlign::center);
	Para_table.column(1).format().width(30);
	if (2 == Para.Nbeam)
	{
		Para_table.column(2).format().font_align(FontAlign::center);
		Para_table.column(2).format().width(30);
	}

	/////////////////////////////////////////// configure table end /////////////////////////////////////////////

	std::cout << Para_table << std::endl;

	std::ofstream outputFile(Para.path_logfile, std::ios::app);
	std::streambuf* coutbuf = std::cout.rdbuf();
	std::cout.rdbuf(outputFile.rdbuf());

	std::cout << Para_table << std::endl;
	outputFile.flush();
	outputFile.close();

	std::cout.rdbuf(coutbuf);
}

void print_beam_parameter(const Parameter& Para, const std::vector<Bunch>& Beam0, const std::vector<Bunch>& Beam1) {

	using namespace tabulate;
	int Nbeam = (0 == Beam1.size()) ? 1 : 2;

	for (size_t i = 0; i < Beam0.size(); i++)
	{
		Table Beam_table;

		Beam_table.add_row({ "Parameter", "Bunch " + std::to_string(Beam0[i].bunchId), (1 != Nbeam) ? "Bunch " + std::to_string(Beam1[i].bunchId) : "^_^" });
		Beam_table.add_row({ "Proton number",std::to_string(Beam0[i].Nproton),(1 != Nbeam) ? std::to_string(Beam1[i].Nproton) : "^_^" });
		Beam_table.add_row({ "Neutron number",std::to_string(Beam0[i].Nneutron),(1 != Nbeam) ? std::to_string(Beam1[i].Nneutron) : "^_^" });
		Beam_table.add_row({ "Charge number",std::to_string(Beam0[i].Ncharge),(1 != Nbeam) ? std::to_string(Beam1[i].Ncharge) : "^_^" });
		Beam_table.add_row({ "Real  particles/bunch",to_scientific_string(Beam0[i].Nrp, 11),(1 != Nbeam) ? to_scientific_string(Beam1[i].Nrp, 11) : "^_^" });
		Beam_table.add_row({ "Macro particles/bunch",std::to_string(Beam0[i].Np),(1 != Nbeam) ? std::to_string(Beam1[i].Np) : "^_^" });
		Beam_table.add_row({ "Ratio Nrp/Np",std::to_string(Beam0[i].ratio),(1 != Nbeam) ? std::to_string(Beam1[i].ratio) : "^_^" });
		Beam_table.add_row({ "Kinetic energy (GeV/u)", std::to_string(Beam0[i].Ek / 1e9), (1 != Nbeam) ? std::to_string(Beam1[i].Ek / 1e9) : "^_^" });
		Beam_table.add_row({ "Statistic mass (MeV/c2)", std::to_string(Beam0[i].m0 / 1e6), (1 != Nbeam) ? std::to_string(Beam1[i].m0 / 1e6) : "^_^" });
		Beam_table.add_row({ "Momentum (kg*m/s)", to_scientific_string(Beam0[i].p0_kg, 12),(1 != Nbeam) ? to_scientific_string(Beam1[i].p0_kg, 12) : "^_^" });
		Beam_table.add_row({ "Momentum (GeV/c)", std::to_string(Beam0[i].p0 / 1e9),(1 != Nbeam) ? std::to_string(Beam1[i].p0 / 1e9) : "^_^" });
		Beam_table.add_row({ "Brho (T*m)",std::to_string(Beam0[i].Brho),(1 != Nbeam) ? std::to_string(Beam1[i].Brho) : "^_^" });
		Beam_table.add_row({ "Beta",to_scientific_string(Beam0[i].beta, 12),(1 != Nbeam) ? to_scientific_string(Beam1[i].beta, 12) : "^_^" });
		Beam_table.add_row({ "Gamma",std::to_string(Beam0[i].gamma),(1 != Nbeam) ? std::to_string(Beam1[i].gamma) : "^_^" });
		Beam_table.add_row({ "Geometric emittance x (m*rad)",to_scientific_string(Beam0[i].emitx, 9),(1 != Nbeam) ? to_scientific_string(Beam1[i].emitx, 9) : "^_^" });
		Beam_table.add_row({ "Geometric emittance y (m*rad)",to_scientific_string(Beam0[i].emity, 9),(1 != Nbeam) ? to_scientific_string(Beam1[i].emity, 9) : "^_^" });
		Beam_table.add_row({ "Normalized emittance x (m*rad)",to_scientific_string(Beam0[i].emitx_norm, 9),(1 != Nbeam) ? to_scientific_string(Beam1[i].emitx_norm, 9) : "^_^" });
		Beam_table.add_row({ "Normalized emittance y (m*rad)",to_scientific_string(Beam0[i].emity_norm,9),(1 != Nbeam) ? to_scientific_string(Beam1[i].emity_norm,9) : "^_^" });
		Beam_table.add_row({ "Twiss alpha x",std::to_string(Beam0[i].alphax),(1 != Nbeam) ? std::to_string(Beam1[i].alphax) : "^_^" });
		Beam_table.add_row({ "Twiss alpha y",std::to_string(Beam0[i].alphay),(1 != Nbeam) ? std::to_string(Beam1[i].alphay) : "^_^" });
		Beam_table.add_row({ "Twiss beta x (m)",std::to_string(Beam0[i].betax),(1 != Nbeam) ? std::to_string(Beam1[i].betax) : "^_^" });
		Beam_table.add_row({ "Twiss beta y (m)",std::to_string(Beam0[i].betay),(1 != Nbeam) ? std::to_string(Beam1[i].betay) : "^_^" });
		Beam_table.add_row({ "Twiss gamma x (m-1)",std::to_string(Beam0[i].gammax),(1 != Nbeam) ? std::to_string(Beam1[i].gammax) : "^_^" });
		Beam_table.add_row({ "Twiss gamma y (m-1)",std::to_string(Beam0[i].gammay),(1 != Nbeam) ? std::to_string(Beam1[i].gammay) : "^_^" });
		Beam_table.add_row({ "Sigma x (mm)",std::to_string(Beam0[i].sigmax * 1e3),(1 != Nbeam) ? std::to_string(Beam1[i].sigmax * 1e3) : "^_^" });
		Beam_table.add_row({ "Sigma y (mm)",std::to_string(Beam0[i].sigmay * 1e3),(1 != Nbeam) ? std::to_string(Beam1[i].sigmay * 1e3) : "^_^" });
		Beam_table.add_row({ "Sigma px (mrad)",std::to_string(Beam0[i].sigmapx * 1e3),(1 != Nbeam) ? std::to_string(Beam1[i].sigmapx * 1e3) : "^_^" });
		Beam_table.add_row({ "Sigma py (mrad)",std::to_string(Beam0[i].sigmapy * 1e3),(1 != Nbeam) ? std::to_string(Beam1[i].sigmapy * 1e3) : "^_^" });
		Beam_table.add_row({ "Sigma z (mm)",std::to_string(Beam0[i].sigmaz * 1e3),(1 != Nbeam) ? std::to_string(Beam1[i].sigmaz * 1e3) : "^_^" });
		Beam_table.add_row({ "Deltap/P (1e-3)",std::to_string(Beam0[i].dp * 1e3),(1 != Nbeam) ? std::to_string(Beam1[i].dp * 1e3) : "^_^" });
		Beam_table.add_row({ "Qx",std::to_string(Beam0[i].Qx),(1 != Nbeam) ? std::to_string(Beam1[i].Qx) : "^_^" });
		Beam_table.add_row({ "Qy",std::to_string(Beam0[i].Qy),(1 != Nbeam) ? std::to_string(Beam1[i].Qy) : "^_^" });
		Beam_table.add_row({ "Qz",std::to_string(Beam0[i].Qz),(1 != Nbeam) ? std::to_string(Beam1[i].Qz) : "^_^" });
		Beam_table.add_row({ "Chromaticity x",std::to_string(Beam0[i].chromx),(1 != Nbeam) ? std::to_string(Beam1[i].chromx) : "^_^" });
		Beam_table.add_row({ "Chromaticity x",std::to_string(Beam0[i].chromy),(1 != Nbeam) ? std::to_string(Beam1[i].chromy) : "^_^" });
		Beam_table.add_row({ "Gamma T",std::to_string(Beam0[i].gammat),(1 != Nbeam) ? std::to_string(Beam1[i].gammat) : "^_^" });


		/////////////////////////////////////////// configure table start ///////////////////////////////////////////

		Beam_table[0][0].format().font_color(Color::green).font_align(FontAlign::center).font_style({ FontStyle::bold });
		Beam_table[0][1].format().font_color(Color::green).font_align(FontAlign::center).font_style({ FontStyle::bold });
		if (2 == Nbeam)
			Beam_table[0][2].format().font_color(Color::green).font_align(FontAlign::center).font_style({ FontStyle::bold });

		Beam_table.column(0).format().font_align(FontAlign::left);
		Beam_table.column(0).format().width(25);
		Beam_table.column(1).format().font_align(FontAlign::center);
		Beam_table.column(1).format().width(30);
		if (2 == Nbeam)
		{
			Beam_table.column(2).format().font_align(FontAlign::center);
			Beam_table.column(2).format().width(30);
		}

		/////////////////////////////////////////// configure table end /////////////////////////////////////////////

		std::cout << Beam_table << std::endl;

		std::ofstream outputFile(Para.path_logfile, std::ios::app);
		std::streambuf* coutbuf = std::cout.rdbuf();
		std::cout.rdbuf(outputFile.rdbuf());

		std::cout << Beam_table << std::endl;
		outputFile.flush();
		outputFile.close();

		std::cout.rdbuf(coutbuf);
	}

}

void create_logger(const Parameter& Para) {

	std::string logfile = Para.path_logfile.string();

	auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
	console_sink->set_level(spdlog::level::debug);  // Set the console log level
	console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v"); // Set tho console log pattern

	auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logfile, true);
	file_sink->set_level(spdlog::level::debug);
	file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

	std::vector<spdlog::sink_ptr> sinks{ console_sink, file_sink };
	auto logger = std::make_shared<spdlog::logger>("logger", sinks.begin(), sinks.end());

	spdlog::set_default_logger(logger);

	spdlog::set_level(spdlog::level::debug);

	spdlog::flush_on(spdlog::level::debug);

}


void show_device_info() {

	auto logger = spdlog::get("logger");
	logger->set_pattern("%v");

	logger->info("\n********** Runtime environment ************\n");

	cudaDeviceProp prop; //"cuda_runtime.h"
	int count;
	cudaGetDeviceCount(&count);

	logger->info("Operation system:        {}", MY_SYSTEM);
	logger->info("CUDA toolkit version:    {}.{}", CUDART_VERSION / 1000, CUDART_VERSION / 10 % 100);
	//printf("CPU threads available:   %d\n", omp_get_num_procs());
	logger->info("GPU available:           {}", count);
	//printf("GPU used number:         %d\n", numtask);
#ifdef _MSC_VER
	logger->info("MSVC version:            {}", _MSC_VER);
#endif
#ifdef __GNUC__
	logger->info("GCC version:             {}.{}.{}", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#endif //

	//if (omp_get_num_procs() < numtask)
	//{
	//	printf("\nUser warning: cpu threads is less than gpu devices, this may cause unknown error. Please change gpu device number.\n");
	//}

	//printf("\n%-9s %-9s\n", "rank", "device");
	//for (size_t i = 0; i < device.size(); i++)
	//{
	//	printf("%-9d %-9d\n", device[i].rank, device[i].deviceId);
	//}

	for (int i = 0; i < count; ++i)
	{
		cudaGetDeviceProperties(&prop, i);
		logger->info("\n--- General Information for device {} ---\n", i);
		logger->info("Name:                     {}", prop.name);
		logger->info("Compute capability:       {}.{}", prop.major, prop.minor);
		logger->info("Device copy overlap:      {}", (prop.deviceOverlap) ? "Enabled" : "Disabled");
		logger->info("Kernel execition timeout: {}", (prop.kernelExecTimeoutEnabled) ? "Enabled" : "Disabled");
		logger->info("Total global Men:         {} GiB", prop.totalGlobalMem / (1024 * 1024 * 1024));
		logger->info("Multiprocessor count:     {}", prop.multiProcessorCount);
		logger->info("Shared men per mp:        {} bytes", prop.sharedMemPerBlock);
		logger->info("Threads in warp:          {}", prop.warpSize);
		logger->info("Max threads per block:    {}", prop.maxThreadsPerBlock);
		logger->info("Max thread dimensions:    ({}, {}, {})", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		logger->info("Max grid dimensions:      ({}, {}, {})", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	}

	////Check peer-to-peer connectivity
	//printf("\nP2P Connectivity Matrix\n");
	//printf("     D\\D");

	//for (int j = 0; j < numtask; j++) {
	//	printf("%6d", device[j].deviceId);
	//}
	//printf("\n");

	//for (int i = 0; i < numtask; i++) {
	//	printf("%6d\t", device[i].deviceId);
	//	for (int j = 0; j < numtask; j++) {
	//		if (i != j) {
	//			int access;
	//			callCuda(cudaDeviceCanAccessPeer(&access, device[i].deviceId, device[j].deviceId));
	//			printf("%6d", (access) ? 1 : 0);
	//		}
	//		else {
	//			printf("%6d", 1);
	//		}
	//	}
	//	printf("\n");
	//}
	logger->info("\n*******************************************\n");

	logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
}