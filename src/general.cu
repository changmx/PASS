#include "general.h"
#include "command.h"

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
		Beam_table.add_row({ "Kinetic energy per nucleon (GeV/u)", std::to_string(Beam0[i].Ek / 1e9), (1 != Nbeam) ? std::to_string(Beam1[i].Ek / 1e9) : "^_^" });
		Beam_table.add_row({ "Statistic mass per nucleon (MeV/c2/u)", std::to_string(Beam0[i].m0 / 1e6), (1 != Nbeam) ? std::to_string(Beam1[i].m0 / 1e6) : "^_^" });
		Beam_table.add_row({ "Momentum (kg*m/s)", to_scientific_string(Beam0[i].p0_kg, 12),(1 != Nbeam) ? to_scientific_string(Beam1[i].p0_kg, 12) : "^_^" });
		Beam_table.add_row({ "Momentum (GeV/c)", std::to_string(Beam0[i].p0 / 1e9),(1 != Nbeam) ? std::to_string(Beam1[i].p0 / 1e9) : "^_^" });
		Beam_table.add_row({ "Brho (T*m)",std::to_string(Beam0[i].Brho),(1 != Nbeam) ? std::to_string(Beam1[i].Brho) : "^_^" });
		Beam_table.add_row({ "Beta",to_scientific_string(Beam0[i].beta, 12),(1 != Nbeam) ? to_scientific_string(Beam1[i].beta, 12) : "^_^" });
		Beam_table.add_row({ "Gamma",std::to_string(Beam0[i].gamma),(1 != Nbeam) ? std::to_string(Beam1[i].gamma) : "^_^" });
		//Beam_table.add_row({ "Geometric emittance x (m*rad)",to_scientific_string(Beam0[i].emitx, 9),(1 != Nbeam) ? to_scientific_string(Beam1[i].emitx, 9) : "^_^" });
		//Beam_table.add_row({ "Geometric emittance y (m*rad)",to_scientific_string(Beam0[i].emity, 9),(1 != Nbeam) ? to_scientific_string(Beam1[i].emity, 9) : "^_^" });
		//Beam_table.add_row({ "Normalized emittance x (m*rad)",to_scientific_string(Beam0[i].emitx_norm, 9),(1 != Nbeam) ? to_scientific_string(Beam1[i].emitx_norm, 9) : "^_^" });
		//Beam_table.add_row({ "Normalized emittance y (m*rad)",to_scientific_string(Beam0[i].emity_norm,9),(1 != Nbeam) ? to_scientific_string(Beam1[i].emity_norm,9) : "^_^" });
		//Beam_table.add_row({ "Twiss alpha x",std::to_string(Beam0[i].alphax),(1 != Nbeam) ? std::to_string(Beam1[i].alphax) : "^_^" });
		//Beam_table.add_row({ "Twiss alpha y",std::to_string(Beam0[i].alphay),(1 != Nbeam) ? std::to_string(Beam1[i].alphay) : "^_^" });
		//Beam_table.add_row({ "Twiss beta x (m)",std::to_string(Beam0[i].betax),(1 != Nbeam) ? std::to_string(Beam1[i].betax) : "^_^" });
		//Beam_table.add_row({ "Twiss beta y (m)",std::to_string(Beam0[i].betay),(1 != Nbeam) ? std::to_string(Beam1[i].betay) : "^_^" });
		//Beam_table.add_row({ "Twiss gamma x (m-1)",std::to_string(Beam0[i].gammax),(1 != Nbeam) ? std::to_string(Beam1[i].gammax) : "^_^" });
		//Beam_table.add_row({ "Twiss gamma y (m-1)",std::to_string(Beam0[i].gammay),(1 != Nbeam) ? std::to_string(Beam1[i].gammay) : "^_^" });
		//Beam_table.add_row({ "Sigma x (mm)",std::to_string(Beam0[i].sigmax * 1e3),(1 != Nbeam) ? std::to_string(Beam1[i].sigmax * 1e3) : "^_^" });
		//Beam_table.add_row({ "Sigma y (mm)",std::to_string(Beam0[i].sigmay * 1e3),(1 != Nbeam) ? std::to_string(Beam1[i].sigmay * 1e3) : "^_^" });
		//Beam_table.add_row({ "Sigma px (mrad)",std::to_string(Beam0[i].sigmapx * 1e3),(1 != Nbeam) ? std::to_string(Beam1[i].sigmapx * 1e3) : "^_^" });
		//Beam_table.add_row({ "Sigma py (mrad)",std::to_string(Beam0[i].sigmapy * 1e3),(1 != Nbeam) ? std::to_string(Beam1[i].sigmapy * 1e3) : "^_^" });
		//Beam_table.add_row({ "Sigma z (mm)",std::to_string(Beam0[i].sigmaz * 1e3),(1 != Nbeam) ? std::to_string(Beam1[i].sigmaz * 1e3) : "^_^" });
		//Beam_table.add_row({ "Deltap/P (1e-3)",std::to_string(Beam0[i].dp * 1e3),(1 != Nbeam) ? std::to_string(Beam1[i].dp * 1e3) : "^_^" });
		//Beam_table.add_row({ "Qx",std::to_string(Beam0[i].Qx),(1 != Nbeam) ? std::to_string(Beam1[i].Qx) : "^_^" });
		//Beam_table.add_row({ "Qy",std::to_string(Beam0[i].Qy),(1 != Nbeam) ? std::to_string(Beam1[i].Qy) : "^_^" });
		//Beam_table.add_row({ "Qz",std::to_string(Beam0[i].Qz),(1 != Nbeam) ? std::to_string(Beam1[i].Qz) : "^_^" });
		//Beam_table.add_row({ "Chromaticity x",std::to_string(Beam0[i].chromx),(1 != Nbeam) ? std::to_string(Beam1[i].chromx) : "^_^" });
		//Beam_table.add_row({ "Chromaticity x",std::to_string(Beam0[i].chromy),(1 != Nbeam) ? std::to_string(Beam1[i].chromy) : "^_^" });
		//Beam_table.add_row({ "Gamma T",std::to_string(Beam0[i].gammat),(1 != Nbeam) ? std::to_string(Beam1[i].gammat) : "^_^" });


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

	constexpr auto active_level = static_cast<spdlog::level::level_enum>(PASS_SPDLOG_ACTIVE_LEVEL);
	spdlog::set_level(active_level);

	spdlog::flush_on(active_level);

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


std::string timeStamp()
{

	time_t now = time(0);
	tm* ltm = localtime(&now);

	int year = 1900 + ltm->tm_year;
	int month = 1 + ltm->tm_mon;
	int day = ltm->tm_mday;
	int hour = ltm->tm_hour;
	int minute = ltm->tm_min;
	int second = ltm->tm_sec;

	std::string syear = std::to_string(year);
	std::string smonth, sday, shour, sminute, ssecond;

	smonth = month < 10 ? "0" + std::to_string(month) : std::to_string(month);
	sday = day < 10 ? "0" + std::to_string(day) : std::to_string(day);
	shour = hour < 10 ? "0" + std::to_string(hour) : std::to_string(hour);
	sminute = minute < 10 ? "0" + std::to_string(minute) : std::to_string(minute);
	ssecond = second < 10 ? "0" + std::to_string(second) : std::to_string(second);

	//std::string fullTime = syear + "_" + smonth + sday + "_" + shour + sminute + "_" + ssecond;
	std::string fullTime = syear + "-" + smonth + "-" + sday + "," + shour + ":" + sminute + ":" + ssecond;

	return fullTime;
}


void TimeEvent::initial(int deviceid, bool timeOrNot) {

	callCuda(cudaSetDevice(deviceid));

	callCuda(cudaEventCreate(&start));
	callCuda(cudaEventCreate(&stop));

	callCuda(cudaEventCreate(&startPerTurn));
	callCuda(cudaEventCreate(&stopPerTurn));

	if (timeOrNot)
	{
		isTime = true;
	}
	else
	{
		isTime = false;
	}
}

void TimeEvent::free(int deviceid) {

	callCuda(cudaSetDevice(deviceid));

	callCuda(cudaEventDestroy(start));
	callCuda(cudaEventDestroy(stop));

	callCuda(cudaEventDestroy(startPerTurn));
	callCuda(cudaEventDestroy(stopPerTurn));
}

TimeEvent& TimeEvent::add(const TimeEvent& rhs) {

	allocate2gridSC += rhs.allocate2gridSC;
	calBoundarySC += rhs.calBoundarySC;
	calPotentialSC += rhs.calPotentialSC;
	calElectricSC += rhs.calElectricSC;
	calKickSC += rhs.calKickSC;

	allocate2gridBB += rhs.allocate2gridBB;
	calBoundaryBB += rhs.calBoundaryBB;
	calPotentialBB += rhs.calPotentialBB;
	calElectricBB += rhs.calElectricBB;
	calKickBB += rhs.calKickBB;

	sort += rhs.sort;
	slice += rhs.slice;
	hourGlass += rhs.hourGlass;
	calPhase += rhs.calPhase;
	statistic += rhs.statistic;
	crossingAngle += rhs.crossingAngle;
	crabCavity += rhs.crabCavity;
	floatWaist += rhs.floatWaist;
	oneTurnMap += rhs.oneTurnMap;
	transferFixPoint += rhs.transferFixPoint;
	saveStatistic += rhs.saveStatistic;
	savePhase += rhs.savePhase;
	saveBunch += rhs.saveBunch;
	saveFixpoint += rhs.saveFixpoint;
	saveLuminosity += rhs.saveLuminosity;

	twiss += rhs.twiss;
	transferElement += rhs.transferElement;

	total += rhs.total;

	turn += rhs.turn;

	return *this;
}

void TimeEvent::print(int totalTurn, double cpuTime, int deviceid) {

	total
		= allocate2gridSC + calBoundarySC + calPotentialSC + calElectricSC + calKickSC
		+ allocate2gridBB + calBoundaryBB + calPotentialBB + calElectricBB + calKickBB
		+ sort + slice + hourGlass + calPhase + statistic + crossingAngle + crabCavity
		+ floatWaist + oneTurnMap + transferFixPoint
		+ saveStatistic + savePhase + saveBunch + saveFixpoint + saveLuminosity
		+ twiss + transferElement;

	std::string name = "process(device " + std::to_string(deviceid) + ")";

	auto logger = spdlog::get("logger");
	logger->info("{:<35} {:>10s} {:>10s} {:>10s}", name.c_str(), "GPU time", "GPU/GPU", "GPU/CPU");
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time sort:", sort / totalTurn, sort / turn * 100, sort / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time cut slice:", slice / totalTurn, slice / turn * 100, slice / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time SC allocate to grids:", allocate2gridSC / totalTurn, allocate2gridSC / turn * 100, allocate2gridSC / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time SC calculate boundary:", calBoundarySC / totalTurn, calBoundarySC / turn * 100, calBoundarySC / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time SC calculate potential:", calPotentialSC / totalTurn, calPotentialSC / turn * 100, calPotentialSC / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time SC calculate electric field:", calElectricSC / totalTurn, calElectricSC / turn * 100, calElectricSC / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time SC calculate kick:", calKickSC / totalTurn, calKickSC / turn * 100, calKickSC / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time BB allocate to grids:", allocate2gridBB / totalTurn, allocate2gridBB / turn * 100, allocate2gridBB / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time BB calculate boundary:", calBoundaryBB / totalTurn, calBoundaryBB / turn * 100, calBoundaryBB / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time BB calculate potential:", calPotentialBB / totalTurn, calPotentialBB / turn * 100, calPotentialBB / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time BB calculate electric field:", calElectricBB / totalTurn, calElectricBB / turn * 100, calElectricBB / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time BB calculate kick:", calKickBB / totalTurn, calKickBB / turn * 100, calKickBB / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time hourglass:", hourGlass / totalTurn, hourGlass / turn * 100, hourGlass / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time cal phase:", calPhase / totalTurn, calPhase / turn * 100, calPhase / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time statistic:", statistic / totalTurn, statistic / turn * 100, statistic / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time crossing angle:", crossingAngle / totalTurn, crossingAngle / turn * 100, crossingAngle / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time crab cavity:", crabCavity / totalTurn, crabCavity / turn * 100, crabCavity / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time floatWaist:", floatWaist / totalTurn, floatWaist / turn * 100, floatWaist / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time one turn map:", oneTurnMap / totalTurn, oneTurnMap / turn * 100, oneTurnMap / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time select fix points:", transferFixPoint / totalTurn, transferFixPoint / turn * 100, transferFixPoint / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time save statistic:", saveStatistic / totalTurn, saveStatistic / turn * 100, saveStatistic / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time save phase:", savePhase / totalTurn, savePhase / turn * 100, savePhase / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time save bunch:", saveBunch / totalTurn, saveBunch / turn * 100, saveBunch / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time save fix points:", saveFixpoint / totalTurn, saveFixpoint / turn * 100, saveFixpoint / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time save luminosity:", saveLuminosity / totalTurn, saveLuminosity / turn * 100, saveLuminosity / 1000 / cpuTime * 100);

	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time twiss transfer:", twiss / totalTurn, twiss / turn * 100, twiss / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "time element transfer:", transferElement / totalTurn, transferElement / turn * 100, transferElement / 1000 / cpuTime * 100);

	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%", "summary:", total / totalTurn, total / turn * 100, total / 1000 / cpuTime * 100);
	logger->info("{:<35} {:8.2f}ms, {:8.2f}%, {:8.2f}%\n", "time per turn:", turn / totalTurn, turn / turn * 100, turn / 1000 / cpuTime * 100);

}


bool is_value_in_turn_ranges(int value, const std::vector<CycleRange>& ranges) {
	for (const auto& range : ranges) {

		const int step = range.step;

		// 快速范围检查
		if ((step > 0 && (value < range.start || value > range.end)) ||
			(step < 0 && (value > range.start || value < range.end))) {
			continue;
		}

		// 计算步数并验证
		const int delta = value - range.start;
		if (delta % step != 0) continue;  // 必须能被步长整除

		const int steps = delta / step;
		if (steps >= 0) {  // 非负步数
			return true;
		}
	}
	return false;
}


bool is_value_in_turn_ranges(int value, const std::vector<CycleRange>& ranges, int& index) {

	for (int i = 0; i < ranges.size(); ++i) {
		const auto& range = ranges[i];

		const int step = range.step;

		// 快速范围检查
		if ((step > 0 && (value < range.start || value > range.end)) ||
			(step < 0 && (value > range.start || value < range.end))) {
			continue;  // 跳过不满足范围条件的项
		}

		// 计算步数并验证整除性
		const int delta = value - range.start;
		if (delta % step != 0) continue;  // 必须能被步长整除

		// 验证步数非负
		const int steps = delta / step;
		if (steps >= 0) {	// 非负步数
			index = i;   // 设置匹配索引
			return true; // 匹配成功
		}
	}
	index = -1; // 未找到任何匹配
	return false;
}


bool is_value_firstPoint_in_turn_ranges(int value, const std::vector<CycleRange>& ranges) {
	for (const auto& range : ranges) {
		if (value == range.start) {
			return true;
		}
	}
	return false;
}


bool is_value_lastPoint_in_turn_ranges(int value, const std::vector<CycleRange>& ranges) {
	for (const auto& range : ranges) {
		if (value == range.end) {
			return true;
		}
	}
	return false;
}


void print_cycleRange(const std::vector<CycleRange>& ranges) {
	for (const auto& range : ranges) {

		spdlog::get("logger")->info("[CycleRange] Start = {}, end = {}, step = {}, totalPoints = {}", range.start, range.end, range.step, range.totalPoints);
	}
}


std::string ms_to_timeString(double ms) {

	int second = static_cast<int>(std::round(ms / 1000.0));
	int eta_hour = div(second, 3600).quot;
	int eta_minute = div(div(second, 3600).rem, 60).quot;
	int eta_second = div(div(second, 3600).rem, 60).rem;

	return std::to_string(eta_hour) + "h " + std::to_string(eta_minute) + "m " + std::to_string(eta_second) + "s";
}