#include "general.h"

#include <ctime>

void print_logo() {
	std::cout << "                                                                                 " << std::endl;
	std::cout << " PPPPPPPPPPPPPPPPP        AAA                 SSSSSSSSSSSSSSS    SSSSSSSSSSSSSSS " << std::endl;
	std::cout << " P::::::::::::::::P      A:::A              SS:::::::::::::::S SS:::::::::::::::S" << std::endl;
	std::cout << " P::::::PPPPPP:::::P    A:::::A            S:::::SSSSSS::::::SS:::::SSSSSS::::::S" << std::endl;
	std::cout << " PP:::::P     P:::::P  A:::::::A           S:::::S     SSSSSSSS:::::S     SSSSSSS" << std::endl;
	std::cout << "   P::::P     P:::::P A:::::::::A          S:::::S            S:::::S            " << std::endl;
	std::cout << "   P::::P     P:::::PA:::::A:::::A         S:::::S            S:::::S            " << std::endl;
	std::cout << "   P::::PPPPPP:::::PA:::::A A:::::A         S::::SSSS          S::::SSSS         " << std::endl;
	std::cout << "   P:::::::::::::PPA:::::A   A:::::A         SS::::::SSSSS      SS::::::SSSSS    " << std::endl;
	std::cout << "   P::::PPPPPPPPP A:::::A     A:::::A          SSS::::::::SS      SSS::::::::SS  " << std::endl;
	std::cout << "   P::::P        A:::::AAAAAAAAA:::::A            SSSSSS::::S        SSSSSS::::S " << std::endl;
	std::cout << "   P::::P       A:::::::::::::::::::::A                S:::::S            S:::::S" << std::endl;
	std::cout << "   P::::P      A:::::AAAAAAAAAAAAA:::::A               S:::::S            S:::::S" << std::endl;
	std::cout << " PP::::::PP   A:::::A             A:::::A  SSSSSSS     S:::::SSSSSSSS     S:::::S" << std::endl;
	std::cout << " P::::::::P  A:::::A               A:::::A S::::::SSSSSS:::::SS::::::SSSSSS:::::S" << std::endl;
	std::cout << " P::::::::P A:::::A                 A:::::AS:::::::::::::::SS S:::::::::::::::SS " << std::endl;
	std::cout << " PPPPPPPPPPAAAAAAA                   AAAAAAASSSSSSSSSSSSSSS    SSSSSSSSSSSSSSS   " << std::endl;
	std::cout << "                                                                                 " << std::endl;
}

void print_copyright() {};

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

void print_beam_and_bunch_parameter(const Parameter& Para) {

	using namespace tabulate;

	Table tables;

	tables.add_row({ "Parameter", "Beam0", (1 != Para.Nbeam) ? "Beam1" : "^_^" });
	tables.add_row({ "Beam name", Para.beam_name[0], (1 != Para.Nbeam) ? Para.beam_name[1] : "^_^" });
	tables.add_row({ "Beam Id", std::to_string(Para.beamId[0]), (1 != Para.Nbeam) ? std::to_string(Para.beamId[1]) : "^_^" });
	tables.add_row({ "Number of bunches", std::to_string(Para.beamId[0]), (1 != Para.Nbeam) ? std::to_string(Para.beamId[1]) : "^_^" });
	tables.add_row({ "Number of turns", std::to_string(Para.Nturn), (1 != Para.Nbeam) ? "<--" : "^_^" });
	tables.add_row({ "Number of GPU devices", std::to_string(Para.Ngpu), (1 != Para.Nbeam) ? "<--" : "^_^" });

	std::string tmp_gpuId;
	for (auto i : Para.gpuId)
	{
		tmp_gpuId += std::to_string(i);
		tmp_gpuId += ", ";
	}
	tables.add_row({ "GPU device Id", tmp_gpuId, (1 != Para.Nbeam) ? "<--" : "^_^" });

	//tables.add_row({ "Beam-beam effect", Para.is_beambeam ? "on" : "off", (1 != Para.Nbeam) ? "<--" : "^_^" });
	//tables.add_row({ "Space charge effect", Para.is_spacecharge ? "on" : "off", (1 != Para.Nbeam) ? "<--" : "^_^" });

	tables.add_row({ "Qx", std::to_string(Para.Qx[0]), (1 != Para.Nbeam) ? std::to_string(Para.Qx[1]) : "^_^" });
	tables.add_row({ "Qy", std::to_string(Para.Qy[0]), (1 != Para.Nbeam) ? std::to_string(Para.Qy[1]) : "^_^" });
	tables.add_row({ "Qz", std::to_string(Para.Qz[0]), (1 != Para.Nbeam) ? std::to_string(Para.Qz[1]) : "^_^" });
	tables.add_row({ "Chromaticity x", std::to_string(Para.chromx[0]), (1 != Para.Nbeam) ? std::to_string(Para.chromx[1]) : "^_^" });
	tables.add_row({ "Chromaticity y", std::to_string(Para.chromy[0]), (1 != Para.Nbeam) ? std::to_string(Para.chromy[1]) : "^_^" });
	tables.add_row({ "Gamma T", std::to_string(Para.gammaT[0]), (1 != Para.Nbeam) ? std::to_string(Para.gammaT[1]) : "^_^" });
	tables.add_row({ "Input path", Para.path_input_para[0].string(), (1 != Para.Nbeam) ? Para.path_input_para[1].string() : "^_^" });
	tables.add_row({ "Output directory", Para.dir_output.string(), (1 != Para.Nbeam) ? "<--" : "^_^" });
	tables.add_row({ "Starting time", Para.yearMonDay + ", " + Para.hourMinSec, (1 != Para.Nbeam) ? "<--" : "^_^" });



	/////////////////////////////////////////// configure table start ///////////////////////////////////////////

	for (size_t i = 0; i < (Para.Nbeam + 1); i++)
	{
		tables[0][i].format().font_color(Color::yellow).font_align(FontAlign::center).font_style({ FontStyle::bold });
	}
	for (size_t i = 0; i < Para.Nbeam; i++)
	{
		tables.column(i + 1).format().font_align(FontAlign::center);
		tables.column(i + 1).format().width(30);
	}

	/////////////////////////////////////////// configure table end /////////////////////////////////////////////

	std::cout << tables << std::endl;
}