#include "parameter.h"
#include "general.h"

Parameter::Parameter(int argc, char** argv)
{
	get_cmd_input(argc, argv, path_input_para, yearMonDay, hourMinSec);

	using json = nlohmann::json;
	namespace fs = std::filesystem;

	if (1 == path_input_para.size() || 2 == path_input_para.size())
	{
		Nbeam = path_input_para.size();

		for (size_t i = 0; i < path_input_para.size(); i++)
		{
			if (fs::exists(path_input_para[i])) {

				beamId.push_back(i);

				std::ifstream jsonFile(path_input_para[i]);
				json data = json::parse(jsonFile);

				try {
					beam_name.push_back(data.at("Name"));
					Nbunch.push_back(data.at("Number of bunches per beam"));
					Qx.push_back(data.at("Qx"));
					Qy.push_back(data.at("Qy"));
					Qz.push_back(data.at("Qz"));
					chromx.push_back(data.at("Chromaticity x"));
					chromy.push_back(data.at("Chromaticity y"));
					gammaT.push_back(data.at("GammaT"));

					if (0 == i)
					{
						Nturn = data.at("Number of turns");

						Ngpu = data.at("Number of GPU devices");
						for (size_t idevice = 0; idevice < data.at("Device Id").size(); idevice++)
						{
							gpuId.push_back(data.at("Device Id")[idevice]);
						}

						fs::path dir_output_tmp = data.at("Output directory");
						dir_output = dir_output_tmp;

						is_plot = data.at("Is plot figure");
					}
				}
				catch (json::exception e) {
					std::cout << e.what() << std::endl;
					std::exit(EXIT_FAILURE);
				}


			}
			else {
				std::cerr << "Error: File \"" << path_input_para[i] << "\" is not exist." << std::endl;
				std::exit(EXIT_FAILURE);
			}
		}
	}
	else
	{
		std::cerr << "Number of input file path should be 1 or 2, but now is " << path_input_para.size() << std::endl;
		for (auto i : path_input_para)
		{
			std::cerr << "\t" << i << std::endl;
		}
		std::exit(EXIT_FAILURE);
	}

	dir_output_statistic = dir_output / "statistic" / yearMonDay / hourMinSec;
	dir_output_distribution = dir_output / "distribution" / yearMonDay / hourMinSec;
	dir_output_tuneSpread = dir_output / "tuneSpread" / yearMonDay / hourMinSec;
	dir_output_chargeDensity = dir_output / "chargeDensity" / yearMonDay / hourMinSec;
	dir_output_plot = dir_output / "plot" / yearMonDay / hourMinSec;

	try
	{
		fs::create_directories(dir_output);
		fs::create_directories(dir_output_statistic);
		fs::create_directories(dir_output_distribution);
		fs::create_directories(dir_output_tuneSpread);
		fs::create_directories(dir_output_chargeDensity);
		fs::create_directories(dir_output_plot);
	}
	catch (const fs::filesystem_error& e)
	{
		std::cerr << "Can not create directory: " << e.what() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	try
	{
		if (1 == Nbeam)
		{
			fs::copy(path_input_para[0], dir_output_statistic / (hourMinSec + "_beam0.json"), fs::copy_options::overwrite_existing);
		}
		else
		{
			fs::copy(path_input_para[0], dir_output_statistic / (hourMinSec + "_beam0.json"), fs::copy_options::overwrite_existing);
			fs::copy(path_input_para[1], dir_output_statistic / (hourMinSec + "_beam1.json"), fs::copy_options::overwrite_existing);
		}
	}
	catch (const fs::filesystem_error& e)
	{
		std::cerr << "Can not copy file: " << e.what() << std::endl;
		std::exit(EXIT_FAILURE);
	}
}