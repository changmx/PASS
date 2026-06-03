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
			if (fs::exists(path_input_para[i]))
			{
				beamId.push_back(i);

				std::ifstream jsonFile(path_input_para[i]);
				json data = json::parse(jsonFile);

				beam_name.push_back(data.at("Name"));
				Nbunch.push_back(data.at("Number of bunches per beam"));

				if (0 == i)
				{
					Nturn = data.at("Number of turns");
					circumference = data.at("Circumference (m)");

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
			else
			{
				std::string error_msg = "[Parameter] Input file path is not exist: " + path_input_para[i];
				throw std::runtime_error(error_msg);
			}
		}
	}
	else
	{
		std::string error_msg = "[Parameter] Number of input file path should be 1 or 2, but now is " + std::to_string(path_input_para.size());
		throw std::runtime_error(error_msg);
	}

	if (1 == path_input_para.size())
	{
		beam_name.push_back("empty");
		beamId.push_back(-1);
		Nbunch.push_back(0);
	}

	dir_output_statistic = dir_output / "statistic" / yearMonDay / hourMinSec;
	dir_output_parameter = dir_output / "parameter" / yearMonDay / hourMinSec;
	dir_output_distribution = dir_output / "distribution" / yearMonDay / hourMinSec;
	dir_output_tuneSpread = dir_output / "tuneSpread" / yearMonDay / hourMinSec;
	dir_output_chargeDensity = dir_output / "chargeDensity" / yearMonDay / hourMinSec;
	dir_output_plot = dir_output / "plot" / yearMonDay / hourMinSec;
	dir_output_particle = dir_output / "particleMonitor" / yearMonDay / hourMinSec;
	dir_output_slowExt_particle = dir_output / "slowExtraction" / yearMonDay / hourMinSec;

	dir_load_distribution = dir_output / "distribution" / "fixed";

	path_logfile = dir_output_statistic / (hourMinSec + ".log");

	fs::create_directories(dir_output);
	fs::create_directories(dir_output_statistic);
	fs::create_directories(dir_output_parameter);
	fs::create_directories(dir_output_distribution);
	fs::create_directories(dir_output_tuneSpread);
	fs::create_directories(dir_output_chargeDensity);
	fs::create_directories(dir_output_plot);
	fs::create_directories(dir_output_particle);
	fs::create_directories(dir_output_slowExt_particle);

	fs::create_directories(dir_load_distribution);

	std::ofstream outputFile_tmp(path_logfile, std::ios::out | std::ios::trunc);
	outputFile_tmp.close();

	if (1 == Nbeam)
	{
		fs::copy(path_input_para[0], dir_output_statistic / (hourMinSec + "_beam0.json"), fs::copy_options::overwrite_existing);
		fs::copy(path_input_para[0], dir_output_parameter / (hourMinSec + "_beam0.json"), fs::copy_options::overwrite_existing);
	}
	else
	{
		fs::copy(path_input_para[0], dir_output_statistic / (hourMinSec + "_beam0.json"), fs::copy_options::overwrite_existing);
		fs::copy(path_input_para[1], dir_output_statistic / (hourMinSec + "_beam1.json"), fs::copy_options::overwrite_existing);
		fs::copy(path_input_para[0], dir_output_parameter / (hourMinSec + "_beam0.json"), fs::copy_options::overwrite_existing);
		fs::copy(path_input_para[1], dir_output_parameter / (hourMinSec + "_beam1.json"), fs::copy_options::overwrite_existing);
	}
}
