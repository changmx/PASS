#include <iostream>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

#include "general.h"
#include "particle.h"
#include "parameter.h"
#include "command.h"
#include "readCommand.h"
#include "injection.h"



int main(int argc, char** argv)
{
	Parameter Para(argc, argv);	// read parameter from json file

	// create a logger named "logger" which can be called globally
	create_logger(Para);
	std::shared_ptr<spdlog::logger> logger = spdlog::get("logger");

	print_logo(Para);
	print_copyright(Para);
	show_device_info();

	// Beam0 and Beam1 are two vectors of Bunch class
	std::vector<Bunch> Beam0;
	std::vector<Bunch> Beam1;

	// create Bunch objects according input parameters(Para object) and push them into Beam0 and Beam1 vector
	for (size_t i = 0; i < Para.Nbunch[0]; i++)
	{
		Bunch bunch(Para, 0, i);
		Beam0.push_back(bunch);
	}
	for (size_t i = 0; i < Para.Nbunch[1]; i++)
	{
		Bunch bunch(Para, 1, i);
		Beam1.push_back(bunch);
	}

	print_config_parameter(Para);
	print_beam_parameter(Para, Beam0, Beam1);

	cudaSetDevice(Para.gpuId[0]);

	//	initialize GPU memory of Bunch objects
	for (auto iter = Beam0.begin(); iter != Beam0.end(); iter++)
	{
		iter->init_memory();
	}
	for (auto iter = Beam1.begin(); iter != Beam1.end(); iter++)
	{
		iter->init_memory();
	}

	//	create vector to store all commands
	std::vector<Command*> command_beam0;
	std::vector<Command*> command_beam1;

	read_command_sequence(Para, Beam0, 0, command_beam0);
	read_command_sequence(Para, Beam1, 1, command_beam1);

	logger->debug("Command vector size of beam0: {}", command_beam0.size());
	logger->debug("Command vector size of beam1: {}", command_beam1.size());

	for (int turn = 0; turn < Para.Nturn; turn++)
	{
		logger->info("Turn: {}/{}", turn, Para.Nturn);

		int icb0 = 0;	// i-th command of beam0
		int icb1 = 0;	// i-th command of beam1

		int N_break_off = (0 == Para.Ncollision) ? 1 : Para.Ncollision + 1;

		for (int ibo = 0; ibo < N_break_off; ibo++)
		{
			for (int icb0_tmp = icb0; icb0_tmp < command_beam0.size(); icb0_tmp++)
			{
				if ("BeamBeam" == command_beam0[icb0_tmp]->name)
				{
					icb0 = icb0_tmp;
					break;
				}

				command_beam0[icb0_tmp]->execute(turn);

			}

			for (int icb1_tmp = icb1; icb1_tmp < command_beam1.size(); icb1_tmp++)
			{
				if ("BeamBeam" == command_beam1[icb1_tmp]->name)
				{
					icb1 = icb1_tmp;
					break;
				}

				command_beam1[icb1_tmp]->execute(turn);

			}

			if (icb0 < command_beam0.size() && icb1 < command_beam1.size())
			{
				// Beam-beam
				//command_beam0[icb0]->execute(turn);
				//command_beam1[icb1]->execute(turn);
			}

			icb0++;
			icb1++;

		}
	}

	for (auto iter = Beam0.begin(); iter != Beam0.end(); iter++)
	{
		iter->free_memory();
	}
	for (auto iter = Beam1.begin(); iter != Beam1.end(); iter++)
	{
		iter->free_memory();
	}
}