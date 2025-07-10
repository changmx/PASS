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
	std::chrono::steady_clock::time_point simulatorStart = std::chrono::steady_clock::now();
	std::string startTime = timeStamp();

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

	callCuda(cudaSetDevice(Para.gpuId[0]));

	// create Bunch objects according input parameters(Para object) and push them into Beam0 and Beam1 vector
	for (size_t i = 0; i < Para.Nbunch[0]; i++)
	{
		Beam0.emplace_back(Para, 0, i);
	}
	for (size_t i = 0; i < Para.Nbunch[1]; i++)
	{
		Beam0.emplace_back(Para, 1, i);
	}

	print_config_parameter(Para);
	print_beam_parameter(Para, Beam0, Beam1);


	TimeEvent simTime;
	simTime.initial(Para.gpuId[0], true);

	//	create vector to store all commands
	std::vector<std::unique_ptr<Command>> command_beam0;
	std::vector<std::unique_ptr<Command>> command_beam1;

	read_command_sequence(Para, Beam0, 0, command_beam0, simTime);
	read_command_sequence(Para, Beam1, 1, command_beam1, simTime);

	logger->debug("Command vector size of beam0: {}", command_beam0.size());
	logger->debug("Command vector size of beam1: {}", command_beam1.size());

	for (int turn = 1; turn < Para.Nturn + 1; turn++)
	{
		callCuda(cudaEventRecord(simTime.startPerTurn, 0));

		logger->info("Turn: {}/{}", turn, Para.Nturn);

		int icb0 = 0;	// i-th command of beam0
		int icb1 = 0;	// i-th command of beam1

		int N_break_off = (0 == Para.Ncollision) ? 1 : Para.Ncollision + 1;

		for (int ibo = 0; ibo < N_break_off; ibo++)
		{
			for (int icb0_tmp = icb0; icb0_tmp < command_beam0.size(); icb0_tmp++)
			{
				//logger->debug("Ececuting: turn = {:<5} command = {:<20} s = {:<10} name = {}",
				//	turn, command_beam0[icb0_tmp]->get_commandType(), command_beam0[icb0_tmp]->get_s(), command_beam0[icb0_tmp]->get_name());

				if ("BeamBeam" == command_beam0[icb0_tmp]->get_commandType())
				{
					icb0 = icb0_tmp;
					break;
				}
				command_beam0[icb0_tmp]->execute(turn);

			}

			for (int icb1_tmp = icb1; icb1_tmp < command_beam1.size(); icb1_tmp++)
			{
				if ("BeamBeam" == command_beam1[icb1_tmp]->get_commandType())
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

		callCuda(cudaEventRecord(simTime.stopPerTurn, 0));
		callCuda(cudaEventSynchronize(simTime.stopPerTurn));
		float time_perTurn_tmp;
		callCuda(cudaEventElapsedTime(&time_perTurn_tmp, simTime.startPerTurn, simTime.stopPerTurn));
		simTime.turn += time_perTurn_tmp;

	}

	simTime.free(Para.gpuId[0]);

	logger->info("All resources have been released.\n");

	logger->info("*********************************** Simulation  Time ***********************************\n");
	std::string endTime = timeStamp();
	std::chrono::steady_clock::time_point simulatorEnd = std::chrono::steady_clock::now();
	double simulatorTime = std::chrono::duration<double>(simulatorEnd - simulatorStart).count();	// in second
	//double simulatorTime = std::chrono::duration<double, std::milli>(simulatorEnd - simulatorStart).count();	// in millisecond

	simTime.print(Para.Nturn, simulatorTime, Para.gpuId[0]);

	logger->info("{:<30} {}", "simulator starts at:", startTime.c_str());
	logger->info("{:<30} {}", "simulator ends at:", endTime.c_str());
	logger->info("{:<30} {:d}h:{:d}min:{:d}s\n", "simulator running:",
		div(simulatorTime, 3600).quot, div(div(simulatorTime, 3600).rem, 60).quot, div(simulatorTime, 60).rem);

}