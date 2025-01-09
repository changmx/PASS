#include <iostream>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

#include "general.h"
#include "particle.h"
#include "parameter.h"
#include "command.h"
#include "injection.h"


int main(int argc, char** argv)
{
	Parameter Para(argc, argv);

	create_logger(Para);
	std::shared_ptr<spdlog::logger> logger = spdlog::get("logger");

	print_logo(Para);
	print_copyright(Para);

	std::vector<Bunch> Beam0;
	std::vector<Bunch> Beam1;
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

	for (auto iter = Beam0.begin(); iter != Beam0.end(); iter++)
	{
		iter->init_gpu_memory();
	}
	for (auto iter = Beam1.begin(); iter != Beam1.end(); iter++)
	{
		iter->init_gpu_memory();
	}


	Injection* inj_beam0 = new Injection(Para, 0, Beam0[0]);
	Injection* inj_beam1 = new Injection(Para, 1, Beam1[0]);

	Command* command0 = new InjectionCommand(inj_beam0);
	Command* command1 = new InjectionCommand(inj_beam1);

	command0->execute();
	command1->execute();

	for (auto iter = Beam0.begin(); iter != Beam0.end(); iter++)
	{
		iter->free_gpu_memory();
	}
	for (auto iter = Beam1.begin(); iter != Beam1.end(); iter++)
	{
		iter->free_gpu_memory();
	}
}