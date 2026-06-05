#include <fstream>

#include "cuda_runtime.h"
#include "cutSlice.h"
#include "element.h"
#include "injection.h"
#include "monitor.h"
#include "parallelPlan.h"
#include "readCommand.h"
#include "spaceCharge.h"
#include "twiss.h"

void read_command_sequence(const Parameter& Para, std::vector<Bunch>& bunch, int input_beamId, std::vector<std::unique_ptr<Command>>& command_vec,
						   TimeEvent& simTime)
{
	if (1 == Para.Nbeam && 1 == input_beamId)
	{
		return;	 // if we only have one beam, we don't need to read the json file of beam1. The size of beam1 vector will be zero.
	}

	auto logger = spdlog::get("logger");

	// cudaDeviceProp prop; //"cuda_runtime.h"
	// cudaGetDeviceProperties(&prop, Para.gpuId[0]);
	// int maxThreadsPerBlock = prop.maxThreadsPerBlock;

	// nlohmann::order_json means that the data read will remain in the original order of the file
	// if use nlohmann::json, the data read from file will be sorted alphabetically
	using json = nlohmann::ordered_json;

	std::ifstream jsonFile(Para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	/*std::string a = "Sequence";
	double s = data.at(a).at("Injection").at("S (m)");
	std::cout << "s = " << s << ", " << typeid(a).name() << std::endl;*/

	for (auto item = data.at("Sequence").begin(); item != data.at("Sequence").end(); item++)
	{
		try
		{
			/******
			key is the name of the object in the sequence
			command is the keyword used to distinguish the type of the object

			Now, we have these commands (modules):
				Injection: generate particles of a bunch in one turn or multi turns
				Twiss: Point-to-point linear transfer according to the input twiss parameters
				Element: Element-to-element transfer according to the input element parameters
					--> MarkerElement: marker, used to mark the position of an element in the sequence
					--> DriftElement: drift
					--> SBendElement: Sector dipole
					--> RBendElement: rectangular dipole
					--> QuadrupoleElement: quadrupole
					--> SextupoleElement: sextupole
					--> OctupoleElement: octupole
					--> MultipoleElement: multipole
					--> KickerElement: horizontal and vertical kicker
					--> RFElement: RF cavity, perform acceleration or logituninal phase space manipulation
					--> ElSeparatorElement: electrostatic separator
					--> TuneExciterElement: tune exciter
				Monitor: save information
					--> DistMonitor: save the distribution of all particles in a bunch
					--> PhaseMonitor: save the phase advance of all particles in a bunch
					--> StatMonitor: save statistical data of a bunch (e.g. centroid, size, emittance ...)
					--> LumiMonitor: save the luminosity of collision
				SortBunch: sort particles in a bunch according z position and calculate slice information
				SpaceCharge: perform space charge simulation at specified position
				BeamBeam: perform beam-beam simulation at specified position
				Wakefield: perform wakefield simulation at specified position
				Aperture: define PIC boundary and calculate particle loss
					--> CircleAperture: circular aperture
					--> RectangleAperture: rectangular aperture
					--> EllipseAperture: elliptical aperture

				StatMonitor: calculate and save bunch's information like centroid, size, emittance ...
				LumiMonitor: calculate and save collision luminosity
				DistMonitor: save bunch's distribution
				PhaseMonitor: save particle's phase advance for tune spread analysis, FMA analysis ...
			*******/

			std::string ikey = std::string(item.key());

			if ("Injection" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					command_vec.emplace_back(
						std::make_unique<ConcreteCommand<Injection>>(std::make_unique<Injection>(Para, input_beamId, bunch[i], ikey)));
				}
			}
			else if ("Twiss" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(256, 2, bunch[i].Np);

					command_vec.emplace_back(
						std::make_unique<ConcreteCommand<Twiss>>(std::make_unique<Twiss>(Para, input_beamId, bunch[i], ikey, plan1d, simTime)));
				}
			}
			else if ("MarkerElement" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(512, 2, bunch[i].Np);

					command_vec.emplace_back(std::make_unique<ConcreteCommand<MarkerElement>>(
						std::make_unique<MarkerElement>(Para, input_beamId, bunch[i], ikey, plan1d, simTime)));
				}
			}
			else if ("DriftElement" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(512, 2, bunch[i].Np);

					command_vec.emplace_back(std::make_unique<ConcreteCommand<DriftElement>>(
						std::make_unique<DriftElement>(Para, input_beamId, bunch[i], ikey, plan1d, simTime)));
				}
			}
			else if ("SBendElement" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(256, 2, bunch[i].Np);

					command_vec.emplace_back(std::make_unique<ConcreteCommand<SBendElement>>(
						std::make_unique<SBendElement>(Para, input_beamId, bunch[i], ikey, plan1d, simTime)));
				}
			}
			else if ("RBendElement" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(256, 2, bunch[i].Np);

					command_vec.emplace_back(std::make_unique<ConcreteCommand<RBendElement>>(
						std::make_unique<RBendElement>(Para, input_beamId, bunch[i], ikey, plan1d, simTime)));
				}
			}
			else if ("QuadrupoleElement" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(512, 2, bunch[i].Np);

					command_vec.emplace_back(std::make_unique<ConcreteCommand<QuadrupoleElement>>(
						std::make_unique<QuadrupoleElement>(Para, input_beamId, bunch[i], ikey, plan1d, simTime)));
				}
			}
			else if ("SextupoleElement" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(512, 2, bunch[i].Np);

					command_vec.emplace_back(std::make_unique<ConcreteCommand<SextupoleElement>>(
						std::make_unique<SextupoleElement>(Para, input_beamId, bunch[i], ikey, plan1d, simTime)));
				}
			}
			else if ("OctupoleElement" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(512, 2, bunch[i].Np);

					command_vec.emplace_back(std::make_unique<ConcreteCommand<OctupoleElement>>(
						std::make_unique<OctupoleElement>(Para, input_beamId, bunch[i], ikey, plan1d, simTime)));
				}
			}
			else if ("MultipoleElement" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(512, 2, bunch[i].Np);

					command_vec.emplace_back(std::make_unique<ConcreteCommand<MultipoleElement>>(
						std::make_unique<MultipoleElement>(Para, input_beamId, bunch[i], ikey, plan1d, simTime)));
				}
			}
			else if ("KickerElement" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(512, 2, bunch[i].Np);

					command_vec.emplace_back(std::make_unique<ConcreteCommand<KickerElement>>(
						std::make_unique<KickerElement>(Para, input_beamId, bunch[i], ikey, plan1d, simTime)));
				}
			}
			else if ("RFElement" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(512, 2, bunch[i].Np);

					command_vec.emplace_back(std::make_unique<ConcreteCommand<RFElement>>(
						std::make_unique<RFElement>(Para, input_beamId, bunch[i], ikey, plan1d, simTime)));
				}
			}
			else if ("ElSeparatorElement" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(512, 2, bunch[i].Np);

					command_vec.emplace_back(std::make_unique<ConcreteCommand<ElSeparatorElement>>(
						std::make_unique<ElSeparatorElement>(Para, input_beamId, bunch[i], ikey, plan1d, simTime)));
				}
			}
			else if ("TuneExciterElement" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(512, 2, bunch[i].Np);

					command_vec.emplace_back(std::make_unique<ConcreteCommand<TuneExciterElement>>(
						std::make_unique<TuneExciterElement>(Para, input_beamId, bunch[i], ikey, plan1d, simTime)));
				}
			}
			else if ("DistMonitor" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					command_vec.emplace_back(
						std::make_unique<ConcreteCommand<DistMonitor>>(std::make_unique<DistMonitor>(Para, input_beamId, bunch[i], ikey, simTime)));
				}
			}
			else if ("StatMonitor" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					// Important!!! block_x in cal_statistic_perblock() is 256, so we set it to 256 here. Do not change it !!!
					ParallelPlan1d plan1d(256, 4, bunch[i].Np);

					command_vec.emplace_back(std::make_unique<ConcreteCommand<StatMonitor>>(
						std::make_unique<StatMonitor>(Para, input_beamId, bunch[i], ikey, plan1d, simTime)));
				}
			}
			else if ("ParticleMonitor" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(512, 1, bunch[i].Np);

					command_vec.emplace_back(std::make_unique<ConcreteCommand<ParticleMonitor>>(
						std::make_unique<ParticleMonitor>(Para, input_beamId, bunch[i], ikey, plan1d, simTime)));
				}
			}
			else if ("PhaseMonitor" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(512, 1, bunch[i].Np);

					command_vec.emplace_back(std::make_unique<ConcreteCommand<PhaseMonitor>>(
						std::make_unique<PhaseMonitor>(Para, input_beamId, bunch[i], ikey, plan1d, simTime)));
				}
			}
			else if ("SortBunch" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(512, 1, bunch[i].Np);

					command_vec.emplace_back(std::make_unique<ConcreteCommand<SortBunch>>(
						std::make_unique<SortBunch>(Para, input_beamId, bunch[i], ikey, plan1d, simTime)));
				}
			}
			else if ("SpaceCharge" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(512, 1, bunch[i].Np);

					command_vec.emplace_back(std::make_unique<ConcreteCommand<SpaceCharge>>(
						std::make_unique<SpaceCharge>(Para, input_beamId, bunch[i], ikey, plan1d, simTime)));
				}
			}
			else
			{
				spdlog::get("logger")->warn("[Read sequence] We don't suppoert '{}' command of object: {}",
											data.at("Sequence").at(ikey).at("Command"), ikey);
			}
			// std::string test1 = std::string(item.key());

			// std::cout << data.at("Sequence").at(test1).at("Command") << std::endl;
			// std::cout << data.at("Sequence").at(test1).at("S (m)") << std::endl;
		}
		catch (json::exception e)
		{
			std::string error_msg = std::string("[Read Command] ") + e.what();
			throw std::runtime_error(error_msg);
		}
	}

	sort_commands(command_vec);

	logger->set_pattern("%v");
	info_centered(logger, "Sequence Brief Info");
	logger->info("");
	logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

	for (const auto& cmd : command_vec)
	{
		spdlog::get("logger")->info("[Read Command] Command = {:<10} s = {:<10} name = {}", cmd->get_commandType(), cmd->get_s(), cmd->get_name());
	}
}

int get_priority(const std::string& commandType)
{
	if (commandType == "Injection")
		return 0;  // The highest priority

	else if (commandType == "SortBunch")
		return 100;	 // The second priority

	else if (commandType == "Twiss")
		return 200;
	else if (commandType == "MarkerElement")
		return 300;
	else if (commandType == "DriftElement")
		return 300;
	else if (commandType == "SBendElement")
		return 300;
	else if (commandType == "RBendElement")
		return 300;
	else if (commandType == "QuadrupoleElement")
		return 300;
	else if (commandType == "SextupoleElement")
		return 300;
	else if (commandType == "OctupoleElement")
		return 300;
	else if (commandType == "MultipoleElement")
		return 300;
	else if (commandType == "KickerElement")
		return 300;
	else if (commandType == "RFElement")
		return 300;
	else if (commandType == "ElSeparatorElement")
		return 300;
	else if (commandType == "TuneExciterElement")
		return 300;

	else if (commandType == "SpaceCharge")
		return 400;
	else if (commandType == "Wakefield")
		return 500;
	else if (commandType == "BeamBeam")
		return 600;
	else if (commandType == "ElectronCloud")
		return 700;

	else if (commandType == "LumiMonitor")
		return 800;
	else if (commandType == "PhaseMonitor")
		return 800;
	else if (commandType == "DistMonitor")
		return 800;
	else if (commandType == "StatMonitor")
		return 800;
	else if (commandType == "ParticleMonitor")
		return 800;

	else
		return 999;	 // The last priority
}

void sort_commands(std::vector<std::unique_ptr<Command>>& command_vec)
{
	std::sort(command_vec.begin(), command_vec.end(),
			  [](const std::unique_ptr<Command>& a, const std::unique_ptr<Command>& b)
			  {
				  if (fabs(a->get_s() - b->get_s()) > 1e-10)
				  {
					  return a->get_s() < b->get_s();  // �� s ����
				  }
				  else
				  {
					  return get_priority(a->get_commandType()) < get_priority(b->get_commandType());
				  }
			  });
}