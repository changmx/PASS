#include "readCommand.h"
#include "injection.h"
#include "twiss.h"
#include "element.h"
#include "parallelPlan.h"
#include "monitor.h"
#include "cutSlice.h"

#include <fstream>
#include "cuda_runtime.h"

void read_command_sequence(const Parameter& Para, std::vector<Bunch>& bunch, int input_beamId, std::vector<std::unique_ptr<Command>>& command_vec, TimeEvent& simTime) {

	if (1 == Para.Nbeam && 1 == input_beamId)
	{
		return;	// if we only have one beam, we don't need to read the json file of beam1. The size of beam1 vector will be zero.
	}

	cudaDeviceProp prop; //"cuda_runtime.h"
	cudaGetDeviceProperties(&prop, Para.gpuId[0]);
	int maxThreadsPerBlock = prop.maxThreadsPerBlock;

	// nlohmann::order_json means that the data read will remain in the original order of the file
	// if use nlohmann::json, the data read from file will be sorted alphabetically
	using json = nlohmann::ordered_json;

	std::ifstream jsonFile(Para.path_input_para[input_beamId]);
	json data = json::parse(jsonFile);

	/*std::string a = "Sequence";
	double s = data.at(a).at("Injection").at("S (m)");
	std::cout << "s = " << s << ", " << typeid(a).name() << std::endl;*/

	for (auto item = data.at("Sequence").begin(); item != data.at("Sequence").end(); item++) {

		try
		{
			/******
			key is the name of the object in the sequence
			command is the keyword used to distinguish the type of the object

			Now, we have these commands (modules):
				Injection: generate particles of a bunch in one turn or multi turns
				Twiss: Point-to-point linear transfer according to the input twiss parameters
				Element: Element-to-element transfer according to the input element parameters
					--> SBendElement: Sector dipole
					--> RBendElement: rectangular dipole
					--> QuadrupoleElement: quadrupole
					--> SextupoleElement: sextupole
					--> OctupoleElement: octupole
					--> HKickerElement: horizontal kicker
					--> VKickerElement: vertical kicker
					--> MarkerElement: marker, used to mark the position of an element in the sequence
					--> RFElement: RF cavity, perform acceleration or logituninal phase space manipulation
				Monitor: save information
					--> DistMonitor: save the distribution of all particles in a bunch
					--> PhaseMonitor: save the phase advance of all particles in a bunch
					--> StatMonitor: save statistical data of a bunch (e.g. centroid, size, emittance ...)
					--> LumiMonitor: save the luminosity of collision
				SortBunch: sort particles in a bunch according z position and calculate slice information
				SpaceCharge: perform space charge simulation at specified position
				BeamBeam: perform beam-beam simulation at specified position
				Wakefield: perform wakefield simulation at specified position

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
						std::make_unique<ConcreteCommand<Injection>>(
							std::make_unique<Injection>(Para, input_beamId, bunch[i], ikey))
					);
				}

			}
			else if ("Twiss" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(maxThreadsPerBlock, 1, bunch[i].Np);

					command_vec.emplace_back(
						std::make_unique<ConcreteCommand<Twiss>>(
							std::make_unique<Twiss>(Para, input_beamId, bunch[i], ikey, plan1d, simTime))
					);
				}
			}
			else if ("MarkerElement" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(maxThreadsPerBlock, 1, bunch[i].Np);

					command_vec.emplace_back(
						std::make_unique<ConcreteCommand<MarkerElement>>(
							std::make_unique<MarkerElement>(Para, input_beamId, bunch[i], ikey, plan1d, simTime))
					);
				}
			}
			else if ("SBendElement" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(maxThreadsPerBlock, 1, bunch[i].Np);

					command_vec.emplace_back(
						std::make_unique<ConcreteCommand<SBendElement>>(
							std::make_unique<SBendElement>(Para, input_beamId, bunch[i], ikey, plan1d, simTime))
					);
				}
			}
			else if ("RBendElement" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(maxThreadsPerBlock, 1, bunch[i].Np);

					command_vec.emplace_back(
						std::make_unique<ConcreteCommand<RBendElement>>(
							std::make_unique<RBendElement>(Para, input_beamId, bunch[i], ikey, plan1d, simTime))
					);
				}
			}
			else if ("QuadrupoleElement" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(maxThreadsPerBlock, 1, bunch[i].Np);

					command_vec.emplace_back(
						std::make_unique<ConcreteCommand<QuadrupoleElement>>(
							std::make_unique<QuadrupoleElement>(Para, input_beamId, bunch[i], ikey, plan1d, simTime))
					);
				}
			}
			else if ("SextupoleElement" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(maxThreadsPerBlock, 1, bunch[i].Np);

					command_vec.emplace_back(
						std::make_unique<ConcreteCommand<SextupoleElement>>(
							std::make_unique<SextupoleElement>(Para, input_beamId, bunch[i], ikey, plan1d, simTime))
					);
				}
			}
			else if ("OctupoleElement" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(maxThreadsPerBlock, 1, bunch[i].Np);

					command_vec.emplace_back(
						std::make_unique<ConcreteCommand<OctupoleElement>>(
							std::make_unique<OctupoleElement>(Para, input_beamId, bunch[i], ikey, plan1d, simTime))
					);
				}
			}
			else if ("HKickerElement" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(maxThreadsPerBlock, 1, bunch[i].Np);

					command_vec.emplace_back(
						std::make_unique<ConcreteCommand<HKickerElement>>(
							std::make_unique<HKickerElement>(Para, input_beamId, bunch[i], ikey, plan1d, simTime))
					);
				}
			}
			else if ("VKickerElement" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(maxThreadsPerBlock, 1, bunch[i].Np);

					command_vec.emplace_back(
						std::make_unique<ConcreteCommand<VKickerElement>>(
							std::make_unique<VKickerElement>(Para, input_beamId, bunch[i], ikey, plan1d, simTime))
					);
				}
			}
			else if ("DistMonitor" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					command_vec.emplace_back(
						std::make_unique<ConcreteCommand<DistMonitor>>(
							std::make_unique<DistMonitor>(Para, input_beamId, bunch[i], ikey, simTime))
					);
				}
			}
			else if ("StatMonitor" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					// Important!!! block_x in cal_statistic_perblock() is 256, so we set it to 256 here. Do not change it !!!
					ParallelPlan1d plan1d(256, 4, bunch[i].Np);

					command_vec.emplace_back(
						std::make_unique<ConcreteCommand<StatMonitor>>(
							std::make_unique<StatMonitor>(Para, input_beamId, bunch[i], ikey, plan1d, simTime))
					);
				}
			}
			else if ("SortBunch" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					ParallelPlan1d plan1d(maxThreadsPerBlock, 1, bunch[i].Np);

					command_vec.emplace_back(
						std::make_unique<ConcreteCommand<SortBunch>>(
							std::make_unique<SortBunch>(Para, input_beamId, bunch[i], ikey, plan1d, simTime))
					);
				}
			}
			else
			{
				spdlog::get("logger")->warn("[Read sequence] We don't suppoert '{}' command of object: {}", data.at("Sequence").at(ikey).at("Command"), ikey);
				//std::exit(EXIT_FAILURE);
			}
			//std::string test1 = std::string(item.key());

			//std::cout << data.at("Sequence").at(test1).at("Command") << std::endl;
			//std::cout << data.at("Sequence").at(test1).at("S (m)") << std::endl;

		}
		catch (json::exception e)
		{
			spdlog::get("logger")->error(e.what());
			std::exit(EXIT_FAILURE);
		}

	}

	sort_commands(command_vec);

	for (const auto& cmd : command_vec)
	{
		spdlog::get("logger")->info("[ReadCommand] Print sequence: command = {:<20} s = {:<10} name = {}", cmd->get_commandType(), cmd->get_s(), cmd->get_name());
	}
}

int get_priority(const std::string& commandType) {
	// 将commandType转换为优先级数值
	if (commandType == "Injection") return 0;	// 最高优先级
	else if (commandType == "Twiss") return 50;	// 次级优先级
	else if (commandType == "MarkerElement") return 100;
	else if (commandType == "SBendElement") return 100;
	else if (commandType == "RBendElement") return 100;
	else if (commandType == "QuadrupoleElement") return 100;
	else if (commandType == "SextupoleElement") return 100;
	else if (commandType == "OctupoleElement") return 100;
	else if (commandType == "HKickerElement") return 100;
	else if (commandType == "VKickerElement") return 100;
	else if (commandType == "RFElement") return 100;
	else if (commandType == "SpaceCharge") return 120;
	//else if (commandType == "Wakefield") return 150;
	//else if (commandType == "LumiMonitor") return 150;
	//else if (commandType == "PhaseMonitor") return 150;
	else if (commandType == "DistMonitor") return 150;
	else if (commandType == "StatMonitor") return 150;
	else if (commandType == "SortBunch") return 200;
	else if (commandType == "BeamBeam") return 300;
	else return 999; // 最低优先级，其他情况
}

//void sort_commands(std::vector<Command*>& vec) {
//	std::sort(vec.begin(), vec.end(), [](const Command* a, const Command* b) {
//		// 分步比较指针指向的对象的成员
//		if (a->s != b->s) {
//			return a->s < b->s;  // 按 s 升序
//		}
//		else {
//			// s 相等时，按 name 的优先级升序
//			return get_priority(a->name) < get_priority(b->name);
//		}
//		});
//}

void sort_commands(std::vector<std::unique_ptr<Command>>& command_vec) {
	std::sort(command_vec.begin(), command_vec.end(),
		[](const std::unique_ptr<Command>& a, const std::unique_ptr<Command>& b) {
			// 分步比较指针指向的对象的成员
			if (a->get_s() != b->get_s()) {
				return a->get_s() < b->get_s();  // 按 s 升序
			}
			else {
				// s 相等时，按 name 的优先级升序
				return get_priority(a->get_commandType()) < get_priority(b->get_commandType());
			}
		});
}