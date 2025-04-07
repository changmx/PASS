#include "readCommand.h"
#include "injection.h"
#include "twiss.h"

#include <fstream>

void read_command_sequence(const Parameter& Para, std::vector<Bunch>& bunch, int input_beamId, std::vector<Command*>& command_vec) {

	if (1 == Para.Nbeam && 1 == input_beamId)
	{
		return;	// if we only have one beam, we don't need to read the json file of beam1. The size of beam1 vector will be zero.
	}

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
			Key is the keyword that indicates the types of simulation module
			Now, we have these keys (modules):
				Injection: generate particles of a bunch in one turn or multi turns
				Transfer: transfer particles of a bunch from s0 to s1
				SpaceCharge: perform space charge simulation at specified position
				BeamBeam: perform beam-beam simulation at specified position
				Wakefield: perform wakefield simulation at specified position
				RFCavity: perform acceleration or logituninal phase space manipulation
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
					Injection* inj = new Injection(Para, input_beamId, bunch[i], ikey);
					Command* command = new InjectionCommand(inj);
					command_vec.push_back(command);
				}

			}
			else if ("Twiss" == data.at("Sequence").at(ikey).at("Command"))
			{
				for (size_t i = 0; i < Para.Nbunch[input_beamId]; i++)
				{
					Twiss* twiss = new Twiss(Para, input_beamId, bunch[i], ikey);
					//twiss->print();
					Command* command = new TwissCommand(twiss);
					command_vec.push_back(command);
				}
			}
			else
			{
				spdlog::get("logger")->warn("[Read sequence] We don't suppoert '{}' command of object: {}.", data.at("Sequence").at(ikey).at("Command"), ikey);
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

	for (const Command* cmd : command_vec)
	{
		spdlog::get("logger")->debug("[Print command sequence] name = {}, s = {}", cmd->name, cmd->s);
	}
}

int get_priority(const std::string& name) {
	// 将name转换为优先级数值
	if (name == "Injection") return 0;	// 最高优先级
	else if (name == "Twiss") return 1;	// 次级优先级
	else if (name == "Element") return 2;
	else return 999; // 最低优先级，其他情况
}

void sort_commands(std::vector<Command*>& vec) {
	std::sort(vec.begin(), vec.end(), [](const Command* a, const Command* b) {
		// 分步比较指针指向的对象的成员
		if (a->s != b->s) {
			return a->s < b->s;  // 按 s 升序
		}
		else {
			// s 相等时，按 name 的优先级升序
			return get_priority(a->name) < get_priority(b->name);
		}
		});
}