#pragma once
#include <algorithm>
#include <tuple>
#include <vector>

#include "bunch.h"
#include "command.h"
#include "general.h"
#include "parameter.h"
#include "particle.h"

void read_command_sequence(const Parameter& Para, std::vector<Bunch>& bunch, int input_beamId, std::vector<std::unique_ptr<Command>>& command_vec,
						   TimeEvent& simTime);

int get_priority(const std::string& name);

void sort_commands(std::vector<std::unique_ptr<Command>>& vec);