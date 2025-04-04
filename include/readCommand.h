#pragma once 
#include "command.h"
#include "parameter.h"
#include "particle.h"

#include <vector>
#include <algorithm>
#include <tuple>

void read_command_sequence(const Parameter& Para, std::vector<Bunch>& beam, int input_beamId, std::vector<Command*>& command_vec);

int get_priority(const std::string& name);

void sort_commands(std::vector<Command*>& vec);