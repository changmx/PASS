#pragma once 
#include "command.h"
#include "parameter.h"
#include "particle.h"

void read_command_sequence(const Parameter& Para, const std::vector<Bunch>& beam, int input_beamId, std::vector<Command*>& command_vec);