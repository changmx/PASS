#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <filesystem>

#include <cmdline/cmdline.h>
#include <tabulate/tabulate.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

#include "parameter.h"
#include "particle.h"
#include "command.h"

#ifndef STRNCASECMP
#define STRNCASECMP _strnicmp
#endif

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define MY_SYSTEM "Windows"
#elif __linux__
#define MY_SYSTEM "Linux"
#elif __APPLE__
printf("\nSorry, we don't support Apple system now.\n");
exit(1);
#elif __unix__
printf("\nSorry, we don't support Unix system now.\n");
exit(1);
#else
printf("\nUnknown runtime system.\n");
exit(1);
#endif

void print_logo(const Parameter& Para);

void print_copyright(const Parameter& Para);

void get_cmd_input(int argc, char** argv, std::vector<std::filesystem::path>& path_input_para, std::string& yearMonDay, std::string& hourMinSec);

void print_config_parameter(const Parameter&);

void print_beam_parameter(const Parameter& Para, const std::vector<Bunch>& Beam0, const std::vector<Bunch>& Beam1);

void create_logger(const Parameter& Para);

void show_device_info();

void read_simulation_config(const Parameter& Para, const std::vector<Bunch>& beam, int input_beamId, std::vector<Command*>& command_vec);