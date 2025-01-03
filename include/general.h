#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <filesystem>

#include <cmdline/cmdline.h>
#include <tabulate/tabulate.hpp>

#include "parameter.h"

#ifndef STRNCASECMP
#define STRNCASECMP _strnicmp
#endif

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define MY_SYSTEM Windows
#elif __linux__
#define MY_SYSTEM Linux
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

void print_logo();

void print_copyright();

void get_cmd_input(int argc, char** argv, std::vector<std::filesystem::path>& path_input_para, std::string& yearMonDay, std::string& hourMinSec);

void print_beam_and_bunch_parameter(const Parameter&);