#pragma once

#include <iostream>

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

int string_remove_delimiter(char delimiter, const char* string);

bool check_cmd_line_flag(const int argc, const char** argv,
	const char* string_ref);

void show_help();