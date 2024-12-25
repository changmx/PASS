#include <iostream>

#include "general.h"

int main(int argc, char** argv)
{
	print_logo();
	print_copyright();

	if (check_cmd_line_flag(argc, (const char**)argv, "help")) {
		std::cout << "\n> Command line options" << std::endl;
		show_help();
		return 0;
	}

	if (argc == 1)
	{
		std::cout << "\n> Input files with absolute or relative paths should be specified using the command:\n\t--para=<filepath/input.json>" << std::endl;
		return 1;
	}


}