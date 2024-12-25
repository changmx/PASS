#include "general.h"

void print_logo() {
	std::cout << "                                                                                 " << std::endl;
	std::cout << " PPPPPPPPPPPPPPPPP        AAA                 SSSSSSSSSSSSSSS    SSSSSSSSSSSSSSS " << std::endl;
	std::cout << " P::::::::::::::::P      A:::A              SS:::::::::::::::S SS:::::::::::::::S" << std::endl;
	std::cout << " P::::::PPPPPP:::::P    A:::::A            S:::::SSSSSS::::::SS:::::SSSSSS::::::S" << std::endl;
	std::cout << " PP:::::P     P:::::P  A:::::::A           S:::::S     SSSSSSSS:::::S     SSSSSSS" << std::endl;
	std::cout << "   P::::P     P:::::P A:::::::::A          S:::::S            S:::::S            " << std::endl;
	std::cout << "   P::::P     P:::::PA:::::A:::::A         S:::::S            S:::::S            " << std::endl;
	std::cout << "   P::::PPPPPP:::::PA:::::A A:::::A         S::::SSSS          S::::SSSS         " << std::endl;
	std::cout << "   P:::::::::::::PPA:::::A   A:::::A         SS::::::SSSSS      SS::::::SSSSS    " << std::endl;
	std::cout << "   P::::PPPPPPPPP A:::::A     A:::::A          SSS::::::::SS      SSS::::::::SS  " << std::endl;
	std::cout << "   P::::P        A:::::AAAAAAAAA:::::A            SSSSSS::::S        SSSSSS::::S " << std::endl;
	std::cout << "   P::::P       A:::::::::::::::::::::A                S:::::S            S:::::S" << std::endl;
	std::cout << "   P::::P      A:::::AAAAAAAAAAAAA:::::A               S:::::S            S:::::S" << std::endl;
	std::cout << " PP::::::PP   A:::::A             A:::::A  SSSSSSS     S:::::SSSSSSSS     S:::::S" << std::endl;
	std::cout << " P::::::::P  A:::::A               A:::::A S::::::SSSSSS:::::SS::::::SSSSSS:::::S" << std::endl;
	std::cout << " P::::::::P A:::::A                 A:::::AS:::::::::::::::SS S:::::::::::::::SS " << std::endl;
	std::cout << " PPPPPPPPPPAAAAAAA                   AAAAAAASSSSSSSSSSSSSSS    SSSSSSSSSSSSSSS   " << std::endl;
	std::cout << "                                                                                 " << std::endl;
}

void print_copyright() {};

int string_remove_delimiter(char delimiter, const char* string) {
	int string_start = 0;

	while (string[string_start] == delimiter) {
		string_start++;
	}

	if (string_start >= static_cast<int>(strlen(string) - 1)) {
		return 0;
	}

	return string_start;
}

bool check_cmd_line_flag(const int argc, const char** argv,
	const char* string_ref) {
	bool bFound = false;

	if (argc >= 1) {
		for (int i = 1; i < argc; i++) {
			int string_start = string_remove_delimiter('-', argv[i]);
			const char* string_argv = &argv[i][string_start];

			const char* equal_pos = strchr(string_argv, '=');
			int argv_length = static_cast<int>(
				equal_pos == 0 ? strlen(string_argv) : equal_pos - string_argv);

			int length = static_cast<int>(strlen(string_ref));

			if (length == argv_length &&
				!STRNCASECMP(string_argv, string_ref, length)) {
				bFound = true;
				continue;
			}
		}
	}

	return bFound;
}

void show_help() {
	printf("\t--para=<filepath/input.json>       (load a parameter file for simulation)\n");

}