#include <iostream>

#include "general.h"
#include "parameter.h"

int main(int argc, char** argv)
{
	print_logo();
	print_copyright();

	Parameter Para(argc, argv);

	print_beam_and_bunch_parameter(Para);
}