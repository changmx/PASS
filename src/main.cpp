#include <iostream>

#include "general.h"
#include "particle.h"
#include "parameter.h"

int main(int argc, char** argv)
{
	print_logo();
	print_copyright();

	Parameter para(argc, argv);

	std::vector<Bunch> beam0;
	std::vector<Bunch> beam1;
	for (size_t i = 0; i < para.Nbunch[0]; i++)
	{
		Bunch bunch;
		beam0.push_back(bunch);
	}
	for (size_t i = 0; i < para.Nbunch[1]; i++)
	{
		Bunch bunch;
		beam0.push_back(bunch);
	}

	print_beam_and_bunch_parameter(para);
}