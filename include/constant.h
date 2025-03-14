#ifndef PASS_CONSTANT
#define PASS_CONSTANT

namespace PassConstant {
	// Data from Particle Data Group 2024. https://pdg.lbl.gov/2024/reviews/contents_sports.html

	constexpr double PI = 3.141592653589793;

	constexpr double c = 299792458;	// speed of light in vacuum (m/s)
	constexpr double e = 1.602176634e-19;	// electron charge magnitude (C)

	constexpr double me = 0.51099895000e6;	// electron mass (eV/c2)
	constexpr double me_kg = 9.1093837015e-31;	// electron mass (kg)

	constexpr double mp = 938.27208816e6;	// proton mass (eV/c2)
	constexpr double mp_kg = 1.67262192369e-27;	// proton mass (kg)

	constexpr double mn = 939.56542052e6;	// neutron mass (eV/c2)

	constexpr double md = 1875.61294257e6;	// deuteron mass (eV/c2)

	constexpr double mu = 931.49410242e6;	// unified atomic mass unit, (mass 12C atom)/12, (eV/c2)
	constexpr double mu_kg = 1.66053906660e-27;	// unified atomic mass unit, (mass 12C atom)/12, (kg)

	constexpr double epsilon0 = 8.8541878128e-12;	// permittivity of free space, epsilon0=1/mu0c2 (F/m)
	constexpr double mu0 = 1.00000000055 * 4 * PI * 1e-7;	// permeability of free space

	constexpr double re = 2.8179403262e-15;	// classical electron radius (m)
	// constexpr double rp = re * me / mp;	// classical proton radius (m)
	constexpr double rp = 1.5346982672e-18;	// classical proton radius (m)
}

#endif // !PASS_CONSTANT
