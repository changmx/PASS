#include "aperture.h"

bool compare_apertures(const Aperture* a1, const Aperture* a2) {
	// handle null pointers
	if (a1 == nullptr && a2 == nullptr) return true;
	if (a1 == nullptr || a2 == nullptr) return false;

	// use the is_equal() method to compare apertures
	return a1->is_equal(a2);
}