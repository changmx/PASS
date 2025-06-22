#pragma once

#include <typeinfo>


class Aperture
{
public:

	virtual ~Aperture() = default;

	virtual bool is_equal(const Aperture* other) const = 0;

	virtual __host__ __device__ int get_particle_position(double x, double y) = 0;

	virtual double get_xmin() = 0;
	virtual double get_xmax() = 0;
	virtual double get_ymin() = 0;
	virtual double get_ymax() = 0;

private:

};


class CircleAperture :public Aperture
{
public:

	CircleAperture(double radius_) {
		radius = fabs(radius_);
		radius_square = radius_ * radius_;
	}

	~CircleAperture() = default;

	bool is_equal(const Aperture* other) const override {

		const CircleAperture* circle = dynamic_cast<const CircleAperture*>(other);

		// if dynamic_cast fails, it means the types are different
		if (circle == nullptr)
			return false;

		// compare all members
		return (fabs(radius - circle->radius) < 1.0e-10);

	}

	__host__ __device__ int get_particle_position(double x, double y) override {

		const double epsilon = 1.0e-10;

		double dist_square = x * x + y * y;
		double diff = dist_square - radius_square;

		int is_inside = (diff < -epsilon);
		int is_outside = (diff > epsilon);
		int is_boundary = (fabs(diff) <= epsilon);

		// -1: outside, 1: inside; 2: on boundary
		return is_outside * (-1) + is_inside * 1 + is_boundary * 2;
	}

	double get_xmin() override {
		return -radius;
	}
	double get_xmax() override {
		return radius;
	}
	double get_ymin() override {
		return -radius;
	}
	double get_ymax() override {
		return radius;
	}

private:

	double radius = 0;	// in unit of m
	double radius_square = 0;

};


class RectangleAperture :public Aperture
{
public:

	RectangleAperture(double half_width_, double half_height_) {
		half_width = fabs(half_width_);
		half_height = fabs(half_height_);
	}

	~RectangleAperture() = default;

	bool is_equal(const Aperture* other)const override {
		const RectangleAperture* rectangle = dynamic_cast<const RectangleAperture*>(other);

		// if dynamic_cast fails, it means the types are different
		if (rectangle == nullptr)
			return false;

		// compare all members
		return (fabs(half_width - rectangle->half_width) < 1.0e-10) &&
			(fabs(half_height - rectangle->half_height) < 1.0e-10);
	}

	__host__ __device__ int get_particle_position(double x, double y) override {

		const double epsilon = 1.0e-10;

		double dist_left = -x - half_width;	// if inside, dist_left < 0
		double dist_right = x - half_width;	// if inside, dist_right < 0
		double dist_bottom = -y - half_height;	// if inside, dist_bottom < 0
		double dist_top = y - half_height;	// if inside, dist_top < 0

		int is_inside = (dist_left < -epsilon) & (dist_right < -epsilon) & (dist_bottom < -epsilon) & (dist_top < -epsilon);
		int is_boundary = (fabs(dist_left) <= epsilon) | (fabs(dist_right) <= epsilon) | (fabs(dist_bottom) <= epsilon) | (fabs(dist_top) <= epsilon);
		int is_outside = 1 - (is_inside | is_boundary);

		// -1: outside, 1: inside; 2: on boundary
		return is_outside * (-1) + is_inside * 1 + is_boundary * 2;
	}

	double get_xmin() override {
		return -half_width;
	}
	double get_xmax() override {
		return half_width;
	}
	double get_ymin() override {
		return -half_height;
	}
	double get_ymax() override {
		return half_height;
	}

private:

	double half_width = 0;	// in unit of m
	double half_height = 0;	// in unit of m

};


class EllipseAperture :public Aperture
{
public:

	EllipseAperture(double hor_semi_axis_, double ver_semi_axis_) {
		hor_semi_axis = fabs(hor_semi_axis_);
		ver_semi_axis = fabs(ver_semi_axis_);
	}

	~EllipseAperture() = default;

	bool is_equal(const Aperture* other) const override {

		const EllipseAperture* ellipse = dynamic_cast<const EllipseAperture*>(other);

		// if dynamic_cast fails, it means the types are different
		if (ellipse == nullptr)
			return false;

		// compare all members
		return (fabs(hor_semi_axis - ellipse->hor_semi_axis) < 1.0e-10) &&
			(fabs(ver_semi_axis - ellipse->ver_semi_axis) < 1.0e-10);

	}

	__host__ __device__ int get_particle_position(double x, double y) override {

		const double epsilon = 1.0e-10;

		double x1 = x / hor_semi_axis;
		double y1 = y / ver_semi_axis;

		double dist_square = x1 * x1 + y1 * y1;

		int is_inside = (dist_square < (1.0 - epsilon));
		int is_boundary = (fabs(dist_square - 1.0) <= epsilon);
		int is_outside = 1 - (is_inside | is_boundary);

		// -1: outside, 1: inside; 2: on boundary
		return is_outside * (-1) + is_inside * 1 + is_boundary * 2;

	}

	double get_xmin() override {
		return -hor_semi_axis;
	}
	double get_xmax() override {
		return hor_semi_axis;
	}
	double get_ymin() override {
		return -ver_semi_axis;
	}
	double get_ymax() override {
		return ver_semi_axis;
	}

private:

	double hor_semi_axis = 0;	// Half the length of the horizontal axis (a), in unit of m
	double ver_semi_axis = 0;	// Half the length of the verticle axis (b), in unit of m

};


bool compare_apertures(const Aperture* a1, const Aperture* a2);