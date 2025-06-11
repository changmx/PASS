#pragma once

#include "command.h"
#include "particle.h"
#include "parameter.h"
#include "general.h"
#include "constant.h"


class Aperture
{
public:

	virtual ~Aperture() = default;

	std::string aperture_type = "empty";

	virtual __host__ __device__ int get_particle_position(double x, double y) = 0;

private:

};


class CircleAperture :public Aperture
{
public:

	CircleAperture(double radius_) {
		aperture_type = "Circle";
		radius = radius_;
		radius_square = radius_ * radius_;
	}

	~CircleAperture() = default;

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

private:

	double radius = 0;	// in unit of m
	double radius_square = 0;

};


class RectangleAperture :public Aperture
{
public:

	RectangleAperture(double half_width_, double half_height_) {
		aperture_type = "Rectangle";
		half_width = half_width_;
		half_height = half_height_;
	}

	~RectangleAperture() = default;

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

private:

	double half_width = 0;	// in unit of m
	double half_height = 0;	// in unit of m

};


class EllipseAperture :public Aperture
{
public:

	EllipseAperture(double hor_semi_axis_, double ver_semi_axis_) {
		aperture_type = "Ellipse";
		hor_semi_axis = hor_semi_axis_;
		ver_semi_axis = ver_semi_axis_;
	}

	~EllipseAperture() = default;

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

private:

	double hor_semi_axis = 0;	// Half the length of the horizontal axis (a), in unit of m
	double ver_semi_axis = 0;	// Half the length of the verticle axis (b), in unit of m

};