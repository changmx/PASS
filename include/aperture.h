#pragma once

#include <typeinfo>


struct IntersectionPoints {
	double x_left;  // y=y0 的左交点 x 值
	double x_right; // y=y0 的右交点 x 值
	double y_bottom; // x=x0 的下交点 y 值
	double y_top; // x=x0 的上交点 y 值
};


class Aperture
{
public:

	virtual ~Aperture() = default;

	virtual bool is_equal(const Aperture* other) const = 0;

	virtual __host__ __device__ int get_particle_position(double x, double y) const = 0;

	virtual double get_xmin() const = 0;
	virtual double get_xmax() const = 0;
	virtual double get_ymin() const = 0;
	virtual double get_ymax() const = 0;

	virtual IntersectionPoints get_intersection_points(double x, double y) const = 0;

	enum Type
	{
		CIRCLE, RECTANGLE, ELLIPSE
	};

	Type type;
private:

};


class CircleAperture :public Aperture
{
public:

	CircleAperture(double radius_) {
		radius = fabs(radius_);
		radius_square = radius_ * radius_;
		type = CIRCLE;
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

	__host__ __device__ int get_particle_position(double x, double y) const override {

		const double epsilon = 1.0e-10;

		double dist_square = x * x + y * y;
		double diff = dist_square - radius_square;

		int is_inside = (diff < -epsilon);

		// -1: outside, 1: inside;
		return (is_inside << 1) - 1;	// means is_inside*2-1
	}

	double get_xmin() const override {
		return -radius;
	}
	double get_xmax() const override {
		return radius;
	}
	double get_ymin() const override {
		return -radius;
	}
	double get_ymax() const override {
		return radius;
	}

	IntersectionPoints get_intersection_points(double x, double y) const override {

		double x_left = std::numeric_limits<double>::quiet_NaN();
		double x_right = std::numeric_limits<double>::quiet_NaN();

		double x_square = radius_square - y * y;
		if (x_square >= 0)
		{
			x_left = -sqrt(x_square);
			x_right = sqrt(x_square);
		}

		double y_bottom = std::numeric_limits<double>::quiet_NaN();
		double y_top = std::numeric_limits<double>::quiet_NaN();

		double y_square = radius_square - x * x;
		if (y_square >= 0)
		{
			y_bottom = -sqrt(y_square);
			y_top = sqrt(y_square);
		}

		return { x_left, x_right, y_bottom, y_top };
	}

	double radius = 0;	// in unit of m
	double radius_square = 0;

private:

};


class RectangleAperture :public Aperture
{
public:

	RectangleAperture(double half_width_, double half_height_) {
		half_width = fabs(half_width_);
		half_height = fabs(half_height_);
		type = RECTANGLE;
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

	__host__ __device__ int get_particle_position(double x, double y) const override {

		const double epsilon = 1.0e-10;

		double dist_left = -x - half_width;	// if inside, dist_left < 0
		double dist_right = x - half_width;	// if inside, dist_right < 0
		double dist_bottom = -y - half_height;	// if inside, dist_bottom < 0
		double dist_top = y - half_height;	// if inside, dist_top < 0

		int is_inside = (dist_left < -epsilon) & (dist_right < -epsilon) & (dist_bottom < -epsilon) & (dist_top < -epsilon);

		// -1: outside, 1: inside;
		return (is_inside << 1) - 1;	// means is_inside*2-1
	}

	double get_xmin()  const override {
		return -half_width;
	}
	double get_xmax()  const override {
		return half_width;
	}
	double get_ymin()  const override {
		return -half_height;
	}
	double get_ymax()  const override {
		return half_height;
	}

	IntersectionPoints get_intersection_points(double x, double y) const override {

		double x_left = std::numeric_limits<double>::quiet_NaN();
		double x_right = std::numeric_limits<double>::quiet_NaN();
		double y_bottom = std::numeric_limits<double>::quiet_NaN();
		double y_top = std::numeric_limits<double>::quiet_NaN();

		if (x >= -half_width && x <= half_width) {
			y_bottom = -half_height;   // 下边界
			y_top = half_height;   // 上边界
		}

		if (y >= -half_height && y <= half_height) {
			x_left = -half_width;     // 左边界
			x_right = half_width;    // 右边界
		}

		return { x_left, x_right, y_bottom, y_top };
	}

	double half_width = 0;	// in unit of m
	double half_height = 0;	// in unit of m

private:

};


class EllipseAperture :public Aperture
{
public:

	EllipseAperture(double hor_semi_axis_, double ver_semi_axis_) {
		hor_semi_axis = fabs(hor_semi_axis_);
		ver_semi_axis = fabs(ver_semi_axis_);
		type = ELLIPSE;
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

	__host__ __device__ int get_particle_position(double x, double y) const override {

		const double epsilon = 1.0e-10;

		double x1 = x / hor_semi_axis;	// x/a
		double y1 = y / ver_semi_axis;	// y/b

		double dist_square = x1 * x1 + y1 * y1;

		int is_inside = (dist_square < (1.0 - epsilon));

		// -1: outside, 1: inside;
		return (is_inside << 1) - 1;	// means is_inside*2-1

	}

	double get_xmin()  const override {
		return -hor_semi_axis;
	}
	double get_xmax()  const override {
		return hor_semi_axis;
	}
	double get_ymin() const  override {
		return -ver_semi_axis;
	}
	double get_ymax() const  override {
		return ver_semi_axis;
	}

	IntersectionPoints get_intersection_points(double x, double y) const override {

		double x_left = std::numeric_limits<double>::quiet_NaN();
		double x_right = std::numeric_limits<double>::quiet_NaN();
		double y_bottom = std::numeric_limits<double>::quiet_NaN();
		double y_top = std::numeric_limits<double>::quiet_NaN();

		// 计算 x=x 的 y 交点, (x/a)^2 + (y/b)^2 = 1
		const double x_ratio = x / hor_semi_axis;
		const double x_ratio_sq = x_ratio * x_ratio;

		if (x_ratio_sq <= 1.0) {
			const double y_term = ver_semi_axis * std::sqrt(1.0 - x_ratio_sq);
			y_top = y_term;   // 上半部分交点
			y_bottom = -y_term;  // 下半部分交点
		}

		// 计算 y=y0 的 x 交点
		const double y_ratio = y / ver_semi_axis;
		const double y_ratio_sq = y_ratio * y_ratio;

		if (y_ratio_sq <= 1.0) {
			const double x_term = hor_semi_axis * std::sqrt(1.0 - y_ratio_sq);
			x_left = -x_term;   // 左半部分交点
			x_right = x_term;   // 右半部分交点
		}

		return { x_left, x_right, y_bottom, y_top };
	}

	double hor_semi_axis = 0;	// Half the length of the horizontal axis (a), in unit of m
	double ver_semi_axis = 0;	// Half the length of the verticle axis (b), in unit of m

private:

};


bool compare_apertures(const Aperture* a1, const Aperture* a2);