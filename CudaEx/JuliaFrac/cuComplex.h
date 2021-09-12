#pragma once

struct cuComplex
{
	float r;
	float i;
	//
	explicit cuComplex(float r_ = 0.0f, float i_ = 0.0f) : r(r_), i(i_) {};
	//
	float magnitude2(void) {
		return r * r + i * i;
	}
	cuComplex operator*(const cuComplex& c) {
						//
		return cuComplex(r * c.r - i * c.i, i * c.r + r * c.i);
	};
	cuComplex operator+(const cuComplex& c) {
		return cuComplex(r + c.r, i + c.i);
	}
};