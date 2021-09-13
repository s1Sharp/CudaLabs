#pragma once

struct Complex
{
	float r;
	float i;
	//
	explicit Complex(float r_ = 0.0f, float i_ = 0.0f) : r(r_), i(i_) {};
	//
	float magnitude2(void) {
		return r * r + i * i;
	}
	Complex operator*(const Complex& c) {
						//
		return Complex(r * c.r - i * c.i, i * c.r + r * c.i);
	};
	Complex operator+(const Complex& c) {
		return Complex(r + c.r, i + c.i);
	}
};
struct cuComplex 
{
	float r;
	float i;
	__device__ cuComplex(float a, float b) : r(a), i(b) {}
	__device__ float magnitude2(void) {
		return r * r + i * i;
	}
	__device__ cuComplex operator*(const cuComplex& a) {
		return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
	}
	__device__ cuComplex operator+(const cuComplex& a) {
		return cuComplex(r + a.r, i + a.i);
	}
};