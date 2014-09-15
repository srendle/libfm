// Copyright (C) 2010, 2011, 2012, 2013, 2014 Steffen Rendle
// Contact:   srendle@libfm.org, http://www.libfm.org/
//
// This file is part of libFM.
//
// libFM is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// libFM is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with libFM.  If not, see <http://www.gnu.org/licenses/>.
//
//
// random.h: Sampling methods

#ifndef RANDOM_H_
#define RANDOM_H_

#include <stdlib.h>
#include <cmath>
#include <assert.h>


double ran_gaussian();
double ran_gaussian(double mean, double stdev);
double ran_left_tgaussian(double left);
double ran_left_tgaussian(double left, double mean, double stdev);
double ran_left_tgaussian_naive(double left);
double ran_uniform();
double ran_exp();			
double ran_gamma(double alpha, double beta);
double ran_gamma(double alpha);
bool ran_bernoulli(double p);

double erf(double x);	
double cdf_gaussian(double x, double mean, double stdev);
double cdf_gaussian(double x);



double erf(double x) {
	double t;
	if (x >= 0) {
		t = 1.0 / (1.0 + 0.3275911 * x);
	} else {
		t = 1.0 / (1.0 - 0.3275911 * x);
	}

	double result = 1.0 - (t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))))*exp(-x*x);
	if (x >= 0) {
		return result;
	} else {
		return -result;
	}
}

double cdf_gaussian(double x, double mean, double stdev) {
	return 0.5 + 0.5 * erf(0.707106781 * (x-mean) / stdev);
}

double cdf_gaussian(double x) {
	return 0.5 + 0.5 * erf(0.707106781 * x );
}


double ran_left_tgaussian(double left) {
	// draw a trunctated normal: acceptance region are values larger than <left>
	if (left <= 0.0) { // acceptance probability > 0.5
		return ran_left_tgaussian_naive(left);
	} else {
		// Robert: Simulation of truncated normal variables
		double alpha_star = 0.5*(left + sqrt(left*left + 4.0));

		// draw from translated exponential distr:
		// f(alpha,left) = alpha * exp(-alpha*(z-left)) * I(z>=left)
		double z,d,u;
		do {
			z = ran_exp() / alpha_star + left;
			d = z-alpha_star;
			d = exp(-(d*d)/2);
			u = ran_uniform();
			if (u < d) {
				return z;
			}
		} while (true);
	}
}

double ran_left_tgaussian_naive(double left) {
	// draw a trunctated normal: acceptance region are values larger than <left>
	double result;
	do {
		result = ran_gaussian();
	} while (result < left);
	return result;
}

double ran_left_tgaussian(double left, double mean, double stdev) {
	return mean + stdev * ran_left_tgaussian((left-mean)/stdev); 
}

double ran_right_tgaussian(double right) {
	return -ran_left_tgaussian(-right);
}

double ran_right_tgaussian(double right, double mean, double stdev) {
	return mean + stdev * ran_right_tgaussian((right-mean)/stdev); 
}



double ran_gamma(double alpha) {
	assert(alpha > 0);
	if (alpha < 1.0) {
		double u;
		do {
			u = ran_uniform();
		} while (u == 0.0);
		return ran_gamma(alpha + 1.0) * pow(u, 1.0 / alpha);
	} else {
		// Marsaglia and Tsang: A Simple Method for Generating Gamma Variables
		double d,c,x,v,u;
		d = alpha - 1.0/3.0;
		c = 1.0 / std::sqrt(9.0 * d);
		do {
			do {
				x = ran_gaussian();
				v = 1.0 + c*x;
			} while (v <= 0.0);
			v = v * v * v;
			u = ran_uniform();
		} while ( 
			(u >= (1.0 - 0.0331 * (x*x) * (x*x)))
			 && (log(u) >= (0.5 * x * x + d * (1.0 - v + std::log(v))))
			 );
		return d*v;
	}
}

double ran_gamma(double alpha, double beta) {
	return ran_gamma(alpha) / beta;
}

double ran_gaussian() {
	// Joseph L. Leva: A fast normal Random number generator
	double u,v, x, y, Q;
	do {
		do {
			u = ran_uniform();
		} while (u == 0.0); 
		v = 1.7156 * (ran_uniform() - 0.5);
		x = u - 0.449871;
		y = std::abs(v) + 0.386595;
		Q = x*x + y*(0.19600*y-0.25472*x);
		if (Q < 0.27597) { break; }
	} while ((Q > 0.27846) || ((v*v) > (-4.0*u*u*std::log(u)))); 
	return v / u;
}

double ran_gaussian(double mean, double stdev) {
	if ((stdev == 0.0) || (std::isnan(stdev))) {
		return mean;
	} else {
		return mean + stdev*ran_gaussian();
	}
}

double ran_uniform() {
	return rand()/((double)RAND_MAX + 1);
}

double ran_exp() {
	return -std::log(1-ran_uniform());
}

bool ran_bernoulli(double p) {
	return (ran_uniform() < p);
}

#endif /*RANDOM_H_*/
