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
// fm_model.h: Model for Factorization Machines
//
// Based on the publication(s):
// - Steffen Rendle (2010): Factorization Machines, in Proceedings of the 10th
//   IEEE International Conference on Data Mining (ICDM 2010), Sydney,
//   Australia.

#ifndef FM_MODEL_H_
#define FM_MODEL_H_

#include "../util/matrix.h"
#include "../util/fmatrix.h"

#include "fm_data.h"


class fm_model {
	private:
		DVector<double> m_sum, m_sum_sqr;
	public:
		double w0;
		DVectorDouble w;
		DMatrixDouble v;

	public:
		// the following values should be set:
		uint num_attribute;
		
		bool k0, k1;
		int num_factor;
		
		double reg0;
		double regw, regv;
		
		double init_stdev;
		double init_mean;
		
		fm_model();
		void debug();
		void init();
		double predict(sparse_row<FM_FLOAT>& x);
		double predict(sparse_row<FM_FLOAT>& x, DVector<double> &sum, DVector<double> &sum_sqr);
	
};



fm_model::fm_model() {
	num_factor = 0;
	init_mean = 0;
	init_stdev = 0.01;
	reg0 = 0.0;
	regw = 0.0;
	regv = 0.0; 
	k0 = true;
	k1 = true;
}

void fm_model::debug() {
	std::cout << "num_attributes=" << num_attribute << std::endl;
	std::cout << "use w0=" << k0 << std::endl;
	std::cout << "use w1=" << k1 << std::endl;
	std::cout << "dim v =" << num_factor << std::endl;
	std::cout << "reg_w0=" << reg0 << std::endl;
	std::cout << "reg_w=" << regw << std::endl;
	std::cout << "reg_v=" << regv << std::endl; 
	std::cout << "init ~ N(" << init_mean << "," << init_stdev << ")" << std::endl;
}

void fm_model::init() {
	w0 = 0;
	w.setSize(num_attribute);
	v.setSize(num_factor, num_attribute);
	w.init(0);
	v.init(init_mean, init_stdev);
	m_sum.setSize(num_factor);
	m_sum_sqr.setSize(num_factor);
}

double fm_model::predict(sparse_row<FM_FLOAT>& x) {
	return predict(x, m_sum, m_sum_sqr);		
}

double fm_model::predict(sparse_row<FM_FLOAT>& x, DVector<double> &sum, DVector<double> &sum_sqr) {
	double result = 0;
	if (k0) {	
		result += w0;
	}
	if (k1) {
		for (uint i = 0; i < x.size; i++) {
			assert(x.data[i].id < num_attribute);
			result += w(x.data[i].id) * x.data[i].value;
		}
	}
	for (int f = 0; f < num_factor; f++) {
		sum(f) = 0;
		sum_sqr(f) = 0;
		for (uint i = 0; i < x.size; i++) {
			double d = v(f,x.data[i].id) * x.data[i].value;
			sum(f) += d;
			sum_sqr(f) += d*d;
		}
		result += 0.5 * (sum(f)*sum(f) - sum_sqr(f));
	}
	return result;
}

#endif /*FM_MODEL_H_*/
