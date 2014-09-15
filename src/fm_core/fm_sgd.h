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
// fm_sgd.h: Generic SGD for elementwise and pairwise losses for Factorization
//           Machines
//
// Based on the publication(s):
// - Steffen Rendle (2010): Factorization Machines, in Proceedings of the 10th
//   IEEE International Conference on Data Mining (ICDM 2010), Sydney,
//   Australia.

#ifndef FM_SGD_H_
#define FM_SGD_H_

#include "fm_model.h"

void fm_SGD(fm_model* fm, const double& learn_rate, sparse_row<DATA_FLOAT> &x, const double multiplier, DVector<double> &sum) {
	if (fm->k0) {
		double& w0 = fm->w0;
		w0 -= learn_rate * (multiplier + fm->reg0 * w0);
	}
	if (fm->k1) {
		for (uint i = 0; i < x.size; i++) {
			double& w = fm->w(x.data[i].id);
			w -= learn_rate * (multiplier * x.data[i].value + fm->regw * w);
		}
	}	
	for (int f = 0; f < fm->num_factor; f++) {
		for (uint i = 0; i < x.size; i++) {
			double& v = fm->v(f,x.data[i].id);
			double grad = sum(f) * x.data[i].value - v * x.data[i].value * x.data[i].value; 
			v -= learn_rate * (multiplier * grad + fm->regv * v);
		}
	}	
}
		
void fm_pairSGD(fm_model* fm, const double& learn_rate, sparse_row<DATA_FLOAT> &x_pos, sparse_row<DATA_FLOAT> &x_neg, const double multiplier, DVector<double> &sum_pos, DVector<double> &sum_neg, DVector<bool> &grad_visited, DVector<double> &grad) {
	if (fm->k0) {
		double& w0 = fm->w0;
		w0 -= fm->reg0 * w0; // w0 should always be 0			
	}
	if (fm->k1) {
		for (uint i = 0; i < x_pos.size; i++) {
			grad(x_pos.data[i].id) = 0;
			grad_visited(x_pos.data[i].id) = false;
		}
		for (uint i = 0; i < x_neg.size; i++) {
			grad(x_neg.data[i].id) = 0;
			grad_visited(x_neg.data[i].id) = false;
		}
		for (uint i = 0; i < x_pos.size; i++) {
			grad(x_pos.data[i].id) += x_pos.data[i].value;
		}
		for (uint i = 0; i < x_neg.size; i++) {
			grad(x_neg.data[i].id) -= x_neg.data[i].value;
		}
		for (uint i = 0; i < x_pos.size; i++) {
			uint& attr_id = x_pos.data[i].id;
			if (! grad_visited(attr_id)) {
				double& w = fm->w(attr_id);
				w -= learn_rate * (multiplier * grad(attr_id) + fm->regw * w);
				grad_visited(attr_id) = true;
			}
		}
		for (uint i = 0; i < x_neg.size; i++) {
			uint& attr_id = x_neg.data[i].id;
			if (! grad_visited(attr_id)) {
				double& w = fm->w(attr_id);
				w -= learn_rate * (multiplier * grad(attr_id) + fm->regw * w);
				grad_visited(attr_id) = true;
			}
		}			
	}
	
	for (int f = 0; f < fm->num_factor; f++) {
		for (uint i = 0; i < x_pos.size; i++) {
			grad(x_pos.data[i].id) = 0;
			grad_visited(x_pos.data[i].id) = false;
		}
		for (uint i = 0; i < x_neg.size; i++) {
			grad(x_neg.data[i].id) = 0;
			grad_visited(x_neg.data[i].id) = false;
		}
		for (uint i = 0; i < x_pos.size; i++) {
			grad(x_pos.data[i].id) += sum_pos(f) * x_pos.data[i].value - fm->v(f, x_pos.data[i].id) * x_pos.data[i].value * x_pos.data[i].value; 
		}
		for (uint i = 0; i < x_neg.size; i++) {
			grad(x_neg.data[i].id) -= sum_neg(f) * x_neg.data[i].value - fm->v(f, x_neg.data[i].id) * x_neg.data[i].value * x_neg.data[i].value;
		}
		for (uint i = 0; i < x_pos.size; i++) {
			uint& attr_id = x_pos.data[i].id;
			if (! grad_visited(attr_id)) {
				double& v = fm->v(f,attr_id);
				v -= learn_rate * (multiplier * grad(attr_id) + fm->regv * v);
				grad_visited(attr_id) = true;
			}
		}
		for (uint i = 0; i < x_neg.size; i++) {
			uint& attr_id = x_neg.data[i].id;
			if (! grad_visited(attr_id)) {
				double& v = fm->v(f,attr_id);
				v -= learn_rate * (multiplier * grad(attr_id) + fm->regv * v);
				grad_visited(attr_id) = true;
			}
		}	
	

	}
			
} 
#endif /*FM_SGD_H_*/
