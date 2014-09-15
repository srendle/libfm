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
// fm_learn_sgd.h: Stochastic Gradient Descent based learning
//
// Based on the publication(s):
// - Steffen Rendle (2010): Factorization Machines, in Proceedings of the 10th
//   IEEE International Conference on Data Mining (ICDM 2010), Sydney,
//   Australia.

#ifndef FM_LEARN_SGD_H_
#define FM_LEARN_SGD_H_

#include "fm_learn.h"
#include "../../fm_core/fm_sgd.h"

class fm_learn_sgd: public fm_learn {
	protected:
		//DVector<double> sum, sum_sqr;
	public:
		int num_iter;
		double learn_rate;
		DVector<double> learn_rates;		

		virtual void init() {		
			fm_learn::init();	
			learn_rates.setSize(3);
		//	sum.setSize(fm->num_factor);		
		//	sum_sqr.setSize(fm->num_factor);
		}		

		virtual void learn(Data& train, Data& test) { 
			fm_learn::learn(train, test);
			std::cout << "learnrate=" << learn_rate << std::endl;
			std::cout << "learnrates=" << learn_rates(0) << "," << learn_rates(1) << "," << learn_rates(2) << std::endl;
			std::cout << "#iterations=" << num_iter << std::endl;

			if (train.relation.dim > 0) {
				throw "relations are not supported with SGD";
			}
			std::cout.flush();
		}

		void SGD(sparse_row<DATA_FLOAT> &x, const double multiplier, DVector<double> &sum) {
			fm_SGD(fm, learn_rate, x, multiplier, sum); 
		} 
		
		void debug() {
			std::cout << "num_iter=" << num_iter << std::endl;
			fm_learn::debug();			
		}

		virtual void predict(Data& data, DVector<double>& out) {
			assert(data.data->getNumRows() == out.dim);
			for (data.data->begin(); !data.data->end(); data.data->next()) {
				double p = predict_case(data);
				if (task == TASK_REGRESSION ) {
					p = std::min(max_target, p);
					p = std::max(min_target, p);
				} else if (task == TASK_CLASSIFICATION) {
					p = 1.0/(1.0 + exp(-p));
				} else {
					throw "task not supported";
				}
				out(data.data->getRowIndex()) = p;
			}				
		} 

};

#endif /*FM_LEARN_SGD_H_*/
