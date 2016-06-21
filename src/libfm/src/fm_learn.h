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
// fm_learn.h: Generic learning method for factorization machines

#ifndef FM_LEARN_H_
#define FM_LEARN_H_

#include <cmath>
#include "Data.h"
#include "../../fm_core/fm_model.h"
#include "../../util/rlog.h"
#include "../../util/util.h"


class fm_learn {
	protected:
		DVector<double> sum, sum_sqr;
		DMatrix<double> pred_q_term;
		
		// this function can be overwritten (e.g. for SGD)
		virtual double predict_case(Data& data) {
			return fm->predict(data.data->getRow());
		}
		
	public:
		DataMetaInfo* meta;
		fm_model* fm;
		double min_target;
		double max_target;

		int task; //1=classification only;

		const static int TASK_CLASSIFICATION = 1;
 
		RLog* log;

		fm_learn() { log = NULL; task = 1; meta = NULL;} 
		
		virtual void init() {
			if (log != NULL) {
				log->addField("accuracy", std::numeric_limits<double>::quiet_NaN());
				log->addField("time_pred", std::numeric_limits<double>::quiet_NaN());
				log->addField("time_learn", std::numeric_limits<double>::quiet_NaN());
				log->addField("time_learn2", std::numeric_limits<double>::quiet_NaN());
				log->addField("time_learn4", std::numeric_limits<double>::quiet_NaN());
			}
			sum.setSize(fm->num_factor);
			sum_sqr.setSize(fm->num_factor);
			pred_q_term.setSize(fm->num_factor, meta->num_relations + 1);
		}

		virtual double evaluate(Data& data) {
			assert(data.data != NULL);
			return evaluate_logloss(data);
		}


	public:
		virtual void learn(Data& train, Data& test, Data& validaiton) { }
		
		virtual void predict(Data& data, DVector<double>& out) = 0;
		
		virtual void debug() { 
			std::cout << "task=" << task << std::endl;
			std::cout << "min_target=" << min_target << std::endl;
			std::cout << "max_target=" << max_target << std::endl;		
		}

	protected:
		virtual double evaluate_logloss(Data& data) {
			double progressive_loss = 0.0;
			for (data.data->begin(); !data.data->end(); data.data->next()) {
				double p = predict_case(data);
				p = 1.0/(1.0 + exp(-p));
				if (data.target(data.data->getRowIndex()) >= 0) {
					progressive_loss += std::log(p);
				} else {
					progressive_loss += std::log(1 - p);
				}
			}
			return -1 * progressive_loss / (double) data.data->getNumRows();
		}
};

#endif /*FM_LEARN_H_*/
