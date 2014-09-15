// Copyright (C) 2011, 2012, 2013, 2014 Steffen Rendle
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
// fm_learn_sgd_element_adapt_reg.h: Stochastic Gradient Descent based learning
// for classification and regression using adaptive shrinkage
//
// Based on the publication(s):
// - Steffen Rendle (2012): Learning Recommender Systems with Adaptive
//   Regularization, in Proceedings of the 5th ACM International Conference on
//   Web Search and Data Mining (WSDM 2012), Seattle, USA.  
//
// theta' = theta - alpha*(grad_theta + 2*lambda*theta)
//        = theta(1-2*alpha*lambda) - alpha*grad_theta
//
// lambda^* = lambda - alpha*(grad_lambda)
// with 	
//  grad_lambdaw0 = (grad l(y(x),y)) * (-2 * alpha * w_0)
//  grad_lambdawg = (grad l(y(x),y)) * (-2 * alpha * (\sum_{l \in group(g)} x_l * w_l))
//  grad_lambdafg = (grad l(y(x),y)) * (-2 * alpha * (\sum_{l} x_l * v'_lf) * \sum_{l \in group(g)} x_l * v_lf) - \sum_{l \in group(g)} x^2_l * v_lf * v'_lf)

#ifndef FM_LEARN_SGD_ELEMENT_ADAPT_REG_H_
#define FM_LEARN_SGD_ELEMENT_ADAPT_REG_H_

#include <sstream>
#include "fm_learn_sgd.h"


class fm_learn_sgd_element_adapt_reg: public fm_learn_sgd {
	public:
		// regularization parameter
		double reg_0; // shrinking the bias towards the mean of the bias (which is the bias) is the same as no regularization.

		DVector<double> reg_w;
		DMatrix<double> reg_v;

		double mean_w, var_w;
		DVector<double> mean_v, var_v;

		// for each parameter there is one gradient to store
		DVector<double> grad_w; 
		DMatrix<double> grad_v;

		Data* validation;

		// local parameters in the lambda_update step
		DVector<double> lambda_w_grad;
		DVector<double> sum_f, sum_f_dash_f;


		virtual void init() {
			fm_learn_sgd::init();

			reg_0 = 0;
			reg_w.setSize(meta->num_attr_groups);
			reg_v.setSize(meta->num_attr_groups, fm->num_factor);

			mean_v.setSize(fm->num_factor);
			var_v.setSize(fm->num_factor);

			grad_w.setSize(fm->num_attribute);
			grad_v.setSize(fm->num_factor, fm->num_attribute);

			grad_w.init(0.0);
			grad_v.init(0.0);

			lambda_w_grad.setSize(meta->num_attr_groups);
			sum_f.setSize(meta->num_attr_groups);
			sum_f_dash_f.setSize(meta->num_attr_groups);


			if (log != NULL) {
				log->addField("rmse_train", std::numeric_limits<double>::quiet_NaN());
				log->addField("rmse_val", std::numeric_limits<double>::quiet_NaN());	
				
				log->addField("wmean", std::numeric_limits<double>::quiet_NaN());
				log->addField("wvar", std::numeric_limits<double>::quiet_NaN());
				for (int f = 0; f < fm->num_factor; f++) {
					{
						std::ostringstream ss;
						ss << "vmean" << f;
						log->addField(ss.str(), std::numeric_limits<double>::quiet_NaN());
					}
					{
						std::ostringstream ss;
						ss << "vvar" << f;
						log->addField(ss.str(), std::numeric_limits<double>::quiet_NaN());
					}
				}
				for (uint g = 0; g < meta->num_attr_groups; g++) {
					{
						std::ostringstream ss;
						ss << "regw[" << g << "]";
						log->addField(ss.str(), std::numeric_limits<double>::quiet_NaN());
					}
					for (int f = 0; f < fm->num_factor; f++) {
						{
							std::ostringstream ss;
							ss << "regv[" << g << "," << f << "]";
							log->addField(ss.str(), std::numeric_limits<double>::quiet_NaN());
						}
					}
				}
			}
		}


		void sgd_theta_step(sparse_row<FM_FLOAT>& x, const DATA_FLOAT target) {
			double p = fm->predict(x, sum, sum_sqr);
			double mult = 0;
			if (task == 0) {
				p = std::min(max_target, p);
				p = std::max(min_target, p);
				mult = 2 * (p - target);
			} else if (task == 1) {
				mult = target * (  (1.0/(1.0+exp(-target*p))) - 1.0 );
			}

			// make the update with my regularization constants:
			if (fm->k0) {
				double& w0 = fm->w0;
				double grad_0 = mult;
				w0 -= learn_rate * (grad_0 + 2 * reg_0 * w0);
			}
			if (fm->k1) {
				for (uint i = 0; i < x.size; i++) {
					uint g = meta->attr_group(x.data[i].id);
					double& w = fm->w(x.data[i].id);
					grad_w(x.data[i].id) = mult * x.data[i].value;
					w -= learn_rate * (grad_w(x.data[i].id) + 2 * reg_w(g) * w);
				}
			}	
			for (int f = 0; f < fm->num_factor; f++) {
				for (uint i = 0; i < x.size; i++) {
					uint g = meta->attr_group(x.data[i].id);
					double& v = fm->v(f,x.data[i].id);
					grad_v(f,x.data[i].id) = mult * (x.data[i].value * (sum(f) - v * x.data[i].value)); // grad_v_if = (y(x)-y) * [ x_i*(\sum_j x_j v_jf) - v_if*x^2 ]			
					v -= learn_rate * (grad_v(f,x.data[i].id) + 2 * reg_v(g,f) * v);
				}
			}	
		}

		double predict_scaled(sparse_row<FM_FLOAT>& x) {
			double p = 0.0;
			if (fm->k0) {	
				p += fm->w0; 
			}
			if (fm->k1) {
				for (uint i = 0; i < x.size; i++) {
					assert(x.data[i].id < fm->num_attribute);
					uint g = meta->attr_group(x.data[i].id);
					double& w = fm->w(x.data[i].id); 
					double w_dash = w - learn_rate * (grad_w(x.data[i].id) + 2 * reg_w(g) * w);
					p += w_dash * x.data[i].value; 
				}
			}
			for (int f = 0; f < fm->num_factor; f++) {
				sum(f) = 0.0;
				sum_sqr(f) = 0.0;
				for (uint i = 0; i < x.size; i++) {
					uint g = meta->attr_group(x.data[i].id);
					double& v = fm->v(f,x.data[i].id); 
					double v_dash = v - learn_rate * (grad_v(f,x.data[i].id) + 2 * reg_v(g,f) * v);
					double d = v_dash * x.data[i].value;
					sum(f) += d;
					sum_sqr(f) += d*d;
				}
				p += 0.5 * (sum(f)*sum(f) - sum_sqr(f));
			}
			return p;
		}

		void sgd_lambda_step(sparse_row<FM_FLOAT>& x, const DATA_FLOAT target) {
			double p = predict_scaled(x);
			double grad_loss = 0;
			if (task == 0) {
				p = std::min(max_target, p);
				p = std::max(min_target, p);
				grad_loss = 2 * (p - target);
			} else if (task == 1) {
				grad_loss = target * ( (1.0/(1.0+exp(-target*p))) -  1.0);
			}		
					
			if (fm->k1) {
				lambda_w_grad.init(0.0);
				for (uint i = 0; i < x.size; i++) {
					uint g = meta->attr_group(x.data[i].id);
					lambda_w_grad(g) += x.data[i].value * fm->w(x.data[i].id); 
				}
				for (uint g = 0; g < meta->num_attr_groups; g++) {
					lambda_w_grad(g) = -2 * learn_rate * lambda_w_grad(g); 
					reg_w(g) -= learn_rate * grad_loss * lambda_w_grad(g);
					reg_w(g) = std::max(0.0, reg_w(g));
				}
			}	
			for (int f = 0; f < fm->num_factor; f++) {
				// grad_lambdafg = (grad l(y(x),y)) * (-2 * alpha * (\sum_{l} x_l * v'_lf) * (\sum_{l \in group(g)} x_l * v_lf) - \sum_{l \in group(g)} x^2_l * v_lf * v'_lf)
				// sum_f_dash      := \sum_{l} x_l * v'_lf, this is independent of the groups
				// sum_f(g)        := \sum_{l \in group(g)} x_l * v_lf
				// sum_f_dash_f(g) := \sum_{l \in group(g)} x^2_l * v_lf * v'_lf
				double sum_f_dash = 0.0;
				sum_f.init(0.0);
				sum_f_dash_f.init(0.0);
				for (uint i = 0; i < x.size; i++) {
					// v_if' =  [ v_if * (1-alpha*lambda_v_f) - alpha * grad_v_if] 
					uint g = meta->attr_group(x.data[i].id);
					double& v = fm->v(f,x.data[i].id); 
					double v_dash = v - learn_rate * (grad_v(f,x.data[i].id) + 2 * reg_v(g,f) * v);
					
					sum_f_dash += v_dash * x.data[i].value;
					sum_f(g) += v * x.data[i].value; 
					sum_f_dash_f(g) += v_dash * x.data[i].value * v * x.data[i].value;
				}
				for (uint g = 0; g < meta->num_attr_groups; g++) {
					double lambda_v_grad = -2 * learn_rate *  (sum_f_dash * sum_f(g) - sum_f_dash_f(g));  
					reg_v(g,f) -= learn_rate * grad_loss * lambda_v_grad;
					reg_v(g,f) = std::max(0.0, reg_v(g,f));
				}
			}

		}

		void update_means() {
			mean_w = 0;
			mean_v.init(0);
			var_w = 0;
			var_v.init(0);
			for (uint j = 0; j < fm->num_attribute; j++) {
				mean_w += fm->w(j);
				var_w += fm->w(j)*fm->w(j);
				for (int f = 0; f < fm->num_factor; f++) {
					mean_v(f) += fm->v(f,j);
					var_v(f) += fm->v(f,j)*fm->v(f,j);
				}
			}
			mean_w /= (double) fm->num_attribute;
			var_w = var_w/fm->num_attribute - mean_w*mean_w;
			for (int f = 0; f < fm->num_factor; f++) {
				mean_v(f) /= fm->num_attribute;
				var_v(f) = var_v(f)/fm->num_attribute - mean_v(f)*mean_v(f);
			}

			mean_w = 0;
			for (int f = 0; f < fm->num_factor; f++) {
				mean_v(f) = 0;
			}			
		}

		virtual void learn(Data& train, Data& test) {
			fm_learn_sgd::learn(train, test);

			std::cout << "Training using self-adaptive-regularization SGD."<< std::endl << "DON'T FORGET TO SHUFFLE THE ROWS IN TRAINING AND VALIDATION DATA TO GET THE BEST RESULTS." << std::endl; 

			// make sure that fm-parameters are initialized correctly (no other side effects)
			fm->w.init(0);
			fm->reg0 = 0;
			fm->regw = 0; 
			fm->regv = 0; 

			// start with no regularization
			reg_w.init(0.0);
			reg_v.init(0.0);
			
			std::cout << "Using " << train.data->getNumRows() << " rows for training model parameters and " << validation->data->getNumRows() << " for training shrinkage." << std::endl;

			// SGD
			for (int i = 0; i < num_iter; i++) {
				double iteration_time = getusertime();

				// SGD-based learning: both lambda and theta are learned
				update_means();
				validation->data->begin();
				for (train.data->begin(); !train.data->end(); train.data->next()) {
					sgd_theta_step(train.data->getRow(), train.target(train.data->getRowIndex()));
					
					if (i > 0) { // make no lambda steps in the first iteration, because some of the gradients (grad_theta) might not be initialized. 
						if (validation->data->end()) {
							update_means();
							validation->data->begin();					
						}
						sgd_lambda_step(validation->data->getRow(), validation->target(validation->data->getRowIndex()));
						validation->data->next();
					}
				}								
				

				// (3) Evaluation					
				iteration_time = (getusertime() - iteration_time);
	
				double rmse_val = evaluate(*validation);
				double rmse_train = evaluate(train);
				double rmse_test = evaluate(test);
				std::cout << "#Iter=" << std::setw(3) << i << "\tTrain=" << rmse_train << "\tTest=" << rmse_test << std::endl;
				if (log != NULL) {
					log->log("wmean", mean_w);						
					log->log("wvar", var_w);					
					for (int f = 0; f < fm->num_factor; f++) {
						{
							std::ostringstream ss;
							ss << "vmean" << f;
							log->log(ss.str(), mean_v(f));
						}
						{
							std::ostringstream ss;
							ss << "vvar" << f;
							log->log(ss.str(), var_v(f));
						}
					}
					for (uint g = 0; g < meta->num_attr_groups; g++) {
						{
							std::ostringstream ss;
							ss << "regw[" << g << "]";
							log->log(ss.str(), reg_w(g));
						}
						for (int f = 0; f < fm->num_factor; f++) {
							{
								std::ostringstream ss;
								ss << "regv[" << g << "," << f << "]";
								log->log(ss.str(), reg_v(g,f));
							}
						}
					}
					log->log("time_learn", iteration_time);
					log->log("rmse_train", rmse_train);
					log->log("rmse_val", rmse_val);
					log->newLine();	
				}
			}		
		}

		void debug() {
			std::cout << "method=sgda" << std::endl;
			fm_learn_sgd::debug();			
		}

		
};

#endif /*FM_LEARN_SGD_ELEMENT_ADAPT_REG_H_*/
