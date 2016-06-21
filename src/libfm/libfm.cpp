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
// libfm.cpp: main file for libFM (Factorization Machines)
//
// Based on the publication(s):
// - Steffen Rendle (2010): Factorization Machines, in Proceedings of the 10th
//   IEEE International Conference on Data Mining (ICDM 2010), Sydney,
//   Australia.
// - Steffen Rendle, Zeno Gantner, Christoph Freudenthaler, Lars Schmidt-Thieme
//   (2011): Fast Context-aware Recommendations with Factorization Machines, in
//   Proceedings of the 34th international ACM SIGIR conference on Research and
//   development in information retrieval (SIGIR 2011), Beijing, China.
// - Christoph Freudenthaler, Lars Schmidt-Thieme, Steffen Rendle (2011):
//   Bayesian Factorization Machines, in NIPS Workshop on Sparse Representation
//   and Low-rank Approximation (NIPS-WS 2011), Spain.
// - Steffen Rendle (2012): Learning Recommender Systems with Adaptive
//   Regularization, in Proceedings of the 5th ACM International Conference on
//   Web Search and Data Mining (WSDM 2012), Seattle, USA.  
// - Steffen Rendle (2012): Factorization Machines with libFM, ACM Transactions
//   on Intelligent Systems and Technology (TIST 2012).
// - Steffen Rendle (2013): Scaling Factorization Machines to Relational Data,
//   in Proceedings of the 39th international conference on Very Large Data
//   Bases (VLDB 2013), Trento, Italy.

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <iterator>
#include <algorithm>
#include <iomanip>
#include "../util/util.h"
#include "../util/cmdline.h"
#include "../fm_core/fm_model.h"
#include "src/Data.h"
#include "src/fm_learn.h"
#include "src/fm_learn_sgd.h"
#include "src/fm_learn_sgd_element.h"


using namespace std;

int main(int argc, char **argv) { 
 	
	try {
		CMDLine cmdline(argc, argv);
		std::cout << "----------------------------------------------------------------------------" << std::endl;
		std::cout << "libFM" << std::endl;
		std::cout << "  Version: 1.4.2" << std::endl;
		std::cout << "  Author:  Steffen Rendle, srendle@libfm.org" << std::endl;
		std::cout << "  WWW:     http://www.libfm.org/" << std::endl;
		std::cout << "This program comes with ABSOLUTELY NO WARRANTY; for details see license.txt." << std::endl;
		std::cout << "This is free software, and you are welcome to redistribute it under certain" << std::endl;
		std::cout << "conditions; for details see license.txt." << std::endl;
		std::cout << "----------------------------------------------------------------------------" << std::endl;
		
		const std::string param_task		= cmdline.registerParameter("task", "r=regression, c=binary classification [MANDATORY]");
		const std::string param_meta_file	= cmdline.registerParameter("meta", "filename for meta information about data set [DISABLED]");
		const std::string param_train_file	= cmdline.registerParameter("train", "filename for training data [MANDATORY]");
		const std::string param_test_file	= cmdline.registerParameter("test", "filename for test data [MANDATORY]");
		const std::string param_val_file	= cmdline.registerParameter("validation", "filename for validation data [MANDATORY]");
		const std::string param_out		= cmdline.registerParameter("out", "filename for output");

		const std::string param_dim		= cmdline.registerParameter("dim", "'k0,k1,k2': k0=use bias, k1=use 1-way interactions, k2=dim of 2-way interactions; default=1,1,8");
		const std::string param_regular		= cmdline.registerParameter("regular", "'r0,r1,r2' for SGD and ALS: r0=bias regularization, r1=1-way regularization, r2=2-way regularization");
		const std::string param_init_stdev	= cmdline.registerParameter("init_stdev", "stdev for initialization of 2-way factors; default=0.003");
		const std::string param_num_iter	= cmdline.registerParameter("iter", "number of iterations; default=40");
		const std::string param_learn_rate	= cmdline.registerParameter("learn_rate", "learn_rate for SGD; default=0.001");

		const std::string param_method		= cmdline.registerParameter("method", "learning method SGD; default=SGD");

		const std::string param_verbosity	= cmdline.registerParameter("verbosity", "how much infos to print; default=0");
		const std::string param_r_log		= cmdline.registerParameter("rlog", "write measurements within iterations to a file; default=''");
		const std::string param_seed		= cmdline.registerParameter("seed", "integer value, default=None");

		const std::string param_help            = cmdline.registerParameter("help", "this screen");

		const std::string param_relation	= cmdline.registerParameter("relation", "BS: filenames for the relations, default=''");

		const std::string param_cache_size	= cmdline.registerParameter("cache_size", "cache size for data storage (only applicable if data is in binary format), default=infty");

		const std::string param_save_model 	= cmdline.registerParameter("save_model", "filename for writing the FM model");
		const std::string param_load_model 	= cmdline.registerParameter("load_model", "filename for reading the FM model");

		const std::string param_early_stop  = cmdline.registerParameter("early_stop", "is early stopping enabled. Default: enabled.");
		const std::string param_num_stop    = cmdline.registerParameter("num_stop", "number of rounds to check for early stop. Default: 10.");

		const std::string param_do_sampling	= "do_sampling";
		const std::string param_do_multilevel	= "do_multilevel";
		const std::string param_num_eval_cases  = "num_eval_cases";

		if (cmdline.hasParameter(param_help) || (argc == 1)) {
			cmdline.print_help();
			return 0;
		}
		cmdline.checkParameters();
		
		if (cmdline.hasParameter(param_meta_file)) {
			std::cout << "Loading external meta info is not supported" << std::endl;
			return 0;
		}

		if (cmdline.getValue(param_method).compare("sgd") != 0) {
			std::cout << "Wrong Optimization Method." << std::endl;
			return 0;
		}

		if (cmdline.hasParameter(param_load_model)) {
			std::cout << "Model Loading is not supported" << std::endl;
			return 0;
		}
		if (cmdline.getValue("task").compare("c") != 0) {
			std::cout << "Supported only classification task" << std::endl;
			return 0;
		}

		// Seed
		long int seed = cmdline.getValue(param_seed, time(NULL));
		srand ( seed );

		if (! cmdline.hasParameter(param_method)) { 
			cmdline.setValue(param_method, "sgd");
			std::cout << "No Optimization method was found in settings. Default: sgd." << std::endl;
		}
		if (! cmdline.hasParameter(param_init_stdev)) { 
			cmdline.setValue(param_init_stdev, "0.0003");
			std::cout << "No Initial Standard deviation pool. Default: 0.001.";
		}
		if (! cmdline.hasParameter(param_dim)) { 
			cmdline.setValue(param_dim, "1,1,8"); 
			std::cout << "No Initial Dimensions. Default 1,1,8.";
		}

		if (! cmdline.hasParameter(param_do_sampling)) { 
			cmdline.setValue(param_do_sampling, "0"); 
		}
		
		if (! cmdline.hasParameter(param_do_multilevel)) { 
			cmdline.setValue(param_do_multilevel, "0"); 
		}

		if (! cmdline.hasParameter(param_val_file)) {
			std::cout << "Validation data is [MANDATORY]" << std::endl;
			std::cout << "No data for early stopping." << std::endl;
			return 0;
		}

		// (1) Load the data
		std::cout << "Loading train...\t" << std::endl;
		Data train(cmdline.getValue(param_cache_size, 0), true, false);

		train.load(cmdline.getValue(param_train_file));
		if (cmdline.getValue(param_verbosity, 0) > 0) { train.debug(); }

		std::cout << "Loading test... \t" << std::endl;
		Data test(cmdline.getValue(param_cache_size, 0), true, false); 
		
		test.load(cmdline.getValue(param_test_file));

		if (cmdline.getValue(param_verbosity, 0) > 0) {test.debug();}

		std::cout << "Loading validation... \t" << std::endl;
		Data validation(cmdline.getValue(param_cache_size, 0), true, false);

		validation.load(cmdline.getValue(param_val_file));

		if (cmdline.getValue(param_verbosity, 0) > 0) {validation.debug();}

		DVector<RelationData*> relation;
		{
			vector<std::string> rel = cmdline.getStrValues(param_relation);
		
			std::cout << "#relations: " << rel.size() << std::endl;
			relation.setSize(rel.size());
			train.relation.setSize(rel.size());
			test.relation.setSize(rel.size());
			validation.relation.setSize(rel.size());
			for (uint i = 0; i < rel.size(); i++) {
				relation(i) = new RelationData(cmdline.getValue(param_cache_size, 0), true, false);
				relation(i)->load(rel[i]);
				train.relation(i).data = relation(i);
				test.relation(i).data = relation(i);
				validation.relation(i).data = relation(i);
				train.relation(i).load(rel[i] + ".train", train.num_cases);
				test.relation(i).load(rel[i] + ".test", test.num_cases);
				validation.relation(i).load(rel[i] + ".validation", validation.num_cases);
			}
		}
		
		// (1.3) Load meta data
		std::cout << "Loading meta data...\t" << std::endl;
		
		// (main table)
		uint num_all_attribute = std::max(std::max(train.num_feature, test.num_feature), validation.num_feature);
		
		// build the joined meta table
		for (uint r = 0; r < train.relation.dim; r++) {
			train.relation(r).data->attr_offset = num_all_attribute;
			num_all_attribute += train.relation(r).data->num_feature;
		}
		DataMetaInfo meta_main(num_all_attribute);
		DataMetaInfo meta(num_all_attribute);
		
		{
			meta.num_attr_groups = meta_main.num_attr_groups;
			for (uint r = 0; r < relation.dim; r++) {
				meta.num_attr_groups += relation(r)->meta->num_attr_groups;
			}
			meta.num_attr_per_group.setSize(meta.num_attr_groups);
			meta.num_attr_per_group.init(0);		
			for (uint i = 0; i < meta_main.attr_group.dim; i++) {
				meta.attr_group(i) = meta_main.attr_group(i);
				meta.num_attr_per_group(meta.attr_group(i))++;
			}

			uint attr_cntr = meta_main.attr_group.dim;
			uint attr_group_cntr = meta_main.num_attr_groups;
			for (uint r = 0; r < relation.dim; r++) {
				for (uint i = 0; i < relation(r)->meta->attr_group.dim; i++) {
					meta.attr_group(i+attr_cntr) = attr_group_cntr + relation(r)->meta->attr_group(i);
					meta.num_attr_per_group(attr_group_cntr + relation(r)->meta->attr_group(i))++;
				}
				attr_cntr += relation(r)->meta->attr_group.dim;
				attr_group_cntr += relation(r)->meta->num_attr_groups;
			}
			if (cmdline.getValue(param_verbosity, 0) > 0) { meta.debug(); }
	
		}
		meta.num_relations = train.relation.dim;

		// (2) Setup the factorization machine
		fm_model fm;
		{
			fm.num_attribute = num_all_attribute;
			fm.init_stdev = cmdline.getValue(param_init_stdev, 0.1);
			// set the number of dimensions in the factorization
			{ 
				vector<int> dim = cmdline.getIntValues(param_dim);
				assert(dim.size() == 3);
				fm.k0 = dim[0] != 0;
				fm.k1 = dim[1] != 0;
				fm.num_factor = dim[2];					
			}			
			fm.init();		
			
		}
		
		// (3) Setup the learning method:
		fm_learn* fml = new fm_learn_sgd_element();
		((fm_learn_sgd*)fml)->num_iter = cmdline.getValue(param_num_iter, 100);

		fml->fm = &fm;
		fml->max_target = train.max_target;
		fml->min_target = train.min_target;
		fml->meta = &meta;
		fml->task = 1;
		for (uint i = 0; i < train.target.dim; i++) { 
			if (train.target(i) <= 0.0) { 
				train.target(i) = -1.0; 
			} else {
				train.target(i) = 1.0; 
			}
		}
		for (uint i = 0; i < test.target.dim; i++) {
			if (test.target(i) <= 0.0) { 
				test.target(i) = -1.0; 
			} else {
				test.target(i) = 1.0; 
			}
		}
		for (uint i = 0; i < validation.target.dim; i++) {
			if (validation.target(i) <= 0.0) { 
				validation.target(i) = -1.0;
			} else {
				validation.target(i) = 1.0; 
			}
		}
		
		// (4) init the logging
		RLog* rlog = NULL;	 
		if (cmdline.hasParameter(param_r_log)) {
			ofstream* out_rlog = NULL;
			std::string r_log_str = cmdline.getValue(param_r_log);
	 		out_rlog = new ofstream(r_log_str.c_str());
	 		if (! out_rlog->is_open())	{
	 			throw "Unable to open file " + r_log_str;
	 		}
	 		std::cout << "logging to " << r_log_str.c_str() << std::endl;
			rlog = new RLog(out_rlog);
	 	}
	 	
		fml->log = rlog;
		fml->init();
		// set the regularization; for standard SGD, groups are not supported
		{
			vector<double> reg = cmdline.getDblValues(param_regular);
			assert((reg.size() == 0) || (reg.size() == 1) || (reg.size() == 3));
			if (reg.size() == 0) {
				fm.reg0 = 0.0;
				fm.regw = 0.0;
				fm.regv = 0.0;
			} else if (reg.size() == 1) {
				fm.reg0 = reg[0];
				fm.regw = reg[0];
				fm.regv = reg[0];
			} else {
				fm.reg0 = reg[0];
				fm.regw = reg[1];
				fm.regv = reg[2];
			}		
		}

		{
			fm_learn_sgd* fmlsgd = dynamic_cast<fm_learn_sgd*>(fml); 
			if (fmlsgd) {
				// set the learning rates (individual per layer)
				{ 
		 			vector<double> lr = cmdline.getDblValues(param_learn_rate);
					assert((lr.size() == 1) || (lr.size() == 3));
					if (lr.size() == 1) {
						fmlsgd->learn_rate = lr[0];
						fmlsgd->learn_rates.init(lr[0]);
					} else {
						fmlsgd->learn_rate = 0;
						fmlsgd->learn_rates(0) = lr[0];
						fmlsgd->learn_rates(1) = lr[1];
						fmlsgd->learn_rates(2) = lr[2];
					}		
					fmlsgd->early_stop = cmdline.getValue(param_early_stop, false);
					fmlsgd->num_stop   = cmdline.getValue(param_num_stop, 10);
			

				}
			}
		}

		if (rlog != NULL) {
			rlog->init();
		}
		
		if (cmdline.getValue(param_verbosity, 0) > 0) { 
			fm.debug();			
			fml->debug();			
		}	

		// () learn		
		fml->learn(train, test, validation);

		// () Save prediction
		if (cmdline.hasParameter(param_out)) {
			DVector<double> pred;
			pred.setSize(test.num_cases);
			fml->predict(test, pred);
			pred.save(cmdline.getValue(param_out));	
		}
		
		// () save the FM model
		if (cmdline.hasParameter(param_save_model)) {
			std::cout << "Writing FM model... \t" << std::endl;
			fm.saveModel(cmdline.getValue(param_save_model));
			std::cout << "NOTIFICATION: model saved. Model load is DISABLED." << std::endl;
		}

	} catch (std::string &e) {
		std::cerr << std::endl << "ERROR: " << e << std::endl;
	} catch (char const* &e) {
		std::cerr << std::endl << "ERROR: " << e << std::endl;
	}


}
