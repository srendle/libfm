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
#include "src/fm_learn_sgd_element_adapt_reg.h"
#include "src/fm_learn_mcmc_simultaneous.h"


using namespace std;

int main(int argc, char **argv) {

  try {
    CMDLine cmdline(argc, argv);
    std::cout << "----------------------------------------------------------------------------" << std::endl;
    std::cout << "libFM" << std::endl;
    std::cout << "  Version: 1.4.4" << std::endl;
    std::cout << "  Author:  Steffen Rendle, srendle@libfm.org" << std::endl;
    std::cout << "  WWW:     http://www.libfm.org/" << std::endl;
    std::cout << "This program comes with ABSOLUTELY NO WARRANTY; for details see license.txt." << std::endl;
    std::cout << "This is free software, and you are welcome to redistribute it under certain" << std::endl;
    std::cout << "conditions; for details see license.txt." << std::endl;
    std::cout << "----------------------------------------------------------------------------" << std::endl;

    const std::string param_task       = cmdline.registerParameter("task", "r=regression, c=binary classification [MANDATORY]");
    const std::string param_meta_file  = cmdline.registerParameter("meta", "filename for meta information about data set");
    const std::string param_train_file = cmdline.registerParameter("train", "filename for training data [MANDATORY]");
    const std::string param_test_file  = cmdline.registerParameter("test", "filename for test data [MANDATORY]");
    const std::string param_val_file   = cmdline.registerParameter("validation", "filename for validation data (only for SGDA)");
    const std::string param_out        = cmdline.registerParameter("out", "filename for output");

    const std::string param_dim        = cmdline.registerParameter("dim", "'k0,k1,k2': k0=use bias, k1=use 1-way interactions, k2=dim of 2-way interactions; default=1,1,8");
    const std::string param_regular    = cmdline.registerParameter("regular", "'r0,r1,r2' for SGD and ALS: r0=bias regularization, r1=1-way regularization, r2=2-way regularization");
    const std::string param_init_stdev = cmdline.registerParameter("init_stdev", "stdev for initialization of 2-way factors; default=0.1");
    const std::string param_num_iter   = cmdline.registerParameter("iter", "number of iterations; default=100");
    const std::string param_learn_rate = cmdline.registerParameter("learn_rate", "learn_rate for SGD; default=0.1");

    const std::string param_method     = cmdline.registerParameter("method", "learning method (SGD, SGDA, ALS, MCMC); default=MCMC");

    const std::string param_verbosity  = cmdline.registerParameter("verbosity", "how much infos to print; default=0");
    const std::string param_r_log      = cmdline.registerParameter("rlog", "write measurements within iterations to a file; default=''");
    const std::string param_seed       = cmdline.registerParameter("seed", "integer value, default=None");

    const std::string param_help       = cmdline.registerParameter("help", "this screen");

    const std::string param_relation   = cmdline.registerParameter("relation", "BS: filenames for the relations, default=''");

    const std::string param_cache_size = cmdline.registerParameter("cache_size", "cache size for data storage (only applicable if data is in binary format), default=infty");

    const std::string param_save_model = cmdline.registerParameter("save_model", "filename for writing the FM model");
    const std::string param_load_model = cmdline.registerParameter("load_model", "filename for reading the FM model");

    const std::string param_do_sampling  = "do_sampling";
    const std::string param_do_multilevel  = "do_multilevel";
    const std::string param_num_eval_cases  = "num_eval_cases";

    if (cmdline.hasParameter(param_help) || (argc == 1)) {
      cmdline.print_help();
      return 0;
    }
    cmdline.checkParameters();

    // Seed
    long int seed = cmdline.getValue(param_seed, time(NULL));
    srand ( seed );

    if (! cmdline.hasParameter(param_method)) { cmdline.setValue(param_method, "mcmc"); }
    if (! cmdline.hasParameter(param_init_stdev)) { cmdline.setValue(param_init_stdev, "0.1"); }
    if (! cmdline.hasParameter(param_dim)) { cmdline.setValue(param_dim, "1,1,8"); }

    // Check for invalid flags.
    if (! cmdline.getValue(param_method).compare("mcmc") && cmdline.hasParameter(param_save_model)) {
      std::cout << "WARNING: -save_model enabled only for SGD and ALS." << std::endl;
      cmdline.removeParameter(param_save_model);
      return 0;
    }

    if (! cmdline.getValue(param_method).compare("mcmc") && cmdline.hasParameter(param_load_model)) {
      std::cout << "WARNING: -load_model enabled only for SGD and ALS." << std::endl;
      cmdline.removeParameter(param_load_model);
      return 0;
    }

    if (! cmdline.getValue(param_method).compare("als")) { // als is an mcmc without sampling and hyperparameter inference
      cmdline.setValue(param_method, "mcmc");
      if (! cmdline.hasParameter(param_do_sampling)) { cmdline.setValue(param_do_sampling, "0"); }
      if (! cmdline.hasParameter(param_do_multilevel)) { cmdline.setValue(param_do_multilevel, "0"); }
    }

    // (1) Load the data
    std::cout << "Loading train...\t" << std::endl;
    Data train(
      cmdline.getValue(param_cache_size, 0),
      ! (!cmdline.getValue(param_method).compare("mcmc")), // no original data for mcmc
      ! (!cmdline.getValue(param_method).compare("sgd") || !cmdline.getValue(param_method).compare("sgda")) // no transpose data for sgd, sgda
    );
    train.load(cmdline.getValue(param_train_file));
    if (cmdline.getValue(param_verbosity, 0) > 0) { train.debug(); }

    std::cout << "Loading test... \t" << std::endl;
    Data test(
      cmdline.getValue(param_cache_size, 0),
      ! (!cmdline.getValue(param_method).compare("mcmc")), // no original data for mcmc
      ! (!cmdline.getValue(param_method).compare("sgd") || !cmdline.getValue(param_method).compare("sgda")) // no transpose data for sgd, sgda
    );
    test.load(cmdline.getValue(param_test_file));
    if (cmdline.getValue(param_verbosity, 0) > 0) { test.debug(); }

    Data* validation = NULL;
    if (cmdline.hasParameter(param_val_file)) {
      if (cmdline.getValue(param_method).compare("sgda")) {
        std::cout << "WARNING: Validation data is only used for SGDA. The data is ignored." << std::endl;
      } else {
        std::cout << "Loading validation set...\t" << std::endl;
        validation = new Data(
          cmdline.getValue(param_cache_size, 0),
          ! (!cmdline.getValue(param_method).compare("mcmc")), // no original data for mcmc
          ! (!cmdline.getValue(param_method).compare("sgd") || !cmdline.getValue(param_method).compare("sgda")) // no transpose data for sgd, sgda
        );
        validation->load(cmdline.getValue(param_val_file));
        if (cmdline.getValue(param_verbosity, 0) > 0) { validation->debug(); }
      }
    }

    DVector<RelationData*> relation;
    // (1.2) Load relational data
    {
      vector<std::string> rel = cmdline.getStrValues(param_relation);

      std::cout << "#relations: " << rel.size() << std::endl;
      relation.setSize(rel.size());
      train.relation.setSize(rel.size());
      test.relation.setSize(rel.size());
      for (uint i = 0; i < rel.size(); i++) {
        relation(i) = new RelationData(
          cmdline.getValue(param_cache_size, 0),
          ! (!cmdline.getValue(param_method).compare("mcmc")), // no original data for mcmc
          ! (!cmdline.getValue(param_method).compare("sgd") || !cmdline.getValue(param_method).compare("sgda")) // no transpose data for sgd, sgda
        );
        relation(i)->load(rel[i]);
        train.relation(i).data = relation(i);
        test.relation(i).data = relation(i);
        train.relation(i).load(rel[i] + ".train", train.num_cases);
        test.relation(i).load(rel[i] + ".test", test.num_cases);
      }
    }

    // (1.3) Load meta data
    std::cout << "Loading meta data...\t" << std::endl;

    // (main table)
    uint num_all_attribute = std::max(train.num_feature, test.num_feature);
    if (validation != NULL) {
      num_all_attribute = std::max(num_all_attribute, (uint) validation->num_feature);
    }
    DataMetaInfo meta_main(num_all_attribute);
    if (cmdline.hasParameter(param_meta_file)) {
      meta_main.loadGroupsFromFile(cmdline.getValue(param_meta_file));
    }

    // build the joined meta table
    for (uint r = 0; r < train.relation.dim; r++) {
      train.relation(r).data->attr_offset = num_all_attribute;
      num_all_attribute += train.relation(r).data->num_feature;
    }
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

    // (2.1) load the FM model
    if (cmdline.hasParameter(param_load_model)) {
      std::cout << "Reading FM model... \t" << std::endl;
      if(!fm.loadModel(cmdline.getValue(param_load_model))){
        std::cout << "WARNING: malformed model file. Nothing will be loaded." << std::endl;
        fm.init();
      }
    }

    // (3) Setup the learning method:
    fm_learn* fml;
    if (! cmdline.getValue(param_method).compare("sgd")) {
      fml = new fm_learn_sgd_element();
      ((fm_learn_sgd*)fml)->num_iter = cmdline.getValue(param_num_iter, 100);

    } else if (! cmdline.getValue(param_method).compare("sgda")) {
      assert(validation != NULL);
      fml = new fm_learn_sgd_element_adapt_reg();
      ((fm_learn_sgd*)fml)->num_iter = cmdline.getValue(param_num_iter, 100);
      ((fm_learn_sgd_element_adapt_reg*)fml)->validation = validation;

    } else if (! cmdline.getValue(param_method).compare("mcmc")) {
      fm.w.init_normal(fm.init_mean, fm.init_stdev);
      fml = new fm_learn_mcmc_simultaneous();
      fml->validation = validation;
      ((fm_learn_mcmc*)fml)->num_iter = cmdline.getValue(param_num_iter, 100);
      ((fm_learn_mcmc*)fml)->num_eval_cases = cmdline.getValue(param_num_eval_cases, test.num_cases);

      ((fm_learn_mcmc*)fml)->do_sample = cmdline.getValue(param_do_sampling, true);
      ((fm_learn_mcmc*)fml)->do_multilevel = cmdline.getValue(param_do_multilevel, true);
    } else {
      throw "unknown method";
    }
    fml->fm = &fm;
    fml->max_target = train.max_target;
    fml->min_target = train.min_target;
    fml->meta = &meta;
    if (! cmdline.getValue("task").compare("r") ) {
      fml->task = 0;
    } else if (! cmdline.getValue("task").compare("c") ) {
      fml->task = 1;
      for (uint i = 0; i < train.target.dim; i++) { if (train.target(i) <= 0.0) { train.target(i) = -1.0; } else {train.target(i) = 1.0; } }
      for (uint i = 0; i < test.target.dim; i++) { if (test.target(i) <= 0.0) { test.target(i) = -1.0; } else {test.target(i) = 1.0; } }
      if (validation != NULL) {
        for (uint i = 0; i < validation->target.dim; i++) { if (validation->target(i) <= 0.0) { validation->target(i) = -1.0; } else {validation->target(i) = 1.0; } }
      }
    } else {
      throw "unknown task";
    }

    // (4) init the logging
    RLog* rlog = NULL;
    if (cmdline.hasParameter(param_r_log)) {
      ofstream* out_rlog = NULL;
      std::string r_log_str = cmdline.getValue(param_r_log);
      out_rlog = new ofstream(r_log_str.c_str());
      if (! out_rlog->is_open())  {
        throw "Unable to open file " + r_log_str;
      }
      std::cout << "logging to " << r_log_str.c_str() << std::endl;
      rlog = new RLog(out_rlog);
    }

    fml->log = rlog;
    fml->init();
    if (! cmdline.getValue(param_method).compare("mcmc")) {
      // set the regularization; for als and mcmc this can be individual per group
      {
        vector<double> reg = cmdline.getDblValues(param_regular);
        assert((reg.size() == 0) || (reg.size() == 1) || (reg.size() == 3) || (reg.size() == (1+meta.num_attr_groups*2)));
        if (reg.size() == 0) {
          fm.reg0 = 0.0;
          fm.regw = 0.0;
          fm.regv = 0.0;
          ((fm_learn_mcmc*)fml)->w_lambda.init(fm.regw);
          ((fm_learn_mcmc*)fml)->v_lambda.init(fm.regv);
        } else if (reg.size() == 1) {
          fm.reg0 = reg[0];
          fm.regw = reg[0];
          fm.regv = reg[0];
          ((fm_learn_mcmc*)fml)->w_lambda.init(fm.regw);
          ((fm_learn_mcmc*)fml)->v_lambda.init(fm.regv);
        } else if (reg.size() == 3) {
          fm.reg0 = reg[0];
          fm.regw = reg[1];
          fm.regv = reg[2];
          ((fm_learn_mcmc*)fml)->w_lambda.init(fm.regw);
          ((fm_learn_mcmc*)fml)->v_lambda.init(fm.regv);
        } else {
          fm.reg0 = reg[0];
          fm.regw = 0.0;
          fm.regv = 0.0;
          int j = 1;
          for (uint g = 0; g < meta.num_attr_groups; g++) {
            ((fm_learn_mcmc*)fml)->w_lambda(g) = reg[j];
            j++;
          }
          for (uint g = 0; g < meta.num_attr_groups; g++) {
            for (int f = 0; f < fm.num_factor; f++) {
              ((fm_learn_mcmc*)fml)->v_lambda(g,f) = reg[j];
            }
            j++;
          }
        }
      }
    } else {
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
    }
    {
      fm_learn_sgd* fmlsgd= dynamic_cast<fm_learn_sgd*>(fml);
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
    fml->learn(train, test);

    // () Prediction at the end  (not for mcmc and als)
    if (cmdline.getValue(param_method).compare("mcmc")) {
      std::cout << "Final\t" << "Train=" << fml->evaluate(train) << "\tTest=" << fml->evaluate(test) << std::endl;
    }

    // () Save prediction
    if (cmdline.hasParameter(param_out)) {
      DVector<double> pred;
      pred.setSize(test.num_cases);
      fml->predict(test, pred);
      pred.save(cmdline.getValue(param_out));
    }

    // () save the FM model
    if (cmdline.hasParameter(param_save_model)) {
      std::cout << "Writing FM model to "<< cmdline.getValue(param_save_model) << std::endl;
      fm.saveModel(cmdline.getValue(param_save_model));
    }

  } catch (std::string &e) {
    std::cerr << std::endl << "ERROR: " << e << std::endl;
  } catch (char const* &e) {
    std::cerr << std::endl << "ERROR: " << e << std::endl;
  }
}
