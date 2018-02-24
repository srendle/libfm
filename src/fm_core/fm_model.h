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
 public:
  fm_model();
  void debug();
  void init();
  double predict(sparse_row<FM_FLOAT>& x);
  double predict(sparse_row<FM_FLOAT>& x, DVector<double> &sum, DVector<double> &sum_sqr);
  void saveModel(std::string model_file_path);
  int loadModel(std::string model_file_path);

  double w0;
  DVectorDouble w;
  DMatrixDouble v;

  // the following values should be set:
  uint num_attribute;

  bool k0, k1;
  int num_factor;

  double reg0;
  double regw, regv;

  double init_stdev;
  double init_mean;

 private:
  void splitString(const std::string& s, char c, std::vector<std::string>& v);

  DVector<double> m_sum, m_sum_sqr;
};

// Implementation
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

/*
 * Write the FM model (all the parameters) in a file.
 */
void fm_model::saveModel(std::string model_file_path){
  std::ofstream out_model;
  out_model.open(model_file_path.c_str());
  if (k0) {
    out_model << "#global bias W0" << std::endl;
    out_model << w0 << std::endl;
  }
  if (k1) {
    out_model << "#unary interactions Wj" << std::endl;
    for (uint i = 0; i<num_attribute; i++){
      out_model <<  w(i) << std::endl;
    }
  }
  out_model << "#pairwise interactions Vj,f" << std::endl;
  for (uint i = 0; i<num_attribute; i++){
    for (int f = 0; f < num_factor; f++) {
      out_model << v(f,i);
      if (f!=num_factor-1){ out_model << ' '; }
    }
    out_model << std::endl;
  }
  out_model.close();
}

/*
 * Read the FM model (all the parameters) from a file.
 * If no valid conversion could be performed, the function std::atof returns zero (0.0).
 */
int fm_model::loadModel(std::string model_file_path) {
  std::string line;
  std::ifstream model_file (model_file_path.c_str());
  if (model_file.is_open()){
    if (k0) {
      if(!std::getline(model_file,line)){return 0;} // "#global bias W0"
      if(!std::getline(model_file,line)){return 0;}
      w0 = std::atof(line.c_str());
    }
    if (k1) {
      if(!std::getline(model_file,line)){return 0;} //"#unary interactions Wj"
      for (uint i = 0; i<num_attribute; i++){
        if(!std::getline(model_file,line)){return 0;}
        w(i) = std::atof(line.c_str());
      }
    }
    if(!std::getline(model_file,line)){return 0;}; // "#pairwise interactions Vj,f"
    for (uint i = 0; i<num_attribute; i++){
      if(!std::getline(model_file,line)){return 0;}
      std::vector<std::string> v_str;
      splitString(line, ' ', v_str);
      if ((int)v_str.size() != num_factor){return 0;}
      for (int f = 0; f < num_factor; f++) {
        v(f,i) = std::atof(v_str[f].c_str());
      }
    }
    model_file.close();
  }
  else{ return 0;}
  return 1;
}

/*
 * Splits the string s around matches of the given character c, and stores the substrings in the vector v
 */
void fm_model::splitString(const std::string& s, char c, std::vector<std::string>& v) {
  std::string::size_type i = 0;
  std::string::size_type j = s.find(c);
  while (j != std::string::npos) {
    v.push_back(s.substr(i, j-i));
    i = ++j;
    j = s.find(c, j);
    if (j == std::string::npos)
      v.push_back(s.substr(i, s.length()));
  }
}

#endif /*FM_MODEL_H_*/
