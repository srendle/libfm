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
// util.h: Utility functions

#ifndef UTIL_H_
#define UTIL_H_

#include <vector>
#include <ctime>

#ifdef _WIN32
#include <float.h>
#else
#include <sys/resource.h>
#endif

#include <iostream>
#include <fstream>

typedef unsigned int uint;

#ifdef _WIN32
namespace std {
  bool isnan(double d) { return _isnan(d); }
  bool isnan(float f) { return  _isnan(f); }
  bool isinf(double d) { return (! _finite(d)) && (! isnan(d)); }
  bool isinf(float f) { return (! _finite(f)) && (! isnan(f)); }
}
#endif

#include <math.h>

double sqr(double d) { return d*d; }

double sigmoid(double d) { return (double)1.0/(1.0+exp(-d)); }

std::vector<std::string> tokenize(const std::string& str, const std::string& delimiter) {
  std::vector<std::string> result;
  std::string::size_type lastPos = str.find_first_not_of(delimiter, 0);

  std::string::size_type pos = str.find_first_of(delimiter, lastPos);
  while (std::string::npos != pos || std::string::npos != lastPos) {
    result.push_back(str.substr(lastPos, pos - lastPos));
    lastPos = str.find_first_not_of(delimiter, pos);
    pos = str.find_first_of(delimiter, lastPos);
  }
  return result;
}

double getusertime2() {
  return (double) clock_t() / CLOCKS_PER_SEC;
}

double getusertime() {
  #ifdef _WIN32
  return getusertime2();
  #else
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);

  struct timeval tim = ru.ru_utime;
  return (double)tim.tv_sec + (double)tim.tv_usec / 1000000.0;
  #endif
}


double getusertime3() {
  return (double) clock() / CLOCKS_PER_SEC;
}

double getusertime4() {
  return (double) time(NULL);
}

bool fileexists(std::string filename) {
  std::ifstream in_file (filename.c_str());
  return in_file.is_open();
}


#endif /*UTIL_H_*/
