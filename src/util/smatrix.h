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
// smatrix.h: Sparse Matrices and Tensors

#ifndef SMATRIX_H_
#define SMATRIX_H_

#include <iostream>
#include <fstream>
#include <assert.h>
#include <map>
#include <set>

template <typename T> class SparseVector : public std::map<int,T> {
 public:
  T get(int x);
  void toStream(std::ostream &stream);
};

template <typename T>  class SparseMatrix : public std::map<int, SparseVector<T> > {
 public:
  T get(int x, int y);
  void toStream(std::ostream &stream);
  void fromFile(const std::string &filename);
};

template <typename T> class SparseTensor : public std::map<int, SparseMatrix<T> > {
 public:
  T get(int x, int y, int z);
  void toStream(std::ostream &stream);
  void toFile(const std::string &filename);
  void fromFile(const std::string &filename);
};

class SparseVectorInt : public SparseVector<int> {};
class SparseMatrixInt : public SparseMatrix<int> {};
class SparseTensorInt : public SparseTensor<int> {};
class SparseVectorDouble : public SparseVector<double> {};
class SparseMatrixDouble : public SparseMatrix<double> {};
class SparseTensorDouble : public SparseTensor<double> {};

class SparseVectorBoolean : public std::set<int> {
 public:
  bool get(int x);
};

class SparseMatrixBoolean : public std::map<int, SparseVectorBoolean> {
 public:
  bool get(int x, int y);
  void fromFile(const std::string &filename);
};

class SparseTensorBoolean : public std::map<int, SparseMatrixBoolean> {
 public:
  bool get(int x, int y, int z);
  void toStream(std::ostream &stream);
  void toFile(const std::string &filename);
  void fromFile(const std::string &filename);
};

// Implementation
template <typename T> T SparseVector<T>::get(int x) {
  typename SparseVector<T>::iterator iter = this->find(x);
  if (iter != this->end()) {
    return iter->second;
  } else {
    return 0;
  }
}

template <typename T> void SparseVector<T>::toStream(std::ostream &stream) {
  for(typename SparseVector<T>::const_iter it_cell = this->begin(); it_cell != this->end(); ++it_cell) {
    stream << it_cell->first << " " << it_cell->second << std::endl;
  }
}

template <typename T> T SparseMatrix<T>::get(int x, int y) {
  typename SparseMatrix<T>::iterator iter = this->find(x);
  if (iter != this->end()) {
    return iter->second.get(y);
  } else {
    return 0;
  }
}

template <typename T> void SparseMatrix<T>::toStream(std::ostream &stream) {
  for(typename SparseMatrix<T>::const_iter i = this->begin(); i != this->end(); ++i) {
    for(typename SparseVector<T>::const_iter j = i->second->begin(); j != i->second->end(); ++j) {
      stream << i->first << " " << j->first << " " << j->second << std::endl;
    }
  }
}

template <typename T> T SparseTensor<T>::get(int x, int y, int z) {
  typename SparseTensor<T>::iterator iter = this->find(x);
  if (iter != this->end()) {
    return iter->second.get(y, z);
  } else {
    return 0;
  }
}

template <typename T> void SparseTensor<T>::toStream(std::ostream &stream) {
  for(typename SparseTensor<T>::const_iterator t = this->begin(); t != this->end(); ++t) {
    for(typename SparseMatrix<T>::const_iterator i = t->second.begin(); i != t->second.end(); ++i) {
      for(typename SparseVector<T>::const_iterator j = i->second.begin(); j != i->second.end(); ++j) {
        stream << t->first << " " << i->first << " " << j->first << " " << j->second << std::endl;
      }
    }
  }
}

template <typename T> void SparseTensor<T>::toFile(const std::string &filename) {
  std::ofstream out_file (filename.c_str());
  if (out_file.is_open())  {
    toStream(out_file);
    out_file.close();
  } else {
    throw "Unable to open file " + filename;
  }
}

template <typename T> void SparseTensor<T>::fromFile(const std::string &filename) {
  std::ifstream fData (filename.c_str());
  if (! fData.is_open()) {
    throw "Unable to open file " + filename;
  }
  while (! fData.eof()) {
    int t, m, v;
    fData >> t;
    fData >> m;
    fData >> v;
    if (! fData.eof()) {
      T value;
      fData >> value;
      (*this)[t][m][v] = value;
    }
  }
  fData.close();
}

template <typename T> void SparseMatrix<T>::fromFile(const std::string &filename) {
  std::ifstream fData (filename.c_str());
  if (! fData.is_open()) {
    throw "Unable to open file " + filename;
  }
  while (! fData.eof()) {
    int t, m;
    fData >> t;
    fData >> m;
    if (! fData.eof()) {
      T value;
      fData >> value;
      (*this)[t][m] = value;
    }
  }
  fData.close();
}

bool SparseVectorBoolean::get(int x) {
  SparseVectorBoolean::iterator iter = this->find(x);
  if (iter != this->end()) {
    return true;
  } else {
    return false;
  }
}

bool SparseMatrixBoolean::get(int x, int y) {
  SparseMatrixBoolean::iterator iter = this->find(x);
  if (iter != this->end()) {
    return iter->second.get(y);
  } else {
    return 0;
  }
}

bool SparseTensorBoolean::get(int x, int y, int z) {
  SparseTensorBoolean::iterator iter = this->find(x);
  if (iter != this->end()) {
    return iter->second.get(y, z);
  } else {
    return 0;
  }
}

void SparseTensorBoolean::toStream(std::ostream &stream) {
  for(SparseTensorBoolean::const_iterator t = this->begin(); t != this->end(); ++t) {
    for(SparseMatrixBoolean::const_iterator i = t->second.begin(); i != t->second.end(); ++i) {
      for(SparseVectorBoolean::const_iterator j = i->second.begin(); j != i->second.end(); ++j) {
        stream << t->first << " " << i->first << " " << (*j) << std::endl;
      }
    }
  }
}

void SparseTensorBoolean::toFile(const std::string &filename) {
  std::ofstream out_file (filename.c_str());
  if (out_file.is_open())  {
    toStream(out_file);
    out_file.close();
  } else {
    throw "Unable to open file " + filename;
  }

}

void SparseTensorBoolean::fromFile(const std::string &filename) {
  std::ifstream fData (filename.c_str());
    if (! fData.is_open()) {
    throw "Unable to open file " + filename;
  }
  while (! fData.eof()) {
    int t, m, v;
    fData >> t;
    fData >> m;
    if (! fData.eof()) {
      fData >> v;
      (*this)[t][m].insert(v);
    }
  }
  fData.close();
}


void SparseMatrixBoolean::fromFile(const std::string &filename) {
  std::ifstream fData (filename.c_str());
    if (! fData.is_open()) {
    throw "Unable to open file " + filename;
  }
  while (! fData.eof()) {
    int  m, v;
    fData >> m;
    if (! fData.eof()) {
      fData >> v;
      (*this)[m].insert(v);
    }
  }
  fData.close();
}

#endif /*SMATRIX_H_*/
