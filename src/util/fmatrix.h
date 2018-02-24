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
// fmatrix.h: Large-Scale Sparse Matrix

#ifndef FMATRIX_H_
#define FMATRIX_H_

#include <limits>
#include <vector>
#include <assert.h>
#include <iostream>
#include <fstream>
#include "../util/random.h"

const uint FMATRIX_EXPECTED_FILE_ID = 2;

template <typename T> struct sparse_entry {
  uint id;
  T value;
};

template <typename T> struct sparse_row {
  sparse_entry<T>* data;
  uint size;
};

struct file_header {
  uint id;
  uint float_size;
  uint64 num_values;
  uint num_rows;
  uint num_cols;
};

template <typename T> class LargeSparseMatrix {
 public:
  virtual void begin() = 0; // go to the beginning
  virtual bool end() = 0;   // are we at the end?
  virtual void next() = 0; // go to the next line
  virtual sparse_row<T>& getRow() = 0; // pointer to the current row
  virtual uint getRowIndex() = 0; // index of current row (starting with 0)
  virtual uint getNumRows() = 0; // get the number of Rows
  virtual uint getNumCols() = 0; // get the number of Cols
  virtual uint64 getNumValues() = 0; // get the number of Values

  void saveToBinaryFile(std::string filename);

  void saveToTextFile(std::string filename);
};

template <typename T> class LargeSparseMatrixHD : public LargeSparseMatrix<T> {
 public:
  LargeSparseMatrixHD(std::string filename, uint64 cache_size);
  // ~LargeSparseMatrixHD() { in.close(); }

  virtual uint getNumRows();
  virtual uint getNumCols();
  virtual uint64 getNumValues();

  virtual void next();
  virtual void begin();
  virtual bool end();

  virtual sparse_row<T>& getRow();
  virtual uint getRowIndex();

 protected:
  void readcache();

  DVector< sparse_row<T> > data;
  DVector< sparse_entry<T> > cache;
  std::string filename;

  std::ifstream in;

  uint64 position_in_data_cache;
  uint number_of_valid_rows_in_cache;
  uint64 number_of_valid_entries_in_cache;
  uint row_index;

  uint num_cols;
  uint64 num_values;
  uint num_rows;
};

template <typename T> class LargeSparseMatrixMemory : public LargeSparseMatrix<T> {
 public:
  virtual void begin();
  virtual bool end();
  virtual void next();
  virtual sparse_row<T>& getRow();
  virtual uint getRowIndex();
  virtual uint getNumRows();
  virtual uint getNumCols();
  virtual uint64 getNumValues();
  // void loadFromTextFile(std::string filename);

  DVector< sparse_row<T> > data;
  uint num_cols;
  uint64 num_values;

 protected:
  uint index;
};

// Implementation
template <typename T> void LargeSparseMatrix<T>::saveToBinaryFile(std::string filename) {
  std::cout << "printing to " << filename << std::endl; std::cout.flush();
  std::ofstream out(filename.c_str(), std::ios_base::out | std::ios_base::binary);
  if (out.is_open()) {
    file_header fh;
    fh.id = FMATRIX_EXPECTED_FILE_ID;
    fh.num_values = getNumValues();
    fh.num_rows = getNumRows();
    fh.num_cols = getNumCols();
    fh.float_size = sizeof(T);
    out.write(reinterpret_cast<char*>(&fh), sizeof(fh));
    for (begin(); !end(); next()) {
      out.write(reinterpret_cast<char*>(&(getRow().size)), sizeof(uint));
      out.write(reinterpret_cast<char*>(getRow().data), sizeof(sparse_entry<T>)*getRow().size);
    }
    out.close();
  } else {
    throw "could not open " + filename;
  }
}

template <typename T> void LargeSparseMatrix<T>::saveToTextFile(std::string filename) {
  std::cout << "printing to " << filename << std::endl; std::cout.flush();
  std::ofstream out(filename.c_str());
  if (out.is_open()) {
    for (begin(); !end(); next()) {
      for (uint i = 0; i < getRow().size; i++) {
        out << getRow().data[i].id << ":" << getRow().data[i].value;
        if ((i+1) < getRow().size) {
          out << " ";
        } else {
          out << "\n";
        }
      }
    }
    out.close();
  } else {
    throw "could not open " + filename;
  }
}

template <typename T> void LargeSparseMatrixHD<T>::readcache() {
  if (row_index >= num_rows) { return; }
  number_of_valid_rows_in_cache = 0;
  number_of_valid_entries_in_cache = 0;
  position_in_data_cache = 0;
  do {
    if ((row_index + number_of_valid_rows_in_cache) > (num_rows-1)) {
      break;
    }
    if (number_of_valid_rows_in_cache >= data.dim) { break; }

    sparse_row<T>& this_row = data.value[number_of_valid_rows_in_cache];

    in.read(reinterpret_cast<char*>(&(this_row.size)), sizeof(uint));
    if ((this_row.size + number_of_valid_entries_in_cache) > cache.dim) {
      in.seekg(- (long int) sizeof(uint), std::ios::cur);
      break;
    }

    this_row.data = &(cache.value[number_of_valid_entries_in_cache]);
    in.read(reinterpret_cast<char*>(this_row.data), sizeof(sparse_entry<T>)*this_row.size);

    number_of_valid_rows_in_cache++;
    number_of_valid_entries_in_cache += this_row.size;
  } while (true);

}

template <typename T> LargeSparseMatrixHD<T>::LargeSparseMatrixHD(std::string filename, uint64 cache_size) {
  this->filename = filename;
  in.open(filename.c_str(), std::ios_base::in | std::ios_base::binary);
  if (in.is_open()) {
    file_header fh;
    in.read(reinterpret_cast<char*>(&fh), sizeof(fh));
    assert(fh.id == FMATRIX_EXPECTED_FILE_ID);
    assert(fh.float_size == sizeof(T));
    this->num_values = fh.num_values;
    this->num_rows = fh.num_rows;
    this->num_cols = fh.num_cols;
    //in.close();
  } else {
    throw "could not open " + filename;
  }

  if (cache_size == 0) {
    cache_size = std::numeric_limits<uint64>::max();
  }
  // determine cache sizes automatically:
  double avg_entries_per_line = (double) this->num_values / this->num_rows;
  uint num_rows_in_cache;
  {
    uint64 dummy = cache_size / (sizeof(sparse_entry<T>) * avg_entries_per_line + sizeof(uint));
    if (dummy > static_cast<uint64>(std::numeric_limits<uint>::max())) {
      num_rows_in_cache = std::numeric_limits<uint>::max();
    } else {
      num_rows_in_cache = dummy;
    }
  }
  num_rows_in_cache = std::min(num_rows_in_cache, this->num_rows);
  uint64 num_entries_in_cache = (cache_size - sizeof(uint)*num_rows_in_cache) / sizeof(sparse_entry<T>);
  num_entries_in_cache = std::min(num_entries_in_cache, this->num_values);
  std::cout << "num entries in cache=" << num_entries_in_cache << "\tnum rows in cache=" << num_rows_in_cache << std::endl;

  cache.setSize(num_entries_in_cache);
  data.setSize(num_rows_in_cache);
}

template <typename T> uint LargeSparseMatrixHD<T>::getNumRows() {
  return num_rows;
};

template <typename T> uint LargeSparseMatrixHD<T>::getNumCols() {
  return num_cols;
};

template <typename T> uint64 LargeSparseMatrixHD<T>::getNumValues() {
  return num_values;
};

template <typename T> void LargeSparseMatrixHD<T>::next() {
  row_index++;
  position_in_data_cache++;
  if (position_in_data_cache >= number_of_valid_rows_in_cache) {
    readcache();
  }
}

template <typename T> void LargeSparseMatrixHD<T>::begin() {
  if ((row_index == position_in_data_cache) && (number_of_valid_rows_in_cache > 0)) {
    // if the beginning is already in the cache, do nothing
    row_index = 0;
    position_in_data_cache = 0;
    // close the file because everything is in the cache
    if (in.is_open()) {
      in.close();
    }
    return;
  }
  row_index = 0;
  position_in_data_cache = 0;
  number_of_valid_rows_in_cache = 0;
  number_of_valid_entries_in_cache = 0;
  in.seekg(sizeof(file_header), std::ios_base::beg);
  readcache();
}

template <typename T> bool LargeSparseMatrixHD<T>::end() {
  return row_index >= num_rows;
}

template <typename T> sparse_row<T>& LargeSparseMatrixHD<T>::getRow() {
  return data(position_in_data_cache);
}

template <typename T> uint LargeSparseMatrixHD<T>::getRowIndex() {
  return row_index;
}

template <typename T> void LargeSparseMatrixMemory<T>::begin() {
  index = 0;
}

template <typename T> bool LargeSparseMatrixMemory<T>::end() {
  return index >= data.dim;
}

template <typename T> void LargeSparseMatrixMemory<T>::next() {
  index++;
}

template <typename T> sparse_row<T>& LargeSparseMatrixMemory<T>::getRow() {
  return data(index);
}

template <typename T> uint LargeSparseMatrixMemory<T>::getRowIndex() {
  return index;
}

template <typename T> uint LargeSparseMatrixMemory<T>::getNumRows() {
  return data.dim;
}

template <typename T> uint LargeSparseMatrixMemory<T>::getNumCols() {
  return num_cols;
}

template <typename T> uint64 LargeSparseMatrixMemory<T>::getNumValues() {
  return num_values;
}

#endif /*FMATRIX_H_*/
