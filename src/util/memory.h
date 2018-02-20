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
// memory.h: Logging memory consumption of large data structures

#ifndef MEMORY_H_
#define MEMORY_H_

#include <vector>
#include <assert.h>

typedef unsigned long long int uint64;
typedef signed long long int int64;

class MemoryLog {
  private:
    uint64 mem_size;

  public:
    static MemoryLog& getInstance() {
      static MemoryLog instance;
      return instance;
    }

    MemoryLog();

    void logNew(std::string message, uint64 size, uint64 count = 1);

    void logFree(std::string message, uint64 size, uint64 count = 1);
};

// Implementation
MemoryLog::MemoryLog() {
  mem_size = 0;
}

void MemoryLog::logNew(std::string message, uint64 size, uint64 count) {
  mem_size += size*count;
  // std::cout << "total memory consumption=" << mem_size << " bytes" << "\t" << "reserving " << count << "*" << size << " for " << message << std::endl;
}

void MemoryLog::logFree(std::string message, uint64 size, uint64 count) {
  mem_size -= size*count;
  // std::cout << "total memory consumption=" << mem_size << " bytes" << std::endl;
}


#endif
