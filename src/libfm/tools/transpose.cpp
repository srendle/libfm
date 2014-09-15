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
// transpose: Transposes a matrix in binary sparse format. 

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <iterator>
#include <algorithm>
#include <iomanip>
#include "../../util/util.h"
#include "../../util/cmdline.h"
#include "../src/Data.h"

/**
 * 
 * Version history:
 * 1.4.2:
 *	changed license to GPLv3 
 * 1.4.0:
 *	default cache size is 200 MB
 * 1.3.6:
 *	binary mode for file access
 * 1.3.4:
 *	no differences, version numbers are kept in sync over all libfm tools
 * 1.3.2:
 *	no differences, version numbers are kept in sync over all libfm tools
 * 1.0:
 *	first version
 */
 


using namespace std;

int main(int argc, char **argv) { 
 	
 	srand ( time(NULL) );
	try {
		CMDLine cmdline(argc, argv);
		std::cout << "----------------------------------------------------------------------------" << std::endl;
		std::cout << "Transpose" << std::endl;
		std::cout << "  Version: 1.4.2" << std::endl;
		std::cout << "  Author:  Steffen Rendle, srendle@libfm.org" << std::endl;
		std::cout << "  WWW:     http://www.libfm.org/" << std::endl;
		std::cout << "This program comes with ABSOLUTELY NO WARRANTY; for details see license.txt." << std::endl;
		std::cout << "This is free software, and you are welcome to redistribute it under certain" << std::endl;
		std::cout << "conditions; for details see license.txt." << std::endl;
		std::cout << "----------------------------------------------------------------------------" << std::endl;
		
		const std::string param_ifile	= cmdline.registerParameter("ifile", "input file name, file has to be in binary sparse format [MANDATORY]");
		const std::string param_ofile	= cmdline.registerParameter("ofile", "output file name [MANDATORY]");
		
		const std::string param_cache_size = cmdline.registerParameter("cache_size", "cache size for data storage, default=200000000");
		const std::string param_help       = cmdline.registerParameter("help", "this screen");


		if (cmdline.hasParameter(param_help) || (argc == 1)) {
			cmdline.print_help();
			return 0;
		}
		cmdline.checkParameters();


		// (1) Load the data
		long long cache_size = cmdline.getValue(param_cache_size, 200000000);
		cache_size /= 2;
		LargeSparseMatrixHD<DATA_FLOAT> d_in(cmdline.getValue(param_ifile), cache_size);
		std::cout << "num_rows=" << d_in.getNumRows() << "\tnum_values=" << d_in.getNumValues() << "\tnum_features=" << d_in.getNumCols() << std::endl;

		// (2) transpose the data
		// (2.1) count how many entries per col (=transpose-row) there are:
		DVector<uint> entries_per_col(d_in.getNumCols());
		entries_per_col.init(0);
		for (d_in.begin(); !d_in.end(); d_in.next() ) {
			sparse_row<DATA_FLOAT>& row = d_in.getRow();
			for (uint j = 0; j < row.size; j++) {
				entries_per_col(row.data[j].id)++;
			}
		}
		// (2.2) build a 
		std::string ofile = cmdline.getValue(param_ofile);
		std::cout << "output to " << ofile << std::endl; std::cout.flush();
		std::ofstream out(ofile.c_str(), ios_base::out | ios_base::binary);
		if (out.is_open()) {
			file_header fh;
			fh.id = FMATRIX_EXPECTED_FILE_ID;
			fh.num_values = d_in.getNumValues();
			fh.num_rows = d_in.getNumCols();
			fh.num_cols = d_in.getNumRows();
			fh.float_size = sizeof(DATA_FLOAT);
			out.write(reinterpret_cast<char*>(&fh), sizeof(fh));

			DVector< sparse_row<DATA_FLOAT> > out_row_cache;
			DVector< sparse_entry<DATA_FLOAT> > out_entry_cache;
			{
				// determine cache sizes automatically:
				double avg_entries_per_line = (double) d_in.getNumValues() / d_in.getNumCols();
				uint num_rows_in_cache = cache_size / (sizeof(sparse_entry<DATA_FLOAT>) * avg_entries_per_line + sizeof(uint));
				num_rows_in_cache = std::min(num_rows_in_cache, d_in.getNumCols());
				uint64 num_entries_in_cache = (cache_size - sizeof(uint)*num_rows_in_cache) / sizeof(sparse_entry<DATA_FLOAT>);
				num_entries_in_cache = std::min(num_entries_in_cache, d_in.getNumValues());
				std::cout << "num entries in cache=" << num_entries_in_cache << "\tnum rows in cache=" << num_rows_in_cache << std::endl;				
				out_entry_cache.setSize(num_entries_in_cache);
				out_row_cache.setSize(num_rows_in_cache);
			}

			uint out_cache_col_position = 0; // the first column id that is in cache
			uint out_cache_col_num = 0; // how many columns are in the cache
		
			while (out_cache_col_position < d_in.getNumCols()) {
				// assign cache sizes 
				{
					uint entry_cache_pos = 0;
					// while (there is enough space in the entry cache for the next row) and (there is space for another row) and (there is another row in the data) do   
					while (((entry_cache_pos + entries_per_col(out_cache_col_position + out_cache_col_num)) < out_entry_cache.dim) && ((out_cache_col_num+1) < out_row_cache.dim) && ((out_cache_col_position+out_cache_col_num) < d_in.getNumCols())) {
						out_row_cache(out_cache_col_num).size = 0;
						out_row_cache(out_cache_col_num).data = &(out_entry_cache.value[entry_cache_pos]);
						entry_cache_pos += entries_per_col(out_cache_col_position + out_cache_col_num);
						out_cache_col_num++;
					}
				}
				assert(out_cache_col_num > 0);
				// fill the cache
				for (d_in.begin(); !d_in.end(); d_in.next() ) {
					sparse_row<DATA_FLOAT>& row = d_in.getRow();
					for (uint j = 0; j < row.size; j++) {
						if ((row.data[j].id >= out_cache_col_position) && (row.data[j].id < (out_cache_col_position+out_cache_col_num))) {
							uint cache_row_index = row.data[j].id-out_cache_col_position; 
							out_row_cache(cache_row_index).data[out_row_cache(cache_row_index).size].id = d_in.getRowIndex();
							out_row_cache(cache_row_index).data[out_row_cache(cache_row_index).size].value = row.data[j].value;
							out_row_cache(cache_row_index).size++;
						}
					}
				}

				for (uint i = 0; i < out_cache_col_num; i++) {
					assert(out_row_cache(i).size == entries_per_col(i + out_cache_col_position));
					out.write(reinterpret_cast<char*>(&(out_row_cache(i).size)), sizeof(uint));
					out.write(reinterpret_cast<char*>(out_row_cache(i).data), sizeof(sparse_entry<DATA_FLOAT>)*out_row_cache(i).size);
				}
				out_cache_col_position += out_cache_col_num;
				out_cache_col_num = 0;
			}
			out.close();				
		} else {
			throw "could not open " + ofile;
		}

	} catch (std::string &e) {
		std::cerr << e << std::endl;
	}

}
