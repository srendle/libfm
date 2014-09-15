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
// transpose: Convert a libfm-format file in a binary sparse matrix for x and
// a dense vector for the target y. 

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
 *	no differences, version numbers are kept in sync over all libfm tools
 * 1.3.6:
 *	binary mode for file access
 * 1.3.4:
 *	no differences, version numbers are kept in sync over all libfm tools
 * 1.3.2:
 *	reading without token reader class
 * 1.0:
 *	first version
 */
 


using namespace std;

int main(int argc, char **argv) { 
 	
 	srand ( time(NULL) );
	try {
		CMDLine cmdline(argc, argv);
		std::cout << "----------------------------------------------------------------------------" << std::endl;
		std::cout << "Convert" << std::endl;
		std::cout << "  Version: 1.4.2" << std::endl;
		std::cout << "  Author:  Steffen Rendle, srendle@libfm.org" << std::endl;
		std::cout << "  WWW:     http://www.libfm.org/" << std::endl;
		std::cout << "This program comes with ABSOLUTELY NO WARRANTY; for details see license.txt." << std::endl;
		std::cout << "This is free software, and you are welcome to redistribute it under certain" << std::endl;
		std::cout << "conditions; for details see license.txt." << std::endl;
		std::cout << "----------------------------------------------------------------------------" << std::endl;
		
		const std::string param_ifile	= cmdline.registerParameter("ifile", "input file name, file has to be in binary sparse format [MANDATORY]");
		const std::string param_ofilex	= cmdline.registerParameter("ofilex", "output file name for x [MANDATORY]");
		const std::string param_ofiley	= cmdline.registerParameter("ofiley", "output file name for y [MANDATORY]");
		const std::string param_help       = cmdline.registerParameter("help", "this screen");


		if (cmdline.hasParameter(param_help) || (argc == 1)) {
			cmdline.print_help();
			return 0;
		}
		cmdline.checkParameters();

		std::string ifile = cmdline.getValue(param_ifile);
		std::string ofilex = cmdline.getValue(param_ofilex);
		std::string ofiley = cmdline.getValue(param_ofiley);

		uint num_rows = 0;
		uint64 num_values = 0;
		uint num_feature = 0;
		bool has_feature = false;
		DATA_FLOAT min_target = +std::numeric_limits<DATA_FLOAT>::max();
		DATA_FLOAT max_target = -std::numeric_limits<DATA_FLOAT>::max();

		// (1) determine the number of rows and the maximum feature_id
		{
			std::ifstream fData(ifile.c_str());
			if (! fData.is_open()) {
				throw "unable to open " + ifile;
			}
			DATA_FLOAT _value;
			int nchar;
			uint _feature;
			while (!fData.eof()) {
				std::string line;
				std::getline(fData, line);
				const char *pline = line.c_str();
				while ((*pline == ' ')  || (*pline == 9)) { pline++; } // skip leading spaces
				if ((*pline == 0)  || (*pline == '#')) { continue; }  // skip empty rows
				if (sscanf(pline, "%f%n", &_value, &nchar) >=1) {
					pline += nchar;
					min_target = std::min(_value, min_target);
					max_target = std::max(_value, max_target);			
					num_rows++;
					while (sscanf(pline, "%d:%f%n", &_feature, &_value, &nchar) >= 2) {
						pline += nchar;	
						num_feature = std::max(_feature, num_feature);
						has_feature = true;
						num_values++;	
					}
					while ((*pline != 0) && ((*pline == ' ')  || (*pline == 9))) { pline++; } // skip trailing spaces
					if ((*pline != 0)  && (*pline != '#')) { 
						throw "cannot parse line \"" + line + "\" at character " + pline[0];
					}
				} else {
					throw "cannot parse line \"" + line + "\" at character " + pline[0];
				}
			} 
			fData.close();
		}
		if (has_feature) {	
			num_feature++; // number of feature is bigger (by one) than the largest value
		}
		std::cout << "num_rows=" << num_rows << "\tnum_values=" << num_values << "\tnum_features=" << num_feature << "\tmin_target=" << min_target << "\tmax_target=" << max_target << std::endl;
		
		sparse_row<DATA_FLOAT> row;
		row.data = new sparse_entry<DATA_FLOAT>[num_feature];

		// (2) read the data and write it back simultaneously
		{
			std::ifstream fData(ifile.c_str());
			if (! fData.is_open()) {
				throw "unable to open " + ifile;
			}
			std::ofstream out_x(ofilex.c_str(), ios_base::out | ios_base::binary);
			if (! out_x.is_open()) {
				throw "unable to open " + ofilex;
			} else {
				file_header fh;
				fh.id = FMATRIX_EXPECTED_FILE_ID;
				fh.num_values = num_values;
				fh.num_rows = num_rows;
				fh.num_cols = num_feature;
				fh.float_size = sizeof(DATA_FLOAT);
				out_x.write(reinterpret_cast<char*>(&fh), sizeof(fh));
			}
			std::ofstream out_y(ofiley.c_str(), ios_base::out | ios_base::binary);
			if (! out_y.is_open()) {
				throw "unable to open " + ofiley;
			} else {
				uint file_version = 1;
				uint data_size = sizeof(DATA_FLOAT);
				out_y.write(reinterpret_cast<char*>(&file_version), sizeof(file_version));
				out_y.write(reinterpret_cast<char*>(&data_size), sizeof(data_size));
				out_y.write(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
			}

			DATA_FLOAT _value;
			int nchar;
			uint _feature;
			while (!fData.eof()) {
				std::string line;
				std::getline(fData, line);
				const char *pline = line.c_str();
				while ((*pline == ' ')  || (*pline == 9)) { pline++; } // skip leading spaces
				if ((*pline == 0)  || (*pline == '#')) { continue; }  // skip empty rows
				if (sscanf(pline, "%f%n", &_value, &nchar) >=1) {
					pline += nchar;
					out_y.write(reinterpret_cast<char*>(&(_value)), sizeof(DATA_FLOAT));
					row.size = 0;
					while (sscanf(pline, "%d:%f%n", &_feature, &_value, &nchar) >= 2) {
						pline += nchar;	
						assert(row.size < num_feature);
						row.data[row.size].id = _feature;
						row.data[row.size].value = _value;
						row.size++;	
					}
					out_x.write(reinterpret_cast<char*>(&(row.size)), sizeof(uint));
					out_x.write(reinterpret_cast<char*>(row.data), sizeof(sparse_entry<DATA_FLOAT>)*row.size);
					while ((*pline != 0) && ((*pline == ' ')  || (*pline == 9))) { pline++; } // skip trailing spaces
					if ((*pline != 0)  && (*pline != '#')) { 
						throw "cannot parse line \"" + line + "\" at character " + pline[0];
					}
				} else {
					throw "cannot parse line \"" + line + "\" at character " + pline[0];
				}
			}	
			fData.close();
			out_x.close();
			out_y.close();

		}
	} catch (std::string &e) {
		std::cerr << e << std::endl;
	}

}
