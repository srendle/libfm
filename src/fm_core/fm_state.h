
#ifndef FM_STATE_H_
#define FM_STATE_H_

#include "../util/matrix.h"
#include "../util/fmatrix.h"

#include "fm_data.h"


class fm_state {
    public:
        double w0;
        DVectorDouble w;
        DMatrixDouble v;
    public:
        fm_state();
        void saveModel(std::string model_file_path);
};

fm_sate::fm_state(double _w0, DVectorDouble _w, DMatrixDouble _v) {
    w0 = _w0;
    w.assign(_w);
    v.assign(_v);
}

/*
 * Write the FM model (all the parameters) in a file.
 */
void fm_stat::saveModel(std::string model_file_path){
    std::ofstream out_model;
    out_model.open(model_file_path.c_str());
    if (k0) {
        out_model << "#global bias W0" << std::endl;
        out_model << w0 << std::endl;
    }
    if (k1) {
        out_model << "#unary interactions Wj" << std::endl;
        for (uint i = 0; i<num_attribute; i++){
            out_model <<    w(i) << std::endl;
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

#endif /*FM_STATE_H_*/
