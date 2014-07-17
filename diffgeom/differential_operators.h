#ifndef HCCL_DIFFERENTIALOPERATORS_H_
#define HCCL_DIFFERENTIALOPERATORS_H_

#include "../common.h"

namespace hccl{

    void opLaplacian(TriMesh& mesh, SparseMatrix& M, SparseMatrix& L);
    void opGradient_V2T(TriMesh& mesh, SparseMatrix& G);
    // void opGradient_V2V();  //TODO
    void opDivergence_T2V(TriMesh& mesh, SparseMatrix& D);
    //void opDivergence_V2T();  // TODO
    //void opCurl_T2V();  // TODO
    //void opCurl_V2T();  // TODO

    // for the implementation of the above functions,
    // see Baek et al., "Differential operators on a triangular mesh," 2014 Summer Conference of Korean Society of CAD/CAM Engineers.

} // namespace hccl

#endif // HCCL_DIFFERENTIALOPERATORS_H_