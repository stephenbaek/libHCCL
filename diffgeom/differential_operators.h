#ifndef HCCL_DIFFERENTIALOPERATORS_H_
#define HCCL_DIFFERENTIALOPERATORS_H_

#include "../common.h"

namespace hccl{
    void opLaplacian(TriMesh& mesh, SparseMatrix& M, SparseMatrix& L);

    // void opGradient_V2T();
    // void opGradient_V2V();
    // void opDivergence_T2V();
    // void opDivergence_V2T();
    // void opCurl_T2V();
    // void opCurl_V2T();

} // namespace hccl

#endif // HCCL_DIFFERENTIALOPERATORS_H_