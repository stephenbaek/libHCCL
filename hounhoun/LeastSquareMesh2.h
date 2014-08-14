#ifndef HCCL_LeastSquareMesh2_H_
#define HCCL_LeastSquareMesh2_H_

#include "../common.h"
#include "../mesh/mesh.h"
#include "../math/sparsematrix.h"
#include "../diffgeom/differential_operators.h"


namespace hccl{

class LeastSquareMesh2{
public:
    LeastSquareMesh2();
    ~LeastSquareMesh2();

    void set_geometry(TriMesh* _mesh);    
    void solve(const std::vector<int>& src, const std::vector<TriMesh::Point>& con);	

protected:
    TriMesh* mesh;
    SparseSolver system;
    SparseMatrix Lc;
    SparseMatrix M;
};

} // namespace hccl

#endif // HCCL_LeastSquareMesh2_H_