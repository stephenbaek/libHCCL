#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <limits>
#include "OpenMesh/Core/Geometry/VectorT.hh"

namespace hccl{

// CONSTANT VARIABLES
#define INF std::numeric_limits<double>::infinity()


// TYPE DECLARATIONS
typedef OpenMesh::Vec3d Point;
typedef OpenMesh::Vec3d Vector;
class TriMesh;
class SparseMatrix;
class DenseMatrix;
class SparseSolver;

// TODO!!
// class Error{
// public:
//     Error();
//     Error(const char* errmsg);
//     Error(std::string errmsg);
// 
//     const char* what();
// 
// protected:
// 
// };

} // namespace hccl