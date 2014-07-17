#ifndef HCCL_GEODESICS_H_
#define HCCL_GEODESICS_H_

#include "../common.h"
#include "../mesh/mesh.h"
#include "../math/sparsematrix.h"
#include "../diffgeom/differential_operators.h"


namespace hccl{

// Implementation of K. Crane et al., "Geodesics in Heat", ACM TOG, 2013
class Geodesics{
public:
    Geodesics();
    ~Geodesics();

    void set_geometry(TriMesh& _mesh);
    //void set_geometry(PolyMesh& mesh);    TODO: Polygonal mesh
    //void set_geometry(PointCloud& pts);   TODO: Point clouds
    void set_timestep(double t);

    void solve(int src);
    void solve(std::vector<int>& src);

    double get_distance(int vtx_id);

protected:
    void solve_heat_flow(std::vector<int>& src);
    void solve_gradient();
    void solve_poisson();

    TriMesh mesh;

    std::vector<double> u;
    std::vector<Vector> gradu;
    std::vector<double> divX;
    std::vector<double> phi;

    std::vector<int> src;

    SparseSolver sys_heat_flow;
    SparseSolver sys_Poisson;

    SparseMatrix Lc;
    SparseMatrix M;
};

} // namespace hccl

#endif // HCCL_GEODESICS_H_