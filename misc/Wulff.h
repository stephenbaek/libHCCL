#ifndef HCCL_WULFF_H_
#define HCCL_WULFF_H_

#include "../common.h"
#include "../mesh/mesh.h"
#include "../math/sparsematrix.h"
#include "../diffgeom/differential_operators.h"


namespace hccl{

// Implementation of S. Baek et al., "The fast Wulff flow on discrete manifolds"
class Wulff{
public:
    Wulff();
    ~Wulff();

    void set_geometry(TriMesh& _mesh);
    //void set_geometry(PolyMesh& mesh);    TODO: Polygonal mesh
    //void set_geometry(PointCloud& pts);   TODO: Point clouds
    void set_timestep(double t);

    void set_values(std::vector<double>& val);
    void get_values(std::vector<double>& val);
    void set_value(int i, double val);
    double get_value(int i);

    void set_beta(std::vector<double>& val);
    void get_beta(std::vector<double>& val);
    void set_beta(int i, double val);
    double get_beta(int i);

    void set_epsilon(double val);
    double get_epsilon();

    void initialize();

    void solve();


protected:
    void build_beta();
    void build_mass();

    void solve_gradient();
    void solve_backward_Euler();

    TriMesh mesh;

    std::vector<double> phi;
    std::vector<double> beta;
    std::vector<Vector> grad_phi;
    std::vector<double> div;

    double eps;
    double time_step;

    SparseSolver system;



    std::vector<double> u;
    std::vector<Vector> gradu;
    std::vector<double> divX;

    std::vector<int> src;

    SparseMatrix Lc;
    SparseMatrix Mc;

    SparseMatrix S;         // Outer mass |grad phi|
    SparseMatrix B;         // Stopping Function - "Beta"
    SparseMatrix D;         // Discrete Divergence
    SparseMatrix G;         // Discrete Gradient
    SparseMatrix M;         // Inner mass |grad phi|^-1
};

} // namespace hccl

#endif // HCCL_WULFF_H_