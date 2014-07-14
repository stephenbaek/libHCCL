#include "differential_operators.h"
#include "../mesh/mesh.h"
#include "../math/sparsematrix.h"

namespace hccl{


void opLaplacian(TriMesh& mesh, SparseMatrix& M, SparseMatrix& L){
    int NV = mesh.n_vertices();
    int NF = mesh.n_faces();

    Point v[3];
    uint vi[3];
    Point e[3];
    double dblA;
    double cot01, cot12, cot20;
    double diag0, diag1, diag2;
    std::map<size_t, double>::iterator iter;
    std::map<size_t, double> lap_mat, mas_mat;
    M.clear();
    L.clear();

    M.set_size(NV, NV);
    L.set_size(NV, NV);
        
    for(int i = 0; i < NF; ++i)
    {
        // Vertices of current facet
        TriMesh::ConstFaceVertexIter fvit = mesh.fv_begin(mesh.face_handle(i));
        v[0] = mesh.point(fvit); vi[0] = (mesh.handle(mesh.vertex(fvit))).idx();
        v[1] = mesh.point(++fvit); vi[1] = (mesh.handle(mesh.vertex(fvit))).idx();
        v[2] = mesh.point(++fvit); vi[2] = (mesh.handle(mesh.vertex(fvit))).idx();

        // Edge vectors of current facet
        e[0] = Point(v[2][0] - v[1][0], v[2][1] - v[1][1], v[2][2] - v[1][2]);
        e[1] = Point(v[0][0] - v[2][0], v[0][1] - v[2][1], v[0][2] - v[2][2]);
        e[2] = Point(v[1][0] - v[0][0], v[1][1] - v[0][1], v[1][2] - v[0][2]);

        // Area, cotangent
        Point nn = cross(e[0], e[1]);
        dblA = nn.length();
        cot01 = dot(-e[0], e[1]) / dblA / 2;
        cot12 = dot(-e[1], e[2]) / dblA / 2;
        cot20 = dot(-e[2], e[0]) / dblA / 2;
        diag0 = -cot01-cot20;
        diag1 = -cot01-cot12;
        diag2 = -cot20-cot12;

        L.add_entry(vi[0], vi[1], cot01);
        L.add_entry(vi[1], vi[0], cot01);
        L.add_entry(vi[1], vi[2], cot12);
        L.add_entry(vi[2], vi[1], cot12);
        L.add_entry(vi[2], vi[0], cot20);
        L.add_entry(vi[0], vi[2], cot20);
        L.add_entry(vi[0], vi[0], diag0);
        L.add_entry(vi[1], vi[1], diag1);
        L.add_entry(vi[2], vi[2], diag2);

        // Barycentric
        double diag = dblA/6;
        M.add_entry(vi[0], vi[0], diag);
        M.add_entry(vi[1], vi[1], diag);
        M.add_entry(vi[2], vi[2], diag);
    }

}




}   // namespace hccl