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

void opGradient_V2T(TriMesh& mesh,  SparseMatrix& G){
    int NV = mesh.n_vertices();
    int NF = mesh.n_faces();

    Point v[3];
    uint vi[3];
    Point e[3];
    double dblA;

    G.clear();
    G.set_size(3*NF, NV);
    for(int i = 0; i < NF; ++i)
    {
        // Vertices of current facet
        TriMesh::ConstFaceVertexIter fvit = mesh.fv_begin(mesh.face_handle(i));
        v[0] = mesh.point(fvit); vi[0] = (mesh.handle(mesh.vertex(fvit))).idx();
        v[1] = mesh.point(++fvit); vi[1] = (mesh.handle(mesh.vertex(fvit))).idx();
        v[2] = mesh.point(++fvit); vi[2] = (mesh.handle(mesh.vertex(fvit))).idx();

        // Edge vectors of current facet
        e[0] = v[2] - v[1];
        e[1] = v[0] - v[2];
        e[2] = v[1] - v[0];

        // Area, cotangent
        Point nn = cross(e[0], e[1]);
        dblA = nn.length();
        nn /= dblA;

        Point c0 = cross(nn, e[0])/dblA;
        Point c1 = cross(nn, e[1])/dblA;
        Point c2 = cross(nn, e[2])/dblA;

        G.add_entry(3*i  , vi[0], c0[0]);
        G.add_entry(3*i  , vi[1], c1[0]);
        G.add_entry(3*i  , vi[2], c2[0]);
        G.add_entry(3*i+1, vi[0], c0[1]);
        G.add_entry(3*i+1, vi[1], c1[1]);
        G.add_entry(3*i+1, vi[2], c2[1]);
        G.add_entry(3*i+2, vi[0], c0[2]);
        G.add_entry(3*i+2, vi[1], c1[2]);
        G.add_entry(3*i+2, vi[2], c2[2]);
    }
}

void opDivergence_T2V(TriMesh& mesh, SparseMatrix& D){
    int NV = mesh.n_vertices();
    int NF = mesh.n_faces();

    Point v[3];
    uint vi[3];
    Point e[3];
    double dblA;
    double cot0, cot1, cot2;

    D.clear();
    D.set_size(NV, 3*NF);
    std::vector<double> voronoi_area(NV, 0);
    for(int i = 0; i < NF; i++){
        TriMesh::ConstFaceVertexIter fvit = mesh.fv_begin(mesh.face_handle(i));
        v[0] = mesh.point(fvit); vi[0] = (mesh.handle(mesh.vertex(fvit))).idx();
        v[1] = mesh.point(++fvit); vi[1] = (mesh.handle(mesh.vertex(fvit))).idx();
        v[2] = mesh.point(++fvit); vi[2] = (mesh.handle(mesh.vertex(fvit))).idx();

        e[0] = v[2] - v[1];
        e[1] = v[0] - v[2];
        e[2] = v[1] - v[0];

        Point nn = cross(-e[1], e[2]);
        dblA = nn.length();
        cot0 = -dot(e[2], e[1]) / dblA;
        cot1 = -dot(e[0], e[2]) / dblA;
        cot2 = -dot(e[1], e[0]) / dblA;

        double l0 = e[0].length();
        double l1 = e[1].length();
        double l2 = e[2].length();

        voronoi_area[vi[0]] += 0.125*(l1*l1*cot1 + l2*l2*cot2);
        voronoi_area[vi[1]] += 0.125*(l2*l2*cot2 + l0*l0*cot0);
        voronoi_area[vi[2]] += 0.125*(l0*l0*cot0 + l1*l1*cot1);
    }

    for(int i = 0; i < NF; ++i)
    {
        // Vertices of current facet
        TriMesh::ConstFaceVertexIter fvit = mesh.fv_begin(mesh.face_handle(i));
        v[0] = mesh.point(fvit); vi[0] = (mesh.handle(mesh.vertex(fvit))).idx();
        v[1] = mesh.point(++fvit); vi[1] = (mesh.handle(mesh.vertex(fvit))).idx();
        v[2] = mesh.point(++fvit); vi[2] = (mesh.handle(mesh.vertex(fvit))).idx();

        // Edge vectors of current facet
        e[0] = v[2] - v[1];
        e[1] = v[0] - v[2];
        e[2] = v[1] - v[0];

        // Area, cotangent
        Point nn = cross(-e[1], e[2]);
        dblA = nn.length();
        cot0 = -dot(e[2], e[1]) / dblA / 2;
        cot1 = -dot(e[0], e[2]) / dblA / 2;
        cot2 = -dot(e[1], e[0]) / dblA / 2;

        D.add_entry(vi[0], 3*i  , (-cot1*e[1][0] + cot2*e[2][0])/voronoi_area[vi[0]]);
        D.add_entry(vi[0], 3*i+1, (-cot1*e[1][1] + cot2*e[2][1])/voronoi_area[vi[0]]);
        D.add_entry(vi[0], 3*i+2, (-cot1*e[1][2] + cot2*e[2][2])/voronoi_area[vi[0]]);
        D.add_entry(vi[1], 3*i  , (-cot2*e[2][0] + cot0*e[0][0])/voronoi_area[vi[1]]);
        D.add_entry(vi[1], 3*i+1, (-cot2*e[2][1] + cot0*e[0][1])/voronoi_area[vi[1]]);
        D.add_entry(vi[1], 3*i+2, (-cot2*e[2][2] + cot0*e[0][2])/voronoi_area[vi[1]]);
        D.add_entry(vi[2], 3*i  , (-cot0*e[0][0] + cot1*e[1][0])/voronoi_area[vi[2]]);
        D.add_entry(vi[2], 3*i+1, (-cot0*e[0][1] + cot1*e[1][1])/voronoi_area[vi[2]]);
        D.add_entry(vi[2], 3*i+2, (-cot0*e[0][2] + cot1*e[1][2])/voronoi_area[vi[2]]);
    }

}



}   // namespace hccl