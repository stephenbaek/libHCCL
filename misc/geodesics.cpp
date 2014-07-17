#include "geodesics.h"

namespace hccl{

Geodesics::Geodesics(){
}

Geodesics::~Geodesics(){
}

void Geodesics::set_geometry(TriMesh& _mesh){
    mesh = _mesh;
    opLaplacian(mesh, M, Lc);
//     SparseMatrix eye;
//     eye.set_identity(Lc.n_rows());
    sys_Poisson.clear();
    //sys_Poisson.set_matrix(ssadd(Lc, eye, 1.0, 0.000001));
    sys_Poisson.set_matrix(Lc);
    sys_Poisson.factor_gen();
}

void Geodesics::set_timestep(double t){
    if(M.n_nonzeros() == 0 || M.n_rows() != mesh.n_vertices() || mesh.n_vertices() == 0){
        // TODO: error handling (set_geometry must be called before to call set_timestep)
    }

    sys_heat_flow.clear();
    sys_heat_flow.set_matrix(ssadd(M, Lc, 1.0, -t));
    sys_heat_flow.factor_sym();
}

void Geodesics::solve(int src){
    std::vector<int> _src(1, src);
    solve(_src);
}

void Geodesics::solve(std::vector<int>& src){
    solve_heat_flow(src);
    solve_gradient();
    solve_poisson();
}

double Geodesics::get_distance(int vtx_id){
    return phi[vtx_id];
}

void Geodesics::solve_heat_flow(std::vector<int>& src){
    int NV = mesh.n_vertices();
    DenseMatrix d(NV, 1), x;
    d.set_null();
    for(int i = 0; i < src.size(); i++)
        d.set(src[i], 0, 1.0);

    sys_heat_flow.set_matrix(d);
    sys_heat_flow.solve_sym(x);

    u.resize(NV);
    double min_val = INF;
    int min_idx = -1;
    double max_val = -INF;
    int max_idx = -1;
    for(size_t i = 0; i < NV; i++){
        u[i] = x.get(i, 0);
        if(i < 10){
            printf("%lf\n", u[i]);
        }
        if(min_val > u[i])
        {
            min_val = u[i];
            min_idx = i;
        }
        if(max_val < u[i])
        {
            max_val = u[i];
            max_idx = i;
        }
    }

    // TODO: erase
    std::cout << max_val << std::endl;
    std::cout << max_idx << std::endl;
}

void Geodesics::solve_gradient(){
    int NF = mesh.n_faces();
    gradu.resize(NF);

    // For each facet
    for(TriMesh::ConstFaceIter fit = mesh.faces_begin(), fend = mesh.faces_end(); fit != fend; ++fit){
        // unit normal N of the face
        TriMesh::Normal N = mesh.normal(fit);

        // compute the summation part of the equation
        TriMesh::Point sum(0.0, 0.0, 0.0);
        for(TriMesh::ConstFaceHalfedgeIter fhit = mesh.fh_begin(fit), fhend = mesh.fh_end(fit); fhit != fhend; ++fhit){
            // for each halfedge attached to the face,
            // compute the edge vector
            TriMesh::Point ei = mesh.point(mesh.to_vertex_handle(fhit)) - mesh.point(mesh.from_vertex_handle(fhit));
            // and then add ui*cross(N, ei) to the sum 
            sum += u[mesh.opposite_vh(fhit).idx()]*cross(N, ei);
        }

        // Compute the area
        TriMesh::ConstFaceHalfedgeIter fhit = mesh.fh_begin(fit);
        TriMesh::Point e0 = mesh.point(mesh.from_vertex_handle(fhit)) - mesh.point(mesh.to_vertex_handle(fhit));
        ++fhit;
        TriMesh::Point e1 = mesh.point(mesh.to_vertex_handle(fhit)) - mesh.point(mesh.from_vertex_handle(fhit));
        double dblA = cross(e0, e1).length();

        sum /= dblA;

        // normalize and invert the direction
        gradu[fit.handle().idx()] = -sum.normalized();
    }
}

void Geodesics::solve_poisson(){
    int NV = mesh.n_vertices();

    // Compute the integrated divergence field divX:
    // divX = 0.5*sum{ cot1*dot(e1, X) + cot2*dot(e2, X) }
    // (see Crane's paper for the notation)
    divX.resize(NV);
    // for each vertex
    for(TriMesh::VertexIter vit = mesh.vertices_begin(), vend = mesh.vertices_end(); vit != vend; ++vit){
        int id = vit.handle().idx();        // vertex index
        divX[id] = 0;

        for(TriMesh::VertexFaceIter vfit = mesh.vf_begin(vit), vfend = mesh.vf_end(vit); vfit != vfend; ++vfit){

            // find a halfedge handle in the face that emanates from the current vertex.
            TriMesh::FaceHalfedgeIter fhit = mesh.fh_begin(vfit);
            while(1){
                if(mesh.from_vertex_handle(fhit) == vit)
                    break;
                ++fhit;
            }

            // e1, e2 are as the same notion in Crane's paper
            // e0 is set to be the remaining edge vector
            // p0 is the current vertex,
            // p1 and p2 are the end vertices of e1 and e2 separately.
            TriMesh::Point e0, e1, e2;
            TriMesh::Point p0, p1, p2;
            p0 = mesh.point(vit);
            p1 = mesh.point(mesh.to_vertex_handle(fhit));
            p2 = mesh.point(mesh.from_vertex_handle(mesh.prev_halfedge_handle(fhit)));
            e0 = p2 - p1;
            e1 = p1 - p0;
            e2 = p2 - p0;

            // compute the cotangents... (multiplied by 0.5)
            double dblA = cross(e1, e2).length();
            double cot1 = dot(e2, e0)/dblA;
            double cot2 = dot(-e1, e0)/dblA;

            // 			if(cot1 > 0 || cot2 > 0)
            // 				printf("error\n");

            // type casting: ofVec3f --> MyMesh::Point
            TriMesh::Point X = gradu[vfit.handle().idx()];
            
            // the summation part
            divX[id] += (cot1*dot(e1, X) + cot2*dot(e2, X))/2;
        }
    }

    DenseMatrix b(NV, 1);
    // right-hand-side of the Poisson equation
    for(size_t i = 0; i < divX.size(); i++){
        b.set(i, 0, divX[i]);
    }

    DenseMatrix x;
    sys_Poisson.set_matrix(b);
    sys_Poisson.solve_gen(x);

    // and then the answer is phi
    // Note that phi should be shifted such that the smallest distance is zero
    // (see the paragraph below the Algorithm 1 in Crane's paper)
    phi.resize(NV);
    double smallest = 1000000000;
    double biggest = 0;
    int smallest_idx = -1;
    for(size_t i = 0; i < NV; i++){
        phi[i] = x.get(i, 0);
        if(phi[i] < smallest)
        {
            smallest = phi[i];
            smallest_idx = i;
        }
        if(phi[i] > biggest)
            biggest = phi[i];
    }
    printf("%lf\n", biggest);
    printf("%lf\n", smallest);
    for(size_t i = 0; i < NV; i++){
        phi[i] -= smallest;
    }
}



}   // namespace hccl