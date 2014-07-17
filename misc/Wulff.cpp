#include "Wulff.h"

namespace hccl{

Wulff::Wulff(){
    eps = 0.00001;
    time_step = 10;
}

Wulff::~Wulff(){

}

void Wulff::set_geometry(TriMesh& _mesh){
    mesh = _mesh;
    opLaplacian(mesh, M, Lc);
}

void Wulff::set_timestep(double t){
    time_step = t;
}

void Wulff::set_values(std::vector<double>& val){
    if(val.size() != mesh.n_vertices()){
        // TODO: Error handling
    }

    phi = val;
}

void Wulff::get_values(std::vector<double>& val){
    val = phi;
}

void Wulff::set_value(int i, double val){
    phi[i] = val;
}

double Wulff::get_value(int i){
    return phi[i];
}

void Wulff::set_beta(std::vector<double>& val){
    if(val.size() != mesh.n_faces()){
        // TODO: Error handling
    }

    beta = val;
}

void Wulff::get_beta(std::vector<double>& val){
    val = beta;
}

void Wulff::set_beta(int i, double val){
    beta[i] = val;
}

double Wulff::get_beta(int i){
    return beta[i];
}

void Wulff::set_epsilon(double val){
    eps = val;
}

double Wulff::get_epsilon(){
    return eps;
}

void Wulff::initialize(){
    // TODO: check if everything is fine.

    build_beta();
    build_mass();
    opGradient_V2T(mesh, G);
    opDivergence_T2V(mesh, D);
}

void Wulff::solve(){
    solve_gradient();
    build_mass();

    int NV = mesh.n_vertices();
    SparseMatrix EYE;
    EYE.set_identity(NV);
    system.set_matrix(EYE - time_step*S*D*B*M*G);
    system.factor_gen();
    
    solve_backward_Euler();
}


void Wulff::build_beta(){
    int NV = mesh.n_vertices();
    int NF = mesh.n_faces();

    B.clear();
    B.set_size(3*NF, 3*NF);

    if(beta.size() != NF){
        B.set_identity(3*NF);
    }else{
        for(int i = 0; i < NF; ++i)
        {
            double betaf = 1.0 / (1.0 + 10*beta[i]*beta[i]);

            B.add_entry(3*i  , 3*i  , betaf);
            B.add_entry(3*i+1, 3*i+1, betaf);
            B.add_entry(3*i+2, 3*i+2, betaf);
        }
    }
}

void Wulff::build_mass(){
    int NV = mesh.n_vertices();
    int NF = mesh.n_faces();
    Point v[3];
    Vector e[3];
    Vector nn;

    S.clear();
    S.set_size(NV, NV);
    M.clear();
    M.set_size(3*NF, 3*NF);

    if(grad_phi.size() != NF){
        solve_gradient();
    }

    for(int i = 0; i < NV; i++){
        TriMesh::VertexHandle vh = mesh.vertex_handle(i);
        Vector grad(0, 0, 0);
        double totalArea = 0.0;
        for(TriMesh::VertexFaceIter vfit = mesh.vf_begin(vh), vfend = mesh.vf_end(vh); vfit!=vfend; ++vfit){
            TriMesh::HalfedgeHandle h1 = mesh.halfedge_handle(vfit.handle());
            TriMesh::HalfedgeHandle h2 = mesh.next_halfedge_handle(h1);
            Point p0 = mesh.point(mesh.to_vertex_handle(h1));
            Point p1 = mesh.point(mesh.from_vertex_handle(h1));
            Point p2 = mesh.point(mesh.to_vertex_handle(h2));
            double area = cross(p1-p0, p2-p0).length()/6;
            grad += area*grad_phi[vfit.handle().idx()];
            totalArea += area;
        }
        grad /= totalArea;

        S.add_entry(i, i, grad.length());
    }

    for(int i = 0; i < NF; i++){
        double norm_grad_phi = grad_phi[i].length();
        double val = 1.0/sqrt(eps + norm_grad_phi);
        M.add_entry(3*i  , 3*i  , val);
        M.add_entry(3*i+1, 3*i+1, val);
        M.add_entry(3*i+2, 3*i+2, val);
    }
}

void Wulff::solve_gradient(){
    int NF = mesh.n_faces();
    grad_phi.resize(NF);

    // For each facet
    for(TriMesh::ConstFaceIter fit = mesh.faces_begin(), fend = mesh.faces_end(); fit != fend; ++fit){
        // unit normal N of the face
        Vector N = mesh.normal(fit);

        // compute the summation part of the equation
        Vector sum(0.0, 0.0, 0.0);
        for(TriMesh::ConstFaceHalfedgeIter fhit = mesh.fh_begin(fit), fhend = mesh.fh_end(fit); fhit != fhend; ++fhit){
            // for each halfedge attached to the face,
            // compute the edge vector
            Vector ei = mesh.point(mesh.to_vertex_handle(fhit)) - mesh.point(mesh.from_vertex_handle(fhit));
            //com += point(from_vertex_handle(fhit));
            // and then add ui*cross(N, ei) to the sum 
            sum += phi[mesh.opposite_vh(fhit).idx()]*cross(N, ei);
        }

        // Compute the area by Heron's formula
        TriMesh::ConstFaceEdgeIter feit = mesh.fe_begin(fit);
        double a = mesh.calc_edge_length(feit);
        double b = mesh.calc_edge_length(++feit);
        double c = mesh.calc_edge_length(++feit);
        double s = 0.5*(a+b+c);
        double Af = sqrt(s*(s-a)*(s-b)*(s-c));

        sum /= 2*Af;

        // normalize and inverse the direction
        grad_phi[fit.handle().idx()] = sum;
    }
}

void Wulff::solve_backward_Euler(){
    int NV = mesh.n_vertices();

    DenseMatrix d;
    d.set_size(NV, 1);
    for(size_t i = 0; i < NV; i++){
        d.set(i, 0, phi[i]);
    }

    system.set_matrix(d);

    DenseMatrix x;
    system.solve_gen(x);
    for(size_t i = 0; i < NV; i++){
        phi[i] = x.get(i, 0);
    }
}




}   // namespace hccl