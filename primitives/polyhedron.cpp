#include "polyhedron.h"
#include "../mesh/mesh.h"

namespace hccl{


void tetrahedron(PolyMesh& mesh){
    double g = 1.0/sqrt(2.0);
    PolyMesh::VertexHandle vh[4];
    vh[0] = mesh.add_vertex(Point(1,0,-g));
    vh[1] = mesh.add_vertex(Point(-1,0,-g));
    vh[2] = mesh.add_vertex(Point(0,1,g));
    vh[3] = mesh.add_vertex(Point(0,-1,g));

    mesh.add_face(vh[0], vh[1], vh[2]);
    mesh.add_face(vh[0], vh[2], vh[3]);
    mesh.add_face(vh[0], vh[3], vh[1]);
    mesh.add_face(vh[3], vh[2], vh[1]);

    mesh.update_normals();
}

void hexahedron(PolyMesh& mesh){
    PolyMesh::VertexHandle vh[8];
    vh[0] = mesh.add_vertex(Point(1,1,1));
    vh[1] = mesh.add_vertex(Point(-1,1,1));
    vh[2] = mesh.add_vertex(Point(1,-1,1));
    vh[3] = mesh.add_vertex(Point(1,1,-1));
    vh[4] = mesh.add_vertex(Point(1,-1,-1));
    vh[5] = mesh.add_vertex(Point(-1,1,-1));
    vh[6] = mesh.add_vertex(Point(-1,-1,1));
    vh[7] = mesh.add_vertex(Point(-1,-1,-1));

    mesh.add_face(vh[0], vh[1], vh[6], vh[2]);
    mesh.add_face(vh[0], vh[3], vh[5], vh[1]);
    mesh.add_face(vh[0], vh[2], vh[4], vh[3]);
    mesh.add_face(vh[7], vh[5], vh[3], vh[4]);
    mesh.add_face(vh[7], vh[6], vh[1], vh[5]);
    mesh.add_face(vh[7], vh[4], vh[2], vh[6]);

    mesh.update_normals();

}

void octahedron(PolyMesh& mesh){
    PolyMesh::VertexHandle vh[6];
    vh[0] = mesh.add_vertex(Point(1,0,0));
    vh[1] = mesh.add_vertex(Point(-1,0,0));
    vh[2] = mesh.add_vertex(Point(0,1,0));
    vh[3] = mesh.add_vertex(Point(0,-1,0));
    vh[4] = mesh.add_vertex(Point(0,0,1));
    vh[5] = mesh.add_vertex(Point(0,0,-1));

    mesh.add_face(vh[0], vh[2], vh[4]);
    mesh.add_face(vh[0], vh[5], vh[2]);
    mesh.add_face(vh[0], vh[3], vh[5]);
    mesh.add_face(vh[0], vh[4], vh[3]);
    mesh.add_face(vh[1], vh[4], vh[2]);
    mesh.add_face(vh[1], vh[2], vh[5]);
    mesh.add_face(vh[1], vh[5], vh[3]);
    mesh.add_face(vh[1], vh[3], vh[4]);
    mesh.update_normals();
}

void dodecahedron(PolyMesh& mesh){
    double g = 0.5*(1.0 + sqrt(5.0));

    PolyMesh::VertexHandle vh[20];
    vh[0] = mesh.add_vertex(Point(1,1,1));
    vh[1] = mesh.add_vertex(Point(-1,1,1));
    vh[2] = mesh.add_vertex(Point(1,-1,1));
    vh[3] = mesh.add_vertex(Point(1,1,-1));
    vh[4] = mesh.add_vertex(Point(1,-1,-1));
    vh[5] = mesh.add_vertex(Point(-1,1,-1));
    vh[6] = mesh.add_vertex(Point(-1,-1,1));
    vh[7] = mesh.add_vertex(Point(-1,-1,-1));

    vh[8] = mesh.add_vertex(Point(0,1.0/g,g));
    vh[9] = mesh.add_vertex(Point(0,-1.0/g,g));
    vh[10] = mesh.add_vertex(Point(0,1.0/g,-g));
    vh[11] = mesh.add_vertex(Point(0,-1.0/g,-g));
    vh[12] = mesh.add_vertex(Point(1.0/g,g,0));
    vh[13] = mesh.add_vertex(Point(-1.0/g,g,0));
    vh[14] = mesh.add_vertex(Point(1.0/g,-g,0));
    vh[15] = mesh.add_vertex(Point(-1.0/g,-g,0));
    vh[16] = mesh.add_vertex(Point(g,0,1.0/g));
    vh[17] = mesh.add_vertex(Point(g,0,-1.0/g));
    vh[18] = mesh.add_vertex(Point(-g,0,1.0/g));
    vh[19] = mesh.add_vertex(Point(-g,0,-1.0/g));

    PolyMesh::VertexHandle f0[] = {vh[0], vh[8], vh[9], vh[2], vh[16]};
    PolyMesh::VertexHandle f1[] = {vh[0], vh[16], vh[17], vh[3], vh[12]};
    PolyMesh::VertexHandle f2[] = {vh[0], vh[12], vh[13], vh[1], vh[8]};
    PolyMesh::VertexHandle f3[] = {vh[1], vh[13], vh[5], vh[19], vh[18]};
    PolyMesh::VertexHandle f4[] = {vh[1], vh[18], vh[6], vh[9], vh[8]};
    PolyMesh::VertexHandle f5[] = {vh[2], vh[9], vh[6], vh[15], vh[14]};
    PolyMesh::VertexHandle f6[] = {vh[2], vh[14], vh[4], vh[17], vh[16]};
    PolyMesh::VertexHandle f7[] = {vh[3], vh[17], vh[4], vh[11], vh[10]};
    PolyMesh::VertexHandle f8[] = {vh[3], vh[10], vh[5], vh[13], vh[12]};
    PolyMesh::VertexHandle f9[] = {vh[4], vh[14], vh[15], vh[7], vh[11]};
    PolyMesh::VertexHandle f10[] = {vh[5], vh[10], vh[11], vh[7], vh[19]};
    PolyMesh::VertexHandle f11[] = {vh[6], vh[18], vh[19], vh[7], vh[15]};
    mesh.add_face(f0, 5);
    mesh.add_face(f1, 5);
    mesh.add_face(f2, 5);
    mesh.add_face(f3, 5);
    mesh.add_face(f4, 5);
    mesh.add_face(f5, 5);
    mesh.add_face(f6, 5);
    mesh.add_face(f7, 5);
    mesh.add_face(f8, 5);
    mesh.add_face(f9, 5);
    mesh.add_face(f10, 5);
    mesh.add_face(f11, 5);
    mesh.update_normals();
}

void icosahedron(PolyMesh& mesh){
    double g = 0.5*(1.0 + sqrt(5.0));

    PolyMesh::VertexHandle vh[12];
    vh[0] = mesh.add_vertex(Point(0,1,g));
    vh[1] = mesh.add_vertex(Point(1,g,0));
    vh[2] = mesh.add_vertex(Point(g,0,1));
    vh[3] = mesh.add_vertex(Point(0,-1,g));
    vh[4] = mesh.add_vertex(Point(-1,g,0));
    vh[5] = mesh.add_vertex(Point(g,0,-1));
    vh[6] = mesh.add_vertex(Point(0,1,-g));
    vh[7] = mesh.add_vertex(Point(1,-g,0));
    vh[8] = mesh.add_vertex(Point(-g,0,1));
    vh[9] = mesh.add_vertex(Point(0,-1,-g));
    vh[10] = mesh.add_vertex(Point(-1,-g,0));
    vh[11] = mesh.add_vertex(Point(-g,0,-1));

    mesh.add_face(vh[0], vh[1], vh[4]);
    mesh.add_face(vh[0], vh[2], vh[1]);
    mesh.add_face(vh[0], vh[3], vh[2]);
    mesh.add_face(vh[0], vh[4], vh[8]);
    mesh.add_face(vh[0], vh[8], vh[3]);
    mesh.add_face(vh[1], vh[2], vh[5]);
    mesh.add_face(vh[1], vh[5], vh[6]);
    mesh.add_face(vh[1], vh[6], vh[4]);
    mesh.add_face(vh[2], vh[3], vh[7]);
    mesh.add_face(vh[2], vh[7], vh[5]);
    mesh.add_face(vh[3], vh[8], vh[10]);
    mesh.add_face(vh[3], vh[10], vh[7]);
    mesh.add_face(vh[4], vh[6], vh[11]);
    mesh.add_face(vh[4], vh[11], vh[8]);
    mesh.add_face(vh[5], vh[9], vh[6]);
    mesh.add_face(vh[5], vh[7], vh[9]);
    mesh.add_face(vh[6], vh[9], vh[11]);
    mesh.add_face(vh[7], vh[10], vh[9]);
    mesh.add_face(vh[8], vh[11], vh[10]);
    mesh.add_face(vh[9], vh[10], vh[11]);
    mesh.update_normals();
}





void truncated_tetrahedron(PolyMesh& mesh){
    PolyMesh::VertexHandle vh[12];
    vh[0] = mesh.add_vertex(Point(3,1,1));
    vh[1] = mesh.add_vertex(Point(1,3,1));
    vh[2] = mesh.add_vertex(Point(1,1,3));
    vh[3] = mesh.add_vertex(Point(-3,-1,1));
    vh[4] = mesh.add_vertex(Point(-1,-3,1));
    vh[5] = mesh.add_vertex(Point(-1,-1,3));
    vh[6] = mesh.add_vertex(Point(-3,1,-1));
    vh[7] = mesh.add_vertex(Point(-1,3,-1));
    vh[8] = mesh.add_vertex(Point(-1,1,-3));
    vh[9] = mesh.add_vertex(Point(3,-1,-1));
    vh[10] = mesh.add_vertex(Point(1,-3,-1));
    vh[11] = mesh.add_vertex(Point(1,-1,-3));

    mesh.add_face(vh[0], vh[1], vh[2]);
    mesh.add_face(vh[3], vh[4], vh[5]);
    mesh.add_face(vh[6], vh[7], vh[8]);
    mesh.add_face(vh[9], vh[10], vh[11]);

    PolyMesh::VertexHandle f0[] = {vh[0], vh[9], vh[11], vh[8], vh[7], vh[1]};
    PolyMesh::VertexHandle f1[] = {vh[0], vh[2], vh[5], vh[4], vh[10], vh[9]};
    PolyMesh::VertexHandle f2[] = {vh[1], vh[7], vh[6], vh[3], vh[5], vh[2]};
    PolyMesh::VertexHandle f3[] = {vh[3], vh[6], vh[8], vh[11], vh[10], vh[4]};
    mesh.add_face(f0, 6);
    mesh.add_face(f1, 6);
    mesh.add_face(f2, 6);
    mesh.add_face(f3, 6);

    mesh.scale(1.0/3.0);

    mesh.update_normals();
}

void cuboctahedron(PolyMesh& mesh){
    PolyMesh::VertexHandle vh[12];
    vh[0] = mesh.add_vertex(Point(1,1,0));
    vh[1] = mesh.add_vertex(Point(1,-1,0));
    vh[2] = mesh.add_vertex(Point(-1,-1,0));
    vh[3] = mesh.add_vertex(Point(-1,1,0));

    vh[4] = mesh.add_vertex(Point(1,0,1));
    vh[5] = mesh.add_vertex(Point(1,0,-1));
    vh[6] = mesh.add_vertex(Point(-1,0,-1));
    vh[7] = mesh.add_vertex(Point(-1,0,1));

    vh[8] = mesh.add_vertex(Point(0,1,1));
    vh[9] = mesh.add_vertex(Point(0,1,-1));
    vh[10] = mesh.add_vertex(Point(0,-1,-1));
    vh[11] = mesh.add_vertex(Point(0,-1,1));

    mesh.add_face(vh[0], vh[9], vh[3], vh[8]);
    mesh.add_face(vh[1], vh[11], vh[2], vh[10]);
    mesh.add_face(vh[2], vh[7], vh[3], vh[6]);
    mesh.add_face(vh[10], vh[6], vh[9], vh[5]);
    mesh.add_face(vh[1], vh[5], vh[0], vh[4]);
    mesh.add_face(vh[11], vh[4], vh[8], vh[7]);

    mesh.add_face(vh[0], vh[8], vh[4]);
    mesh.add_face(vh[7], vh[8], vh[3]);
    mesh.add_face(vh[6], vh[3], vh[9]);
    mesh.add_face(vh[5], vh[9], vh[0]);
    mesh.add_face(vh[7], vh[2], vh[11]);
    mesh.add_face(vh[2], vh[6], vh[10]);
    mesh.add_face(vh[10], vh[5], vh[1]);
    mesh.add_face(vh[4], vh[11], vh[1]);

    mesh.update_normals();
}

// void truncated_cube(PolyMesh& mesh);

void truncated_octahedron(PolyMesh& mesh){
    PolyMesh::VertexHandle vh[24];
    vh[0] = mesh.add_vertex(Point(0,1,2));
    vh[1] = mesh.add_vertex(Point(0,2,1));
    vh[2] = mesh.add_vertex(Point(1,0,2));
    vh[3] = mesh.add_vertex(Point(1,2,0));
    vh[4] = mesh.add_vertex(Point(2,0,1));
    vh[5] = mesh.add_vertex(Point(2,1,0));

    vh[6] = mesh.add_vertex(Point(0,-1,-2));
    vh[7] = mesh.add_vertex(Point(0,-2,-1));
    vh[8] = mesh.add_vertex(Point(-1,0,-2));
    vh[9] = mesh.add_vertex(Point(-1,-2,0));
    vh[10] = mesh.add_vertex(Point(-2,0,-1));
    vh[11] = mesh.add_vertex(Point(-2,-1,0));

    vh[12] = mesh.add_vertex(Point(0,-1,2));
    vh[13] = mesh.add_vertex(Point(0,2,-1));
    vh[14] = mesh.add_vertex(Point(-1,0,2));
    vh[15] = mesh.add_vertex(Point(-1,2,0));
    vh[16] = mesh.add_vertex(Point(2,0,-1));
    vh[17] = mesh.add_vertex(Point(2,-1,0));

    vh[18] = mesh.add_vertex(Point(0,1,-2));
    vh[19] = mesh.add_vertex(Point(0,-2,1));
    vh[20] = mesh.add_vertex(Point(1,0,-2));
    vh[21] = mesh.add_vertex(Point(1,-2,0));
    vh[22] = mesh.add_vertex(Point(-2,0,1));
    vh[23] = mesh.add_vertex(Point(-2,1,0));

    PolyMesh::VertexHandle f0[] = {vh[0], vh[2], vh[4], vh[5], vh[3], vh[1]};
    PolyMesh::VertexHandle f1[] = {vh[0], vh[1], vh[15], vh[23], vh[22], vh[14]};
    PolyMesh::VertexHandle f2[] = {vh[3], vh[5], vh[16], vh[20], vh[18], vh[13]};
    PolyMesh::VertexHandle f3[] = {vh[15], vh[13], vh[18], vh[8], vh[10], vh[23]};
    PolyMesh::VertexHandle f4[] = {vh[10], vh[8], vh[6], vh[7], vh[9], vh[11]};
    PolyMesh::VertexHandle f5[] = {vh[20], vh[16], vh[17], vh[21], vh[7], vh[6]};
    PolyMesh::VertexHandle f6[] = {vh[4], vh[2], vh[12], vh[19], vh[21], vh[17]};
    PolyMesh::VertexHandle f7[] = {vh[11], vh[9], vh[19], vh[12], vh[14], vh[22]};
    mesh.add_face(f0, 6);
    mesh.add_face(f1, 6);
    mesh.add_face(f2, 6);
    mesh.add_face(f3, 6);
    mesh.add_face(f4, 6);
    mesh.add_face(f5, 6);
    mesh.add_face(f6, 6);
    mesh.add_face(f7, 6);

    mesh.add_face(vh[0], vh[14], vh[12], vh[2]);
    mesh.add_face(vh[1], vh[3], vh[13], vh[15]);
    mesh.add_face(vh[4], vh[17], vh[16], vh[5]);
    mesh.add_face(vh[6], vh[8], vh[18], vh[20]);
    mesh.add_face(vh[7], vh[21], vh[19], vh[9]);
    mesh.add_face(vh[10], vh[11], vh[22], vh[23]);

    mesh.scale(1.0/3.0);

    mesh.update_normals();
}

//     void rhombicuboctahedron(PolyMesh& mesh);
//     void truncated_cuboctahedron(PolyMesh& mesh);
//     void snub_hexahedron(PolyMesh& mesh);

void icosidodecahedron(PolyMesh& mesh){
    double g = 0.5*(1.0 + sqrt(5.0));

    PolyMesh::VertexHandle vh[30];
    vh[0] = mesh.add_vertex(Point(g,0,0));
    vh[1] = mesh.add_vertex(Point(-g,0,0));
    vh[2] = mesh.add_vertex(Point(0,g,0));
    vh[3] = mesh.add_vertex(Point(0,-g,0));
    vh[4] = mesh.add_vertex(Point(0,0,g));
    vh[5] = mesh.add_vertex(Point(0,0,-g));

    vh[6] = mesh.add_vertex(Point(0.5,0.5*g,0.5*(1+g)));
    vh[7] = mesh.add_vertex(Point(-0.5,0.5*g,0.5*(1+g)));
    vh[8] = mesh.add_vertex(Point(0.5,-0.5*g,0.5*(1+g)));
    vh[9] = mesh.add_vertex(Point(0.5,0.5*g,-0.5*(1+g)));
    vh[10] = mesh.add_vertex(Point(0.5,-0.5*g,-0.5*(1+g)));
    vh[11] = mesh.add_vertex(Point(-0.5,0.5*g,-0.5*(1+g)));
    vh[12] = mesh.add_vertex(Point(-0.5,-0.5*g,0.5*(1+g)));
    vh[13] = mesh.add_vertex(Point(-0.5,-0.5*g,-0.5*(1+g)));

    vh[14] = mesh.add_vertex(Point(0.5*g,0.5*(1+g),0.5));
    vh[15] = mesh.add_vertex(Point(0.5*g,0.5*(1+g),-0.5));
    vh[16] = mesh.add_vertex(Point(-0.5*g,0.5*(1+g),0.5));
    vh[17] = mesh.add_vertex(Point(0.5*g,-0.5*(1+g),0.5));
    vh[18] = mesh.add_vertex(Point(-0.5*g,-0.5*(1+g),0.5));
    vh[19] = mesh.add_vertex(Point(0.5*g,-0.5*(1+g),-0.5));
    vh[20] = mesh.add_vertex(Point(-0.5*g,0.5*(1+g),-0.5));
    vh[21] = mesh.add_vertex(Point(-0.5*g,-0.5*(1+g),-0.5));

    vh[22] = mesh.add_vertex(Point(0.5*(1+g),0.5,0.5*g));
    vh[23] = mesh.add_vertex(Point(0.5*(1+g),-0.5,0.5*g));
    vh[24] = mesh.add_vertex(Point(0.5*(1+g),0.5,-0.5*g));
    vh[25] = mesh.add_vertex(Point(-0.5*(1+g),0.5,0.5*g));
    vh[26] = mesh.add_vertex(Point(-0.5*(1+g),0.5,-0.5*g));
    vh[27] = mesh.add_vertex(Point(-0.5*(1+g),-0.5,0.5*g));
    vh[28] = mesh.add_vertex(Point(0.5*(1+g),-0.5,-0.5*g));
    vh[29] = mesh.add_vertex(Point(-0.5*(1+g),-0.5,-0.5*g));

    PolyMesh::VertexHandle f0[] = {vh[0], vh[24], vh[15], vh[14], vh[22]};
    PolyMesh::VertexHandle f1[] = {vh[0], vh[23], vh[17], vh[19], vh[28]};
    PolyMesh::VertexHandle f2[] = {vh[1], vh[25], vh[16], vh[20], vh[26]};
    PolyMesh::VertexHandle f3[] = {vh[1], vh[29], vh[21], vh[18], vh[27]};
    PolyMesh::VertexHandle f4[] = {vh[2], vh[16], vh[7], vh[6], vh[14]};
    PolyMesh::VertexHandle f5[] = {vh[2], vh[15], vh[9], vh[11], vh[20]};
    PolyMesh::VertexHandle f6[] = {vh[3], vh[17], vh[8], vh[12], vh[18]};
    PolyMesh::VertexHandle f7[] = {vh[3], vh[21], vh[13], vh[10], vh[19]};
    PolyMesh::VertexHandle f8[] = {vh[4], vh[8], vh[23], vh[22], vh[6]};
    PolyMesh::VertexHandle f9[] = {vh[4], vh[7], vh[25], vh[27], vh[12]};
    PolyMesh::VertexHandle f10[] = {vh[5], vh[13], vh[29], vh[26], vh[11]};
    PolyMesh::VertexHandle f11[] = {vh[5], vh[9], vh[24], vh[28], vh[10]};
    mesh.add_face(f0, 5);
    mesh.add_face(f1, 5);
    mesh.add_face(f2, 5);
    mesh.add_face(f3, 5);
    mesh.add_face(f4, 5);
    mesh.add_face(f5, 5);
    mesh.add_face(f6, 5);
    mesh.add_face(f7, 5);
    mesh.add_face(f8, 5);
    mesh.add_face(f9, 5);
    mesh.add_face(f10, 5);
    mesh.add_face(f11, 5);

    mesh.add_face(vh[0], vh[22], vh[23]);
    mesh.add_face(vh[0], vh[28], vh[24]);
    mesh.add_face(vh[1], vh[27], vh[25]);
    mesh.add_face(vh[1], vh[26], vh[29]);
    mesh.add_face(vh[2], vh[14], vh[15]);
    mesh.add_face(vh[2], vh[20], vh[16]);
    mesh.add_face(vh[3], vh[18], vh[21]);
    mesh.add_face(vh[3], vh[19], vh[17]);
    mesh.add_face(vh[4], vh[12], vh[8]);
    mesh.add_face(vh[5], vh[11], vh[9]);
    mesh.add_face(vh[5], vh[10], vh[13]);
    mesh.add_face(vh[16], vh[25], vh[7]);
    mesh.add_face(vh[7], vh[4], vh[6]);
    mesh.add_face(vh[6], vh[22], vh[14]);
    mesh.add_face(vh[8], vh[17], vh[23]);
    mesh.add_face(vh[10], vh[28], vh[19]);
    mesh.add_face(vh[12], vh[27], vh[18]);
    mesh.add_face(vh[13], vh[21], vh[29]);
    mesh.add_face(vh[20], vh[11], vh[26]);
    mesh.add_face(vh[15], vh[24], vh[9]);
    mesh.update_normals();
}

//void truncated_dodecahedron(PolyMesh& mesh);

void truncated_icosahedron(PolyMesh& mesh){
    double g = 0.5*(1.0 + sqrt(5.0));

    PolyMesh::VertexHandle vh[60];
    vh[0] = mesh.add_vertex(Point(0, 1, 3*g));
    vh[1] = mesh.add_vertex(Point(1, 3*g, 0));
    vh[2] = mesh.add_vertex(Point(3*g, 0, 1));
    vh[3] = mesh.add_vertex(Point(0, -1, 3*g));
    vh[4] = mesh.add_vertex(Point(-1, 3*g, 0));
    vh[5] = mesh.add_vertex(Point(3*g, 0, -1));
    vh[6] = mesh.add_vertex(Point(0, 1, -3*g));
    vh[7] = mesh.add_vertex(Point(1, -3*g, 0));
    vh[8] = mesh.add_vertex(Point(-3*g, 0, 1));
    vh[9] = mesh.add_vertex(Point(0, -1, -3*g));
    vh[10] = mesh.add_vertex(Point(-1, -3*g, 0));
    vh[11] = mesh.add_vertex(Point(-3*g, 0, -1));

    vh[12] = mesh.add_vertex(Point(2, (1+2*g), g));
    vh[13] = mesh.add_vertex(Point((1+2*g), g, 2));
    vh[14] = mesh.add_vertex(Point(g, 2, (1+2*g)));
    vh[15] = mesh.add_vertex(Point(-2, (1+2*g), g));
    vh[16] = mesh.add_vertex(Point((1+2*g), g, -2));
    vh[17] = mesh.add_vertex(Point(g, -2, (1+2*g)));
    vh[18] = mesh.add_vertex(Point(2, -(1+2*g), g));
    vh[19] = mesh.add_vertex(Point(-(1+2*g), g, 2));
    vh[20] = mesh.add_vertex(Point(g, 2, -(1+2*g)));
    vh[21] = mesh.add_vertex(Point(2, (1+2*g), -g));
    vh[22] = mesh.add_vertex(Point((1+2*g), -g, 2));
    vh[23] = mesh.add_vertex(Point(-g, 2, (1+2*g)));
    vh[24] = mesh.add_vertex(Point(-2, -(1+2*g), g));
    vh[25] = mesh.add_vertex(Point(-(1+2*g), g, -2));
    vh[26] = mesh.add_vertex(Point(g, -2, -(1+2*g)));
    vh[27] = mesh.add_vertex(Point(-2, (1+2*g), -g));
    vh[28] = mesh.add_vertex(Point((1+2*g), -g, -2));
    vh[29] = mesh.add_vertex(Point(-g, -2, (1+2*g)));
    vh[30] = mesh.add_vertex(Point(2, -(1+2*g), -g));
    vh[31] = mesh.add_vertex(Point(-(1+2*g), -g, 2));
    vh[32] = mesh.add_vertex(Point(-g, 2, -(1+2*g)));
    vh[33] = mesh.add_vertex(Point(-2, -(1+2*g), -g));
    vh[34] = mesh.add_vertex(Point(-(1+2*g), -g, -2));
    vh[35] = mesh.add_vertex(Point(-g, -2, -(1+2*g)));

    vh[36] = mesh.add_vertex(Point(1, (2+g), 2*g));
    vh[37] = mesh.add_vertex(Point((2+g), 2*g, 1));
    vh[38] = mesh.add_vertex(Point(2*g, 1, (2+g)));
    vh[39] = mesh.add_vertex(Point(-1, (2+g), 2*g));
    vh[40] = mesh.add_vertex(Point((2+g), 2*g, -1));
    vh[41] = mesh.add_vertex(Point(2*g, -1, (2+g)));
    vh[42] = mesh.add_vertex(Point(1, -(2+g), 2*g));
    vh[43] = mesh.add_vertex(Point(-(2+g), 2*g, 1));
    vh[44] = mesh.add_vertex(Point(2*g, 1, -(2+g)));
    vh[45] = mesh.add_vertex(Point(1, (2+g), -2*g));
    vh[46] = mesh.add_vertex(Point((2+g), -2*g, 1));
    vh[47] = mesh.add_vertex(Point(-2*g, 1, (2+g)));
    vh[48] = mesh.add_vertex(Point(1, -(2+g), -2*g));
    vh[49] = mesh.add_vertex(Point(-(2+g), -2*g, 1));
    vh[50] = mesh.add_vertex(Point(-2*g, 1, -(2+g)));
    vh[51] = mesh.add_vertex(Point(-1, (2+g), -2*g));
    vh[52] = mesh.add_vertex(Point((2+g), -2*g, -1));
    vh[53] = mesh.add_vertex(Point(-2*g, -1, (2+g)));
    vh[54] = mesh.add_vertex(Point(-1, -(2+g), 2*g));
    vh[55] = mesh.add_vertex(Point(-(2+g), 2*g, -1));
    vh[56] = mesh.add_vertex(Point(2*g, -1, -(2+g)));
    vh[57] = mesh.add_vertex(Point(-1, -(2+g), -2*g));
    vh[58] = mesh.add_vertex(Point(-(2+g), -2*g, -1));
    vh[59] = mesh.add_vertex(Point(-2*g, -1, -(2+g)));

    PolyMesh::VertexHandle f0[] = {vh[0], vh[23], vh[47], vh[53], vh[29], vh[3]};
    PolyMesh::VertexHandle f1[] = {vh[0], vh[3], vh[17], vh[41], vh[38], vh[14]};
    PolyMesh::VertexHandle f2[] = {vh[14], vh[38], vh[13], vh[37], vh[12], vh[36]};
    PolyMesh::VertexHandle f3[] = {vh[37], vh[13], vh[2], vh[5], vh[16], vh[40]};
    PolyMesh::VertexHandle f4[] = {vh[40], vh[16], vh[44], vh[20], vh[45], vh[21]};
    PolyMesh::VertexHandle f5[] = {vh[20], vh[44], vh[56], vh[26], vh[9], vh[6]};
    PolyMesh::VertexHandle f6[] = {vh[6], vh[9], vh[35], vh[59], vh[50], vh[32]};
    PolyMesh::VertexHandle f7[] = {vh[59], vh[35], vh[57], vh[33], vh[58], vh[34]};
    PolyMesh::VertexHandle f8[] = {vh[34], vh[58], vh[49], vh[31], vh[8], vh[11]};
    PolyMesh::VertexHandle f9[] = {vh[24], vh[54], vh[29], vh[53], vh[31], vh[49]};
    PolyMesh::VertexHandle f10[] = {vh[1], vh[21], vh[45], vh[51], vh[27], vh[4]};
    PolyMesh::VertexHandle f11[] = {vh[51], vh[32], vh[50], vh[25], vh[55], vh[27]};
    PolyMesh::VertexHandle f12[] = {vh[55], vh[25], vh[11], vh[8], vh[19], vh[43]};
    PolyMesh::VertexHandle f13[] = {vh[15], vh[39], vh[36], vh[12], vh[1], vh[4]};
    PolyMesh::VertexHandle f14[] = {vh[39], vh[15], vh[43], vh[19], vh[47], vh[23]};
    PolyMesh::VertexHandle f15[] = {vh[30], vh[7], vh[10], vh[33], vh[57], vh[48]};
    PolyMesh::VertexHandle f16[] = {vh[54], vh[24], vh[10], vh[7], vh[18], vh[42]};
    PolyMesh::VertexHandle f17[] = {vh[17], vh[42], vh[18], vh[46], vh[22], vh[41]};
    PolyMesh::VertexHandle f18[] = {vh[5], vh[2], vh[22], vh[46], vh[52], vh[28]};
    PolyMesh::VertexHandle f19[] = {vh[26], vh[56], vh[28], vh[52], vh[30], vh[48]};
    mesh.add_face(f0, 6);
    mesh.add_face(f1, 6);
    mesh.add_face(f2, 6);
    mesh.add_face(f3, 6);
    mesh.add_face(f4, 6);
    mesh.add_face(f5, 6);
    mesh.add_face(f6, 6);
    mesh.add_face(f7, 6);
    mesh.add_face(f8, 6);
    mesh.add_face(f9, 6);
    mesh.add_face(f10, 6);
    mesh.add_face(f11, 6);
    mesh.add_face(f12, 6);
    mesh.add_face(f13, 6);
    mesh.add_face(f14, 6);
    mesh.add_face(f15, 6);
    mesh.add_face(f16, 6);
    mesh.add_face(f17, 6);
    mesh.add_face(f18, 6);
    mesh.add_face(f19, 6);

    PolyMesh::VertexHandle f20[] = {vh[0], vh[14], vh[36], vh[39], vh[23]};
    PolyMesh::VertexHandle f21[] = {vh[35], vh[9], vh[26], vh[48], vh[57]};
    PolyMesh::VertexHandle f22[] = {vh[25], vh[50], vh[59], vh[34], vh[11]};
    PolyMesh::VertexHandle f23[] = {vh[43], vh[15], vh[4], vh[27], vh[55]};
    PolyMesh::VertexHandle f24[] = {vh[47], vh[19], vh[8], vh[31], vh[53]};
    PolyMesh::VertexHandle f25[] = {vh[51], vh[45], vh[20], vh[6], vh[32]};
    PolyMesh::VertexHandle f26[] = {vh[16], vh[5], vh[28], vh[56], vh[44]};
    PolyMesh::VertexHandle f27[] = {vh[38], vh[41], vh[22], vh[2], vh[13]};
    PolyMesh::VertexHandle f28[] = {vh[3], vh[29], vh[54], vh[42], vh[17]};
    PolyMesh::VertexHandle f29[] = {vh[49], vh[58], vh[33], vh[10], vh[24]};
    PolyMesh::VertexHandle f30[] = {vh[12], vh[37], vh[40], vh[21], vh[1]};
    PolyMesh::VertexHandle f31[] = {vh[30], vh[52], vh[46], vh[18], vh[7]};
    mesh.add_face(f20, 5);
    mesh.add_face(f21, 5);
    mesh.add_face(f22, 5);
    mesh.add_face(f23, 5);
    mesh.add_face(f24, 5);
    mesh.add_face(f25, 5);
    mesh.add_face(f26, 5);
    mesh.add_face(f27, 5);
    mesh.add_face(f28, 5);
    mesh.add_face(f29, 5);
    mesh.add_face(f30, 5);
    mesh.add_face(f31, 5);
    mesh.scale(1.0/3.0);
    mesh.update_normals();
}

//     void rhombicosidodecahedron(PolyMesh& mesh);
//     void truncated_icosidodecahedron(PolyMesh& mesh);
//     void snub_dodecahedron(PolyMesh& mesh);



}