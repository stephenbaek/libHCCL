#pragma once

#include "ofMain.h"
#include "../common.h"

// TODO: Need TriMesh versions

namespace hccl{
    //////////////////////////////////////////////////////////////////////////
    // Platonic Solids (http://en.wikipedia.org/wiki/Platonic_solid)
    //////////////////////////////////////////////////////////////////////////
    void tetrahedron(PolyMesh& mesh);
    void hexahedron(PolyMesh& mesh);
    void octahedron(PolyMesh& mesh);
    void dodecahedron(PolyMesh& mesh);
    void icosahedron(PolyMesh& mesh);


    //////////////////////////////////////////////////////////////////////////
    // Archimedean Solids (http://en.wikipedia.org/wiki/Archimedean_solid)
    //////////////////////////////////////////////////////////////////////////
    void truncated_tetrahedron(PolyMesh& mesh);
    void cuboctahedron(PolyMesh& mesh);
    // void truncated_cube(PolyMesh& mesh);
    void truncated_octahedron(PolyMesh& mesh);
//     void rhombicuboctahedron(PolyMesh& mesh);
//     void truncated_cuboctahedron(PolyMesh& mesh);
//     void snub_hexahedron(PolyMesh& mesh);
    void icosidodecahedron(PolyMesh& mesh);
    //void truncated_dodecahedron(PolyMesh& mesh);
    void truncated_icosahedron(PolyMesh& mesh);
//     void rhombicosidodecahedron(PolyMesh& mesh);
//     void truncated_icosidodecahedron(PolyMesh& mesh);
//     void snub_dodecahedron(PolyMesh& mesh);
}
