#include "mesh.h"

#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <OpenMesh/Tools/Decimater/ModQuadricT.hh>

using namespace std;

namespace hccl{

#ifndef min
#define min(a, b) (((a) < (b))? (a) : (b))
#endif
#ifndef max
#define max(a, b) (((a) > (b))? (a) : (b))
#endif
    
TriMesh::TriMesh(void)
{
}

TriMesh::~TriMesh(void)
{
}

bool TriMesh::read(std::string strFilePath)
{
    OpenMesh::IO::Options opt;
    if(!has_vertex_normals())
    {
        std::cerr << "File Open Error: Standard vertex property 'Vertex Normals' not available!\n";
        return false;
    }
    if(!has_vertex_colors())
    {
        std::cerr << "File Open Error: Standard vertex property 'Vertex Colors' not available!\n";
        return false;
    }
    if(!has_face_normals())
    {
        std::cerr << "File Open Error: Standard vertex property 'Face Normals' not available!\n";
        return false;
    }
    if(!has_face_colors())
    {
        std::cerr << "File Open Error: Standard vertex property 'Face Colors' not available!\n";
        return false;
    }

    if( !OpenMesh::IO::read_mesh(*this, strFilePath, opt) )
    {
        std::cerr << "File Open Error: Error loading mesh from file " << strFilePath << std::endl;
        return false;
    }
    if( !opt.check( OpenMesh::IO::Options::FaceNormal) )
        update_face_normals();
    if( !opt.check( OpenMesh::IO::Options::VertexNormal) )
        update_vertex_normals();

    return true;
}

bool TriMesh::write(std::string strFilePath) const
{
    if ( !OpenMesh::IO::write_mesh(*this, strFilePath) )
    {
        std::cerr << "Cannot write mesh to file" << strFilePath << std::endl;
        return false;
    }
    return true;
}

void TriMesh::translate(double x, double y, double z)
{
    for(TriMesh::VertexIter vit = vertices_begin(), vend = vertices_end(); vit != vend; ++vit){
        set_point(vit, point(vit) + Point(x,y,z));
    }
}

void TriMesh::translate(Vector v)
{
    translate(v[0], v[1], v[2]);
}

// void SNUMesh::rotate( double angle, SNUMesh::Point axis )
// {
// 
// }

}   // namespace hccl