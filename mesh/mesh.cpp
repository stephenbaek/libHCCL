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

void TriMesh::scale(double s){
    for(TriMesh::VertexIter vit = vertices_begin(), vend = vertices_end(); vit != vend; ++vit){
        set_point(*vit, point(*vit)*s);
    }    
}

void TriMesh::translate(double x, double y, double z)
{
    for(TriMesh::VertexIter vit = vertices_begin(), vend = vertices_end(); vit != vend; ++vit){
        set_point(*vit, point(*vit) + Point(x,y,z));
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

Point TriMesh::calc_center_of_geometry(){
    Point cog(0,0,0);
    if(n_vertices() == 0)
        return cog;
    for(TriMesh::VertexIter vit = vertices_begin(), vend = vertices_end(); vit != vend; ++vit){
        cog += point(*vit);
    }
    cog /= n_vertices();
    return cog;
}

double TriMesh::calc_max_radius(){
    double maxr = 0.0, r;
    for(TriMesh::VertexIter vit = vertices_begin(), vend = vertices_end(); vit != vend; ++vit){
        r = point(*vit).sqrnorm();
        if(r > maxr)
            maxr = r;
    }
    return sqrt(maxr);
}

double TriMesh::calc_min_radius(){
    double minr = INF, r;
    for(TriMesh::VertexIter vit = vertices_begin(), vend = vertices_end(); vit != vend; ++vit){
        r = point(*vit).sqrnorm();
        if(r < minr)
            minr = r;
    }
    return sqrt(minr);
}

double TriMesh::calc_max_x(){
    if(n_vertices() == 0)
        return 0.0;
    double maxx = -INF, x;
    for(TriMesh::VertexIter vit = vertices_begin(), vend = vertices_end(); vit != vend; ++vit){
        x = point(*vit)[0];
        if(x > maxx)
            maxx = x;
    }
    return maxx;
}

double TriMesh::calc_min_x(){
    if(n_vertices() == 0)
        return 0.0;
    double minx = INF, x;
    for(TriMesh::VertexIter vit = vertices_begin(), vend = vertices_end(); vit != vend; ++vit){
        x = point(*vit)[0];
        if(x < minx)
            minx = x;
    }
    return minx;
}

double TriMesh::calc_max_y(){
    if(n_vertices() == 0)
        return 0.0;
    double maxy = -INF, y;
    for(TriMesh::VertexIter vit = vertices_begin(), vend = vertices_end(); vit != vend; ++vit){
        y = point(*vit)[1];
        if(y > maxy)
            maxy = y;
    }
    return maxy;
}

double TriMesh::calc_min_y(){
    if(n_vertices() == 0)
        return 0.0;
    double miny = INF, y;
    for(TriMesh::VertexIter vit = vertices_begin(), vend = vertices_end(); vit != vend; ++vit){
        y = point(*vit)[1];
        if(y < miny)
            miny = y;
    }
    return miny;
}

double TriMesh::calc_max_z(){
    if(n_vertices() == 0)
        return 0.0;
    double maxz = -INF, z;
    for(TriMesh::VertexIter vit = vertices_begin(), vend = vertices_end(); vit != vend; ++vit){
        z = point(*vit)[2];
        if(z > maxz)
            maxz = z;
    }
    return maxz;
}

double TriMesh::calc_min_z(){
    if(n_vertices() == 0)
        return 0.0;
    double minz = INF, z;
    for(TriMesh::VertexIter vit = vertices_begin(), vend = vertices_end(); vit != vend; ++vit){
        z = point(*vit)[2];
        if(z < minz)
            minz = z;
    }
    return minz;
}

double TriMesh::calc_avg_edgeLength()
{	
	double avg_edge_length = 0.0;
	for(ConstEdgeIter eit = edges_begin(), eend = edges_end(); eit != eend; ++eit){
		avg_edge_length += calc_edge_length(eit);
	}
	avg_edge_length /= n_edges();
	return avg_edge_length;
}




}   // namespace hccl