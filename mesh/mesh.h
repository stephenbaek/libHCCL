#ifndef HCCL_MESH_H_
#define HCCL_MESH_H_

#include "../common.h"

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <OpenMesh/Tools/Smoother/SmootherT.hh>
#include <OpenMesh/Tools/Smoother/smooth_mesh.hh>

namespace hccl{

struct Traits : public OpenMesh::DefaultTraits
{
    typedef OpenMesh::Vec3d Point;
    typedef OpenMesh::Vec3d Vector;
    typedef OpenMesh::Vec3d Normal;
    typedef OpenMesh::Vec4d Color;

    VertexAttributes(
        OpenMesh::Attributes::Status |
        OpenMesh::Attributes::Normal |
        OpenMesh::Attributes::Color);
    FaceAttributes(
        OpenMesh::Attributes::Status |
        OpenMesh::Attributes::Normal |
        OpenMesh::Attributes::Color);
    EdgeAttributes(OpenMesh::Attributes::Status);
};
typedef OpenMesh::TriMesh_ArrayKernelT<Traits> TriKernel;
typedef OpenMesh::PolyMesh_ArrayKernelT<Traits> PolyKernel;

// typedef Kernel::Point Point;
// typedef Kernel::Normal Vector;
typedef OpenMesh::Vec4d Color;


typedef class TriMesh : public TriKernel
{
public:
    TriMesh();
    ~TriMesh();

    bool read(std::string strFilePath);
    bool write(std::string strFilePath) const;

public:
    void scale(double s);
    void translate(double x, double y, double z);
    void translate(Vector v);
    // void rotate(double angle, OpenMesh::Vec3d axis); // TODO

public:
    Point calc_center_of_geometry();
    double calc_max_radius();
    double calc_min_radius();
    double calc_max_x();
    double calc_min_x();
    double calc_max_y();
    double calc_min_y();
    double calc_max_z();
    double calc_min_z();
}TriMesh;


typedef class PolyMesh : public PolyKernel
{
public:
    PolyMesh();
    ~PolyMesh();

    bool read(std::string strFilePath);
    bool write(std::string strFilePath) const;

public:
    void scale(double s);
    void translate(double x, double y, double z);
    void translate(Vector v);
    // void rotate(double angle, OpenMesh::Vec3d axis); // TODO

public:
    Point calc_center_of_geometry();
    double calc_max_radius();
    double calc_min_radius();
    double calc_max_x();
    double calc_min_x();
    double calc_max_y();
    double calc_min_y();
    double calc_max_z();
    double calc_min_z();
}PolyMesh;

} // namespace hccl

#endif // HCCL_MESH_H_