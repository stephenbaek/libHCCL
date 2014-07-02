#ifndef HCCL_MESH_H_
#define HCCL_MESH_H_

#include <vector>

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Tools/Smoother/SmootherT.hh>
#include <OpenMesh/Tools/Smoother/smooth_mesh.hh>

namespace hccl{
typedef OpenMesh::Vec3d Point;
typedef OpenMesh::Vec3d Vector;
typedef OpenMesh::Vec3d Normal;
typedef OpenMesh::Vec4d Color;

struct Traits : public OpenMesh::DefaultTraits
{
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
typedef OpenMesh::TriMesh_ArrayKernelT<Traits> Kernel;

class TriMesh : public Kernel
{
public:
    TriMesh();
    ~TriMesh();

    bool read(std::string strFilePath);
    bool write(std::string strFilePath) const;

public:
    void translate(double x, double y, double z);
    void translate(Vector v);
    // void rotate(double angle, OpenMesh::Vec3d axis); // TODO
};

} // namespace hccl

#endif // HCCL_MESH_H_