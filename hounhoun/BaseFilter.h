#ifndef BaseFilter_h__
#define BaseFilter_h__

#include "hccl/mesh/mesh.h"

namespace hccl
{

class BaseFilter
{
public:
	typedef TriMesh MeshType;
	typedef MeshType::Point PointType;
	typedef MeshType::Scalar ScalarType;


    BaseFilter(MeshType* mesh = NULL) : m_Mesh(mesh){}
    ~BaseFilter(void){}

    virtual void setMesh(MeshType* mesh){m_Mesh = mesh;};

protected:
    MeshType* m_Mesh;
};

}
#endif // BaseFilter_h__