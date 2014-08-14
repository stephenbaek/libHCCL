#ifndef RemeshFilter_h__
#define RemeshFilter_h__

#include "BaseFilter.h"

namespace hccl
{

class RemeshFilter : public BaseFilter
{
public:
    RemeshFilter(MeshType* mesh = NULL);
    ~RemeshFilter();

	void setMesh(MeshType* mesh);
	void update(const ScalarType tl, const PointType& ptIn = PointType(), const ScalarType r = 0, const int nit = 1);
	void update(const ScalarType tl, const std::vector<int>& c, const int nit = 1, const bool reverseConstraint = false);
	void update_only_collapse(const ScalarType tl);

private:
	ScalarType target_length;
	ScalarType low;
	ScalarType high;

	PointType pt;
	ScalarType radius;
	
	int numIter;

	std::vector<bool> constraint;
	std::vector<bool> b_edge;

	OpenMesh::EPropHandleT<bool> checkEdge;
	OpenMesh::VPropHandleT<bool> checkVertex;


	void split_Long_Edges();
	void collapse_Short_Edges();
	void tangential_Relaxation();
};

}
#endif // RemeshFilter_h__
