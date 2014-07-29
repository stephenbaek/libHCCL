#ifndef CurvatureFilter_h__
#define CurvatureFilter_h__

#include "BaseFilter.h"

namespace hccl
{

class CurvatureFilter : public BaseFilter
{
public:
	CurvatureFilter(MeshType* mesh = NULL) : BaseFilter(mesh){}
	~CurvatureFilter(){}

	void setMesh(MeshType* mesh);
	void clear();
	void updateCurvature();

	std::vector<PointType> k1_dir;
	std::vector<PointType> k2_dir;
	std::vector<ScalarType> k1;
	std::vector<ScalarType> k2;
private:	
	BaseFilter::PointType rotateAxis(PointType vec, PointType axis, ScalarType theta);

};

}
#endif // CurvatureFilter_h__