#ifndef MeshSampler_h__
#define MeshSampler_h__

#include "BaseFilter.h"

namespace hccl
{

class MeshSampler : public BaseFilter
{
public:
	MeshSampler(MeshType* mesh = NULL) : BaseFilter(mesh){}
	~MeshSampler(){}

	void sampleRandom(int nSamples, std::vector<std::pair<PointType, int>>& samples) const;
	void sampleUniform(int nSamples, std::vector<PointType>& samples, std::vector<int>& indx) const;
	void sampleUniformBySamplingDistance(const ScalarType samplingD, std::vector<PointType>& samples, std::vector<int>& indx) const;
};

}
#endif // MeshSampler_h__
