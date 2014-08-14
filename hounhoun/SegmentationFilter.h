#ifndef SegmentationFilter_h__
#define SegmentationFilter_h__

#include "BaseFilter.h"

namespace hccl
{

class SegmentationFilter : public BaseFilter
{
public:
    SegmentationFilter(MeshType* mesh = NULL) : BaseFilter(mesh){};
    ~SegmentationFilter(){};

    bool segmentation(const std::vector<MeshType::VertexHandle>& divVertices, std::vector<int>& label, int step);
};

}
#endif // SegmentationFilter_h__