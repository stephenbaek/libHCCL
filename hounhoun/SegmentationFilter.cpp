#include "SegmentationFilter.h"

using namespace hccl;
using namespace std;

bool SegmentationFilter::segmentation( const std::vector<MeshType::VertexHandle>& divVertices, std::vector<int>& label, int step )
{
    if (m_Mesh == NULL || divVertices.empty())
    {
        cout<<"empty input!!"<<endl;
    }

    // mark boundary edges
    std::vector<bool> edge_isBoundary(m_Mesh->n_edges(), false);    
    for(size_t i = 0; i < divVertices.size()-1; i++)
    {
        MeshType::EdgeHandle eh = m_Mesh->edge_handle(m_Mesh->find_halfedge(divVertices[i], divVertices[i+1]));
        if (eh.is_valid()) edge_isBoundary[eh.idx()] = true;
        else
        {
            cout<<i<<" : edge is not valid!"<<endl;
            return false;
        }
    }

    // expand labels (region growing)
    // generate random seeds
    std::vector<int> label_map(m_Mesh->n_faces());
    label.clear();
    label.resize(m_Mesh->n_faces());
    for(int i = 0; i < m_Mesh->n_faces(); i++)
    {
        label_map[i] = i;
        label[i] = i;
    }

    bool changed;
    for(int cnt = 0; cnt < step; cnt++)
    {
        changed = false;
        std::vector<bool> visited(m_Mesh->n_faces(), false);
        for(auto fit = m_Mesh->faces_begin(); fit != m_Mesh->faces_end(); ++fit)
        {
            int f_indx = fit.handle().idx();
            if(visited[f_indx]) continue;

            for(auto fhit = m_Mesh->fh_begin(fit); fhit != m_Mesh->fh_end(fit); ++fhit)
            {
                MeshType::EdgeHandle eh = m_Mesh->edge_handle(fhit);
                int e_indx = eh.idx();

                if (!edge_isBoundary[e_indx] && !m_Mesh->is_boundary(eh))
                {
                    int f0 = f_indx;
                    int f1 = m_Mesh->opposite_face_handle(fhit).idx();
                    int newLabel = std::min(label[f0], label[f1]);
                    int oldLabel = std::max(label[f0], label[f1]);
                    if(newLabel != oldLabel)
                    {
                        label_map[label[f0]] = newLabel;
                        label_map[label[f1]] = newLabel;
                        label[f0] = newLabel;
                        label[f1] = newLabel;
                        changed = true;
                    }
                }
            }
            visited[f_indx] = true;
        }

        if(!changed)
            break;

        for(int i = 0; i < m_Mesh->n_faces(); i++)
        {
            int newLabel = label[i];
            while(newLabel != label_map[newLabel])
            {
                newLabel = label_map[newLabel];
            }
            if(newLabel != label[i])
            {
                label_map[label[i]] = newLabel;
                label[i] = newLabel;
            }
        }
    }

    std::vector<int> ordered_label = label;
    std::sort(ordered_label.begin(), ordered_label.end());
    ordered_label.erase(std::unique(ordered_label.begin(), ordered_label.end()), ordered_label.end());

    std::map<int, int> label_order_map;
    for(size_t i = 0; i < ordered_label.size(); i++)
        label_order_map[ordered_label[i]] = i+1;

    for(size_t i = 0; i < label.size(); i++)
        label[i] = label_order_map[label[i]];

    return true;
}
