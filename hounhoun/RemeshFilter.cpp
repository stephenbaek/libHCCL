#include "RemeshFilter.h"

using namespace hccl;
using namespace std;

hccl::RemeshFilter::RemeshFilter( MeshType* mesh /*= NULL*/ )
	: BaseFilter(mesh)
{


}


hccl::RemeshFilter::~RemeshFilter()
{
	m_Mesh->remove_property(checkEdge);
	m_Mesh->remove_property(checkVertex);
}

void hccl::RemeshFilter::setMesh( MeshType* mesh )
{
	BaseFilter::setMesh(mesh);

	m_Mesh->add_property(checkEdge, "checkEdge");
	m_Mesh->add_property(checkVertex, "checkVertex");
}


void hccl::RemeshFilter::update( const ScalarType tl, const PointType& ptIn, const ScalarType r, const int nit )
{
	target_length = tl;
	pt = ptIn;
	radius = r;
	numIter = nit;

	low = 4/5.0*target_length;
	high = 4/3.0*target_length;

	constraint.clear();

	for(int iter = 0; iter < numIter; ++iter)
	{
		split_Long_Edges();
		collapse_Short_Edges();
		tangential_Relaxation();
	}
}

void hccl::RemeshFilter::update( const ScalarType tl, const std::vector<int>& c, const int nit /*= 1*/, const bool reverseConstraint /*= false*/ )
{
	target_length = tl;
	numIter = nit;

	low = 4/5.0*target_length;
	high = 4/3.0*target_length;

	constraint.clear();
	if (reverseConstraint)
	{
		constraint.resize(m_Mesh->n_vertices(), true);
#pragma omp parallel for
		for (int i = 0; i<c.size() ;++i)
			constraint[c[i]] = false;
	}
	else
	{
		constraint.resize(m_Mesh->n_vertices(), false);
#pragma omp parallel for
		for (int i = 0; i<c.size() ;++i)
			constraint[c[i]] = true;
	}


	for(int iter = 0; iter < numIter; ++iter)
	{
		split_Long_Edges();
		collapse_Short_Edges();
		tangential_Relaxation();
	}
}

void hccl::RemeshFilter::update_only_collapse( const ScalarType tl )
{
	target_length = tl;
	low = 0.8*target_length;

	radius = 0;
	numIter = 5;

	for(int iter = 0; iter < numIter; ++iter)
	{		
		b_edge.resize(m_Mesh->n_edges(), true);
		for(int i = 0; i < m_Mesh->n_edges(); ++i)
		{
			MeshType::HalfedgeHandle heh = m_Mesh->halfedge_handle(m_Mesh->edge_handle(i), 0);
			MeshType::VertexHandle vh1 = m_Mesh->to_vertex_handle(heh);
			MeshType::VertexHandle vh2 = m_Mesh->from_vertex_handle(heh);

			if(m_Mesh->is_boundary(vh1) || m_Mesh->is_boundary(vh2))
			{ 
				b_edge[i] = false;
			}
		}


		bool finished = false;
		while(!finished)
		{
			finished = true;
			for(int i = 0; i < m_Mesh->n_edges(); ++i)
			{
				if(b_edge[i] == false)
					continue;

				MeshType::EdgeHandle eh = m_Mesh->edge_handle(i);
				MeshType::HalfedgeHandle heh = m_Mesh->halfedge_handle(eh, 0);
				MeshType::VertexHandle vh1 = m_Mesh->from_vertex_handle(heh);
				MeshType::VertexHandle vh2 = m_Mesh->to_vertex_handle(heh);

				if(m_Mesh->is_boundary(vh1) || m_Mesh->is_boundary(vh2))
					continue;

				if(m_Mesh->calc_edge_length(eh) < low)
				{
					if(m_Mesh->is_collapse_ok(heh))
					{
						int idx1 = m_Mesh->edge_handle(heh).idx();
						int idx2 = m_Mesh->edge_handle(m_Mesh->prev_halfedge_handle(heh)).idx();
						int idx3 = m_Mesh->edge_handle(m_Mesh->next_halfedge_handle(m_Mesh->opposite_halfedge_handle(heh))).idx();
						b_edge[idx1] = false;
						b_edge[idx2] = false;
						b_edge[idx3] = false;
						m_Mesh->collapse(heh);
						finished = false;
					}
				}
			}
		}
		m_Mesh->garbage_collection();
	}

}

void hccl::RemeshFilter::split_Long_Edges()
{
	bool finished = false;
	while(!finished)
	{
		finished = true;
		int n = m_Mesh->n_edges();

		// update edge list
		b_edge.resize(m_Mesh->n_edges(), false);
		for(int i = 0; i < m_Mesh->n_edges(); ++i)
		{
			MeshType::HalfedgeHandle heh = m_Mesh->halfedge_handle(m_Mesh->edge_handle(i), 0);
			MeshType::VertexHandle vh1 = m_Mesh->to_vertex_handle(heh);
			MeshType::VertexHandle vh2 = m_Mesh->from_vertex_handle(heh);
			
			if(m_Mesh->is_boundary(vh1) || m_Mesh->is_boundary(vh2))
				continue;


			if (constraint.empty())
			{
				if (radius > 0)
				{
					double dist1 = (m_Mesh->point(vh1) - pt).norm();
					double dist2 = (m_Mesh->point(vh2) - pt).norm();
					if(dist1 < radius && dist2 < radius)
						b_edge[i] = true;
				}
				else
				{
	
					b_edge[i] = true;
				}
			}
			else
			{
				if (constraint[vh1.idx()] && constraint[vh2.idx()])
					b_edge[i] = true;
			}
		}
		// end update edge list

		for(int i = 0; i < n; ++i)
		{
			if(m_Mesh->calc_edge_length(m_Mesh->edge_handle(i)) > high && b_edge[i])
			{
				MeshType::HalfedgeHandle heh = m_Mesh->halfedge_handle(m_Mesh->edge_handle(i), 0);
				MeshType::VertexHandle vh1 = m_Mesh->to_vertex_handle(heh);
				MeshType::VertexHandle vh2 = m_Mesh->from_vertex_handle(heh);
				
				PointType add_pt = ( m_Mesh->point(vh1) + m_Mesh->point(vh2) )*0.5;
				MeshType::VertexHandle Vh = m_Mesh->add_vertex(add_pt);
				m_Mesh->split(m_Mesh->edge_handle(i), Vh);
				finished = false;
			}
		}
	}
}
void hccl::RemeshFilter::collapse_Short_Edges()
{
	bool finished = false;
	while(!finished)
	{
		finished = true;
		for(int i = 0; i < m_Mesh->n_edges(); ++i)
		{
			if(b_edge[i] == false)
				continue;
			if(m_Mesh->calc_edge_length(m_Mesh->edge_handle(i)) < low)
			{
				MeshType::HalfedgeHandle heh = m_Mesh->halfedge_handle(m_Mesh->edge_handle(i), 0);
				MeshType::VertexHandle vh1 = m_Mesh->from_vertex_handle(heh);
				MeshType::VertexHandle vh2 = m_Mesh->to_vertex_handle(heh);

				if(m_Mesh->is_boundary(vh1) || m_Mesh->is_boundary(vh2))
					continue;

				bool collapse_ok = true;
				for(MeshType::VertexVertexIter vv_it = m_Mesh->vv_iter(vh1); vv_it; ++vv_it)
				{
					PointType vec = m_Mesh->point(vv_it.handle()) - m_Mesh->point(vh2);
					if(vec.length() > high)
					{
						collapse_ok = false;
						b_edge[i] = false;
						break;
					}
				}
				if(m_Mesh->is_collapse_ok(heh) && collapse_ok)
				{
					int idx1 = m_Mesh->edge_handle(heh).idx();
					int idx2 = m_Mesh->edge_handle(m_Mesh->prev_halfedge_handle(heh)).idx();
					int idx3 = m_Mesh->edge_handle(m_Mesh->next_halfedge_handle(m_Mesh->opposite_halfedge_handle(heh))).idx();
					b_edge[idx1] = false;
					b_edge[idx2] = false;
					b_edge[idx3] = false;
					m_Mesh->collapse(heh);
					finished = false;
				}
			}
		}
	}
	m_Mesh->garbage_collection();
}
void hccl::RemeshFilter::tangential_Relaxation()
{
	// update vertex list
	std::vector<bool> b_vertex(m_Mesh->n_vertices(), false);	
	for(int i = 0; i < m_Mesh->n_vertices(); ++i)
	{
		if(m_Mesh->is_boundary(m_Mesh->vertex_handle(i)))
			continue;

		if (constraint.empty())
		{
			if (radius>0)
			{
				double dist = (m_Mesh->point(m_Mesh->vertex_handle(i)) - pt).norm();
				if(dist < radius)
					b_vertex[i] = true;
			} 
			else
			{
				b_vertex[i] = true;
			}
		} 
		else
		{
			b_vertex[i] = constraint[i];
		}
	}

	// end update edge list
	std::vector<PointType> pts;
	pts.resize(m_Mesh->n_vertices());
	m_Mesh->update_normals();
	for(int i = 0; i < pts.size(); ++i)
	{
		if(m_Mesh->is_boundary(m_Mesh->vertex_handle(i)) || b_vertex[i] == false)
		{
			pts[i] = m_Mesh->point(m_Mesh->vertex_handle(i));
			continue;
		}
		PointType p = m_Mesh->point(m_Mesh->vertex_handle(i));
		MeshType::Normal n = m_Mesh->normal(m_Mesh->vertex_handle(i));
		PointType q = PointType(0,0,0);
		int cnt = 0;
		for(MeshType::VertexVertexIter vv_it = m_Mesh->vv_iter(m_Mesh->vertex_handle(i)); vv_it; ++vv_it, cnt++)
			q += m_Mesh->point(vv_it.handle());
		q /= cnt;

		pts[i] = q + n*dot(n, p-q);
	}

	for(int i = 0; i < pts.size(); ++i)
		m_Mesh->set_point(m_Mesh->vertex_handle(i), pts[i]);

}

