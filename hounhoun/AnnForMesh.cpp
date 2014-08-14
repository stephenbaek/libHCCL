#include "AnnForMesh.h"
#include "StaticFunctions.h"

using namespace hccl;
using namespace std;

AnnForMesh::AnnForMesh( MeshType* mesh /*= NULL*/ )
	 : m_Mesh(mesh)
	 , allPoints(0)
	 , ann_allPoints(allPoints)
{	
	updateAnn();
}

void AnnForMesh::setMesh( MeshType* mesh )
{
	m_Mesh = mesh;
	updateAnn();
}

void AnnForMesh::updateAnn()
{
	if (!m_Mesh)
		return;
	if(m_Mesh->n_vertices() == 0)
		return;

	allPoints.resize(m_Mesh->n_vertices());
	memcpy(allPoints[0].getPtr(), m_Mesh->points(), sizeof(PointType)*m_Mesh->n_vertices());
	ann_allPoints.buildIndex();
}

int AnnForMesh::findNearestIndx( const PointType& pt )
{
	if (allPoints.empty()) updateAnn();

	AnnPoints::result_knn nearst = ann_allPoints.findNearest(pt);

	if (nearst.size()!=1)
	{
		cout<<"nearest search error!"<<endl;
	}

	return nearst.indices[0];
}
std::vector<int> AnnForMesh::findNeighborsIndx( const PointType& pt, int num_pt )
{
	if (allPoints.empty()) updateAnn();

	vector<int> result;

	AnnPoints::result_knn neighbors = ann_allPoints.findNeighbors(pt, num_pt);
	result.reserve(neighbors.size());

	for (neighbors.indices_it = neighbors.indices.begin();
		neighbors.indices_it != neighbors.indices.end();
		neighbors.indices_it++)
	{
		result.push_back(*neighbors.indices_it);
	}
	return result;
}
std::vector<int> AnnForMesh::findWidthinIndx( const PointType& pt, ScalarType radius )
{
	if (allPoints.empty()) updateAnn();

	vector<int> result;
	AnnPoints::result_radius neighbors = ann_allPoints.findWithin(pt, radius);
	result.reserve(neighbors.size());

	for_each(neighbors.begin(), neighbors.end(), [this, &result](const std::pair<int,ScalarType>& pair)
	{
		result.push_back(pair.first);
	});

	return result;
}
AnnForMesh::PointType AnnForMesh::findNearest( const PointType& pt )
{
	if (allPoints.empty()) updateAnn();

	AnnPoints::result_knn nearst = ann_allPoints.findNearest(pt);

	if (nearst.size()!=1)
	{
		cout<<"nearest search error!"<<endl;
	}

	return allPoints[nearst.indices[0]].getPoint();
}
std::vector<AnnForMesh::PointType> AnnForMesh::findNeighbors( const PointType& pt, int num_pt )
{
	if (allPoints.empty()) updateAnn();

	vector<PointType> result;

	AnnPoints::result_knn neighbors = ann_allPoints.findNeighbors(pt, num_pt);
	result.reserve(neighbors.size());

	for (neighbors.indices_it = neighbors.indices.begin();
		neighbors.indices_it != neighbors.indices.end();
		neighbors.indices_it++)
	{
		result.push_back(allPoints[*neighbors.indices_it].getPoint());
	}
	return result;
}
std::vector<AnnForMesh::PointType> AnnForMesh::findWidthin( const PointType& pt, ScalarType radius )
{
	if (allPoints.empty()) updateAnn();

	vector<PointType> result;
	AnnPoints::result_radius neighbors = ann_allPoints.findWithin(pt, radius);
	result.reserve(neighbors.size());

	for_each(neighbors.begin(), neighbors.end(), [this, &result](const std::pair<int,ScalarType>& pair)
	{
		result.push_back(allPoints[pair.first].getPoint());
	});

	return result;
}

int AnnForMesh::getProjectedPoint( const PointType& p, PointType& proj, ScalarType threshold /*= 0.5*/ )
{
	MeshType::VertexHandle vh = m_Mesh->vertex_handle(findNearestIndx(p));

	if (!vh.is_valid())
	{
// 		printf("invalid vertex handle!\n");
		return -1;
	}

	ScalarType dist = (p-m_Mesh->point(vh)).norm();
	if (dist>threshold)
	{
// 		printf("exceed threshold!!(distance : %3f)\n", dist);
		proj = p;
		return -1;
	}

	ScalarType mink = 99999999;
	int fid = -1;
	for (auto cvvit = m_Mesh->cvv_begin(vh); cvvit != m_Mesh->cvv_end(vh); ++cvvit)
	{
		for (auto cvfit = m_Mesh->cvf_begin(cvvit); cvfit != m_Mesh->cvf_end(cvvit); ++cvfit)
		{
			MeshType::ConstFaceVertexIter cfvit = m_Mesh->cfv_iter(cvfit);
			PointType v0 = m_Mesh->point(cfvit);
			PointType v1 = m_Mesh->point(++cfvit);
			PointType v2 = m_Mesh->point(++cfvit);		
			PointType fn = m_Mesh->normal(cvfit);

			ScalarType k = dot(v0-p, fn);
			PointType temp = p + k*fn;

			ScalarType u,v;

			if(isInsideTriangle(temp, v0, v1, v2, u, v) && k<mink)
			{
				proj = temp;
				fid = cvfit.handle().idx();
				mink = k;
			}
		}
	}

	if (fid == -1)
	{
		proj = m_Mesh->point(vh);
		fid = m_Mesh->vf_begin(vh).handle().idx();
	}

	return fid;

}

int hccl::AnnForMesh::getProjectedPoint( const PointType& p, const PointType& dir, PointType& proj, ScalarType threshold /*= 0.5*/, bool bidirection /*= true*/ )
{
	ScalarType mink = 99999999;
	int fid = -1;
	proj = p;


	std::vector<bool> b_facet(m_Mesh->n_faces(), false);

	std::vector<int> neighbors = this->findWidthinIndx(p, threshold);



// 	// #pragma omp parallel for shared(fid, mink)
// 	for (int i = 0; i<m_Mesh->n_faces(); ++i)
// 	{
// 		MeshType::FaceHandle fh = m_Mesh->face_handle(i);
// 
// 		Point tem;
// 		ScalarType k = m_Mesh->getProjectedPoint(p, dir, fh, mink, tem, bidirection);
// 
// 		if (-1 != k && abs(k) < mink)
// 		{
// 			mink = abs(k);
// 			proj = tem;
// 			fid = i;
// 		}
// 	}
	return fid;	

}
