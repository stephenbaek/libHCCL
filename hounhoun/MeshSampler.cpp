#include "MeshSampler.h"
#include <QTime>

using namespace hccl;
void MeshSampler::sampleRandom( int nSamples, std::vector<std::pair<PointType, int>>& samples ) const
{
	if(m_Mesh->n_vertices() <= 0)
		return;

	//////////////////////////////////////////////////////////////////////////
	// Random Sampling Method
	samples.resize(nSamples);

	// Vertices List
	std::vector<int> vtxIdxList;
	std::vector<int> vtxSelect(nSamples);

	vtxIdxList.resize(m_Mesh->n_vertices());
	std::iota(vtxIdxList.begin(), vtxIdxList.end(), 0); // Generate a vertex index list (Non-zero base indexing)

	// Random Shuffle
	std::random_shuffle(vtxIdxList.begin(), vtxIdxList.end()); // Randomly shuffle
	std::copy(vtxIdxList.begin(), vtxIdxList.begin()+nSamples, vtxSelect.begin());
	std::sort(vtxSelect.begin(), vtxSelect.end());

	TriMesh::ConstVertexIter vit(m_Mesh->vertices_begin()), vEnd(m_Mesh->vertices_end());
	TriMesh::Point temp;
	int cnt = 0;
	int idxCnt = 0;
	for(; vit != vEnd; ++vit, ++cnt)
	{		
		if(cnt==vtxSelect[idxCnt])
		{
			TriMesh::Point pt = m_Mesh->point(vit);
			samples[idxCnt].first = PointType(pt[0], pt[1], pt[2]);
			samples[idxCnt].second = cnt;

			idxCnt++;
			if(idxCnt==nSamples)
				break;
		}				
	}
}
void MeshSampler::sampleUniform( int nSamples, std::vector<PointType>& samples, std::vector<int>& indx ) const
{
	int NV = m_Mesh->n_vertices();
	if(NV <= 0 || nSamples >= NV)
		return;	

	TriMesh::ConstFaceIter fIt(m_Mesh->faces_begin()), fEnd(m_Mesh->faces_end());
	TriMesh::ConstFaceVertexIter fvIt;
	TriMesh::Point pt1, pt2, pt3;		
	double area = 0;
	double mean_dist = 0;
	for (; fIt!=fEnd; ++fIt)
	{
		fvIt = m_Mesh->cfv_iter(fIt.handle());

		pt1 = m_Mesh->point(fvIt.handle());
		pt2 = m_Mesh->point((++fvIt).handle());
		pt3 = m_Mesh->point((++fvIt).handle());

		TriMesh::Point t1 = pt1-pt2;
		TriMesh::Point t2 = pt1-pt3;
		area += 0.5*(t1%t2).norm();
		mean_dist += 0.5*(t1.norm()+t2.norm());			
	}	

	mean_dist /= m_Mesh->n_vertices();


	std::vector<std::pair<PointType, int>> samplse_pair;
	std::vector<std::pair<PointType, int>> D_nodes(nSamples);

	// 원하는 샘플 수 보다 10배로 많은 샘플을 뽑아낸다.
	sampleRandom(10*nSamples > m_Mesh->n_vertices() ? m_Mesh->n_vertices() : 10*nSamples, samplse_pair);

	if(samplse_pair.size() == 0)
		return;

	int n_nodes = samplse_pair.size();
	double dist = sqrt(area/(double)(sqrt(3.0)*nSamples));			

	PointType temp;
	auto f1 = [this, &temp, &dist](std::pair<PointType, int> v)->bool
	{
		if((temp - v.first).length() < dist)
			return true;
		else
			return false;
	};

	int cnt = 0;
	int r;
	int num_iter = 0;

	double margin = 0.1;
	double m_plus = 1+margin;
	double m_minus = 1-margin;

	while(1)
	{
		n_nodes = samplse_pair.size();
		r = rand()%n_nodes;

		temp = D_nodes[cnt].first = samplse_pair[r].first;
		D_nodes[cnt].second = samplse_pair[r].second;
		++cnt;

		samplse_pair.erase(std::remove_if(samplse_pair.begin(), samplse_pair.end(), f1), samplse_pair.end());

		if(cnt < m_minus*nSamples && samplse_pair.size() == 0)
		{			
			sampleRandom(10*nSamples > m_Mesh->n_vertices() ? m_Mesh->n_vertices() : 10*nSamples, samplse_pair);

			cnt = 0;
			dist *= m_minus;
			num_iter++;
		}
		else if(cnt == nSamples && samplse_pair.size() != 0)
		{			
			sampleRandom(10*nSamples > m_Mesh->n_vertices() ? m_Mesh->n_vertices() : 10*nSamples, samplse_pair);

			cnt = 0;
			dist *= m_plus;
			num_iter++;
		}
		else if(cnt >= m_minus*nSamples && samplse_pair.size() == 0)
			break;
	}

// 	auto f2 = [](std::pair<PointType, int> v)->bool
// 	{
// 		if(v.first.length()==0)
// 			return true;
// 		else
// 			return false;
// 	};
// 
// 	printf("d-node size : %d\n", D_nodes.size());
// 	D_nodes.erase(std::remove_if(D_nodes.begin(), D_nodes.end(), f2), D_nodes.end());
// 	printf("d-node size : %d\n", D_nodes.size());

	D_nodes.resize(cnt);
	samples.resize(D_nodes.size());
	indx.resize(D_nodes.size());
	for (int i = 0; i<D_nodes.size(); ++i)
	{
		samples[i] = D_nodes[i].first;
		indx[i] = D_nodes[i].second;
	}
}

void MeshSampler::sampleUniformBySamplingDistance( const ScalarType samplingD, std::vector<PointType>& samples, std::vector<int>& indx ) const
{
	if (samplingD <= 0)
	{
		printf("Wrong sampling distance!\n");
		return;
	}

	// 	QTime t;
	// 	t.start();	

	std::vector<std::pair<PointType, int>> samplse_pair(m_Mesh->n_vertices());
	for (int i = 0; i<samplse_pair.size(); ++i)
	{
		samplse_pair[i].first = m_Mesh->point(m_Mesh->vertex_handle(i));
		samplse_pair[i].second = i;
	}
	std::vector<std::pair<PointType, int>> D_nodes;
	D_nodes.reserve(samplse_pair.size());

	PointType temp;
	auto f1 = [this, &temp, &samplingD](std::pair<PointType, int> v)->bool
	{
		if((temp - v.first).length() < samplingD)
			return true;
		else
			return false;
	};

	while(!samplse_pair.empty())
	{
		int n_nodes = samplse_pair.size();
		int curIndx = samplse_pair.size()-1;
		std::pair<PointType, int> temPair;
		temPair.first = samplse_pair[curIndx].first;
		temPair.second = samplse_pair[curIndx].second;
		
		D_nodes.push_back(temPair);
		samplse_pair.pop_back();

		samplse_pair.erase(std::remove_if(samplse_pair.begin(), samplse_pair.end(), f1), samplse_pair.end());
	}
	
	samples.resize(D_nodes.size());
	indx.resize(D_nodes.size());
	for (int i = 0; i<D_nodes.size(); ++i)
	{
		samples[i] = D_nodes[i].first;
		indx[i] = D_nodes[i].second;
	}
}

