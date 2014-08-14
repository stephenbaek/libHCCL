#include "CurvatureFilter.h"
#include <opencv2/opencv.hpp>

using namespace hccl;
void CurvatureFilter::updateCurvature()
{
	int tic = clock();

	k1.clear();
	k2.clear();
	k1.resize(m_Mesh->n_vertices(), 0);
	k2.resize(m_Mesh->n_vertices(), 0);
	k1_dir.clear();
	k1_dir.resize(m_Mesh->n_vertices(), PointType(0,0,0));
	k2_dir.clear();
	k2_dir.resize(m_Mesh->n_vertices(), PointType(0,0,0));

	for(int i = 0; i < m_Mesh->n_vertices(); ++i)
	{
		PointType normal_v;
		m_Mesh->calc_vertex_normal_correct(m_Mesh->vertex_handle(i), normal_v);
		ScalarType norm = normal_v.length();
		if (norm != 0.0) normal_v *= (1.0/norm);
		m_Mesh->set_normal(m_Mesh->vertex_handle(i), normal_v);
	}

	std::vector<PointType> vertex_u;
	std::vector<PointType> vertex_v;

	vertex_u.resize(m_Mesh->n_vertices());
	vertex_v.resize(m_Mesh->n_vertices());

	for(int i = 0; i < m_Mesh->n_vertices(); ++i)
	{
		MeshType::ConstVertexVertexIter vvit = m_Mesh->cvv_begin(m_Mesh->vertex_handle(i));
		vertex_u[i] = m_Mesh->point(m_Mesh->vertex_handle(vvit.handle().idx())) - m_Mesh->point(m_Mesh->vertex_handle(i));
		vertex_v[i] = cross(m_Mesh->normal(m_Mesh->vertex_handle(i)), vertex_u[i]);
		vertex_u[i] = cross(vertex_v[i], m_Mesh->normal(m_Mesh->vertex_handle(i)));
		vertex_u[i] = vertex_u[i].normalize();
		vertex_v[i] = vertex_v[i].normalize();
	}

	std::vector<ScalarType> Ep;
	std::vector<ScalarType> Fp;
	std::vector<ScalarType> Gp;
	Ep.resize(m_Mesh->n_vertices(), 0);
	Fp.resize(m_Mesh->n_vertices(), 0);
	Gp.resize(m_Mesh->n_vertices(), 0);

	for(int i = 0; i < m_Mesh->n_faces(); ++i)
	{
		MeshType::ConstFaceVertexIter fvit = m_Mesh->cfv_begin(m_Mesh->face_handle(i));
		int idx[3];
		idx[0] = fvit.handle().idx();
		idx[1] = (++fvit).handle().idx();
		idx[2] = (++fvit).handle().idx();
		PointType u,v,w;
		PointType e1,e2,e3;
		e1 = m_Mesh->point(m_Mesh->vertex_handle(idx[1])) - m_Mesh->point(m_Mesh->vertex_handle(idx[0]));
		e2 = m_Mesh->point(m_Mesh->vertex_handle(idx[2])) - m_Mesh->point(m_Mesh->vertex_handle(idx[1]));
		e3 = m_Mesh->point(m_Mesh->vertex_handle(idx[0])) - m_Mesh->point(m_Mesh->vertex_handle(idx[2]));
		u = e1.normalize();
		v = e2.normalize();
		w = cross(u,v).normalize();
		v = cross(w,u).normalize();

		ScalarType a[6];
		ScalarType b[6];
		a[0] = dot(e1, u);
		a[1] = dot(e1, v);
		a[2] = dot(e2, u);
		a[3] = dot(e2, v);
		a[4] = dot(e3, u);
		a[5] = dot(e3, v);

		b[0] = dot(m_Mesh->normal(m_Mesh->vertex_handle(idx[1]))-m_Mesh->normal(m_Mesh->vertex_handle(idx[0])), u);
		b[1] = dot(m_Mesh->normal(m_Mesh->vertex_handle(idx[1]))-m_Mesh->normal(m_Mesh->vertex_handle(idx[0])), v);
		b[2] = dot(m_Mesh->normal(m_Mesh->vertex_handle(idx[2]))-m_Mesh->normal(m_Mesh->vertex_handle(idx[1])), u);
		b[3] = dot(m_Mesh->normal(m_Mesh->vertex_handle(idx[2]))-m_Mesh->normal(m_Mesh->vertex_handle(idx[1])), v);
		b[4] = dot(m_Mesh->normal(m_Mesh->vertex_handle(idx[0]))-m_Mesh->normal(m_Mesh->vertex_handle(idx[2])), u);
		b[5] = dot(m_Mesh->normal(m_Mesh->vertex_handle(idx[0]))-m_Mesh->normal(m_Mesh->vertex_handle(idx[2])), v);

		cv::Mat A = cv::Mat::zeros(6,3,CV_32FC1);
		cv::Mat B = cv::Mat::zeros(6,1,CV_32FC1);
		if (sizeof(ScalarType) == sizeof(double))
		{
			A = cv::Mat::zeros(6,3,CV_64FC1);
			B = cv::Mat::zeros(6,1,CV_64FC1);
		}

		for(int j = 0; j < 3; ++j)
		{
			A.at<ScalarType>(2*j,0) = a[2*j];
			A.at<ScalarType>(2*j,1) = a[2*j+1];
			A.at<ScalarType>(2*j,2) = 0;

			A.at<ScalarType>(2*j+1,0) = 0;
			A.at<ScalarType>(2*j+1,1) = a[2*j];
			A.at<ScalarType>(2*j+1,2) = a[2*j+1];
		}
		for(int j = 0; j < 6; ++j)
			B.at<ScalarType>(j,0) = b[j];

		cv::Mat SFT = (A.t()*A).inv()*A.t()*B; //second fundamental tensor

		ScalarType area_tri = m_Mesh->calc_sector_area(m_Mesh->halfedge_handle(m_Mesh->face_handle(i)));

		for(int i = 0; i < 3; ++i)
		{
			PointType axis = cross(w, m_Mesh->normal(m_Mesh->vertex_handle(idx[i]))).normalize();
			ScalarType theta = acos(dot(w, m_Mesh->normal(m_Mesh->vertex_handle(idx[i]))));

			PointType new_normal_f[3];
			new_normal_f[0] = rotateAxis(u, axis, theta);
			new_normal_f[1] = rotateAxis(v, axis, theta);
			new_normal_f[2] = rotateAxis(w, axis, theta);

			ScalarType u_p[2], v_p[2];
			u_p[0] = dot(vertex_u[idx[i]], new_normal_f[0]);
			u_p[1] = dot(vertex_u[idx[i]], new_normal_f[1]);

			v_p[0] = dot(vertex_v[idx[i]], new_normal_f[0]);
			v_p[1] = dot(vertex_v[idx[i]], new_normal_f[1]);

			Ep[idx[i]] += area_tri*(u_p[0]*SFT.at<ScalarType>(0,0)*u_p[0] + u_p[1]*SFT.at<ScalarType>(1,0)*u_p[0] + u_p[0]*SFT.at<ScalarType>(1,0)*u_p[1] + u_p[1]*SFT.at<ScalarType>(2,0)*u_p[1]);
			Fp[idx[i]] += area_tri*(u_p[0]*SFT.at<ScalarType>(0,0)*v_p[0] + u_p[1]*SFT.at<ScalarType>(1,0)*v_p[0] + u_p[0]*SFT.at<ScalarType>(1,0)*v_p[1] + u_p[1]*SFT.at<ScalarType>(2,0)*v_p[1]);
			Gp[idx[i]] += area_tri*(v_p[0]*SFT.at<ScalarType>(0,0)*v_p[0] + v_p[1]*SFT.at<ScalarType>(1,0)*v_p[0] + v_p[0]*SFT.at<ScalarType>(1,0)*v_p[1] + v_p[1]*SFT.at<ScalarType>(2,0)*v_p[1]);
		}	
	}

	for(int i = 0; i < m_Mesh->n_vertices(); ++i)
	{
		MeshType::ConstVertexFaceIter vfit = m_Mesh->cvf_begin(m_Mesh->vertex_handle(i));
		MeshType::ConstVertexFaceIter vfit_end = m_Mesh->cvf_end(m_Mesh->vertex_handle(i));
		ScalarType total_area = 0;
		for(;vfit != vfit_end; ++vfit)
		{
			total_area += m_Mesh->calc_sector_area(m_Mesh->halfedge_handle(vfit));
		}
		Ep[i] /= total_area;
		Fp[i] /= total_area;
		Gp[i] /= total_area;

		cv::Mat SFT_v = cv::Mat::zeros(2,2, CV_32F);
		if (sizeof(ScalarType) == sizeof(double))
		{
			SFT_v = cv::Mat::zeros(2,2,CV_64FC1);
		}
		SFT_v.at<ScalarType>(0,0) = Ep[i];
		SFT_v.at<ScalarType>(1,0) = Fp[i];
		SFT_v.at<ScalarType>(0,1) = Fp[i];
		SFT_v.at<ScalarType>(1,1) = Gp[i];

		cv::Mat eigenvalue, eigenvector;

		cv::eigen(SFT_v, eigenvalue, eigenvector);
		int max_idx, min_idx;

		max_idx = eigenvalue.at<ScalarType>(0) > eigenvalue.at<ScalarType>(1) ? 0 : 1;
		min_idx = eigenvalue.at<ScalarType>(0) > eigenvalue.at<ScalarType>(1) ? 1 : 0;
		k1[i] = eigenvalue.at<ScalarType>(max_idx);
		k2[i] = eigenvalue.at<ScalarType>(min_idx);

		k1_dir[i] = eigenvector.at<ScalarType>(max_idx, 0) * vertex_u[i] + eigenvector.at<ScalarType>(max_idx, 1) * vertex_v[i];
		k2_dir[i] = eigenvector.at<ScalarType>(min_idx, 0) * vertex_u[i] + eigenvector.at<ScalarType>(min_idx, 1) * vertex_v[i];
		k1_dir[i] = k1_dir[i].normalize();
		k2_dir[i] = k2_dir[i].normalize();
	}

	printf("update curvature time : %lfsec\n", (clock() - tic) / (ScalarType)CLOCKS_PER_SEC);

}

BaseFilter::PointType CurvatureFilter::rotateAxis( PointType vec, PointType axis, ScalarType theta )
{
	if( axis[0] == 0 && axis[1] == 0 && axis[2] == 0 )
		return PointType(1,0,0);
	if( theta == 0 )
		return PointType(1,0,0);

	PointType Vp, Vn, Vn_, W;

	axis.normalize();
	ScalarType dot_ = dot(vec, axis);
	if( dot_ == 0 )
	{
		Vp = axis;
		Vn = vec;
		W = cross(Vp, Vn);
		W.normalize();
		W *= Vn.norm();

		Vn_ = Vn*cos(theta) + W*sin(theta);

		vec = Vn_;
	}
	else if( dot_ == vec.norm()*axis.norm() )
		return vec;
	else
	{
		Vp = axis*dot_;
		Vn = vec - Vp;
		W = cross(Vp, Vn);
		W.normalize();
		W *= Vn.norm();
		if( dot_ < 0 )
			W *= -1;

		Vn_ = Vn*cos(theta) + W*sin(theta);

		vec = Vp + Vn_;
	}
	return vec;

}

void CurvatureFilter::setMesh( MeshType* mesh )
{
	clear();
	m_Mesh = mesh;
}

void CurvatureFilter::clear()
{
	k1.clear();
	k1_dir.clear();
	k2.clear();
	k2_dir.clear();
}

