#include "LeastSquareMesh2.h"

namespace hccl{

LeastSquareMesh2::LeastSquareMesh2()
{
	system.clear();
}

LeastSquareMesh2::~LeastSquareMesh2()
{

}

void LeastSquareMesh2::set_geometry(TriMesh* _mesh){
    mesh = _mesh;
    opLaplacian(*mesh, M, Lc);
    system.clear();	
    system.set_matrix(ssmul(M, Lc));    
}


void LeastSquareMesh2::solve( const std::vector<int>& src, const std::vector<TriMesh::Point>& con )
{
	DenseMatrix c(src.size(), 3);
	c.set_null();
	for(int i = 0; i < src.size(); ++i)
	{
		c.set(i, 0, con[i][0]);
		c.set(i, 1, con[i][1]);
		c.set(i, 2, con[i][2]);
	}
	system.set_constraints(src, c);

	DenseMatrix b(mesh->n_vertices(), 3), x;
	b.set_null();

	system.set_matrix(b);	
	system.factor_linear_least_squares_soft_constraints();
	system.solve_linear_least_squares_soft_constraints(x);

	
	for (auto vit = mesh->vertices_begin(); vit != mesh->vertices_end(); ++vit)
	{
		int indx = vit.handle().idx();
		mesh->set_point(vit, TriMesh::Point(x.get(indx,0), x.get(indx,1), x.get(indx,2)));
	}
}

}   // namespace hccl