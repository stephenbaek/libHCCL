#include "sparsematrix.h"

namespace hccl{

//////////////////////////////////////////////////////////////////////////
// SparseMatrix
// - A sparse matrix container
// - by Stephen Baek (3d@cad.snu.ac.kr)
//////////////////////////////////////////////////////////////////////////

SparseMatrix::SparseMatrix():rows(0),cols(0){
}

SparseMatrix::SparseMatrix(int _rows, int _cols):rows(_rows),cols(_cols){
}

SparseMatrix::~SparseMatrix(){
}

void SparseMatrix::set_size(int _rows, int _cols){
    rows = _rows;
    cols = _cols;
}

int SparseMatrix::n_rows() const{
    return rows;
}
int SparseMatrix::n_cols() const{
    return cols;
}

void SparseMatrix::clear(){
    i.clear(); j.clear(); x.clear();
}

int SparseMatrix::add_entry(int _i, int _j, double _x){
    if(_i >= rows || _j >= cols || _i < 0 || _j < 0)
        return -1;
    i.push_back(_i); j.push_back(_j); x.push_back(_x);
    return i.size()-1;
}

size_t SparseMatrix::n_nonzeros() const{
    return i.size();
}

int* SparseMatrix::get_i_ptr(){
    return &(i[0]);
}

int* SparseMatrix::get_j_ptr(){
    return &(j[0]);
}

double* SparseMatrix::get_x_ptr(){
    return &(x[0]);
}

void SparseMatrix::set_null(){
    clear();
}

void SparseMatrix::set_identity(int n){
    rows = cols = n;
    i.resize(n);
    j.resize(n);
    x.resize(n, 0);
    for(int k = 0; k < n; k++){
        i[k] = k;
        j[k] = k;
        x[k] = 1.0;
    }
}

void SparseMatrix::print(const char* name){
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Matrix " << name << std::endl;
    std::cout << "Rows: " << rows << std::endl;
    std::cout << "Cols: " << cols << std::endl;
    std::cout << "Nonzeros: " << x.size() << std::endl << std::endl;
    for(int k = 0; k < x.size(); k++){
        std::cout << "   (" << i[k] << ", " << j[k] << "):\t" << x[k] << std::endl;
    }
    std::cout << "--------------------------------" << std::endl;
}

void SparseMatrix::print_info(const char* name){
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Matrix " << name << std::endl;
    std::cout << "Rows: " << rows << std::endl;
    std::cout << "Cols: " << cols << std::endl;
    std::cout << "Nonzeros: " << x.size() << std::endl;
    std::cout << "--------------------------------" << std::endl;
}





//////////////////////////////////////////////////////////////////////////
// DenseMatrix
// - A dense matrix container
// - by Stephen Baek (3d@cad.snu.ac.kr)
//////////////////////////////////////////////////////////////////////////

DenseMatrix::DenseMatrix():rows(0),cols(0){
}

DenseMatrix::DenseMatrix(int _rows, int _cols){
    set_size(_rows, _cols);
}

DenseMatrix::~DenseMatrix(){
}

void DenseMatrix::set_size(int _rows, int _cols){
    rows = _rows;
    cols = _cols;
    x.resize(cols);
    for(int i = 0; i < cols; i++){
        x[i].resize(rows, 0);
    }
}

int DenseMatrix::n_rows() const{
    return rows;
}
int DenseMatrix::n_cols() const{
    return cols;
}

void DenseMatrix::clear(){
    x.clear();
    rows = 0;
    cols = 0;
}

int DenseMatrix::set(int _i, int _j, double _x){
    if(_i >= rows || _j >= cols || _i < 0 || _j < 0)
        return -1;
    x[_j][_i] = _x;
    
    return 0;
}

void DenseMatrix::set(int _rows, int _cols, int _step_size, double* _x){
    clear();
    set_size(_rows, _cols);
    for(int i = 0; i < _cols; i++){
        memcpy(&(x[i][0]), &(_x[i*_step_size]), sizeof(double)*_rows);
    }
}

void DenseMatrix::set(int _rows, int _cols, double** _x){
    clear();
    set_size(_rows, _cols);
    for(int i = 0; i < _cols; i++){
        memcpy(&(x[i][0]), &(x[i*_cols][0]), sizeof(double)*_rows);
    }
}

void DenseMatrix::set(std::vector< std::vector<double> >& _x){
    clear();
    x = _x;
}

double DenseMatrix::get(int _i, int _j) const{
    return x[_j][_i];
}

double* DenseMatrix::get_col_ptr(int i){
    return &(x[i][0]);
}

std::vector< std::vector<double> >& DenseMatrix::get_data(){
    return x;
}

const std::vector< std::vector<double> >& DenseMatrix::get_data() const
{
	return x;
}

void DenseMatrix::set_null(){
    int _rows = rows;
    int _cols = cols;
    clear();
    set_size(_rows, _cols);
}

void DenseMatrix::set_identity(int n){
    rows = cols = n;
    x.resize(n);
    for(int i = 0; i < n; i++){
        x[i].clear();
        x[i].resize(n,0);
        x[i][i] = 1.0;
    }
}

void DenseMatrix::print(const char* name){
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Matrix " << name << std::endl;
    std::cout << "Rows: " << rows << std::endl;
    std::cout << "Cols: " << cols << std::endl << std::endl;
    for(int i = 0; i < cols; i++){
        std::cout << "Column " << i << ":" << std::endl;
        for(int j = 0; j < rows; j++){
            std::cout << "   Row " << j << ": " << x[i][j] << std::endl;
        }
    }
    std::cout << "--------------------------------" << std::endl;
}

void DenseMatrix::print_info(const char* name){
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Matrix " << name << std::endl;
    std::cout << "Rows: " << rows << std::endl;
    std::cout << "Cols: " << cols << std::endl;
    std::cout << "--------------------------------" << std::endl;
}





SparseMatrix ssadd(SparseMatrix& A, SparseMatrix& B, double alpha/* = 1.0*/, double beta/* = 1.0*/){
    CholmodSparseMatrix AA(A), BB(B);
    double aa[2] = {alpha, alpha};
    double bb[2] = {beta, beta};
    cholmod_start(&chol);
    CholmodSparseMatrix CC(cholmod_add(AA.get_chol(), BB.get_chol(), aa, bb, 1, 1, &chol));
    cholmod_finish(&chol);
    
    return CC.get();
}
SparseMatrix ssmul( SparseMatrix& A, SparseMatrix& B )
{
	CholmodSparseMatrix AA(A), BB(B);
	cholmod_start(&chol);
	CholmodSparseMatrix CC(cholmod_ssmult(AA.get_chol(), BB.get_chol(), 0, 1, 1, &chol));
	cholmod_finish(&chol);
	return CC.get();	
}
SparseMatrix ssvercat( SparseMatrix& A, SparseMatrix& B )
{
	CholmodSparseMatrix AA(A), BB(B);
	cholmod_start(&chol);
	CholmodSparseMatrix CC(cholmod_vertcat(AA.get_chol(), BB.get_chol(), 1, &chol));
	cholmod_finish(&chol);
	return CC.get();
}


SparseSolver::SparseSolver(){

}

SparseSolver::SparseSolver(SparseMatrix& _A, DenseMatrix& _b){
    set_matrices(_A, _b);
}

SparseSolver::SparseSolver(SparseMatrix& _A, DenseMatrix& _b, std::vector<int>& _constraint_idx, DenseMatrix& _constraints){
    set_matrices(_A, _b);
    set_constraints(_constraint_idx, _constraints);
}

SparseSolver::~SparseSolver(){
    clear();
}

void SparseSolver::clear(){
    A.clear();
    b.clear();
    con_id.clear();
    con.clear();

    F.clear();
    F_gen.clear();
    LHS.clear();
    RHS.clear();
}

// TODO: validity test
bool SparseSolver::is_valid(){
    if(A.n_rows() != b.n_rows()){
        std::cerr << "Matrix dimension mismatch!" << std::endl;
        return false;
    }
    return true;
}

void SparseSolver::set_matrices(SparseMatrix& _A, DenseMatrix& _b){
    A.set(_A);
    b.set(_b);
    //A.set(_A.n_rows(), _A.n_cols(), _A.n_nonzeros(), _A.get_i_ptr(), _A.get_j_ptr(), _A.get_x_ptr());
    //b.set(_b.get_data());
}

void SparseSolver::set_matrix(SparseMatrix& _A){
    A.set(_A);
}

void SparseSolver::set_matrix(DenseMatrix& _b){
    b.set(_b);
}

void SparseSolver::set_constraints(const std::vector<int>& _constraint_idx, const DenseMatrix& _constraints){
    con_id = _constraint_idx;
    con.set(_constraints.get_data());
}

void SparseSolver::factor_sym(){
    F.clear();

    cholmod_start(&chol);
    F.set(cholmod_analyze(A.get_chol(), &chol));
    cholmod_factorize(A.get_chol(), F.get(), &chol);
    cholmod_finish(&chol);
}

void SparseSolver::factor_gen(){
    F_gen.clear();

    int n = A.n_rows();
    UF_long* Ap = (UF_long*) A.get_chol()->p;
    UF_long* Ai = (UF_long*) A.get_chol()->i;
    double*  Ax =  (double*) A.get_chol()->x;
    umfpack_dl_symbolic( n, n, Ap, Ai, Ax, &(F_gen.symbolic), NULL, NULL );
    umfpack_dl_numeric( Ap, Ai, Ax, F_gen.symbolic, &(F_gen.numeric), NULL, NULL );
}

void SparseSolver::factor_linear_least_squares(){
    if(!is_valid()){
        // TODO: Exception handling
    }
    
    F.clear();
    LHS.clear();
    RHS.clear();
    
	cholmod_start(&chol);
    
	cholmod_sparse* AT = cholmod_transpose(A.get_chol(), 1, &chol);
	cholmod_sparse* ATA = cholmod_aat(AT, NULL, -1, 1, &chol);
    LHS.set(ATA);
	F.set(cholmod_analyze(LHS.get_chol(), &chol));
	cholmod_factorize(LHS.get_chol(), F.get(), &chol);
	double alpha[2] = {1, 1}, beta[2] = {0, 0};
	cholmod_dense* rhs = cholmod_zeros(AT->nrow, b.n_cols(), CHOLMOD_REAL, &chol);
	cholmod_sdmult(AT, 0, alpha, beta, b.get_chol(), rhs, &chol);
	RHS.set(rhs);
	
	cholmod_free_sparse(&AT, &chol);


    cholmod_finish(&chol);
}

void SparseSolver::factor_linear_least_squares_soft_constraints(){


	if(!is_valid()){
		// TODO: Exception handling
	}

	F.clear();
	LHS.clear();
	RHS.clear();

	cholmod_start(&chol);



	const double weight = 100000;
	cholmod_triplet* lowA_trip = cholmod_allocate_triplet(con_id.size(), A.n_cols(), con_id.size(), 0, CHOLMOD_REAL, &chol);
	for(int i = 0; i < con_id.size(); i++){
		((int*)lowA_trip->i)[i] = i;
		((int*)lowA_trip->j)[i] = con_id[i];
		((double*)lowA_trip->x)[i] = weight;
	}
	lowA_trip->nnz = con_id.size();
	cholmod_sparse* lowA = cholmod_triplet_to_sparse(lowA_trip, lowA_trip->nnz, &chol);
	cholmod_free_triplet(&lowA_trip, &chol);
	
	cholmod_sparse* newA = cholmod_vertcat(A.get_chol(), lowA, 1, &chol);
	cholmod_free_sparse(&lowA, &chol);
	

	cholmod_dense* tem = cholmod_zeros(b.n_rows()+con.n_rows(), b.n_cols(), CHOLMOD_REAL, &chol);
	CholmodDenseMatrix newb;
	newb.set(tem);
	for(int i = 0; i < b.n_rows(); ++i)
	{
		for (int j = 0; j<b.n_cols(); ++j)
			newb.get_ptr()[i+j*newb.get_step_size()] = b.get_ptr()[i+j*b.get_step_size()];
	}
	for(int i = 0; i < con.n_rows(); ++i)
	{
		for (int j = 0; j<b.n_cols(); ++j)
			newb.get_ptr()[con.n_rows()+i+j*newb.get_step_size()] = weight*con.get_ptr()[i+j*b.get_step_size()];
	}
	



	cholmod_sparse* AT = cholmod_transpose(newA, 1, &chol);
	cholmod_free_sparse(&newA, &chol);

	cholmod_sparse* ATA = cholmod_aat(AT, NULL, -1, 1, &chol);
	LHS.set(ATA);
	F.set(cholmod_analyze(LHS.get_chol(), &chol));
	cholmod_factorize(LHS.get_chol(), F.get(), &chol);
	double alpha[2] = {1, 1}, beta[2] = {0, 0};

	cholmod_dense* rhs = cholmod_zeros(AT->nrow, newb.n_cols(), CHOLMOD_REAL, &chol);
	cholmod_sdmult(AT, 0, alpha, beta, newb.get_chol(), rhs, &chol);
	RHS.set(rhs);

	cholmod_free_sparse(&AT, &chol);
	newb.clear();

	cholmod_finish(&chol);

}

void SparseSolver::factor_linear_least_squares_hard_constraints(){

}

void SparseSolver::solve_sym(DenseMatrix& x){
    x.clear();
    CholmodDenseMatrix ans;
    cholmod_start(&chol);
    ans.set(cholmod_solve(CHOLMOD_A, F.get(), b.get_chol(), &chol));
    x.set(ans.n_rows(), ans.n_cols(), ans.get_step_size(), ans.get_ptr());
    cholmod_finish(&chol);
}

void SparseSolver::solve_gen(DenseMatrix& x){
    x.clear();
    CholmodDenseMatrix ans;
    int n = A.n_rows();
    UF_long* Ap = (UF_long*) A.get_chol()->p;
    UF_long* Ai = (UF_long*) A.get_chol()->i;
    double*  Ax =  (double*) A.get_chol()->x;
    std::vector<double> temp(n);
    umfpack_dl_solve( UMFPACK_A, Ap, Ai, Ax, &temp[0], b.get_ptr(), F_gen.numeric, NULL, NULL );
    ans.set(n, 1, &temp[0]);
    x.set(ans.n_rows(), ans.n_cols(), ans.get_step_size(), ans.get_ptr());
}

void SparseSolver::solve_linear_least_squares(DenseMatrix& x){	
	x.clear();
	CholmodDenseMatrix ans;	
	cholmod_start(&chol);	
	ans.set(cholmod_solve(CHOLMOD_A, F.get(), RHS.get_chol(), &chol));
	x.set(ans.n_rows(), ans.n_cols(), ans.get_step_size(), ans.get_ptr());
	cholmod_finish(&chol);
}

void SparseSolver::solve_linear_least_squares_soft_constraints(DenseMatrix& x){
	x.clear();
	CholmodDenseMatrix ans;	
	cholmod_start(&chol);	
	ans.set(cholmod_solve(CHOLMOD_A, F.get(), RHS.get_chol(), &chol));
	x.set(ans.n_rows(), ans.n_cols(), ans.get_step_size(), ans.get_ptr());
	cholmod_finish(&chol);
}

void SparseSolver::solve_linear_least_squares_hard_constraints(DenseMatrix& x){

}


}   // namespace hccl