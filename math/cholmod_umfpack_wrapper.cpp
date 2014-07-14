#include "sparsematrix.h"

namespace hccl{

//////////////////////////////////////////////////////////////////////////
// CholmodSparseMatrix
// - A CHOLMOD sparse matrix wrapper
// - by Stephen Baek (3d@cad.snu.ac.kr)
//////////////////////////////////////////////////////////////////////////

CholmodSparseMatrix::CholmodSparseMatrix():M(NULL){
}

CholmodSparseMatrix::CholmodSparseMatrix(SparseMatrix& _M):M(NULL){
    set(_M);
}

CholmodSparseMatrix::CholmodSparseMatrix(int n_rows, int n_cols, int n_nonzeros, int* i, int* j, double* x):M(NULL){
    set(n_rows, n_cols, n_nonzeros, i, j, x);
}

CholmodSparseMatrix::CholmodSparseMatrix(cholmod_sparse* _M):M(NULL){
    set(_M);
}

CholmodSparseMatrix::~CholmodSparseMatrix(){
    clear();
}

void CholmodSparseMatrix::clear(){
    cholmod_start(&chol);
    if(M != NULL && cholmod_check_sparse(M, &chol) != 0)
        cholmod_free_sparse(&M, &chol);
    cholmod_finish(&chol);
}

void CholmodSparseMatrix::set(SparseMatrix& _M){
    set(_M.n_rows(), _M.n_cols(), _M.n_nonzeros(), _M.get_i_ptr(), _M.get_j_ptr(), _M.get_x_ptr());
}

void CholmodSparseMatrix::set(int n_rows, int n_cols, int n_nonzeros, int* i, int* j, double* x){
    clear();
    cholmod_start(&chol);
    cholmod_triplet* trip = cholmod_allocate_triplet(n_rows, n_cols, n_nonzeros, 0, CHOLMOD_REAL, &chol);
    memcpy(trip->i, i, sizeof(int)*n_nonzeros);
    memcpy(trip->j, j, sizeof(int)*n_nonzeros);
    memcpy(trip->x, x, sizeof(double)*n_nonzeros);
    trip->nnz = n_nonzeros;

    M = cholmod_triplet_to_sparse(trip, n_nonzeros, &chol);
    
    cholmod_free_triplet(&trip, &chol);
    cholmod_finish(&chol);
}

void CholmodSparseMatrix::set(cholmod_sparse* _M){
    clear();
    M = _M;
}

int CholmodSparseMatrix::n_rows() const{
    return M->nrow;
}

int CholmodSparseMatrix::n_cols() const{
    return M->ncol;
}

size_t CholmodSparseMatrix::n_nonzeros() const{
    return M->nzmax;        // is it correct to do so?
}

int* CholmodSparseMatrix::get_i_ptr(){
    return (int*)M->i;
}

int* CholmodSparseMatrix::get_p_ptr(){
    return (int*)M->p;
}

double* CholmodSparseMatrix::get_x_ptr(){
    return (double*)M->x;
}

cholmod_sparse* CholmodSparseMatrix::get_chol(){
    return M;
}

SparseMatrix CholmodSparseMatrix::get(){
    SparseMatrix ret(M->nrow, M->ncol);

    cholmod_start(&chol);
    cholmod_triplet* trip = cholmod_sparse_to_triplet(M, &chol);
    for(int i = 0; i < trip->nnz; i++){
        ret.add_entry( ((int*)trip->i)[i], ((int*)trip->j)[i], ((double*)trip->x)[i] );
    }
    cholmod_free_triplet(&trip, &chol);
    cholmod_finish(&chol);

    return ret;
}

void CholmodSparseMatrix::set_identity(int n){
    int rows = M->nrow;
    int cols = M->ncol;
    clear();
    cholmod_start(&chol);
    M = cholmod_speye(rows, cols, CHOLMOD_REAL, &chol);
    cholmod_finish(&chol);
}


//////////////////////////////////////////////////////////////////////////
// CholmodTripletMatrix
// - A CHOLMOD triplet matrix wrapper
// - by Stephen Baek (3d@cad.snu.ac.kr)
//////////////////////////////////////////////////////////////////////////
CholmodTripletMatrix::CholmodTripletMatrix():M(NULL){

}

CholmodTripletMatrix::CholmodTripletMatrix(SparseMatrix& _M):M(NULL){
    set(_M);
}

CholmodTripletMatrix::CholmodTripletMatrix(int n_rows, int n_cols, int n_nonzeros, int* i, int* j, double* x):M(NULL){
    set(n_rows, n_cols, n_nonzeros, i, j, x);
}

CholmodTripletMatrix::CholmodTripletMatrix(cholmod_triplet* _M):M(NULL){
    set(_M);
}

CholmodTripletMatrix::~CholmodTripletMatrix(){
    clear();
}

void CholmodTripletMatrix::clear(){
    cholmod_start(&chol);
    if(M != NULL && cholmod_check_triplet(M, &chol) != 0)
        cholmod_free_triplet(&M, &chol);
    cholmod_finish(&chol);
}

void CholmodTripletMatrix::set(SparseMatrix& _M){
    set(_M.n_rows(), _M.n_cols(), _M.n_nonzeros(), _M.get_i_ptr(), _M.get_j_ptr(), _M.get_x_ptr());
}

void CholmodTripletMatrix::set(int n_rows, int n_cols, int n_nonzeros, int* i, int* j, double* x){
    clear();
    cholmod_start(&chol);
    M = cholmod_allocate_triplet(n_rows, n_cols, n_nonzeros, 0, CHOLMOD_REAL, &chol);
    memcpy(M->i, i, sizeof(int)*n_nonzeros);
    memcpy(M->j, j, sizeof(int)*n_nonzeros);
    memcpy(M->x, x, sizeof(double)*n_nonzeros);
    M->nnz = n_nonzeros;
    cholmod_finish(&chol);
}

void CholmodTripletMatrix::set(cholmod_triplet* _M){
    clear();
    M = _M;
}

int CholmodTripletMatrix::n_rows() const{
    return M->nrow;
}

int CholmodTripletMatrix::n_cols() const{
    return M->ncol;
}

size_t CholmodTripletMatrix::n_nonzeros() const{
    return M->nnz;
}

int* CholmodTripletMatrix::get_i_ptr(){
    return (int*)M->i;
}

int* CholmodTripletMatrix::get_j_ptr(){
    return (int*)M->j;
}

double* CholmodTripletMatrix::get_x_ptr(){
    return (double*)M->x;
}

cholmod_triplet* CholmodTripletMatrix::get_chol(){
    return M;
}

SparseMatrix CholmodTripletMatrix::get(){
    SparseMatrix ret(M->nrow, M->ncol);

    for(int i = 0; i < M->nnz; i++){
        ret.add_entry( ((int*)M->i)[i], ((int*)M->j)[i], ((double*)M->x)[i] );
    }

    return ret;
}
//////////////////////////////////////////////////////////////////////////
// CholmodDenseMatrix
// - A CHOLMOD dense matrix wrapper
// - by Stephen Baek (3d@cad.snu.ac.kr)
//////////////////////////////////////////////////////////////////////////

CholmodDenseMatrix::CholmodDenseMatrix():M(NULL){

}

CholmodDenseMatrix::CholmodDenseMatrix(const std::vector< std::vector<double> >& x):M(NULL){
    set(x);
}

CholmodDenseMatrix::CholmodDenseMatrix(int n_rows, int n_cols, double* x):M(NULL){
    set(n_rows, n_cols, x);
}

CholmodDenseMatrix::~CholmodDenseMatrix(){
    clear();
}

void CholmodDenseMatrix::clear(){
    cholmod_start(&chol);
    if(M != NULL && cholmod_check_dense(M, &chol) != 0)
        cholmod_free_dense(&M, &chol);
    cholmod_finish(&chol);
}

void CholmodDenseMatrix::set(DenseMatrix& _M){
    set(_M.get_data());
}

void CholmodDenseMatrix::set(const std::vector< std::vector<double> >& x){
    clear();
    int n_cols = x.size();
    int n_rows = x[0].size();
    cholmod_start(&chol);
    M = cholmod_zeros(n_rows, n_cols, CHOLMOD_REAL, &chol);
    for(int i = 0; i < n_cols; i++){
        memcpy(&( ((double*)M->x)[i*M->d] ), &(x[i][0]), sizeof(double)*n_rows);
    }
    cholmod_finish(&chol);
}

void CholmodDenseMatrix::set(int n_rows, int n_cols, double* x){
    clear();

    cholmod_start(&chol);
    M = cholmod_zeros(n_rows, n_cols, CHOLMOD_REAL, &chol);
    for(int i = 0; i < n_cols; i++){
        memcpy(&( ((double*)M->x)[i*M->d] ), x+i*n_cols, sizeof(double)*n_rows);
    }
    cholmod_finish(&chol);
}

void CholmodDenseMatrix::set(cholmod_dense* _M){
    clear();
    M = _M;
}

// bool CholmodDenseMatrix::is_valid(){
//     cholmod_check_dense(M, &chol);
// }

int CholmodDenseMatrix::n_rows() const{
    return M->nrow;
}

int CholmodDenseMatrix::n_cols() const{
    return M->ncol;
}

int CholmodDenseMatrix::get_step_size() const{
    return M->d;
}

double* CholmodDenseMatrix::get_col_ptr(int i){
    return &(((double*)M->x)[i*M->d]);
}

double* CholmodDenseMatrix::get_ptr(){
    return (double*)(M->x);
}

cholmod_dense* CholmodDenseMatrix::get_chol(){
    return M;
}

void CholmodDenseMatrix::set_null(){
    int rows = M->nrow;
    int cols = M->ncol;
    clear();
    cholmod_start(&chol);
    M = cholmod_zeros(rows, cols, CHOLMOD_REAL, &chol);
    cholmod_finish(&chol);
}

void CholmodDenseMatrix::set_identity(int n){
    int rows = M->nrow;
    int cols = M->ncol;
    clear();
    cholmod_start(&chol);
    M = cholmod_eye(rows, cols, CHOLMOD_REAL, &chol);
    cholmod_finish(&chol);
}

//////////////////////////////////////////////////////////////////////////
// CholmodFactorMatrix
// - A CHOLMOD factor matrix wrapper
// - by Stephen Baek (3d@cad.snu.ac.kr)
//////////////////////////////////////////////////////////////////////////

CholmodFactorMatrix::CholmodFactorMatrix():M(NULL){

}

CholmodFactorMatrix::~CholmodFactorMatrix(){
    clear();
}

void CholmodFactorMatrix::clear(){
    cholmod_start(&chol);
    if(M != NULL && cholmod_check_factor(M, &chol) != 0)
        cholmod_free_factor(&M, &chol);
    cholmod_finish(&chol);
}

void CholmodFactorMatrix::set(cholmod_factor* _M){
    M = _M;
}

cholmod_factor* CholmodFactorMatrix::get(){
    return M;
}


//////////////////////////////////////////////////////////////////////////
// UmfpackFactorMatrix
// - A UMFPACK factor matrix wrapper
// - by Stephen Baek (3d@cad.snu.ac.kr)
//////////////////////////////////////////////////////////////////////////

UmfpackFactorMatrix::UmfpackFactorMatrix():symbolic(NULL),numeric(NULL){

}

UmfpackFactorMatrix::~UmfpackFactorMatrix(){
    clear();
}

void UmfpackFactorMatrix::clear(){
    if(symbolic != NULL)
        umfpack_dl_free_symbolic( &symbolic );
    if(numeric != NULL)
        umfpack_dl_free_numeric( &numeric );
}

void* UmfpackFactorMatrix::get_numeric(){
    return numeric;
}

void* UmfpackFactorMatrix::get_symbolic(){
    return symbolic;
}

}   // namespace hccl