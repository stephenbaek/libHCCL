#ifndef HCCL_CHOLMOD_UMFPACK_WRAPPER_H_
#define HCCL_CHOLMOD_UMFPACK_WRAPPER_H_

#include "../common.h"
#include "sparsematrix.h"
#include "SuiteSparse_config.h"
#include "cholmod.h"
#include "umfpack.h"

namespace hccl{

static cholmod_common chol;

class CholmodSparseMatrix{
public:
    CholmodSparseMatrix();
    CholmodSparseMatrix(SparseMatrix& _M);
    CholmodSparseMatrix(int n_rows, int n_cols, int n_nonzeros, int* i, int* j, double* x);
    CholmodSparseMatrix(cholmod_sparse* _M);
    ~CholmodSparseMatrix();

    void clear();
    void set(SparseMatrix& _M);
    void set(int n_rows, int n_cols, int n_nonzeros, int* i, int* j, double* x);
    void set(cholmod_sparse* _M);
    //bool is_valid(); // TODO

    int n_rows() const;
    int n_cols() const;
    size_t n_nonzeros() const;

    int* get_i_ptr();
    int* get_p_ptr();
    double* get_x_ptr();

    cholmod_sparse* get_chol();
    SparseMatrix get();

    void set_identity(int n);

protected:
    cholmod_sparse* M;
};

class CholmodTripletMatrix{
public:
    CholmodTripletMatrix();
    CholmodTripletMatrix(SparseMatrix& _M);
    CholmodTripletMatrix(int n_rows, int n_cols, int n_nonzeros, int* i, int* j, double* x);
    CholmodTripletMatrix(cholmod_triplet* _M);
    ~CholmodTripletMatrix();

    void clear();
    void set(SparseMatrix& _M);
    void set(int n_rows, int n_cols, int n_nonzeros, int* i, int* j, double* x);
    void set(cholmod_triplet* _M);
    //bool is_valid(); // TODO

    int n_rows() const;
    int n_cols() const;
    size_t n_nonzeros() const;

    int* get_i_ptr();
    int* get_j_ptr();
    double* get_x_ptr();

    cholmod_triplet* get_chol();
    SparseMatrix get();

protected:
    cholmod_triplet* M;
};

class CholmodDenseMatrix{
public:
    CholmodDenseMatrix();
    CholmodDenseMatrix(const std::vector< std::vector<double> >& x);
    CholmodDenseMatrix(int n_rows, int n_cols, double* x);
    ~CholmodDenseMatrix();

    void clear();
    void set(DenseMatrix& _M);
    void set(const std::vector< std::vector<double> >& x);
    void set(int n_rows, int n_cols, double* x);
    void set(cholmod_dense* _M);
    //bool is_valid(); // TODO

    int n_rows() const;
    int n_cols() const;

    double* get_col_ptr(int i);
    double* get_ptr();
    int get_step_size() const;

    cholmod_dense* get_chol();
    //DenseMatrix get();  // TODO

    void set_null();
    void set_identity(int n);

protected:
    cholmod_dense* M;
};

class CholmodFactorMatrix{
public:
    CholmodFactorMatrix();
    ~CholmodFactorMatrix();

    void clear();
    void set(cholmod_factor* _M);
    cholmod_factor* get();

protected:
    cholmod_factor* M;
};

class UmfpackFactorMatrix{
public:
    UmfpackFactorMatrix();
    ~UmfpackFactorMatrix();

    void clear();
    void* get_numeric();
    void* get_symbolic();

//protected:
    void* symbolic;
    void* numeric;
};

} // namespace hccl

#endif // HCCL_CHOLMOD_UMFPACK_WRAPPER_H_