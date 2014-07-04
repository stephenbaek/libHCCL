#ifndef HCCL_SPARSEMATRIX_H_
#define HCCL_SPARSEMATRIX_H_

#include "../common.h"
#include "cholmod_umfpack_wrapper.h"

namespace hccl{

class SparseMatrix{
public:
    SparseMatrix();
    SparseMatrix(int _rows, int _cols);
    ~SparseMatrix();

    void set_size(int _rows, int _cols);
    int n_rows() const;
    int n_cols() const;

    void clear();
    int add_entry(int _i, int _j, double _x);
    size_t n_nonzeros() const;

    int* get_i_ptr();
    int* get_j_ptr();
    double* get_x_ptr();

    void set_null();
    void set_identity(int n);

    void print(const char* name);
    void print_info(const char* name);

protected:
    int rows, cols;
    std::vector<int> i;
    std::vector<int> j;
    std::vector<double> x;
};

class DenseMatrix{
public:
    DenseMatrix();
    DenseMatrix(int _rows, int _cols);
    ~DenseMatrix();

    void set_size(int _rows, int _cols);
    int n_rows() const;
    int n_cols() const;

    void clear();
    int set(int _i, int _j, double _x);
    void set(int _rows, int _cols, int _step_size, double* _x);
    void set(int _rows, int _cols, double** _x);
    void set(std::vector< std::vector<double> >& _x);

    double get(int _i, int _j) const;
    double* get_col_ptr(int i);
    std::vector< std::vector<double> >& get_data();

    void set_null();
    void set_identity(int n);

    void print(const char* name);
    void print_info(const char* name);

protected:
    int rows, cols;
    std::vector< std::vector<double> > x;
};

class SparseSolver{
public:
    SparseSolver();
    SparseSolver(SparseMatrix& _A, DenseMatrix& _b);
    SparseSolver(SparseMatrix& _A, DenseMatrix& _b, std::vector<int>& _constraint_idx, DenseMatrix& _constraints);
    ~SparseSolver();

    void clear();

    bool is_valid();

    void set_matrices(SparseMatrix& _A, DenseMatrix& _b);
    void set_constraints(std::vector<int>& _constraint_idx, DenseMatrix& _constraints);

    void factor_sym();
    void factor_gen();
    void factor_linear_least_squares();
    void factor_linear_least_squares_soft_constraints();
    void factor_linear_least_squares_hard_constraints();

    void solve_sym(DenseMatrix& x);
    void solve_gen(DenseMatrix& x);
    void solve_linear_least_squares(DenseMatrix& x);
    void solve_linear_least_squares_soft_constraints(DenseMatrix& x);
    void solve_linear_least_squares_hard_constraints(DenseMatrix& x);
    
protected:
    CholmodSparseMatrix A;
    CholmodDenseMatrix b;
    std::vector<int> con_id;
    CholmodDenseMatrix con;

    CholmodFactorMatrix F;
    CholmodSparseMatrix LHS;
    CholmodDenseMatrix RHS;
};

} // namespace hccl

#endif // HCCL_SPARSEMATRIX_H_