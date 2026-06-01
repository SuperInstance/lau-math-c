/*
 * lau_matrix.h — Fixed-size stack-allocated matrix operations
 *
 * Supports 3x3, 4x4 (robotics), and NxN up to 16x16.
 * C99, no heap allocation, no external dependencies.
 *
 * Copyright (c) 2025 Lau Ecosystem
 */

#ifndef LAU_MATRIX_H
#define LAU_MATRIX_H

#ifdef __cplusplus
extern "C" {
#endif

#define LAU_MATRIX_MAX_N 16

typedef struct {
    int rows;
    int cols;
    double data[LAU_MATRIX_MAX_N * LAU_MATRIX_MAX_N];
} lau_matrix_t;

/* Construction */
static inline lau_matrix_t lau_matrix_zeros(int rows, int cols);
static inline lau_matrix_t lau_matrix_identity(int n);
static inline lau_matrix_t lau_matrix_from_array(int rows, int cols, const double *arr);
static inline double lau_matrix_get(const lau_matrix_t *m, int i, int j);
static inline void lau_matrix_set(lau_matrix_t *m, int i, int j, double val);

/* Arithmetic */
lau_matrix_t lau_matrix_multiply(const lau_matrix_t *a, const lau_matrix_t *b);
lau_matrix_t lau_matrix_add(const lau_matrix_t *a, const lau_matrix_t *b);
lau_matrix_t lau_matrix_sub(const lau_matrix_t *a, const lau_matrix_t *b);
lau_matrix_t lau_matrix_scale(const lau_matrix_t *m, double s);

/* Transformations */
lau_matrix_t lau_matrix_transpose(const lau_matrix_t *m);
int lau_matrix_invert(lau_matrix_t *out, const lau_matrix_t *in); /* Gauss-Jordan, returns 0 on success */
double lau_matrix_determinant(const lau_matrix_t *m);
double lau_matrix_trace(const lau_matrix_t *m);

/* Eigenvalues (Jacobi for symmetric matrices) */
int lau_matrix_eigenvalues_symmetric(const lau_matrix_t *m, double *eigenvalues, double *eigenvectors);
/* eigenvectors is rows*rows, columns are eigenvectors. Returns number of iterations. */

/* Utility */
int lau_matrix_approx_equal(const lau_matrix_t *a, const lau_matrix_t *b, double tol);
void lau_matrix_print(const lau_matrix_t *m);

/* ---- static inline implementations ---- */

static inline lau_matrix_t lau_matrix_zeros(int rows, int cols)
{
    lau_matrix_t m;
    int i;
    m.rows = rows;
    m.cols = cols;
    for (i = 0; i < rows * cols; i++) {
        m.data[i] = 0.0;
    }
    return m;
}

static inline lau_matrix_t lau_matrix_identity(int n)
{
    lau_matrix_t m = lau_matrix_zeros(n, n);
    int i;
    for (i = 0; i < n; i++) {
        m.data[i * n + i] = 1.0;
    }
    return m;
}

static inline lau_matrix_t lau_matrix_from_array(int rows, int cols, const double *arr)
{
    lau_matrix_t m;
    int i;
    m.rows = rows;
    m.cols = cols;
    for (i = 0; i < rows * cols; i++) {
        m.data[i] = arr[i];
    }
    return m;
}

static inline double lau_matrix_get(const lau_matrix_t *m, int i, int j)
{
    return m->data[i * m->cols + j];
}

static inline void lau_matrix_set(lau_matrix_t *m, int i, int j, double val)
{
    m->data[i * m->cols + j] = val;
}

#ifdef __cplusplus
}
#endif

#endif /* LAU_MATRIX_H */
