/*
 * lau_matrix.c — Fixed-size matrix operation implementations
 *
 * Copyright (c) 2025 Lau Ecosystem
 */

#include "lau_matrix.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

lau_matrix_t lau_matrix_multiply(const lau_matrix_t *a, const lau_matrix_t *b)
{
    lau_matrix_t out;
    int i, j, k;
    out.rows = a->rows;
    out.cols = b->cols;

    for (i = 0; i < out.rows; i++) {
        for (j = 0; j < out.cols; j++) {
            double sum = 0.0;
            for (k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            out.data[i * out.cols + j] = sum;
        }
    }
    return out;
}

lau_matrix_t lau_matrix_add(const lau_matrix_t *a, const lau_matrix_t *b)
{
    lau_matrix_t out;
    int i, total;
    out.rows = a->rows;
    out.cols = a->cols;
    total = a->rows * a->cols;
    for (i = 0; i < total; i++) {
        out.data[i] = a->data[i] + b->data[i];
    }
    return out;
}

lau_matrix_t lau_matrix_sub(const lau_matrix_t *a, const lau_matrix_t *b)
{
    lau_matrix_t out;
    int i, total;
    out.rows = a->rows;
    out.cols = a->cols;
    total = a->rows * a->cols;
    for (i = 0; i < total; i++) {
        out.data[i] = a->data[i] - b->data[i];
    }
    return out;
}

lau_matrix_t lau_matrix_scale(const lau_matrix_t *m, double s)
{
    lau_matrix_t out;
    int i, total;
    out.rows = m->rows;
    out.cols = m->cols;
    total = m->rows * m->cols;
    for (i = 0; i < total; i++) {
        out.data[i] = m->data[i] * s;
    }
    return out;
}

lau_matrix_t lau_matrix_transpose(const lau_matrix_t *m)
{
    lau_matrix_t out;
    int i, j;
    out.rows = m->cols;
    out.cols = m->rows;
    for (i = 0; i < m->rows; i++) {
        for (j = 0; j < m->cols; j++) {
            out.data[j * out.cols + i] = m->data[i * m->cols + j];
        }
    }
    return out;
}

int lau_matrix_invert(lau_matrix_t *out, const lau_matrix_t *in)
{
    /* Gauss-Jordan elimination on augmented matrix [A|I] */
    double aug[LAU_MATRIX_MAX_N][2 * LAU_MATRIX_MAX_N];
    int n = in->rows;
    int i, j, k;

    if (in->rows != in->cols) return -1;
    if (n > LAU_MATRIX_MAX_N) return -1;

    out->rows = n;
    out->cols = n;

    /* Build augmented matrix */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            aug[i][j] = in->data[i * n + j];
            aug[i][j + n] = (i == j) ? 1.0 : 0.0;
        }
    }

    /* Forward elimination */
    for (i = 0; i < n; i++) {
        /* Find pivot */
        double max_val = fabs(aug[i][i]);
        int max_row = i;
        for (k = i + 1; k < n; k++) {
            if (fabs(aug[k][i]) > max_val) {
                max_val = fabs(aug[k][i]);
                max_row = k;
            }
        }
        if (max_val < 1e-15) return -1; /* Singular */

        /* Swap rows */
        if (max_row != i) {
            double tmp;
            for (j = 0; j < 2 * n; j++) {
                tmp = aug[i][j];
                aug[i][j] = aug[max_row][j];
                aug[max_row][j] = tmp;
            }
        }

        /* Scale pivot row */
        {
            double pivot = aug[i][i];
            for (j = 0; j < 2 * n; j++) {
                aug[i][j] /= pivot;
            }
        }

        /* Eliminate column */
        for (k = 0; k < n; k++) {
            if (k != i) {
                double factor = aug[k][i];
                for (j = 0; j < 2 * n; j++) {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }

    /* Extract inverse */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            out->data[i * n + j] = aug[i][j + n];
        }
    }

    return 0;
}

double lau_matrix_determinant(const lau_matrix_t *m)
{
    /* LU decomposition approach */
    double lu[LAU_MATRIX_MAX_N][LAU_MATRIX_MAX_N];
    int n = m->rows;
    int i, j, k;
    int swaps = 0;
    double det;

    if (m->rows != m->cols) return 0.0;
    if (n == 1) return m->data[0];
    if (n == 2) return m->data[0] * m->data[3] - m->data[1] * m->data[2];

    /* Copy to LU */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            lu[i][j] = m->data[i * n + j];
        }
    }

    /* Crout's LU with partial pivoting */
    for (k = 0; k < n; k++) {
        /* Pivot */
        double max_val = fabs(lu[k][k]);
        int max_row = k;
        for (i = k + 1; i < n; i++) {
            if (fabs(lu[i][k]) > max_val) {
                max_val = fabs(lu[i][k]);
                max_row = i;
            }
        }
        if (max_row != k) {
            double tmp;
            for (j = 0; j < n; j++) {
                tmp = lu[k][j];
                lu[k][j] = lu[max_row][j];
                lu[max_row][j] = tmp;
            }
            swaps++;
        }

        if (fabs(lu[k][k]) < 1e-15) return 0.0;

        for (i = k + 1; i < n; i++) {
            lu[i][k] /= lu[k][k];
            for (j = k + 1; j < n; j++) {
                lu[i][j] -= lu[i][k] * lu[k][j];
            }
        }
    }

    det = (swaps % 2 == 0) ? 1.0 : -1.0;
    for (i = 0; i < n; i++) {
        det *= lu[i][i];
    }
    return det;
}

double lau_matrix_trace(const lau_matrix_t *m)
{
    double tr = 0.0;
    int i;
    int n = (m->rows < m->cols) ? m->rows : m->cols;
    for (i = 0; i < n; i++) {
        tr += m->data[i * m->cols + i];
    }
    return tr;
}

int lau_matrix_eigenvalues_symmetric(const lau_matrix_t *m,
                                     double *eigenvalues,
                                     double *eigenvectors)
{
    /* Jacobi eigenvalue algorithm for symmetric matrices */
    double a[LAU_MATRIX_MAX_N][LAU_MATRIX_MAX_N];
    double v[LAU_MATRIX_MAX_N][LAU_MATRIX_MAX_N];
    int n = m->rows;
    int iter, i, j, p, q;
    const int max_iter = 1000;

    if (m->rows != m->cols) return -1;

    /* Copy input */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            a[i][j] = m->data[i * n + j];
            v[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (iter = 0; iter < max_iter; iter++) {
        /* Find largest off-diagonal element */
        double max_off = 0.0;
        p = 0; q = 1;
        for (i = 0; i < n; i++) {
            for (j = i + 1; j < n; j++) {
                if (fabs(a[i][j]) > max_off) {
                    max_off = fabs(a[i][j]);
                    p = i;
                    q = j;
                }
            }
        }

        if (max_off < 1e-12) break; /* Converged */

        /* Compute rotation angle */
        double theta;
        if (fabs(a[p][p] - a[q][q]) < 1e-15) {
            theta = M_PI / 4.0;
        } else {
            theta = 0.5 * atan2(2.0 * a[p][q], a[p][p] - a[q][q]);
        }

        {
            double c = cos(theta);
            double s = sin(theta);
            double app = a[p][p], aqq = a[q][q], apq = a[p][q];
            double aip, aiq;

            a[p][p] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
            a[q][q] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;

            for (i = 0; i < n; i++) {
                if (i != p && i != q) {
                    aip = a[i][p];
                    aiq = a[i][q];
                    a[i][p] = c * aip + s * aiq;
                    a[p][i] = a[i][p];
                    a[i][q] = -s * aip + c * aiq;
                    a[q][i] = a[i][q];
                }
            }

            /* Update eigenvectors */
            for (i = 0; i < n; i++) {
                double vip = v[i][p];
                double viq = v[i][q];
                v[i][p] = c * vip + s * viq;
                v[i][q] = -s * vip + c * viq;
            }
        }
    }

    /* Extract eigenvalues */
    for (i = 0; i < n; i++) {
        eigenvalues[i] = a[i][i];
    }

    /* Extract eigenvectors (column-major -> row-major) */
    if (eigenvectors) {
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                eigenvectors[i * n + j] = v[i][j];
            }
        }
    }

    return iter;
}

int lau_matrix_approx_equal(const lau_matrix_t *a, const lau_matrix_t *b, double tol)
{
    int i, total;
    if (a->rows != b->rows || a->cols != b->cols) return 0;
    total = a->rows * a->cols;
    for (i = 0; i < total; i++) {
        if (fabs(a->data[i] - b->data[i]) > tol) return 0;
    }
    return 1;
}

void lau_matrix_print(const lau_matrix_t *m)
{
    int i, j;
    for (i = 0; i < m->rows; i++) {
        for (j = 0; j < m->cols; j++) {
            printf("%10.6f ", m->data[i * m->cols + j]);
        }
        printf("\n");
    }
}
