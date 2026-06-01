/*
 * lau_laplacian.c — Graph Laplacian construction and spectral analysis
 *
 * Copyright (c) 2025 Lau Ecosystem
 */

#include "lau_laplacian.h"
#include <math.h>
#include <string.h>

lau_matrix_t lau_laplacian_from_adjacency(const lau_matrix_t *adj)
{
    int n = adj->rows;
    lau_matrix_t L = lau_matrix_zeros(n, n);
    int i, j;

    for (i = 0; i < n; i++) {
        double degree = 0.0;
        for (j = 0; j < n; j++) {
            degree += adj->data[i * n + j];
        }
        L.data[i * n + i] = degree;
        for (j = 0; j < n; j++) {
            if (i != j) {
                L.data[i * n + j] = -adj->data[i * n + j];
            }
        }
    }
    return L;
}

lau_matrix_t lau_laplacian_normalized(const lau_matrix_t *adj)
{
    int n = adj->rows;
    lau_matrix_t L_norm = lau_matrix_zeros(n, n);
    double d_inv_sqrt[LAU_MATRIX_MAX_N];
    int i, j;

    /* Compute D^{-1/2} */
    for (i = 0; i < n; i++) {
        double degree = 0.0;
        for (j = 0; j < n; j++) {
            degree += adj->data[i * n + j];
        }
        d_inv_sqrt[i] = (degree > 1e-15) ? 1.0 / sqrt(degree) : 0.0;
    }

    /* L_norm = I - D^{-1/2} A D^{-1/2} */
    for (i = 0; i < n; i++) {
        L_norm.data[i * n + i] = 1.0;
        for (j = 0; j < n; j++) {
            if (i != j) {
                L_norm.data[i * n + j] = -d_inv_sqrt[i] * adj->data[i * n + j] * d_inv_sqrt[j];
            }
        }
    }
    return L_norm;
}

int lau_power_iteration(const lau_matrix_t *m, double *eigenvalue,
                        double *eigenvector, int max_iter, double tol)
{
    int n = m->rows;
    double v[LAU_MATRIX_MAX_N];
    double v_new[LAU_MATRIX_MAX_N];
    int iter, i;
    double norm, lambda_old = 0.0;

    /* Initialize with uniform vector */
    for (i = 0; i < n; i++) {
        v[i] = 1.0 / sqrt((double)n);
    }

    for (iter = 0; iter < max_iter; iter++) {
        /* v_new = M * v */
        for (i = 0; i < n; i++) {
            int j;
            v_new[i] = 0.0;
            for (j = 0; j < n; j++) {
                v_new[i] += m->data[i * n + j] * v[j];
            }
        }

        /* Normalize */
        norm = 0.0;
        for (i = 0; i < n; i++) {
            norm += v_new[i] * v_new[i];
        }
        norm = sqrt(norm);
        if (norm < 1e-15) break;

        /* Rayleigh quotient */
        *eigenvalue = 0.0;
        for (i = 0; i < n; i++) {
            v[i] = v_new[i] / norm;
            *eigenvalue += v_new[i] * v[i];
        }

        if (fabs(*eigenvalue - lambda_old) < tol) break;
        lambda_old = *eigenvalue;
    }

    if (eigenvector) {
        for (i = 0; i < n; i++) {
            eigenvector[i] = v[i];
        }
    }

    return iter;
}

int lau_laplacian_topk_eigenvalues(const lau_matrix_t *laplacian,
                                   double *eigenvalues, double *eigenvectors,
                                   int k, int max_iter, double tol)
{
    lau_matrix_t deflated;
    int i, j, idx;

    if (k > laplacian->rows) k = laplacian->rows;

    deflated = *laplacian;

    for (idx = 0; idx < k; idx++) {
        double ev;
        double vec[LAU_MATRIX_MAX_N];
        int n = deflated.rows;

        lau_power_iteration(&deflated, &ev, vec, max_iter, tol);
        eigenvalues[idx] = ev;

        if (eigenvectors) {
            for (i = 0; i < n; i++) {
                eigenvectors[idx * n + i] = vec[i];
            }
        }

        /* Deflate: M' = M - lambda * v * v^T */
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                deflated.data[i * n + j] -= ev * vec[i] * vec[j];
            }
        }
    }

    return k;
}

double lau_spectral_gap(const lau_matrix_t *laplacian, int max_iter, double tol)
{
    int n = laplacian->rows;
    double eigenvalues[LAU_MATRIX_MAX_N];
    double eigenvectors[LAU_MATRIX_MAX_N * LAU_MATRIX_MAX_N];
    int i, count;

    if (n <= 1) return 0.0;

    /* Use Jacobi for symmetric matrices */
    count = lau_matrix_eigenvalues_symmetric(laplacian, eigenvalues, eigenvectors);
    (void)count;

    /* Sort eigenvalues ascending */
    for (i = 0; i < n - 1; i++) {
        int j2;
        for (j2 = i + 1; j2 < n; j2++) {
            if (eigenvalues[j2] < eigenvalues[i]) {
                double tmp = eigenvalues[i];
                eigenvalues[i] = eigenvalues[j2];
                eigenvalues[j2] = tmp;
            }
        }
    }

    /* Find smallest non-zero eigenvalue */
    for (i = 0; i < n; i++) {
        if (eigenvalues[i] > tol) {
            return eigenvalues[i];
        }
    }
    return 0.0;
}

double lau_algebraic_connectivity(const lau_matrix_t *adj, int max_iter, double tol)
{
    lau_matrix_t L = lau_laplacian_from_adjacency(adj);
    return lau_spectral_gap(&L, max_iter, tol);
}
