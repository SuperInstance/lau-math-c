/*
 * lau_dirichlet.c — Dirichlet form, Green's function, heat kernel
 *
 * Copyright (c) 2025 Lau Ecosystem
 */

#include "lau_dirichlet.h"
#include <math.h>
#include <string.h>

double lau_dirichlet_energy(const lau_matrix_t *L, const double *f, int n)
{
    /* E = f^T L f */
    double energy = 0.0;
    int i, j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            energy += f[i] * L->data[i * n + j] * f[j];
        }
    }
    return energy;
}

int lau_greens_function(const lau_matrix_t *L, lau_matrix_t *G)
{
    /* Green's function G ≈ L^+ via eigendecomposition of symmetric L */
    int n = L->rows;
    double eigenvalues[LAU_MATRIX_MAX_N];
    double eigenvectors[LAU_MATRIX_MAX_N * LAU_MATRIX_MAX_N];
    int i, j, k;

    lau_matrix_eigenvalues_symmetric(L, eigenvalues, eigenvectors);

    G->rows = n;
    G->cols = n;

    /* G = sum_{lambda_i != 0} (1/lambda_i) v_i v_i^T */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            double sum = 0.0;
            for (k = 0; k < n; k++) {
                if (fabs(eigenvalues[k]) > 1e-10) {
                    sum += (1.0 / eigenvalues[k]) *
                           eigenvectors[i * n + k] *
                           eigenvectors[j * n + k];
                }
            }
            G->data[i * n + j] = sum;
        }
    }

    return 0;
}

int lau_heat_kernel(const lau_matrix_t *L, double t, lau_matrix_t *H)
{
    /* H(t) = e^{-tL} = V diag(e^{-t*lambda}) V^T */
    int n = L->rows;
    double eigenvalues[LAU_MATRIX_MAX_N];
    double eigenvectors[LAU_MATRIX_MAX_N * LAU_MATRIX_MAX_N];
    int i, j, k;

    lau_matrix_eigenvalues_symmetric(L, eigenvalues, eigenvectors);

    H->rows = n;
    H->cols = n;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            double sum = 0.0;
            for (k = 0; k < n; k++) {
                sum += exp(-t * eigenvalues[k]) *
                       eigenvectors[i * n + k] *
                       eigenvectors[j * n + k];
            }
            H->data[i * n + j] = sum;
        }
    }

    return 0;
}

int lau_harmonic_projection(const lau_matrix_t *L, const double *f,
                            double *result, int n)
{
    /* Project onto kernel(L): sum of projections onto eigenvectors with eigenvalue ≈ 0 */
    double eigenvalues[LAU_MATRIX_MAX_N];
    double eigenvectors[LAU_MATRIX_MAX_N * LAU_MATRIX_MAX_N];
    int i, k;

    lau_matrix_eigenvalues_symmetric(L, eigenvalues, eigenvectors);

    for (i = 0; i < n; i++) {
        result[i] = 0.0;
    }

    for (k = 0; k < n; k++) {
        if (fabs(eigenvalues[k]) < 1e-8) {
            double dot = 0.0;
            for (i = 0; i < n; i++) {
                dot += f[i] * eigenvectors[i * n + k];
            }
            for (i = 0; i < n; i++) {
                result[i] += dot * eigenvectors[i * n + k];
            }
        }
    }

    return 0;
}
