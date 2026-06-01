/*
 * lau_laplacian.h — Graph Laplacian construction and spectral analysis
 *
 * Copyright (c) 2025 Lau Ecosystem
 */

#ifndef LAU_LAPLACIAN_H
#define LAU_LAPLACIAN_H

#ifdef __cplusplus
extern "C" {
#endif

#include "lau_matrix.h"

/* Build combinatorial Laplacian L = D - A from adjacency matrix */
lau_matrix_t lau_laplacian_from_adjacency(const lau_matrix_t *adj);

/* Build normalized Laplacian L_norm = I - D^{-1/2} A D^{-1/2} */
lau_matrix_t lau_laplacian_normalized(const lau_matrix_t *adj);

/* Power iteration to find dominant eigenvalue/eigenvector */
int lau_power_iteration(const lau_matrix_t *m, double *eigenvalue,
                        double *eigenvector, int max_iter, double tol);

/* Top-k eigenvalues via power iteration + deflation */
int lau_laplacian_topk_eigenvalues(const lau_matrix_t *laplacian,
                                   double *eigenvalues, double *eigenvectors,
                                   int k, int max_iter, double tol);

/* Spectral gap: smallest non-zero eigenvalue */
double lau_spectral_gap(const lau_matrix_t *laplacian, int max_iter, double tol);

/* Algebraic connectivity (Fiedler value) */
double lau_algebraic_connectivity(const lau_matrix_t *adj, int max_iter, double tol);

#ifdef __cplusplus
}
#endif

#endif /* LAU_LAPLACIAN_H */
