/*
 * lau_dirichlet.h — Dirichlet form, Green's function, heat kernel
 *
 * Copyright (c) 2025 Lau Ecosystem
 */

#ifndef LAU_DIRICHLET_H
#define LAU_DIRICHLET_H

#ifdef __cplusplus
extern "C" {
#endif

#include "lau_matrix.h"

/* Dirichlet energy: E(f,f) = f^T L f */
double lau_dirichlet_energy(const lau_matrix_t *L, const double *f, int n);

/* Green's function approximation: G ≈ L^+ (pseudoinverse via eigendecomp) */
int lau_greens_function(const lau_matrix_t *L, lau_matrix_t *G);

/* Heat kernel: H(t) = e^{-tL} via eigendecomposition */
int lau_heat_kernel(const lau_matrix_t *L, double t, lau_matrix_t *H);

/* Harmonic projection: project f onto kernel of L */
int lau_harmonic_projection(const lau_matrix_t *L, const double *f,
                            double *result, int n);

#ifdef __cplusplus
}
#endif

#endif /* LAU_DIRICHLET_H */
