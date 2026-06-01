/*
 * lau_dirac.h — Dirac operator, exterior derivative, cohomology
 *
 * Copyright (c) 2025 Lau Ecosystem
 */

#ifndef LAU_DIRAC_H
#define LAU_DIRAC_H

#ifdef __cplusplus
extern "C" {
#endif

#include "lau_matrix.h"

#define LAU_DIRAC_MAX_SIMPLICES 64
#define LAU_DIRAC_MAX_EDGES     256

/* Simplicial complex: vertices + edges */
typedef struct {
    int n_vertices;
    int n_edges;
    int edges[LAU_DIRAC_MAX_EDGES][2]; /* pairs of vertex indices */
} lau_simplicial_complex_t;

/* Build boundary operator d0: edges -> vertices (incidence matrix) */
lau_matrix_t lau_boundary_operator_d0(const lau_simplicial_complex_t *sc);

/* Build coboundary operator d0^T: vertices -> edges */
lau_matrix_t lau_coboundary_operator_d0(const lau_simplicial_complex_t *sc);

/* Graph Dirac operator D = d + d^T on 0-forms and 1-forms */
lau_matrix_t lau_dirac_operator(const lau_simplicial_complex_t *sc);

/* Compute H^0 cohomology (connected components) */
int lau_cohomology_h0(const lau_simplicial_complex_t *sc,
                      double *basis, int *dim);

/* Compute H^1 cohomology (1-cycles / loops) */
int lau_cohomology_h1(const lau_simplicial_complex_t *sc,
                      double *basis, int *dim);

/* Betti numbers: b0 = rank H^0, b1 = rank H^1 */
void lau_betti_numbers(const lau_simplicial_complex_t *sc,
                       int *b0, int *b1);

/* Euler characteristic: chi = V - E (+ F if we had triangles) */
int lau_euler_characteristic(const lau_simplicial_complex_t *sc);

#ifdef __cplusplus
}
#endif

#endif /* LAU_DIRAC_H */
