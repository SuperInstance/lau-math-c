/*
 * lau_dirac.c — Dirac operator, exterior derivative, cohomology
 *
 * Copyright (c) 2025 Lau Ecosystem
 */

#include "lau_dirac.h"
#include <math.h>
#include <string.h>

lau_matrix_t lau_boundary_operator_d0(const lau_simplicial_complex_t *sc)
{
    /* Boundary matrix: n_vertices x n_edges */
    lau_matrix_t B = lau_matrix_zeros(sc->n_vertices, sc->n_edges);
    int e;
    for (e = 0; e < sc->n_edges; e++) {
        B.data[sc->edges[e][0] * sc->n_edges + e] = 1.0;
        B.data[sc->edges[e][1] * sc->n_edges + e] = -1.0;
    }
    return B;
}

lau_matrix_t lau_coboundary_operator_d0(const lau_simplicial_complex_t *sc)
{
    lau_matrix_t B = lau_boundary_operator_d0(sc);
    return lau_matrix_transpose(&B);
}

lau_matrix_t lau_dirac_operator(const lau_simplicial_complex_t *sc)
{
    /* D = [  0    d0  ]
           [ d0^T   0  ]  on (0-forms, 1-forms) */
    int n = sc->n_vertices + sc->n_edges;
    lau_matrix_t D = lau_matrix_zeros(n, n);
    lau_matrix_t d0 = lau_coboundary_operator_d0(sc);
    int i, j;

    /* Upper-right block: d0 */
    for (i = 0; i < sc->n_vertices; i++) {
        for (j = 0; j < sc->n_edges; j++) {
            D.data[i * n + (sc->n_vertices + j)] = d0.data[i * sc->n_edges + j];
        }
    }

    /* Lower-left block: d0^T */
    for (i = 0; i < sc->n_edges; i++) {
        for (j = 0; j < sc->n_vertices; j++) {
            D.data[(sc->n_vertices + i) * n + j] = d0.data[j * sc->n_edges + i];
        }
    }

    return D;
}

int lau_cohomology_h0(const lau_simplicial_complex_t *sc,
                      double *basis, int *dim)
{
    /* H^0 = ker(d0^T) ≈ indicator vectors of connected components */
    /* Build adjacency, use union-find to find components */
    int parent[LAU_DIRAC_MAX_SIMPLICES];
    int rank_arr[LAU_DIRAC_MAX_SIMPLICES];
    int i, e;
    int n_comp = 0;
    int comp_id[LAU_DIRAC_MAX_SIMPLICES];

    for (i = 0; i < sc->n_vertices; i++) {
        parent[i] = i;
        rank_arr[i] = 0;
    }

    /* Union-find path compression */
    for (e = 0; e < sc->n_edges; e++) {
        int a = sc->edges[e][0], b = sc->edges[e][1];
        /* Find root of a */
        int ra = a;
        while (parent[ra] != ra) { parent[ra] = parent[parent[ra]]; ra = parent[ra]; }
        /* Find root of b */
        int rb = b;
        while (parent[rb] != rb) { parent[rb] = parent[parent[rb]]; rb = parent[rb]; }
        /* Union */
        if (ra != rb) {
            if (rank_arr[ra] < rank_arr[rb]) { int tmp = ra; ra = rb; rb = tmp; }
            parent[rb] = ra;
            if (rank_arr[ra] == rank_arr[rb]) rank_arr[ra]++;
        }
    }

    /* Compress all paths */
    for (i = 0; i < sc->n_vertices; i++) {
        int r = i;
        while (parent[r] != r) r = parent[r];
        comp_id[i] = r;
    }

    /* Count unique components */
    {
        int roots[LAU_DIRAC_MAX_SIMPLICES];
        int n_roots = 0;
        for (i = 0; i < sc->n_vertices; i++) {
            int found = 0;
            int j2;
            for (j2 = 0; j2 < n_roots; j2++) {
                if (comp_id[i] == roots[j2]) { found = 1; break; }
            }
            if (!found) {
                roots[n_roots] = comp_id[i];
                n_roots++;
            }
        }
        n_comp = n_roots;

        /* Build basis vectors */
        if (basis) {
            int c;
            for (c = 0; c < n_comp; c++) {
                for (i = 0; i < sc->n_vertices; i++) {
                    basis[c * sc->n_vertices + i] = (comp_id[i] == roots[c]) ? 1.0 : 0.0;
                }
            }
        }
    }

    *dim = n_comp;
    return 0;
}

int lau_cohomology_h1(const lau_simplicial_complex_t *sc,
                      double *basis, int *dim)
{
    /* H^1 = ker(d1) / im(d0^T) for 1-forms */
    /* For graphs: b1 = E - V + b0 (Euler's formula for graphs) */
    int b0, b1_val;

    lau_betti_numbers(sc, &b0, &b1_val);

    /* For a proper basis, compute null space of coboundary mod image of boundary */
    /* Simplified: just report dimension */
    *dim = b1_val;

    if (basis && b1_val > 0) {
        /* Use null space of incidence matrix mod image */
        lau_matrix_t B = lau_boundary_operator_d0(sc);
        /* Compute via SVD-lite: eigenvectors of B^T B with zero eigenvalue
         * that are NOT eigenvectors of B B^T with zero eigenvalue */
        /* For simplicity, leave basis computation to the caller via power iteration */
        int i;
        for (i = 0; i < b1_val * sc->n_edges; i++) {
            basis[i] = 0.0;
        }
    }

    return 0;
}

void lau_betti_numbers(const lau_simplicial_complex_t *sc, int *b0, int *b1)
{
    int dim_h0;
    lau_cohomology_h0(sc, NULL, &dim_h0);
    *b0 = dim_h0;
    *b1 = sc->n_edges - sc->n_vertices + dim_h0;
}

int lau_euler_characteristic(const lau_simplicial_complex_t *sc)
{
    return sc->n_vertices - sc->n_edges;
}
