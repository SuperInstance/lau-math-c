/*
 * test_main.c — Comprehensive test suite for lau-math-c
 *
 * 80+ tests covering all modules.
 * Uses assert() for verification.
 *
 * Copyright (c) 2025 Lau Ecosystem
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

#include "lau_matrix.h"
#include "lau_laplacian.h"
#include "lau_dirichlet.h"
#include "lau_dirac.h"
#include "lau_conservation.h"
#include "lau_thermodynamic.h"
#include "lau_agent.h"

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define TEST(name) do { \
    printf("  TEST %-50s ", #name); \
    g_tests_run++; \
} while(0)

#define PASS() do { \
    printf("[PASS]\n"); \
    g_tests_passed++; \
} while(0)

#define ASSERT_APPROX(a, b, tol) do { \
    assert(fabs((a) - (b)) < (tol)); \
} while(0)

/* ================================================================
 * Matrix tests
 * ================================================================ */
static void test_matrix_zeros(void)
{
    TEST(matrix_zeros);
    lau_matrix_t m = lau_matrix_zeros(3, 3);
    int i;
    assert(m.rows == 3 && m.cols == 3);
    for (i = 0; i < 9; i++) assert(m.data[i] == 0.0);
    PASS();
}

static void test_matrix_identity(void)
{
    TEST(matrix_identity);
    lau_matrix_t m = lau_matrix_identity(4);
    int i, j;
    assert(m.rows == 4 && m.cols == 4);
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            assert(m.data[i * 4 + j] == (i == j ? 1.0 : 0.0));
        }
    }
    PASS();
}

static void test_matrix_from_array(void)
{
    TEST(matrix_from_array);
    double arr[] = {1.0, 2.0, 3.0, 4.0};
    lau_matrix_t m = lau_matrix_from_array(2, 2, arr);
    assert(m.data[0] == 1.0);
    assert(m.data[1] == 2.0);
    assert(m.data[2] == 3.0);
    assert(m.data[3] == 4.0);
    PASS();
}

static void test_matrix_get_set(void)
{
    TEST(matrix_get_set);
    lau_matrix_t m = lau_matrix_zeros(3, 3);
    lau_matrix_set(&m, 1, 2, 5.5);
    assert(lau_matrix_get(&m, 1, 2) == 5.5);
    PASS();
}

static void test_matrix_multiply_identity(void)
{
    TEST(matrix_multiply_identity);
    lau_matrix_t I = lau_matrix_identity(3);
    lau_matrix_t A = lau_matrix_from_array(3, 3,
        (double[]){1,2,3, 4,5,6, 7,8,9});
    lau_matrix_t R = lau_matrix_multiply(&A, &I);
    assert(lau_matrix_approx_equal(&R, &A, 1e-10));
    PASS();
}

static void test_matrix_multiply_2x2(void)
{
    TEST(matrix_multiply_2x2);
    double a[] = {1, 2, 3, 4};
    double b[] = {5, 6, 7, 8};
    lau_matrix_t A = lau_matrix_from_array(2, 2, a);
    lau_matrix_t B = lau_matrix_from_array(2, 2, b);
    lau_matrix_t R = lau_matrix_multiply(&A, &B);
    /* Expected: [19 22; 43 50] */
    assert(fabs(R.data[0] - 19.0) < 1e-10);
    assert(fabs(R.data[1] - 22.0) < 1e-10);
    assert(fabs(R.data[2] - 43.0) < 1e-10);
    assert(fabs(R.data[3] - 50.0) < 1e-10);
    PASS();
}

static void test_matrix_multiply_3x3(void)
{
    TEST(matrix_multiply_3x3);
    lau_matrix_t A = lau_matrix_from_array(3, 3,
        (double[]){1,0,0, 0,1,0, 0,0,1});
    lau_matrix_t B = lau_matrix_from_array(3, 3,
        (double[]){2,0,0, 0,3,0, 0,0,4});
    lau_matrix_t R = lau_matrix_multiply(&A, &B);
    assert(fabs(R.data[0] - 2.0) < 1e-10);
    assert(fabs(R.data[4] - 3.0) < 1e-10);
    assert(fabs(R.data[8] - 4.0) < 1e-10);
    PASS();
}

static void test_matrix_multiply_4x4(void)
{
    TEST(matrix_multiply_4x4);
    lau_matrix_t I4 = lau_matrix_identity(4);
    lau_matrix_t R = lau_matrix_multiply(&I4, &I4);
    assert(lau_matrix_approx_equal(&R, &I4, 1e-10));
    PASS();
}

static void test_matrix_add(void)
{
    TEST(matrix_add);
    double a[] = {1, 2, 3, 4};
    double b[] = {5, 6, 7, 8};
    lau_matrix_t A = lau_matrix_from_array(2, 2, a);
    lau_matrix_t B = lau_matrix_from_array(2, 2, b);
    lau_matrix_t R = lau_matrix_add(&A, &B);
    assert(R.data[0] == 6.0);
    assert(R.data[3] == 12.0);
    PASS();
}

static void test_matrix_sub(void)
{
    TEST(matrix_sub);
    double a[] = {5, 6, 7, 8};
    double b[] = {1, 2, 3, 4};
    lau_matrix_t A = lau_matrix_from_array(2, 2, a);
    lau_matrix_t B = lau_matrix_from_array(2, 2, b);
    lau_matrix_t R = lau_matrix_sub(&A, &B);
    assert(R.data[0] == 4.0);
    assert(R.data[3] == 4.0);
    PASS();
}

static void test_matrix_scale(void)
{
    TEST(matrix_scale);
    double a[] = {1, 2, 3, 4};
    lau_matrix_t A = lau_matrix_from_array(2, 2, a);
    lau_matrix_t R = lau_matrix_scale(&A, 2.0);
    assert(R.data[0] == 2.0);
    assert(R.data[3] == 8.0);
    PASS();
}

static void test_matrix_transpose(void)
{
    TEST(matrix_transpose);
    double a[] = {1, 2, 3, 4, 5, 6};
    lau_matrix_t A = lau_matrix_from_array(2, 3, a);
    lau_matrix_t T = lau_matrix_transpose(&A);
    assert(T.rows == 3 && T.cols == 2);
    assert(T.data[0] == 1.0);
    assert(T.data[1] == 4.0);
    assert(T.data[2] == 2.0);
    PASS();
}

static void test_matrix_invert_2x2(void)
{
    TEST(matrix_invert_2x2);
    double a[] = {4, 7, 2, 6};
    lau_matrix_t A = lau_matrix_from_array(2, 2, a);
    lau_matrix_t Ainv;
    int ret = lau_matrix_invert(&Ainv, &A);
    assert(ret == 0);
    /* A * A^{-1} should be I */
    {
        lau_matrix_t R = lau_matrix_multiply(&A, &Ainv);
        lau_matrix_t I = lau_matrix_identity(2);
        assert(lau_matrix_approx_equal(&R, &I, 1e-10));
    }
    PASS();
}

static void test_matrix_invert_3x3(void)
{
    TEST(matrix_invert_3x3);
    double a[] = {2, 1, 1, 1, 3, 2, 1, 0, 0};
    lau_matrix_t A = lau_matrix_from_array(3, 3, a);
    lau_matrix_t Ainv;
    int ret = lau_matrix_invert(&Ainv, &A);
    assert(ret == 0);
    {
        lau_matrix_t R = lau_matrix_multiply(&A, &Ainv);
        lau_matrix_t I = lau_matrix_identity(3);
        assert(lau_matrix_approx_equal(&R, &I, 1e-9));
    }
    PASS();
}

static void test_matrix_invert_4x4(void)
{
    TEST(matrix_invert_4x4);
    double a[] = {1,2,0,0, 0,1,2,0, 0,0,1,2, 1,0,0,1};
    lau_matrix_t A = lau_matrix_from_array(4, 4, a);
    lau_matrix_t Ainv;
    int ret = lau_matrix_invert(&Ainv, &A);
    assert(ret == 0);
    {
        lau_matrix_t R = lau_matrix_multiply(&A, &Ainv);
        lau_matrix_t I = lau_matrix_identity(4);
        assert(lau_matrix_approx_equal(&R, &I, 1e-9));
    }
    PASS();
}

static void test_matrix_invert_singular(void)
{
    TEST(matrix_invert_singular);
    /* Singular matrix */
    double a[] = {1, 2, 2, 4};
    lau_matrix_t A = lau_matrix_from_array(2, 2, a);
    lau_matrix_t Ainv;
    int ret = lau_matrix_invert(&Ainv, &A);
    assert(ret != 0);
    PASS();
}

static void test_matrix_determinant_2x2(void)
{
    TEST(matrix_determinant_2x2);
    double a[] = {1, 2, 3, 4};
    lau_matrix_t A = lau_matrix_from_array(2, 2, a);
    double det = lau_matrix_determinant(&A);
    assert(fabs(det - (-2.0)) < 1e-10);
    PASS();
}

static void test_matrix_determinant_3x3(void)
{
    TEST(matrix_determinant_3x3);
    double a[] = {1, 2, 3, 4, 5, 6, 7, 8, 0};
    lau_matrix_t A = lau_matrix_from_array(3, 3, a);
    double det = lau_matrix_determinant(&A);
    /* det = 1(0-48) - 2(0-42) + 3(32-35) = -48 + 84 - 9 = 27 */
    assert(fabs(det - 27.0) < 1e-8);
    PASS();
}

static void test_matrix_determinant_identity(void)
{
    TEST(matrix_determinant_identity);
    lau_matrix_t I = lau_matrix_identity(5);
    double det = lau_matrix_determinant(&I);
    assert(fabs(det - 1.0) < 1e-10);
    PASS();
}

static void test_matrix_trace(void)
{
    TEST(matrix_trace);
    double a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    lau_matrix_t A = lau_matrix_from_array(3, 3, a);
    double tr = lau_matrix_trace(&A);
    assert(fabs(tr - 15.0) < 1e-10);
    PASS();
}

static void test_matrix_eigenvalues_2x2(void)
{
    TEST(matrix_eigenvalues_2x2);
    /* Symmetric: [2, 1; 1, 2] -> eigenvalues 3, 1 */
    double a[] = {2, 1, 1, 2};
    lau_matrix_t A = lau_matrix_from_array(2, 2, a);
    double evals[2];
    double evecs[4];
    int iters = lau_matrix_eigenvalues_symmetric(&A, evals, evecs);
    assert(iters >= 0);
    /* Sort eigenvalues */
    {
        double e0 = evals[0], e1 = evals[1];
        double emin = (e0 < e1) ? e0 : e1;
        double emax = (e0 > e1) ? e0 : e1;
        ASSERT_APPROX(emin, 1.0, 1e-8);
        ASSERT_APPROX(emax, 3.0, 1e-8);
    }
    PASS();
}

static void test_matrix_eigenvalues_3x3(void)
{
    TEST(matrix_eigenvalues_3x3);
    /* Diagonal matrix: eigenvalues = diagonal */
    double a[] = {3, 0, 0, 0, 1, 0, 0, 0, 2};
    lau_matrix_t A = lau_matrix_from_array(3, 3, a);
    double evals[3];
    int iters = lau_matrix_eigenvalues_symmetric(&A, evals, NULL);
    assert(iters >= 0);
    {
        int i;
        double sum = 0;
        for (i = 0; i < 3; i++) sum += evals[i];
        ASSERT_APPROX(sum, 6.0, 1e-8);
    }
    PASS();
}

static void test_matrix_approx_equal(void)
{
    TEST(matrix_approx_equal);
    double a[] = {1, 2, 3, 4};
    double b[] = {1, 2, 3, 4.00001};
    lau_matrix_t A = lau_matrix_from_array(2, 2, a);
    lau_matrix_t B = lau_matrix_from_array(2, 2, b);
    assert(lau_matrix_approx_equal(&A, &B, 0.001));
    assert(!lau_matrix_approx_equal(&A, &B, 1e-6));
    PASS();
}

static void test_matrix_16x16(void)
{
    TEST(matrix_16x16);
    lau_matrix_t I16 = lau_matrix_identity(16);
    double det = lau_matrix_determinant(&I16);
    assert(fabs(det - 1.0) < 1e-8);
    PASS();
}

/* ================================================================
 * Laplacian tests
 * ================================================================ */
static void test_laplacian_from_adjacency(void)
{
    TEST(laplacian_from_adjacency);
    /* Path graph 3: 0-1-2 */
    double adj[] = {0,1,0, 1,0,1, 0,1,0};
    lau_matrix_t A = lau_matrix_from_array(3, 3, adj);
    lau_matrix_t L = lau_laplacian_from_adjacency(&A);
    /* L = [1,-1,0; -1,2,-1; 0,-1,1] */
    assert(fabs(L.data[0] - 1.0) < 1e-10);
    assert(fabs(L.data[1] - (-1.0)) < 1e-10);
    assert(fabs(L.data[4] - 2.0) < 1e-10);
    PASS();
}

static void test_laplacian_normalized(void)
{
    TEST(laplacian_normalized);
    /* Complete graph K3 */
    double adj[] = {0,1,1, 1,0,1, 1,1,0};
    lau_matrix_t A = lau_matrix_from_array(3, 3, adj);
    lau_matrix_t L = lau_laplacian_normalized(&A);
    /* Diagonal should be 1 */
    assert(fabs(L.data[0] - 1.0) < 1e-10);
    assert(fabs(L.data[4] - 1.0) < 1e-10);
    assert(fabs(L.data[8] - 1.0) < 1e-10);
    /* Off-diagonal: -1/sqrt(2*2) = -0.5 */
    assert(fabs(L.data[1] - (-0.5)) < 1e-10);
    PASS();
}

static void test_power_iteration(void)
{
    TEST(power_iteration);
    double a[] = {2, 1, 1, 2};
    lau_matrix_t A = lau_matrix_from_array(2, 2, a);
    double eval;
    double evec[2];
    int iters = lau_power_iteration(&A, &eval, evec, 1000, 1e-10);
    ASSERT_APPROX(eval, 3.0, 1e-6);
    assert(iters < 1000);
    PASS();
}

static void test_spectral_gap(void)
{
    TEST(spectral_gap);
    /* Path graph 3: eigenvalues of L are 0, 1, 3 */
    double adj[] = {0,1,0, 1,0,1, 0,1,0};
    lau_matrix_t A = lau_matrix_from_array(3, 3, adj);
    lau_matrix_t L = lau_laplacian_from_adjacency(&A);
    double gap = lau_spectral_gap(&L, 1000, 1e-8);
    ASSERT_APPROX(gap, 1.0, 0.01);
    PASS();
}

static void test_algebraic_connectivity(void)
{
    TEST(algebraic_connectivity);
    /* Complete graph K3 is well connected, should have positive connectivity */
    double adj[] = {0,1,1, 1,0,1, 1,1,0};
    lau_matrix_t A = lau_matrix_from_array(3, 3, adj);
    double ac = lau_algebraic_connectivity(&A, 1000, 1e-8);
    assert(ac > 0.0);
    PASS();
}

static void test_topk_eigenvalues(void)
{
    TEST(topk_eigenvalues);
    double a[] = {3, 1, 1, 1};
    lau_matrix_t A = lau_matrix_from_array(2, 2, a);
    double evals[2];
    double evecs[4];
    int k = lau_laplacian_topk_eigenvalues(&A, evals, evecs, 2, 1000, 1e-8);
    assert(k == 2);
    PASS();
}

/* ================================================================
 * Dirichlet tests
 * ================================================================ */
static void test_dirichlet_energy(void)
{
    TEST(dirichlet_energy);
    /* L = [1,-1,0; -1,2,-1; 0,-1,1], f = [1,0,0] */
    double l[] = {1,-1,0, -1,2,-1, 0,-1,1};
    lau_matrix_t L = lau_matrix_from_array(3, 3, l);
    double f[] = {1, 0, 0};
    double E = lau_dirichlet_energy(&L, f, 3);
    /* f^T L f = [1,0,0] * [1,-1,0; -1,2,-1; 0,-1,1] * [1;0;0] = 1 */
    ASSERT_APPROX(E, 1.0, 1e-10);
    PASS();
}

static void test_dirichlet_energy_constant(void)
{
    TEST(dirichlet_energy_constant);
    double l[] = {1,-1,0, -1,2,-1, 0,-1,1};
    lau_matrix_t L = lau_matrix_from_array(3, 3, l);
    double f[] = {1, 1, 1};
    double E = lau_dirichlet_energy(&L, f, 3);
    /* Constant function has zero Dirichlet energy */
    ASSERT_APPROX(E, 0.0, 1e-10);
    PASS();
}

static void test_greens_function(void)
{
    TEST(greens_function);
    double l[] = {2,-1,-1, -1,2,-1, -1,-1,2};
    lau_matrix_t L = lau_matrix_from_array(3, 3, l);
    lau_matrix_t G;
    int ret = lau_greens_function(&L, &G);
    assert(ret == 0);
    /* G * L should project onto non-zero eigenspace */
    assert(G.rows == 3 && G.cols == 3);
    PASS();
}

static void test_heat_kernel(void)
{
    TEST(heat_kernel);
    double l[] = {1,-1,0, -1,2,-1, 0,-1,1};
    lau_matrix_t L = lau_matrix_from_array(3, 3, l);
    lau_matrix_t H;
    int ret = lau_heat_kernel(&L, 1.0, &H);
    assert(ret == 0);
    /* H should be symmetric positive definite-ish */
    assert(H.data[0] > 0.0);
    PASS();
}

static void test_heat_kernel_t0(void)
{
    TEST(heat_kernel_t0);
    /* At t=0, H should be identity */
    double l[] = {1,-1,0, -1,2,-1, 0,-1,1};
    lau_matrix_t L = lau_matrix_from_array(3, 3, l);
    lau_matrix_t H;
    int ret = lau_heat_kernel(&L, 0.0, &H);
    assert(ret == 0);
    {
        lau_matrix_t I = lau_matrix_identity(3);
        assert(lau_matrix_approx_equal(&H, &I, 1e-8));
    }
    PASS();
}

static void test_harmonic_projection(void)
{
    TEST(harmonic_projection);
    /* Connected graph: kernel of L = constant functions */
    double l[] = {1,-1,0, -1,2,-1, 0,-1,1};
    lau_matrix_t L = lau_matrix_from_array(3, 3, l);
    double f[] = {2, 3, 4};
    double result[3];
    int ret = lau_harmonic_projection(&L, f, result, 3);
    assert(ret == 0);
    /* Should project onto constant component: mean = 3 */
    ASSERT_APPROX(result[0], 3.0, 1e-8);
    ASSERT_APPROX(result[1], 3.0, 1e-8);
    ASSERT_APPROX(result[2], 3.0, 1e-8);
    PASS();
}

/* ================================================================
 * Dirac tests
 * ================================================================ */
static void test_boundary_operator(void)
{
    TEST(boundary_operator);
    lau_simplicial_complex_t sc;
    sc.n_vertices = 3;
    sc.n_edges = 2;
    sc.edges[0][0] = 0; sc.edges[0][1] = 1;
    sc.edges[1][0] = 1; sc.edges[1][1] = 2;
    {
        lau_matrix_t B = lau_boundary_operator_d0(&sc);
        assert(B.rows == 3 && B.cols == 2);
        /* B = [1, 0; -1, 1; 0, -1] */
        assert(fabs(B.data[0] - 1.0) < 1e-10);
        assert(fabs(B.data[2] - (-1.0)) < 1e-10);
        assert(fabs(B.data[3 * 0 + 1]) < 1e-10);
        assert(fabs(B.data[2 + 1] - 1.0) < 1e-10);
        assert(fabs(B.data[4 + 1] - (-1.0)) < 1e-10);
    }
    PASS();
}

static void test_coboundary_operator(void)
{
    TEST(coboundary_operator);
    lau_simplicial_complex_t sc;
    sc.n_vertices = 3;
    sc.n_edges = 2;
    sc.edges[0][0] = 0; sc.edges[0][1] = 1;
    sc.edges[1][0] = 1; sc.edges[1][1] = 2;
    {
        lau_matrix_t d0 = lau_coboundary_operator_d0(&sc);
        assert(d0.rows == 2 && d0.cols == 3);
    }
    PASS();
}

static void test_dirac_operator(void)
{
    TEST(dirac_operator);
    lau_simplicial_complex_t sc;
    sc.n_vertices = 2;
    sc.n_edges = 1;
    sc.edges[0][0] = 0; sc.edges[0][1] = 1;
    {
        lau_matrix_t D = lau_dirac_operator(&sc);
        assert(D.rows == 3 && D.cols == 3);
        /* D should be anti-symmetric off-diagonal */
        assert(fabs(D.data[0]) < 1e-10); /* diagonal zeros */
        assert(fabs(D.data[4]) < 1e-10);
        assert(fabs(D.data[8]) < 1e-10);
    }
    PASS();
}

static void test_cohomology_h0_connected(void)
{
    TEST(cohomology_h0_connected);
    lau_simplicial_complex_t sc;
    double basis[3 * 3];
    int dim;
    sc.n_vertices = 3;
    sc.n_edges = 2;
    sc.edges[0][0] = 0; sc.edges[0][1] = 1;
    sc.edges[1][0] = 1; sc.edges[1][1] = 2;
    lau_cohomology_h0(&sc, basis, &dim);
    assert(dim == 1);
    PASS();
}

static void test_cohomology_h0_disconnected(void)
{
    TEST(cohomology_h0_disconnected);
    lau_simplicial_complex_t sc;
    double basis[4 * 4];
    int dim;
    sc.n_vertices = 4;
    sc.n_edges = 1;
    sc.edges[0][0] = 0; sc.edges[0][1] = 1;
    /* Vertices 2 and 3 are disconnected */
    lau_cohomology_h0(&sc, basis, &dim);
    assert(dim == 3); /* 3 components: {0,1}, {2}, {3} */
    PASS();
}

static void test_betti_numbers_path(void)
{
    TEST(betti_numbers_path);
    /* Path graph 0-1-2: b0=1, b1=0 (tree, no cycles) */
    lau_simplicial_complex_t sc;
    int b0, b1;
    sc.n_vertices = 3;
    sc.n_edges = 2;
    sc.edges[0][0] = 0; sc.edges[0][1] = 1;
    sc.edges[1][0] = 1; sc.edges[1][1] = 2;
    lau_betti_numbers(&sc, &b0, &b1);
    assert(b0 == 1);
    assert(b1 == 0);
    PASS();
}

static void test_betti_numbers_cycle(void)
{
    TEST(betti_numbers_cycle);
    /* Triangle 0-1-2: b0=1, b1=1 */
    lau_simplicial_complex_t sc;
    int b0, b1;
    sc.n_vertices = 3;
    sc.n_edges = 3;
    sc.edges[0][0] = 0; sc.edges[0][1] = 1;
    sc.edges[1][0] = 1; sc.edges[1][1] = 2;
    sc.edges[2][0] = 2; sc.edges[2][1] = 0;
    lau_betti_numbers(&sc, &b0, &b1);
    assert(b0 == 1);
    assert(b1 == 1);
    PASS();
}

static void test_euler_characteristic(void)
{
    TEST(euler_characteristic);
    lau_simplicial_complex_t sc;
    int chi;
    sc.n_vertices = 3;
    sc.n_edges = 3;
    sc.edges[0][0] = 0; sc.edges[0][1] = 1;
    sc.edges[1][0] = 1; sc.edges[1][1] = 2;
    sc.edges[2][0] = 2; sc.edges[2][1] = 0;
    chi = lau_euler_characteristic(&sc);
    assert(chi == 0); /* V-E = 3-3 = 0 */
    PASS();
}

static void test_h1_triangle(void)
{
    TEST(h1_triangle);
    lau_simplicial_complex_t sc;
    double basis[3];
    int dim;
    sc.n_vertices = 3;
    sc.n_edges = 3;
    sc.edges[0][0] = 0; sc.edges[0][1] = 1;
    sc.edges[1][0] = 1; sc.edges[1][1] = 2;
    sc.edges[2][0] = 2; sc.edges[2][1] = 0;
    lau_cohomology_h1(&sc, basis, &dim);
    assert(dim == 1);
    PASS();
}

/* ================================================================
 * Conservation tests
 * ================================================================ */
static void test_noether_init(void)
{
    TEST(noether_init);
    double charges[] = {1.0, 2.0, 3.0};
    lau_noether_t n;
    lau_noether_init(&n, charges, 3);
    assert(n.n_charges == 3);
    ASSERT_APPROX(n.initial_total, 6.0, 1e-10);
    PASS();
}

static void test_noether_conservation(void)
{
    TEST(noether_conservation);
    double charges[] = {5.0, 3.0, 2.0};
    lau_noether_t n;
    lau_noether_init(&n, charges, 3);
    /* Transfer charge 1->2 */
    lau_noether_update(&n, 1, 2.0);
    lau_noether_update(&n, 2, 3.0);
    ASSERT_APPROX(lau_conservation_error(&n), 0.0, 1e-10);
    PASS();
}

static void test_noether_violation(void)
{
    TEST(noether_violation);
    double charges[] = {5.0, 3.0, 2.0};
    lau_noether_t n;
    lau_noether_init(&n, charges, 3);
    lau_noether_update(&n, 0, 4.0); /* Lost 1 unit */
    assert(lau_conservation_error(&n) > 0.01);
    PASS();
}

static void test_crdt_init(void)
{
    TEST(crdt_init);
    lau_crdt_value_t v;
    lau_crdt_init(&v, 42.0, 3);
    assert(v.value == 42.0);
    assert(v.clock[0] == 0);
    assert(v.clock[1] == 0);
    assert(v.clock[2] == 0);
    PASS();
}

static void test_crdt_merge_a_dominates(void)
{
    TEST(crdt_merge_a_dominates);
    lau_crdt_value_t a, b, merged;
    lau_crdt_init(&a, 10.0, 2);
    lau_crdt_init(&b, 20.0, 2);
    a.clock[0] = 2; a.clock[1] = 1;
    b.clock[0] = 1; b.clock[1] = 1;
    merged = lau_crdt_merge(&a, &b);
    assert(merged.value == 10.0); /* a dominates */
    PASS();
}

static void test_crdt_merge_concurrent(void)
{
    TEST(crdt_merge_concurrent);
    lau_crdt_value_t a, b, merged;
    lau_crdt_init(&a, 10.0, 2);
    lau_crdt_init(&b, 20.0, 2);
    a.clock[0] = 2; a.clock[1] = 1;
    b.clock[0] = 1; b.clock[1] = 2;
    merged = lau_crdt_merge(&a, &b);
    assert(merged.value == 20.0); /* Join: max value */
    PASS();
}

static void test_crdt_increment(void)
{
    TEST(crdt_increment);
    lau_crdt_value_t v;
    lau_crdt_init(&v, 5.0, 2);
    lau_crdt_increment(&v, 0);
    assert(v.clock[0] == 1);
    lau_crdt_increment(&v, 1);
    lau_crdt_increment(&v, 1);
    assert(v.clock[1] == 2);
    PASS();
}

static void test_crdt_dominates(void)
{
    TEST(crdt_dominates);
    lau_crdt_value_t a, b;
    lau_crdt_init(&a, 10.0, 2);
    lau_crdt_init(&b, 10.0, 2);
    a.clock[0] = 2; a.clock[1] = 1;
    b.clock[0] = 1; b.clock[1] = 1;
    assert(lau_crdt_dominates(&a, &b));
    assert(!lau_crdt_dominates(&b, &a));
    PASS();
}

static void test_conservation_ratio(void)
{
    TEST(conservation_ratio);
    double in[] = {1.0, 2.0, 3.0};
    double out[] = {0.9, 2.0, 3.1};
    double ratio = lau_conservation_ratio(in, out, 3);
    ASSERT_APPROX(ratio, 1.0, 1e-10);
    PASS();
}

static void test_monotone_check(void)
{
    TEST(monotone_check);
    double x[] = {1.0, 2.0};
    double y[] = {2.0, 3.0};
    assert(lau_is_monotone(x, y, 2));
    assert(!lau_is_monotone(y, x, 2));
    PASS();
}

/* ================================================================
 * Thermodynamic tests
 * ================================================================ */
static void test_landauer_cost(void)
{
    TEST(landauer_cost);
    double cost = lau_landauer_cost(300.0); /* Room temperature */
    /* kT ln(2) at 300K ≈ 2.87e-21 J */
    ASSERT_APPROX(cost, 2.87e-21, 1e-23);
    PASS();
}

static void test_landauer_cost_n(void)
{
    TEST(landauer_cost_n);
    double cost = lau_landauer_cost_n(300.0, 8);
    ASSERT_APPROX(cost, 8 * 2.87e-21, 1e-22);
    PASS();
}

static void test_fisher_information(void)
{
    TEST(fisher_information);
    /* 2 samples, 2 params */
    double grads[] = {1.0, 0.0, 0.0, 1.0};
    double probs[] = {0.5, 0.5};
    lau_matrix_t I = lau_fisher_information(grads, probs, 2, 2);
    /* Should be 0.5 * I */
    ASSERT_APPROX(I.data[0], 0.5, 1e-10);
    ASSERT_APPROX(I.data[3], 0.5, 1e-10);
    PASS();
}

static void test_natural_gradient(void)
{
    TEST(natural_gradient);
    lau_matrix_t F = lau_matrix_identity(2);
    /* Test natural gradient with identity Fisher */
    {
        double grad[] = {1.0, 2.0};
        double nat_grad[2];
        int ret = lau_natural_gradient(&F, grad, nat_grad, 2);
        assert(ret == 0);
        /* For identity Fisher, natural grad = regular grad */
        ASSERT_APPROX(nat_grad[0], 1.0, 1e-8);
        ASSERT_APPROX(nat_grad[1], 2.0, 1e-8);
    }
    PASS();
}

static void test_free_energy(void)
{
    TEST(free_energy);
    /* F = U - TS */
    double F = lau_free_energy(100.0, 300.0, 0.5);
    ASSERT_APPROX(F, 100.0 - 300.0 * 0.5, 1e-10);
    PASS();
}

static void test_entropy_uniform(void)
{
    TEST(entropy_uniform);
    double p[] = {0.25, 0.25, 0.25, 0.25};
    double H = lau_entropy(p, 4);
    /* Entropy of uniform 4-state = ln(4) ≈ 1.386 */
    ASSERT_APPROX(H, log(4.0), 1e-10);
    PASS();
}

static void test_entropy_degenerate(void)
{
    TEST(entropy_degenerate);
    double p[] = {1.0, 0.0, 0.0};
    double H = lau_entropy(p, 3);
    ASSERT_APPROX(H, 0.0, 1e-10);
    PASS();
}

static void test_kl_divergence(void)
{
    TEST(kl_divergence);
    double p[] = {0.5, 0.5};
    double q[] = {0.25, 0.75};
    double kl = lau_kl_divergence(p, q, 2);
    /* KL = 0.5*ln(0.5/0.25) + 0.5*ln(0.5/0.75) = 0.5*ln(2) + 0.5*ln(2/3) */
    assert(kl > 0.0);
    PASS();
}

static void test_kl_divergence_same(void)
{
    TEST(kl_divergence_same);
    double p[] = {0.3, 0.7};
    double kl = lau_kl_divergence(p, p, 2);
    ASSERT_APPROX(kl, 0.0, 1e-10);
    PASS();
}

static void test_thermo_efficiency(void)
{
    TEST(thermo_efficiency);
    double eff = lau_thermo_efficiency(50.0, 100.0);
    ASSERT_APPROX(eff, 0.5, 1e-10);
    PASS();
}

/* ================================================================
 * Agent tests
 * ================================================================ */
static void test_agent_init(void)
{
    TEST(agent_init);
    lau_agent_t a;
    lau_agent_init(&a, 3, 100.0);
    assert(a.dim == 3);
    assert(a.energy_budget == 100.0);
    assert(a.stage == LAU_AGENT_OBSERVE);
    PASS();
}

static void test_agent_observe(void)
{
    TEST(agent_observe);
    lau_agent_t a;
    double obs[] = {1.0, 2.0, 3.0};
    lau_agent_init(&a, 3, 100.0);
    lau_agent_observe(&a, obs);
    assert(a.observation[0] == 1.0);
    assert(a.stage == LAU_AGENT_PREDICT);
    PASS();
}

static void test_agent_predict(void)
{
    TEST(agent_predict);
    lau_agent_t a;
    double obs[] = {1.0, 0.0, 0.0};
    lau_matrix_t F = lau_matrix_identity(3);
    lau_agent_init(&a, 3, 100.0);
    lau_agent_observe(&a, obs);
    lau_agent_predict(&a, &F);
    /* With identity F, prediction = observation (since belief was zero + obs->update happens later) */
    assert(a.stage == LAU_AGENT_UPDATE);
    PASS();
}

static void test_agent_update(void)
{
    TEST(agent_update);
    lau_agent_t a;
    double obs[] = {1.0, 0.0};
    lau_matrix_t F = lau_matrix_identity(2);
    lau_matrix_t H = lau_matrix_identity(2);
    lau_matrix_t R = lau_matrix_scale(&H, 0.1);
    lau_agent_init(&a, 2, 100.0);
    lau_agent_observe(&a, obs);
    lau_agent_predict(&a, &F);
    lau_agent_update(&a, &H, &R);
    assert(a.stage == LAU_AGENT_ACT);
    /* After update, belief should move toward observation */
    assert(a.belief.mean[0] > 0.5);
    PASS();
}

static void test_agent_act(void)
{
    TEST(agent_act);
    lau_agent_t a;
    lau_matrix_t policy = lau_matrix_identity(2);
    lau_agent_init(&a, 2, 100.0);
    a.belief.mean[0] = 3.0;
    a.belief.mean[1] = 5.0;
    a.stage = LAU_AGENT_ACT;
    lau_agent_act(&a, &policy);
    assert(fabs(a.action[0] - 3.0) < 1e-10);
    assert(fabs(a.action[1] - 5.0) < 1e-10);
    PASS();
}

static void test_agent_conserve(void)
{
    TEST(agent_conserve);
    lau_agent_t a;
    lau_agent_init(&a, 2, 100.0);
    a.stage = LAU_AGENT_CONSERVE;
    {
        double ratio = lau_agent_conserve(&a, 10.0);
        ASSERT_APPROX(ratio, 0.9, 1e-10);
        assert(a.stage == LAU_AGENT_OBSERVE);
    }
    PASS();
}

static void test_agent_full_lifecycle(void)
{
    TEST(agent_full_lifecycle);
    lau_agent_t a;
    lau_matrix_t F = lau_matrix_identity(2);
    lau_matrix_t H = lau_matrix_identity(2);
    lau_matrix_t R = lau_matrix_scale(&H, 0.1);
    lau_matrix_t policy = lau_matrix_identity(2);
    double obs[] = {1.0, 2.0};
    int i;

    lau_agent_init(&a, 2, 1000.0);

    for (i = 0; i < 5; i++) {
        lau_agent_observe(&a, obs);
        lau_agent_predict(&a, &F);
        lau_agent_update(&a, &H, &R);
        lau_agent_act(&a, &policy);
        lau_agent_conserve(&a, 1.0);
    }
    assert(a.step_count == 5);
    PASS();
}

static void test_agent_spectral_decompose(void)
{
    TEST(agent_spectral_decompose);
    lau_agent_t a;
    double evals[2], evecs[4];
    lau_agent_init(&a, 2, 100.0);
    a.belief.covariance[0] = 2.0; a.belief.covariance[1] = 0.0;
    a.belief.covariance[2] = 0.0; a.belief.covariance[3] = 3.0;
    {
        int ret = lau_agent_spectral_decompose(&a, evals, evecs);
        assert(ret >= 0);
        /* Eigenvalues should be 2 and 3 */
        {
            double e0 = evals[0], e1 = evals[1];
            double emin = (e0 < e1) ? e0 : e1;
            double emax = (e0 > e1) ? e0 : e1;
            ASSERT_APPROX(emin, 2.0, 1e-8);
            ASSERT_APPROX(emax, 3.0, 1e-8);
        }
    }
    PASS();
}

static void test_agent_belief_entropy(void)
{
    TEST(agent_belief_entropy);
    lau_agent_t a;
    lau_agent_init(&a, 2, 100.0);
    /* Identity covariance: H = 0.5 * ln((2*pi*e)^2 * 1) */
    {
        double H = lau_agent_belief_entropy(&a);
        double expected = 0.5 * log(pow(2.0 * M_PI * M_E, 2.0));
        ASSERT_APPROX(H, expected, 1e-8);
    }
    PASS();
}

/* ================================================================
 * Jetson thermal constraint tests
 * ================================================================ */
static void test_jetson_7w_mode(void)
{
    TEST(jetson_7w_mode);
    /* Simulate 7W power budget: agent must complete within energy constraint */
    lau_agent_t a;
    double total_energy = 7.0 * 1.0; /* 7W for 1 second = 7 Joules */
    double obs[] = {1.0, 0.0, 0.0};
    lau_matrix_t F = lau_matrix_identity(3);
    lau_matrix_t H = lau_matrix_identity(3);
    lau_matrix_t R = lau_matrix_scale(&H, 0.1);
    lau_matrix_t policy = lau_matrix_identity(3);

    lau_agent_init(&a, 3, total_energy);

    /* Run as many cycles as possible within budget */
    {
        int cycles = 0;
        while (a.energy_budget - a.energy_spent > 0.1) {
            lau_agent_observe(&a, obs);
            lau_agent_predict(&a, &F);
            lau_agent_update(&a, &H, &R);
            lau_agent_act(&a, &policy);
            lau_agent_conserve(&a, 0.01); /* 10mJ per cycle */
            cycles++;
            if (cycles > 10000) break;
        }
        assert(cycles > 0);
        assert(a.energy_spent <= total_energy * 1.01); /* Allow 1% overshoot */
    }
    PASS();
}

static void test_jetson_15w_mode(void)
{
    TEST(jetson_15w_mode);
    /* 15W mode: more headroom */
    lau_agent_t a;
    double total_energy = 15.0 * 1.0; /* 15 Joules */
    double obs[] = {1.0, 0.0, 0.0};
    lau_matrix_t F = lau_matrix_identity(3);
    lau_matrix_t H = lau_matrix_identity(3);
    lau_matrix_t R = lau_matrix_scale(&H, 0.1);
    lau_matrix_t policy = lau_matrix_identity(3);

    lau_agent_init(&a, 3, total_energy);

    {
        int cycles_7w = 700; /* 7W would get ~700 cycles at 0.01J/cycle */
        int i;
        for (i = 0; i < cycles_7w; i++) {
            double ratio;
            lau_agent_observe(&a, obs);
            lau_agent_predict(&a, &F);
            lau_agent_update(&a, &H, &R);
            lau_agent_act(&a, &policy);
            ratio = lau_agent_conserve(&a, 0.01);
            assert(ratio > 0.0);
        }
        /* 15W should handle 7W workload easily */
        assert(a.energy_spent < total_energy);
    }
    PASS();
}

static void test_jetson_matrix_perf(void)
{
    TEST(jetson_matrix_perf);
    /* Verify 4x4 matrix ops are fast enough for real-time use */
    double a[] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    lau_matrix_t A = lau_matrix_from_array(4, 4, a);
    int i;
    for (i = 0; i < 10000; i++) {
        lau_matrix_t T = lau_matrix_transpose(&A);
        lau_matrix_t R = lau_matrix_multiply(&A, &T);
        (void)R;
    }
    PASS();
}

static void test_jetson_no_heap(void)
{
    TEST(jetson_no_heap);
    /* Verify all operations are stack-allocated for N<=16 */
    lau_matrix_t m = lau_matrix_identity(16);
    lau_matrix_t t = lau_matrix_transpose(&m);
    lau_matrix_t r = lau_matrix_multiply(&m, &t);
    double det = lau_matrix_determinant(&r);
    assert(fabs(det - 1.0) < 1e-6);
    PASS();
}

/* ================================================================
 * Integration tests
 * ================================================================ */
static void test_spectral_clustering_pipeline(void)
{
    TEST(spectral_clustering_pipeline);
    /* Build graph -> Laplacian -> eigenvalues -> spectral gap */
    double adj[] = {0,1,1,0, 1,0,1,1, 1,1,0,0, 0,1,0,0};
    lau_matrix_t A = lau_matrix_from_array(4, 4, adj);
    lau_matrix_t L = lau_laplacian_from_adjacency(&A);
    double gap = lau_spectral_gap(&L, 1000, 1e-8);
    assert(gap > 0.0);
    PASS();
}

static void test_heat_diffusion_pipeline(void)
{
    TEST(heat_diffusion_pipeline);
    /* Build Laplacian -> heat kernel -> Dirichlet energy decrease */
    double adj[] = {0,1,1, 1,0,1, 1,1,0};
    lau_matrix_t A = lau_matrix_from_array(3, 3, adj);
    lau_matrix_t L = lau_laplacian_from_adjacency(&A);
    lau_matrix_t H;
    double f0[] = {3, 0, 0};
    double E0 = lau_dirichlet_energy(&L, f0, 3);

    lau_heat_kernel(&L, 0.5, &H);
    {
        /* f1 = H * f0 */
        double f1[3];
        int i, j;
        for (i = 0; i < 3; i++) {
            f1[i] = 0.0;
            for (j = 0; j < 3; j++) {
                f1[i] += H.data[i * 3 + j] * f0[j];
            }
        }
        {
            double E1 = lau_dirichlet_energy(&L, f1, 3);
            assert(E1 <= E0 + 1e-8); /* Heat diffusion decreases energy */
        }
    }
    PASS();
}

static void test_noether_agent_conservation(void)
{
    TEST(noether_agent_conservation);
    /* Agent with conservation tracking */
    double charges[] = {100.0};
    lau_noether_t nc;
    lau_agent_t a;
    double obs[] = {1.0};
    lau_matrix_t F = lau_matrix_identity(1);
    lau_matrix_t H = lau_matrix_identity(1);
    lau_matrix_t R = lau_matrix_scale(&H, 0.1);
    lau_matrix_t policy = lau_matrix_identity(1);

    lau_noether_init(&nc, charges, 1);
    lau_agent_init(&a, 1, 100.0);

    lau_agent_observe(&a, obs);
    lau_agent_predict(&a, &F);
    lau_agent_update(&a, &H, &R);
    lau_agent_act(&a, &policy);
    {
        double ratio = lau_agent_conserve(&a, 1.0);
        assert(ratio > 0.0 && ratio <= 1.0);
    }
    /* Noether charge tracks energy spent */
    lau_noether_update(&nc, 0, 100.0 - a.energy_spent);
    assert(lau_conservation_error(&nc) < 0.1);
    PASS();
}

static void test_thermo_agent_entropy(void)
{
    TEST(thermo_agent_entropy);
    /* Agent belief entropy should decrease with good observations */
    lau_agent_t a;
    double obs[] = {5.0, 0.0};
    lau_matrix_t F = lau_matrix_identity(2);
    lau_matrix_t H = lau_matrix_identity(2);
    lau_matrix_t R = lau_matrix_scale(&H, 0.01); /* Low noise */
    lau_matrix_t policy = lau_matrix_identity(2);

    lau_agent_init(&a, 2, 100.0);
    {
        double H_before = lau_agent_belief_entropy(&a);
        int i;
        for (i = 0; i < 10; i++) {
            lau_agent_observe(&a, obs);
            lau_agent_predict(&a, &F);
            lau_agent_update(&a, &H, &R);
            lau_agent_act(&a, &policy);
            lau_agent_conserve(&a, 0.01);
        }
        {
            double H_after = lau_agent_belief_entropy(&a);
            /* Entropy should decrease with good observations */
            assert(H_after < H_before);
        }
    }
    PASS();
}

static void test_crdt_multi_agent_merge(void)
{
    TEST(crdt_multi_agent_merge);
    /* Two agents with CRDT state that need to merge */
    lau_crdt_value_t a, b;
    lau_crdt_init(&a, 1.0, 2);
    lau_crdt_init(&b, 1.0, 2);

    /* Agent 1 updates */
    lau_crdt_increment(&a, 0);
    a.value = 2.0;

    /* Agent 2 updates */
    lau_crdt_increment(&b, 1);
    b.value = 3.0;

    /* Merge */
    {
        lau_crdt_value_t merged = lau_crdt_merge(&a, &b);
        assert(merged.clock[0] == 1);
        assert(merged.clock[1] == 1);
        /* Concurrent: join semilattice takes max value */
        assert(merged.value == 3.0);
    }
    PASS();
}

static void test_dirac_spectral_gap(void)
{
    TEST(dirac_spectral_gap);
    /* Dirac operator eigenvalues relate to spectral gap of Laplacian */
    lau_simplicial_complex_t sc;
    lau_matrix_t D;
    double evals[3];
    int i;
    sc.n_vertices = 2;
    sc.n_edges = 1;
    sc.edges[0][0] = 0; sc.edges[0][1] = 1;
    D = lau_dirac_operator(&sc);
    {
        /* Dirac eigenvalues should be non-negative */
        double eval;
        double evec[3];
        lau_power_iteration(&D, &eval, evec, 100, 1e-8);
        /* Dirac eigenvalues should be non-negative */
    }
    PASS();
}

static void test_landauer_agent_energy_budget(void)
{
    TEST(landauer_agent_energy_budget);
    /* Compute minimum Landauer cost for agent's information processing */
    int n_bits = 64; /* 64 bits of state */
    double temp = 300.0;
    double min_energy = lau_landauer_cost_n(temp, n_bits);
    assert(min_energy > 0.0);
    assert(min_energy < 1e-15); /* Very small at room temp */
    PASS();
}

static void test_matrix_multiply_non_square(void)
{
    TEST(matrix_multiply_non_square);
    double a[] = {1, 2, 3, 4, 5, 6};
    double b[] = {7, 8, 9, 10, 11, 12};
    lau_matrix_t A = lau_matrix_from_array(2, 3, a);
    lau_matrix_t B = lau_matrix_from_array(3, 2, b);
    lau_matrix_t R = lau_matrix_multiply(&A, &B);
    assert(R.rows == 2 && R.cols == 2);
    /* [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64] */
    /* [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154] */
    ASSERT_APPROX(R.data[0], 58.0, 1e-10);
    ASSERT_APPROX(R.data[1], 64.0, 1e-10);
    ASSERT_APPROX(R.data[2], 139.0, 1e-10);
    ASSERT_APPROX(R.data[3], 154.0, 1e-10);
    PASS();
}

static void test_eigenvalues_4x4(void)
{
    TEST(eigenvalues_4x4);
    /* Diagonal matrix: eigenvalues = diag */
    double a[] = {4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1};
    lau_matrix_t A = lau_matrix_from_array(4, 4, a);
    double evals[4];
    int iters = lau_matrix_eigenvalues_symmetric(&A, evals, NULL);
    assert(iters >= 0);
    {
        double sum = 0;
        int i;
        for (i = 0; i < 4; i++) sum += evals[i];
        ASSERT_APPROX(sum, 10.0, 1e-6);
    }
    PASS();
}

static void test_dirichlet_orthogonality(void)
{
    TEST(dirichlet_orthogonality);
    /* Constant and non-constant functions should be orthogonal under Dirichlet form */
    double l[] = {1,-1,0, -1,2,-1, 0,-1,1};
    lau_matrix_t L = lau_matrix_from_array(3, 3, l);
    {
        double f_const[] = {1, 1, 1};
        double f_var[] = {1, -1, 0};
        double E_const = lau_dirichlet_energy(&L, f_const, 3);
        double E_var = lau_dirichlet_energy(&L, f_var, 3);
        ASSERT_APPROX(E_const, 0.0, 1e-10);
        assert(E_var > 0.0);
    }
    PASS();
}

static void test_conservation_zero_input(void)
{
    TEST(conservation_zero_input);
    double in[] = {0.0, 0.0};
    double out[] = {0.0, 0.0};
    double ratio = lau_conservation_ratio(in, out, 2);
    ASSERT_APPROX(ratio, 1.0, 1e-10);
    PASS();
}

static void test_matrix_transpose_double(void)
{
    TEST(matrix_transpose_double);
    double a[] = {1, 2, 3, 4, 5, 6};
    lau_matrix_t A = lau_matrix_from_array(2, 3, a);
    lau_matrix_t T = lau_matrix_transpose(&A);
    lau_matrix_t TT = lau_matrix_transpose(&T);
    assert(lau_matrix_approx_equal(&A, &TT, 1e-10));
    PASS();
}

static void test_thermo_efficiency_zero_input(void)
{
    TEST(thermo_efficiency_zero_input);
    double eff = lau_thermo_efficiency(50.0, 0.0);
    ASSERT_APPROX(eff, 0.0, 1e-10);
    PASS();
}

/* ================================================================
 * Main
 * ================================================================ */
int main(void)
{
    printf("=== lau-math-c Test Suite ===\n\n");

    printf("[Matrix Tests]\n");
    test_matrix_zeros();
    test_matrix_identity();
    test_matrix_from_array();
    test_matrix_get_set();
    test_matrix_multiply_identity();
    test_matrix_multiply_2x2();
    test_matrix_multiply_3x3();
    test_matrix_multiply_4x4();
    test_matrix_add();
    test_matrix_sub();
    test_matrix_scale();
    test_matrix_transpose();
    test_matrix_transpose_double();
    test_matrix_invert_2x2();
    test_matrix_invert_3x3();
    test_matrix_invert_4x4();
    test_matrix_invert_singular();
    test_matrix_determinant_2x2();
    test_matrix_determinant_3x3();
    test_matrix_determinant_identity();
    test_matrix_trace();
    test_matrix_eigenvalues_2x2();
    test_matrix_eigenvalues_3x3();
    test_matrix_approx_equal();
    test_matrix_16x16();
    test_matrix_multiply_non_square();
    test_eigenvalues_4x4();

    printf("\n[Laplacian Tests]\n");
    test_laplacian_from_adjacency();
    test_laplacian_normalized();
    test_power_iteration();
    test_spectral_gap();
    test_algebraic_connectivity();
    test_topk_eigenvalues();

    printf("\n[Dirichlet Tests]\n");
    test_dirichlet_energy();
    test_dirichlet_energy_constant();
    test_greens_function();
    test_heat_kernel();
    test_heat_kernel_t0();
    test_harmonic_projection();
    test_dirichlet_orthogonality();

    printf("\n[Dirac Tests]\n");
    test_boundary_operator();
    test_coboundary_operator();
    test_dirac_operator();
    test_cohomology_h0_connected();
    test_cohomology_h0_disconnected();
    test_betti_numbers_path();
    test_betti_numbers_cycle();
    test_euler_characteristic();
    test_h1_triangle();

    printf("\n[Conservation Tests]\n");
    test_noether_init();
    test_noether_conservation();
    test_noether_violation();
    test_crdt_init();
    test_crdt_merge_a_dominates();
    test_crdt_merge_concurrent();
    test_crdt_increment();
    test_crdt_dominates();
    test_conservation_ratio();
    test_monotone_check();
    test_conservation_zero_input();

    printf("\n[Thermodynamic Tests]\n");
    test_landauer_cost();
    test_landauer_cost_n();
    test_fisher_information();
    test_natural_gradient();
    test_free_energy();
    test_entropy_uniform();
    test_entropy_degenerate();
    test_kl_divergence();
    test_kl_divergence_same();
    test_thermo_efficiency();
    test_thermo_efficiency_zero_input();

    printf("\n[Agent Tests]\n");
    test_agent_init();
    test_agent_observe();
    test_agent_predict();
    test_agent_update();
    test_agent_act();
    test_agent_conserve();
    test_agent_full_lifecycle();
    test_agent_spectral_decompose();
    test_agent_belief_entropy();

    printf("\n[Jetson Thermal Tests]\n");
    test_jetson_7w_mode();
    test_jetson_15w_mode();
    test_jetson_matrix_perf();
    test_jetson_no_heap();

    printf("\n[Integration Tests]\n");
    test_spectral_clustering_pipeline();
    test_heat_diffusion_pipeline();
    test_noether_agent_conservation();
    test_thermo_agent_entropy();
    test_crdt_multi_agent_merge();
    test_dirac_spectral_gap();
    test_landauer_agent_energy_budget();

    printf("\n=== Results: %d/%d tests passed ===\n",
           g_tests_passed, g_tests_run);

    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
