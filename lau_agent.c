/*
 * lau_agent.c — Minimal agent implementation
 *
 * Copyright (c) 2025 Lau Ecosystem
 */

#include "lau_agent.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

void lau_agent_init(lau_agent_t *a, int dim, double energy_budget)
{
    int i;
    a->dim = dim;
    a->stage = LAU_AGENT_OBSERVE;
    a->energy_budget = energy_budget;
    a->energy_spent = 0.0;
    a->step_count = 0;

    for (i = 0; i < dim; i++) {
        a->belief.mean[i] = 0.0;
        a->observation[i] = 0.0;
        a->prediction[i] = 0.0;
        a->action[i] = 0.0;
    }

    /* Identity covariance */
    for (i = 0; i < dim * dim; i++) {
        a->belief.covariance[i] = 0.0;
    }
    for (i = 0; i < dim; i++) {
        a->belief.covariance[i * dim + i] = 1.0;
    }
}

void lau_agent_observe(lau_agent_t *a, const double *observation)
{
    int i;
    for (i = 0; i < a->dim; i++) {
        a->observation[i] = observation[i];
    }
    a->stage = LAU_AGENT_PREDICT;
}

void lau_agent_predict(lau_agent_t *a, const lau_matrix_t *F)
{
    /* x' = F * x */
    int i, j;
    int n = a->dim;

    for (i = 0; i < n; i++) {
        double sum = 0.0;
        for (j = 0; j < n; j++) {
            sum += F->data[i * n + j] * a->belief.mean[j];
        }
        a->prediction[i] = sum;
    }

    /* P' = F * P * F^T (simplified: assume F is small) */
    {
        double new_cov[LAU_AGENT_STATE_DIM * LAU_AGENT_STATE_DIM];
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                double sum = 0.0;
                int k;
                for (k = 0; k < n; k++) {
                    int l;
                    for (l = 0; l < n; l++) {
                        sum += F->data[i * n + k] *
                               a->belief.covariance[k * n + l] *
                               F->data[j * n + l];
                    }
                }
                new_cov[i * n + j] = sum;
            }
        }
        memcpy(a->belief.covariance, new_cov, sizeof(double) * (size_t)(n * n));
    }

    for (i = 0; i < n; i++) {
        a->belief.mean[i] = a->prediction[i];
    }

    a->stage = LAU_AGENT_UPDATE;
}

void lau_agent_update(lau_agent_t *a, const lau_matrix_t *H,
                      const lau_matrix_t *R)
{
    /* Kalman update:
     * K = P * H^T * (H * P * H^T + R)^{-1}
     * x = x + K * (z - H * x)
     * P = (I - K * H) * P */
    int n = a->dim;
    int m = H->rows; /* observation dim */
    lau_matrix_t S_mat, S_inv;
    double K[LAU_AGENT_STATE_DIM * LAU_AGENT_STATE_DIM];
    double innovation[LAU_AGENT_STATE_DIM];
    double I_KH[LAU_AGENT_STATE_DIM * LAU_AGENT_STATE_DIM];
    int i, j, k;

    /* S = H * P * H^T + R */
    {
        lau_matrix_t P_mat;
        P_mat.rows = n;
        P_mat.cols = n;
        memcpy(P_mat.data, a->belief.covariance, sizeof(double) * (size_t)(n * n));

        {
            lau_matrix_t PHt = lau_matrix_multiply(&P_mat, &(lau_matrix_t){.rows = n, .cols = m, .data = {0}});
            /* Build H^T manually */
            lau_matrix_t Ht;
            Ht.rows = m;
            Ht.cols = n;
            /* H is m x n, so H^T is n x m — but we need dimensions right */
            /* Actually H maps state->obs, so H is m x n */
            Ht = lau_matrix_transpose(H);
            PHt = lau_matrix_multiply(&P_mat, &Ht);
            {
                lau_matrix_t HPHt_mat = lau_matrix_multiply(H, &PHt);
                lau_matrix_t S_full = lau_matrix_add(&HPHt_mat, R);
                S_mat = S_full;
            }
        }
    }

    /* S_inv */
    if (lau_matrix_invert(&S_inv, &S_mat) != 0) {
        /* Fallback: skip update */
        a->stage = LAU_AGENT_ACT;
        return;
    }

    /* K = P * H^T * S_inv */
    {
        lau_matrix_t P_mat, Ht;
        P_mat.rows = n;
        P_mat.cols = n;
        memcpy(P_mat.data, a->belief.covariance, sizeof(double) * (size_t)(n * n));
        Ht = lau_matrix_transpose(H);
        {
            lau_matrix_t PHt = lau_matrix_multiply(&P_mat, &Ht);
            lau_matrix_t K_mat = lau_matrix_multiply(&PHt, &S_inv);
            for (i = 0; i < n * m; i++) {
                K[i] = K_mat.data[i];
            }
        }
    }

    /* innovation = z - H * x */
    for (i = 0; i < m; i++) {
        double hdotx = 0.0;
        for (j = 0; j < n; j++) {
            hdotx += H->data[i * n + j] * a->belief.mean[j];
        }
        innovation[i] = a->observation[i] - hdotx;
    }

    /* x = x + K * innovation */
    for (i = 0; i < n; i++) {
        double sum = 0.0;
        for (j = 0; j < m; j++) {
            sum += K[i * m + j] * innovation[j];
        }
        a->belief.mean[i] += sum;
    }

    /* P = (I - K*H) * P */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            double sum = (i == j) ? 1.0 : 0.0;
            for (k = 0; k < m; k++) {
                sum -= K[i * m + k] * H->data[k * n + j];
            }
            I_KH[i * n + j] = sum;
        }
    }

    {
        double new_cov[LAU_AGENT_STATE_DIM * LAU_AGENT_STATE_DIM];
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                double sum = 0.0;
                for (k = 0; k < n; k++) {
                    sum += I_KH[i * n + k] * a->belief.covariance[k * n + j];
                }
                new_cov[i * n + j] = sum;
            }
        }
        memcpy(a->belief.covariance, new_cov, sizeof(double) * (size_t)(n * n));
    }

    a->stage = LAU_AGENT_ACT;
}

void lau_agent_act(lau_agent_t *a, const lau_matrix_t *policy)
{
    /* action = policy * belief_mean */
    int i, j;
    int n = a->dim;
    int m = policy->rows;

    for (i = 0; i < m && i < LAU_AGENT_STATE_DIM; i++) {
        double sum = 0.0;
        for (j = 0; j < n; j++) {
            sum += policy->data[i * n + j] * a->belief.mean[j];
        }
        a->action[i] = sum;
    }

    a->stage = LAU_AGENT_CONSERVE;
}

double lau_agent_conserve(lau_agent_t *a, double cost)
{
    a->energy_spent += cost;
    a->step_count++;
    a->stage = LAU_AGENT_OBSERVE;

    /* Return remaining budget ratio */
    if (a->energy_budget <= 0.0) return 0.0;
    double remaining = a->energy_budget - a->energy_spent;
    return remaining / a->energy_budget;
}

int lau_agent_spectral_decompose(const lau_agent_t *a,
                                 double *eigenvalues,
                                 double *eigenvectors)
{
    lau_matrix_t cov;
    cov.rows = a->dim;
    cov.cols = a->dim;
    memcpy(cov.data, a->belief.covariance,
           sizeof(double) * (size_t)(a->dim * a->dim));

    return lau_matrix_eigenvalues_symmetric(&cov, eigenvalues, eigenvectors);
}

double lau_agent_belief_entropy(const lau_agent_t *a)
{
    /* H = 0.5 * ln((2*pi*e)^d * det(P)) */
    lau_matrix_t cov;
    double det;
    int n = a->dim;

    cov.rows = n;
    cov.cols = n;
    memcpy(cov.data, a->belief.covariance,
           sizeof(double) * (size_t)(n * n));

    det = lau_matrix_determinant(&cov);
    if (det <= 0.0) return 0.0;

    return 0.5 * log(pow(2.0 * M_PI * M_E, (double)n) * det);
}
