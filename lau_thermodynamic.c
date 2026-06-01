/*
 * lau_thermodynamic.c — Landauer cost, Fisher information, free energy
 *
 * Copyright (c) 2025 Lau Ecosystem
 */

#include "lau_thermodynamic.h"
#include <math.h>
#include <string.h>

double lau_landauer_cost(double temperature_kelvin)
{
    /* E = kT ln(2) */
    return LAU_BOLTZMANN * temperature_kelvin * log(2.0);
}

double lau_landauer_cost_n(double temperature_kelvin, int n_bits)
{
    return n_bits * lau_landauer_cost(temperature_kelvin);
}

lau_matrix_t lau_fisher_information(const double *log_likelihood_grads,
                                    const double *probs,
                                    int n_samples, int n_params)
{
    /* grads is n_samples * n_params, probs is n_samples */
    /* I = sum_i p_i * grad_i * grad_i^T */
    lau_matrix_t I = lau_matrix_zeros(n_params, n_params);
    int i, j, k;

    for (i = 0; i < n_samples; i++) {
        for (j = 0; j < n_params; j++) {
            for (k = 0; k < n_params; k++) {
                I.data[j * n_params + k] += probs[i] *
                    log_likelihood_grads[i * n_params + j] *
                    log_likelihood_grads[i * n_params + k];
            }
        }
    }

    return I;
}

int lau_natural_gradient(const lau_matrix_t *fisher_info,
                         const double *gradient, double *nat_grad, int n)
{
    /* nat_grad = F^{-1} * gradient */
    lau_matrix_t F_inv;
    int ret = lau_matrix_invert(&F_inv, fisher_info);
    int i;

    if (ret != 0) {
        /* Pseudoinverse fallback: use regularized version */
        lau_matrix_t F_reg = *fisher_info;
        double lambda = 1e-6;
        int j;
        for (j = 0; j < n; j++) {
            F_reg.data[j * n + j] += lambda;
        }
        ret = lau_matrix_invert(&F_inv, &F_reg);
        if (ret != 0) {
            /* Last resort: use identity (standard gradient) */
            for (i = 0; i < n; i++) {
                nat_grad[i] = gradient[i];
            }
            return -1;
        }
    }

    for (i = 0; i < n; i++) {
        int j;
        nat_grad[i] = 0.0;
        for (j = 0; j < n; j++) {
            nat_grad[i] += F_inv.data[i * n + j] * gradient[j];
        }
    }

    return 0;
}

double lau_free_energy(double internal_energy, double temperature, double entropy)
{
    /* F = U - TS */
    return internal_energy - temperature * entropy;
}

double lau_entropy(const double *probs, int n)
{
    double H = 0.0;
    int i;
    for (i = 0; i < n; i++) {
        if (probs[i] > 1e-15) {
            H -= probs[i] * log(probs[i]);
        }
    }
    return H;
}

double lau_kl_divergence(const double *p, const double *q, int n)
{
    double kl = 0.0;
    int i;
    for (i = 0; i < n; i++) {
        if (p[i] > 1e-15 && q[i] > 1e-15) {
            kl += p[i] * log(p[i] / q[i]);
        }
    }
    return kl;
}

double lau_thermo_efficiency(double useful_work, double total_energy)
{
    if (fabs(total_energy) < 1e-15) return 0.0;
    return useful_work / total_energy;
}
