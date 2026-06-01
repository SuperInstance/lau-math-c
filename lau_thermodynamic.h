/*
 * lau_thermodynamic.h — Landauer cost, Fisher information, free energy
 *
 * Copyright (c) 2025 Lau Ecosystem
 */

#ifndef LAU_THERMODYNAMIC_H
#define LAU_THERMODYNAMIC_H

#ifdef __cplusplus
extern "C" {
#endif

#include "lau_matrix.h"

/* Boltzmann constant in J/K */
#define LAU_BOLTZMANN 1.380649e-23

/* Landauer cost: minimum energy to erase one bit at temperature T */
double lau_landauer_cost(double temperature_kelvin);

/* Landauer cost for n bits */
double lau_landauer_cost_n(double temperature_kelvin, int n_bits);

/* Fisher information matrix for a parameterized distribution */
/* I(theta) = E[(d log p / dtheta)(d log p / dtheta)^T] */
lau_matrix_t lau_fisher_information(const double *log_likelihood_grads,
                                    const double *probs, int n_samples,
                                    int n_params);

/* Natural gradient direction: I^{-1} * gradient */
int lau_natural_gradient(const lau_matrix_t *fisher_info,
                         const double *gradient, double *nat_grad, int n);

/* Free energy: F = U - T*S */
double lau_free_energy(double internal_energy, double temperature,
                       double entropy);

/* Entropy from probability distribution */
double lau_entropy(const double *probs, int n);

/* KL divergence D_KL(P || Q) */
double lau_kl_divergence(const double *p, const double *q, int n);

/* Thermodynamic efficiency: useful_work / total_energy */
double lau_thermo_efficiency(double useful_work, double total_energy);

#ifdef __cplusplus
}
#endif

#endif /* LAU_THERMODYNAMIC_H */
