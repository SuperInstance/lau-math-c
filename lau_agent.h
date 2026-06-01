/*
 * lau_agent.h — Minimal agent: observe->predict->update->act->conserve
 *
 * Copyright (c) 2025 Lau Ecosystem
 */

#ifndef LAU_AGENT_H
#define LAU_AGENT_H

#ifdef __cplusplus
extern "C" {
#endif

#include "lau_matrix.h"

#define LAU_AGENT_STATE_DIM 16

/* Agent lifecycle stages */
typedef enum {
    LAU_AGENT_OBSERVE = 0,
    LAU_AGENT_PREDICT,
    LAU_AGENT_UPDATE,
    LAU_AGENT_ACT,
    LAU_AGENT_CONSERVE,
    LAU_AGENT_NUM_STAGES
} lau_agent_stage_t;

/* Belief state: mean + covariance (Kalman-like) */
typedef struct {
    int dim;
    double mean[LAU_AGENT_STATE_DIM];
    double covariance[LAU_AGENT_STATE_DIM * LAU_AGENT_STATE_DIM];
} lau_belief_t;

/* Agent structure */
typedef struct {
    int dim;
    lau_belief_t belief;
    lau_agent_stage_t stage;
    double energy_budget;
    double energy_spent;
    int step_count;
    double observation[LAU_AGENT_STATE_DIM];
    double prediction[LAU_AGENT_STATE_DIM];
    double action[LAU_AGENT_STATE_DIM];
} lau_agent_t;

/* Initialize agent with dimension and energy budget */
void lau_agent_init(lau_agent_t *a, int dim, double energy_budget);

/* Observe: store observation */
void lau_agent_observe(lau_agent_t *a, const double *observation);

/* Predict: propagate belief forward (simple linear: x' = Fx) */
void lau_agent_predict(lau_agent_t *a, const lau_matrix_t *F);

/* Update: Kalman-like update with observation */
void lau_agent_update(lau_agent_t *a, const lau_matrix_t *H,
                      const lau_matrix_t *R);

/* Act: produce action from belief */
void lau_agent_act(lau_agent_t *a, const lau_matrix_t *policy);

/* Conserve: check energy budget and adjust */
double lau_agent_conserve(lau_agent_t *a, double cost);

/* Get spectral decomposition of belief covariance */
int lau_agent_spectral_decompose(const lau_agent_t *a,
                                 double *eigenvalues,
                                 double *eigenvectors);

/* Belief entropy (differential entropy of Gaussian) */
double lau_agent_belief_entropy(const lau_agent_t *a);

#ifdef __cplusplus
}
#endif

#endif /* LAU_AGENT_H */
