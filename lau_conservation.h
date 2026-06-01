/*
 * lau_conservation.h — Noether charges, CRDT merge, conservation laws
 *
 * Copyright (c) 2025 Lau Ecosystem
 */

#ifndef LAU_CONSERVATION_H
#define LAU_CONSERVATION_H

#ifdef __cplusplus
extern "C" {
#endif

#include "lau_matrix.h"

#define LAU_CONSERVATION_MAX_CHARGES 32

/* Noether charge: a conserved quantity tracked over state transitions */
typedef struct {
    int n_charges;
    double charges[LAU_CONSERVATION_MAX_CHARGES];
    double initial_total;
} lau_noether_t;

/* Initialize with initial charges */
void lau_noether_init(lau_noether_t *n, const double *charges, int count);

/* Update a single charge, return conservation error */
double lau_noether_update(lau_noether_t *n, int idx, double new_val);

/* Get total charge */
double lau_noether_total(const lau_noether_t *n);

/* Conservation error: |current_total - initial_total| / |initial_total| */
double lau_conservation_error(const lau_noether_t *n);

/* ---- CRDT merge (join-semilattice) ---- */

/* CRDT value with vector clock */
typedef struct {
    double value;
    int clock[LAU_CONSERVATION_MAX_CHARGES];
    int n_replicas;
} lau_crdt_value_t;

/* Initialize CRDT value */
void lau_crdt_init(lau_crdt_value_t *v, double val, int n_replicas);

/* Merge two CRDT values (last-writer-wins with vector clock) */
lau_crdt_value_t lau_crdt_merge(const lau_crdt_value_t *a,
                                const lau_crdt_value_t *b);

/* Increment clock for replica */
void lau_crdt_increment(lau_crdt_value_t *v, int replica_id);

/* Check if a dominates b (a >= b in the semilattice) */
int lau_crdt_dominates(const lau_crdt_value_t *a, const lau_crdt_value_t *b);

/* ---- Conservation ratio ---- */

/* Compute conservation ratio: sum(output) / sum(input) */
double lau_conservation_ratio(const double *input, const double *output, int n);

/* ---- CALM monotonicity ---- */

/* Check if a function is monotone (for CALM theorem) */
int lau_is_monotone(const double *x, const double *y, int n);

#ifdef __cplusplus
}
#endif

#endif /* LAU_CONSERVATION_H */
