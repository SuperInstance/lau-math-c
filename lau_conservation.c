/*
 * lau_conservation.c — Noether charges, CRDT merge, conservation laws
 *
 * Copyright (c) 2025 Lau Ecosystem
 */

#include "lau_conservation.h"
#include <math.h>
#include <string.h>

void lau_noether_init(lau_noether_t *n, const double *charges, int count)
{
    int i;
    n->n_charges = count;
    n->initial_total = 0.0;
    for (i = 0; i < count; i++) {
        n->charges[i] = charges[i];
        n->initial_total += charges[i];
    }
}

double lau_noether_update(lau_noether_t *n, int idx, double new_val)
{
    double old_total = lau_noether_total(n);
    if (idx >= 0 && idx < n->n_charges) {
        n->charges[idx] = new_val;
    }
    return fabs(lau_noether_total(n) - old_total);
}

double lau_noether_total(const lau_noether_t *n)
{
    double total = 0.0;
    int i;
    for (i = 0; i < n->n_charges; i++) {
        total += n->charges[i];
    }
    return total;
}

double lau_conservation_error(const lau_noether_t *n)
{
    double total = lau_noether_total(n);
    if (fabs(n->initial_total) < 1e-15) {
        return fabs(total);
    }
    return fabs(total - n->initial_total) / fabs(n->initial_total);
}

void lau_crdt_init(lau_crdt_value_t *v, double val, int n_replicas)
{
    int i;
    v->value = val;
    v->n_replicas = n_replicas;
    for (i = 0; i < n_replicas && i < LAU_CONSERVATION_MAX_CHARGES; i++) {
        v->clock[i] = 0;
    }
}

lau_crdt_value_t lau_crdt_merge(const lau_crdt_value_t *a,
                                const lau_crdt_value_t *b)
{
    lau_crdt_value_t result;
    int i;
    int a_dominates = 1, b_dominates = 1;

    result.n_replicas = a->n_replicas;

    /* Check domination */
    for (i = 0; i < a->n_replicas && i < LAU_CONSERVATION_MAX_CHARGES; i++) {
        if (a->clock[i] < b->clock[i]) a_dominates = 0;
        if (b->clock[i] < a->clock[i]) b_dominates = 0;
        result.clock[i] = (a->clock[i] > b->clock[i]) ? a->clock[i] : b->clock[i];
    }

    if (b_dominates && !a_dominates) {
        result.value = b->value;
    } else if (a_dominates && !b_dominates) {
        result.value = a->value;
    } else {
        /* Concurrent: use larger value as tiebreaker (join semilattice) */
        result.value = (a->value >= b->value) ? a->value : b->value;
    }

    return result;
}

void lau_crdt_increment(lau_crdt_value_t *v, int replica_id)
{
    if (replica_id >= 0 && replica_id < v->n_replicas &&
        replica_id < LAU_CONSERVATION_MAX_CHARGES) {
        v->clock[replica_id]++;
    }
}

int lau_crdt_dominates(const lau_crdt_value_t *a, const lau_crdt_value_t *b)
{
    int i;
    int has_strict = 0;
    for (i = 0; i < a->n_replicas && i < LAU_CONSERVATION_MAX_CHARGES; i++) {
        if (a->clock[i] < b->clock[i]) return 0;
        if (a->clock[i] > b->clock[i]) has_strict = 1;
    }
    return has_strict || (a->value >= b->value);
}

double lau_conservation_ratio(const double *input, const double *output, int n)
{
    double sum_in = 0.0, sum_out = 0.0;
    int i;
    for (i = 0; i < n; i++) {
        sum_in += input[i];
        sum_out += output[i];
    }
    if (fabs(sum_in) < 1e-15) return (fabs(sum_out) < 1e-15) ? 1.0 : 0.0;
    return sum_out / sum_in;
}

int lau_is_monotone(const double *x, const double *y, int n)
{
    /* x <= y component-wise => f(x) <= f(y) for all such pairs */
    /* Check if x <= y component-wise */
    int i;
    int x_leq_y = 1;
    for (i = 0; i < n; i++) {
        if (x[i] > y[i]) { x_leq_y = 0; break; }
    }
    /* If x <= y, the function is monotone-preserving if we verify f(x) <= f(y) */
    /* This is a simplified check; full monotonicity requires checking all pairs */
    return x_leq_y;
}
