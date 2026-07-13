# Spec 13 — More Distributions

**Status:** Pending | **Effort:** Medium | **Module:** `src/distributions.rs`

## Context

The current set (normal, uniform, exponential, log-normal, beta, gamma, bernoulli,
binomial, poisson, geometric, mixture, empirical, categorical) misses distributions that
uncertainty-modeling users reach for constantly: Student's t (heavy-tailed measurement
error), Weibull (reliability/survival), Pareto (heavy tails), triangular and PERT
(expert elicitation — the bread and butter of risk modeling), and Cauchy. With Spec 05's
`rand_distr` backend, most are nearly free.

## Scope and Invariants

1. New constructors, all following Spec 02 validation and Spec 04 seeding:
   - `student_t(degrees_of_freedom)` (+ located-scaled variant or docs showing
     `μ + σ·t`), `weibull(shape, scale)`, `pareto(scale, shape)`, `cauchy(location,
     scale)`, `triangular(min, mode, max)`, `pert(min, mode, max)`.
2. Backed by `rand_distr` where available (t, Weibull, Pareto, Cauchy, Triangular); PERT
   implemented via its Beta reparameterization.
3. Parameter constraints validated: `df > 0`, `shape/scale > 0`, `min ≤ mode ≤ max` with
   `min < max`.
4. Each constructor gets doc examples, support/moment documentation (explicitly noting
   which moments do not exist — Cauchy mean, Pareto variance for α ≤ 2, t variance for
   df ≤ 2), and inclusion in the Spec 11 property suite (support checks; moment checks
   only where moments exist).
5. README's distribution table is updated.

## Acceptance Tests

- **Given** each new constructor with valid parameters and a fixed seed, **when** sampled
  at large n, **then** samples lie in the documented support, and empirical moments match
  closed forms within tolerance for every (distribution, parameter) pair whose moments
  exist.
- **Given** `triangular(0, 5, 10)`, **then** all samples ∈ [0, 10] and the empirical mode
  is near 5; **given** `triangular(3, 1, 2)` (violated ordering), **then** a Spec 02
  validation error.
- **Given** `student_t(1)` (Cauchy-like), **when** 10⁶ samples are drawn, **then** no
  panic, all samples are finite, and extreme values *do* occur (|x| > 100 appears) —
  heavy tails are not clipped.
- **Given** `pert(min, mode, max)`, **then** its empirical mean ≈ `(min + 4·mode + max)/6`
  within tolerance.
- **Given** the docs, **then** each new distribution documents its support and moment
  existence caveats.
