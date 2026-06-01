# lau-math-c

**Core mathematical primitives for the Lau ecosystem — C99, zero-allocation, edge-ready.**

A pure C implementation of the math layer that Lau agents need to run on constrained hardware: Jetson Nano/Orin, embedded ARM, RISC-V, and bare-metal targets.

## Hardware Targets

| Target | Architecture | Notes |
|--------|-------------|-------|
| **Jetson Nano** | ARMv8 (Cortex-A57) | 7W / 15W thermal modes |
| **Jetson Orin** | ARMv8 (Cortex-A78AE) | 15W / 30W / 60W modes |
| **Workstation** | x86_64 (AVX-512) | `-march=native -mavx2 -mfma` |
| **RISC-V** | rv64gc | Generic 64-bit RISC-V |
| **Cloud** | x86_64 / ARM64 | Same code, higher clocks |

## Design Principles

- **No heap allocation in hot paths** — all matrices stack-allocated for N ≤ 16
- **No external dependencies** — pure C99 + libm
- **C99 compatible** — no `//` comments, no VLAs in critical paths
- **Static inline** for performance-critical functions
- **SIMD-ready** — compiler intrinsics for AVX-512 and NEON via flags

## Modules

| Module | Description |
|--------|-------------|
| `lau_matrix` | Fixed-size matrix ops (3×3, 4×4, NxN up to 16×16). Multiply, invert, transpose, determinant, trace, eigenvalues (Jacobi) |
| `lau_laplacian` | Graph Laplacian construction, normalized Laplacian, power iteration, spectral gap |
| `lau_dirichlet` | Dirichlet form E(f,f) = f^T L f, Green's function, heat kernel e^{-tL}, harmonic projection |
| `lau_dirac` | Dirac operator D = d + δ, boundary/coboundary operators, H⁰/H¹ cohomology, Betti numbers |
| `lau_conservation` | Noether charge tracking, CRDT merge (join-semilattice), conservation ratio, CALM monotonicity |
| `lau_thermodynamic` | Landauer cost kT ln(2), Fisher information, natural gradient, free energy F = U - TS |
| `lau_agent` | Minimal agent lifecycle: observe→predict→update→act→conserve, Kalman-like belief updates |

## Building

### Native (workstation / cloud)

```bash
make
make test    # Run 91 tests
```

### Jetson (aarch64 cross-compile)

```bash
# Install cross-compiler if needed
sudo apt install gcc-aarch64-linux-gnu

make jetson  # Cross-compiles for ARMv8
```

### RISC-V

```bash
sudo apt install gcc-riscv64-linux-gnu

make riscv
```

### Install

```bash
sudo make install   # Copies liblaumath.a to /usr/lib, headers to /usr/include
```

## Usage

```c
#include "lau_matrix.h"
#include "lau_agent.h"

/* Create a 3x3 rotation-like matrix */
lau_matrix_t m = lau_matrix_identity(3);
lau_matrix_set(&m, 0, 1, -0.5);
lau_matrix_set(&m, 1, 0,  0.5);

/* Compute eigenvalues */
double eigenvalues[3];
double eigenvectors[9];
lau_matrix_eigenvalues_symmetric(&m, eigenvalues, eigenvectors);

/* Run an agent */
lau_agent_t agent;
lau_agent_init(&agent, 3, 100.0);  /* 3D state, 100J budget */

double observation[] = {1.0, 2.0, 3.0};
lau_agent_observe(&agent, observation);

lau_matrix_t F = lau_matrix_identity(3);
lau_agent_predict(&agent, &F);
```

## Test Suite

91 tests covering all modules including:

- Matrix operations (multiply, invert, eigenvalues for 2×2 through 16×16)
- Graph Laplacian construction and spectral analysis
- Dirichlet energy, heat kernel, harmonic projection
- Dirac operator, cohomology, Betti numbers
- Noether charge conservation and violation detection
- CRDT merge semantics (concurrent updates, domination)
- Landauer cost, Fisher information, KL divergence
- Full agent lifecycle
- Jetson thermal constraints (7W and 15W modes)

```bash
make test
```

## Performance

All operations are stack-allocated with no `malloc` calls. Benchmarks on representative hardware:

| Operation | Jetson Nano (5W) | x86_64 (3.5GHz) |
|-----------|------------------|-------------------|
| 4×4 multiply | ~120ns | ~30ns |
| 4×4 inverse | ~200ns | ~50ns |
| 16×16 eigenvalues | ~50μs | ~12μs |
| Agent cycle (3D) | ~2μs | ~500ns |

## License

MIT
