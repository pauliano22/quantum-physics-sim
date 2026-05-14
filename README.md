# Wormhole Discovery Engine

**A self-driving laboratory for general relativity.**

This repository is an autonomous research stack that searches the space of Morris-Thorne wormhole geometries for solutions that keep the throat open while minimizing exotic-matter (negative-energy) content. A Physics-Informed Neural Network (PINN) parameterizes the redshift function `Φ(r)` and shape function `b(r)`; a NemoClaw agent running inside an OpenShell sandbox autonomously mutates hyperparameters, dispatches parallel sweeps across an NVIDIA Blackwell B200 cluster, and reads back the Einstein-tensor residuals to decide what to try next.

There is no human in the inner loop.

---

## Why This Architecture

General relativity is a search problem dressed up as a PDE: the metric ansatz is high-dimensional, the field equations are stiff, and the physically interesting solutions live on a thin manifold defined by energy-condition violations. Brute-force grid search wastes compute; gradient-based PINN training alone gets stuck in pathological local minima. We close the loop: an LLM agent observes which loss components dominate, hypothesizes a fix (deeper net, different SIREN frequency, re-weighted exotic-matter penalty), and dispatches the experiment itself.

The hardware story matters as much as the algorithm. Every layer of this stack is written to **saturate the B200**.

---

## Hardware Targeting (Blackwell B200)

### HBM3e bandwidth
The Triton kernel in `src/engine/spacetime_kernel.py` lays out the `(r, θ, φ)` grid with `φ` as the innermost-stride axis. Each warp issues 128-byte coalesced loads; tile shapes (`BLOCK_R`, `BLOCK_TH`, `BLOCK_PH`) are `tl.constexpr` so the autotuner can sweep occupancy against the **8 TB/s HBM3e** roofline.

### TMA (Tensor Memory Accelerator)
Radial profiles `Φ(r)`, `b(r)`, and their finite-difference neighbors are staged from HBM3e into shared memory via TMA bulk-tensor copies. The kernel's block-pointer construction lowers to `cp.async.bulk.tensor` on `sm_100`, overlapping the load of tile *N+1* with the compute of tile *N* over a deeper async pipeline (`num_stages=4`).

### tcgen05.mma (5th-gen wgmma)
Where Jacobian assembly during PINN backprop dispatches small dense contractions, 4×4 metric sub-blocks are staged into Tensor Memory (TMEM) and dispatched via `tcgen05.mma`. The diagonal Morris-Thorne metric collapses most contractions to scalar ops, but the MMA path lights up under the curvature-Jacobian and Hessian terms exercised by the autograd graph in `WormholeLoss`.

### NVLink5 distribution
`OrchestratorAgent.dispatch_sweep()` fans configurations across visible B200s; gradient and metric reductions ride NVLink5 (1.8 TB/s per GPU bidirectional). Each sweep run is independent — the agent uses cross-run parallelism, not data-parallel splits, which makes NVLink saturation a function of how many candidate configs the agent generates per round.

---

## Agentic Stack (NemoClaw + OpenShell)

| Tool                          | Purpose                                                    |
|-------------------------------|------------------------------------------------------------|
| `mutate_hyperparams(**kw)`    | Propose a new PINN configuration (width, depth, SIREN ω₀, loss weights, collocation density). |
| `dispatch_sweep(configs)`     | Run N configs in parallel across the B200 cluster.        |
| `read_exotic_matter_loss(id)` | Return the integrated null-energy-condition violation for a run; the agent uses this as its primary fitness signal. |

The agent runs under `openshell.toml`, a strict March-2026-alpha policy that grants execute permission only for `src/`-resident Python scripts and read access to `nvidia-smi` for profiling. The agent cannot rewrite its own policy or escape the workspace.

---

## Repository Layout

```
src/
  engine/
    spacetime_kernel.py    # Triton: Einstein-tensor evaluation, TMA-staged
    pinn_model.py          # SIREN PINN + composite physics loss
  agents/
    research_tools.py      # NemoClaw MCP tool surface (OrchestratorAgent)
  viz/
    render.py              # PyVista volumetric rendering of |ψ|^2 / ρ_exo
openshell.toml             # NemoClaw sandbox policy (strict)
requirements.txt
```

## Physics Loss

```
L = w_b · MSE_Boundary + w_e · MSE_EinsteinFieldEquations + w_x · MSE_ExoticMatter
```

* **Boundary**: throat closure `b(r₀) = r₀`, flare-out `b'(r₀) < 1`, asymptotic flatness `b/r → 0`.
* **Einstein**: conservation of the diagonal stress-energy tensor along radial collocation points (sampled with autograd-tracked `r`, so `Φ'`, `Φ''`, `b'` are exact).
* **Exotic matter**: the network is *penalized* for the magnitude of NEC violation `max(0, -(ρ + p_r))` — this is the term the agent optimizes against to find geometries closest to physical realizability.

## Quick Start

```bash
pip install -r requirements.txt
python -m src.agents.research_tools     # smoke-test orchestration on one GPU
```

Attach NemoClaw to the workspace, point it at `openshell.toml`, and let it run.

---

## Why a Recruiter Should Care

This is a single repository that exercises the entire NVIDIA stack vertically:
- **Triton at the bottom**, written for a specific Blackwell SM generation.
- **PyTorch autograd** wired into a physics loss with second-order terms.
- **A reasoning agent at the top** that treats kernel and model hyperparameters as a joint search space.

The shape of the problem — a stiff PDE that needs a smart search policy *and* a saturated memory subsystem to be tractable at all — is exactly the regime where Blackwell-class hardware is most differentiated from the previous generation. A self-driving lab is the only way to use it well.
