# Quantum Physics Simulation Engine

**An agentically-autotuned, H100-native solver for the time-dependent Schrödinger equation.**

This project pairs a Triton-based 3D wave-evolution kernel with an autonomous optimization agent (NemoClaw, driven via MCP) that iteratively tunes the kernel for peak throughput on NVIDIA Hopper-class GPUs.

---

## Engineering Highlights

### 1. H100-First Kernel Design
- **Triton 3.x kernels** targeting `sm_90a`, with launch parameters (`BLOCK_X`, `BLOCK_Y`, `BLOCK_Z`, `num_warps`, `num_stages`) exposed as `tl.constexpr` so the autotuner can explore the full SM occupancy surface.
- 3D stencil layout chosen to maximize **L2 residency** and exploit the H100's enlarged shared memory (228 KB/SM) and **TMA-style** tiled loads.
- Target workload: complex-valued ψ field updates approximating split-step Schrödinger evolution on grids up to 1024³.
- Roadmap: FP8 / `wgmma`-backed Hamiltonian application, async copies via the H100 **Tensor Memory Accelerator**, and CUDA-graph capture for sub-µs step launches.

### 2. Agentic Autonomy (NemoClaw + MCP)
Traditional autotuners brute-force a grid. This engine instead exposes the kernel knobs as an **MCP tool surface** (`src/agents/mcp_tools.py`) consumed by NemoClaw running in OpenShell:

- `update_simulation_params(...)` — mutate block shape, warp count, pipeline depth, timestep.
- `trigger_gpu_run(...)` — execute a measured workload on the GPU and return wall-clock, GCells/s, and peak HBM usage.

NemoClaw observes the metric history, reasons about the cost model (occupancy vs. register pressure vs. memory-pipe saturation), and proposes the next configuration. The loop is closed: the agent *reads its own previous benchmarks*, hypothesizes a bottleneck, and tests it — no hard-coded search grid.

### 3. Visualization
PyVista-backed volumetric rendering of |ψ|² (`src/viz/render.py`) for qualitative validation against analytic test cases (Gaussian wavepacket dispersion, harmonic oscillator eigenstates, double-slit interference).

---

## Repository Layout

```
src/
  engine/    Triton & CUDA kernels (wave_kernel.py)
  agents/    NemoClaw MCP tool surface (mcp_tools.py)
  viz/       PyVista volume rendering (render.py)
tests/       Analytic validation harness
```

## Quick Start

```bash
pip install -r requirements.txt
python -m src.agents.mcp_tools          # smoke-test the kernel
```

Hook the `SimulationTools` class into your MCP server registration to give NemoClaw control of the optimization loop.

## Why This Matters

Modern HPC and AI codes are bottlenecked not by FLOPs but by **how well a kernel fits the architecture it runs on**. Hopper rewards code that respects its memory hierarchy, async pipelines, and warp-specialization model. This project demonstrates a path where the *kernel author is itself an LLM agent* — one that closes the loop between hypothesis, benchmark, and revision faster than a human can rebuild PyTorch.

The same pattern generalizes to any CUDA/Triton workload where the optimal configuration depends on shape, dtype, and SM-generation — exactly the regime where NVIDIA's stack is hardest to tune by hand.
