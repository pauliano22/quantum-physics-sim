# Wormhole Discovery Engine

I'm trying to find Morris-Thorne wormhole geometries that stay open without needing absurd amounts of exotic (negative-energy) matter. A small neural network learns the redshift function `Φ(r)` and the shape function `b(r)`, and an agent (NemoClaw, running in an OpenShell sandbox) drives the search: it picks hyperparameters, launches runs on the GPUs, looks at the losses, and decides what to try next.

## How it works

- **PINN** (`src/engine/pinn_model.py`): a SIREN-style network that outputs `Φ(r)` and `b(r)`. Final activations enforce the obvious physical bounds (no horizons, `0 ≤ b ≤ r`).
- **Loss** (`WormholeLoss`): three pieces added together.
  - *Boundary*: `b(r₀) = r₀`, flare-out, asymptotic flatness.
  - *Einstein*: residual of the field equations at collocation points.
  - *Exotic matter*: how badly the null energy condition is violated. This is the one we actually want small.
- **Triton kernel** (`src/engine/spacetime_kernel.py`): evaluates the Einstein tensor on an `(r, θ, φ)` grid. Block sizes are `tl.constexpr` so they're autotunable.
- **Agent** (`src/agents/research_tools.py`): `mutate_hyperparams`, `dispatch_sweep`, `read_exotic_matter_loss`. That's the whole MCP surface.

## Hardware

Built for B200. The kernel lays out `φ` as the innermost axis so loads coalesce into HBM3e; radial profiles are staged via TMA; the autograd Jacobian work hits `tcgen05.mma`. Sweeps fan out across GPUs over NVLink5.

## Layout

```
src/engine/    Triton kernel + PINN
src/agents/    NemoClaw tools
src/viz/       PyVista rendering
openshell.toml NemoClaw sandbox policy
```

## Run it

```bash
pip install -r requirements.txt
python -m src.agents.research_tools
```

Then point NemoClaw at `openshell.toml` and let it go.
