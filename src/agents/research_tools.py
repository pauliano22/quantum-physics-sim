"""
MCP tool surface exposed to NemoClaw for autonomous wormhole discovery.

The OrchestratorAgent class registers three tool families:

  (a) mutate_hyperparams       - propose a new PINN configuration
  (b) dispatch_sweep           - fan out N configs across the B200 cluster
  (c) read_exotic_matter_loss  - evaluate physical viability of a run

NemoClaw runs in OpenShell, observes the loss landscape, and proposes the
next experiment. Sweeps execute in parallel via torch.distributed across
NVLink5-connected B200 GPUs.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from src.engine.pinn_model import LossWeights, WormholeLoss, WormholePINN


@dataclass
class PINNHyperparams:
    hidden: int = 128
    depth: int = 5
    w0: float = 30.0
    learning_rate: float = 1e-4
    n_collocation: int = 4096
    n_steps: int = 2000
    r0: float = 1.0
    r_max: float = 50.0
    weight_boundary: float = 1.0
    weight_einstein: float = 1.0
    weight_exotic: float = 0.5
    seed: int = 0


@dataclass
class RunResult:
    run_id: str
    gpu_index: int
    hyperparams: dict[str, Any]
    final_loss: dict[str, float]
    exotic_matter_integral: float
    physically_viable: bool
    elapsed_s: float


def _train_one(hp: PINNHyperparams, gpu_index: int) -> RunResult:
    device = torch.device(f"cuda:{gpu_index}")
    torch.manual_seed(hp.seed)

    model = WormholePINN(hidden=hp.hidden, depth=hp.depth, w0=hp.w0).to(device)
    loss_fn = WormholeLoss(
        r0=hp.r0,
        weights=LossWeights(hp.weight_boundary, hp.weight_einstein, hp.weight_exotic),
    )
    opt = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)

    t0 = time.perf_counter()
    last = None
    for _ in range(hp.n_steps):
        r = (hp.r0 + (hp.r_max - hp.r0) * torch.rand(hp.n_collocation, 1, device=device))
        breakdown = loss_fn(model, r)
        opt.zero_grad(set_to_none=True)
        breakdown.total.backward()
        opt.step()
        last = breakdown

    # Diagnose viability on a dense radial grid.
    with torch.no_grad():
        r_eval = torch.linspace(hp.r0, hp.r_max, 1024, device=device).requires_grad_(True)
        Phi, b = model(r_eval)
        db = torch.autograd.grad(b.sum(), r_eval, create_graph=False)[0]
        rho = db / (8.0 * 3.141592653589793 * r_eval.pow(2))
        exotic_integral = float(torch.relu(-rho).sum().item() * (hp.r_max - hp.r0) / 1024)

    final = last.to_dict() if last else {"total": float("nan")}
    viable = exotic_integral < 1e-3 and final["boundary"] < 1e-2

    return RunResult(
        run_id=uuid.uuid4().hex[:12],
        gpu_index=gpu_index,
        hyperparams=asdict(hp),
        final_loss=final,
        exotic_matter_integral=exotic_integral,
        physically_viable=viable,
        elapsed_s=time.perf_counter() - t0,
    )


class OrchestratorAgent:
    """MCP-exposed orchestration surface for NemoClaw."""

    def __init__(self, base: PINNHyperparams | None = None) -> None:
        self.base = base or PINNHyperparams()
        self.results: list[RunResult] = []
        self._n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # ----- (a) hyperparameter mutation ------------------------------------
    def mutate_hyperparams(self, **overrides: Any) -> dict[str, Any]:
        """Apply overrides to the current base config. Returns the new config."""
        for k, v in overrides.items():
            if not hasattr(self.base, k):
                raise KeyError(f"unknown hyperparameter: {k}")
            setattr(self.base, k, type(getattr(self.base, k))(v))
        return asdict(self.base)

    # ----- (b) parallel sweep dispatch -----------------------------------
    def dispatch_sweep(self, configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run `configs` in parallel, one per visible B200, round-robin if N > GPUs."""
        if self._n_gpus == 0:
            raise RuntimeError("no CUDA devices visible to the orchestrator")

        hps = [PINNHyperparams(**{**asdict(self.base), **c}) for c in configs]
        futures: list[Future[RunResult]] = []
        with ThreadPoolExecutor(max_workers=self._n_gpus) as ex:
            for i, hp in enumerate(hps):
                futures.append(ex.submit(_train_one, hp, i % self._n_gpus))
            results = [f.result() for f in futures]

        self.results.extend(results)
        return [asdict(r) for r in results]

    # ----- (c) physical-viability readout --------------------------------
    def read_exotic_matter_loss(self, run_id: str | None = None) -> dict[str, Any]:
        """Return the exotic-matter integral for a run, or the best so far."""
        if not self.results:
            return {"status": "empty", "results": []}
        if run_id:
            for r in self.results:
                if r.run_id == run_id:
                    return asdict(r)
            raise KeyError(f"unknown run_id: {run_id}")
        best = min(self.results, key=lambda r: r.exotic_matter_integral)
        return {"best": asdict(best), "n_runs": len(self.results)}

    # ----- introspection helpers -----------------------------------------
    def list_runs(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self.results]

    def export_results(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        return os.path.abspath(path)


if __name__ == "__main__":
    agent = OrchestratorAgent()
    agent.mutate_hyperparams(n_steps=100, n_collocation=512)
    print(json.dumps(agent.dispatch_sweep([{"depth": 4}, {"depth": 6}]), indent=2))
    print(json.dumps(agent.read_exotic_matter_loss(), indent=2))
