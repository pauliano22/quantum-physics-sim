"""
MCP tool surface exposed to NemoClaw.

NemoClaw drives an autotuning loop: it calls update_simulation_params() to
mutate kernel configs (block sizes, warps, stages, dt), then trigger_gpu_run()
to measure throughput on H100. The agent uses the returned metrics to propose
the next configuration.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from src.engine.wave_kernel import KernelConfig, WaveSimulation


@dataclass
class RunMetrics:
    steps: int
    elapsed_s: float
    steps_per_s: float
    gcells_per_s: float
    peak_mem_mb: float
    config: dict[str, Any]


class SimulationTools:
    """Methods on this class are registered as MCP tools for NemoClaw."""

    def __init__(self, shape: tuple[int, int, int] = (128, 128, 128)) -> None:
        self.sim = WaveSimulation(shape=shape)
        self.history: list[RunMetrics] = []

    def update_simulation_params(
        self,
        block_x: int | None = None,
        block_y: int | None = None,
        block_z: int | None = None,
        num_warps: int | None = None,
        num_stages: int | None = None,
        dt: float | None = None,
        dx: float | None = None,
    ) -> dict[str, Any]:
        """Update kernel/simulation parameters. Returns the new config."""
        cfg = self.sim.cfg
        if block_x is not None: cfg.block_x = int(block_x)
        if block_y is not None: cfg.block_y = int(block_y)
        if block_z is not None: cfg.block_z = int(block_z)
        if num_warps is not None: cfg.num_warps = int(num_warps)
        if num_stages is not None: cfg.num_stages = int(num_stages)
        if dt is not None: self.sim.dt = float(dt)
        if dx is not None: self.sim.dx = float(dx)
        return {"config": asdict(cfg), "dt": self.sim.dt, "dx": self.sim.dx,
                "shape": list(self.sim.shape)}

    def trigger_gpu_run(self, n_steps: int = 100, warmup: int = 5) -> dict[str, Any]:
        """Execute n_steps of evolution on GPU; return performance metrics."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device not available")

        torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup):
            self.sim.step(1)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        self.sim.step(n_steps)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        Nx, Ny, Nz = self.sim.shape
        cells = Nx * Ny * Nz * n_steps
        metrics = RunMetrics(
            steps=n_steps,
            elapsed_s=elapsed,
            steps_per_s=n_steps / elapsed,
            gcells_per_s=cells / elapsed / 1e9,
            peak_mem_mb=torch.cuda.max_memory_allocated() / (1024 ** 2),
            config=asdict(self.sim.cfg),
        )
        self.history.append(metrics)
        return asdict(metrics)

    def best_run(self) -> dict[str, Any] | None:
        if not self.history:
            return None
        return asdict(max(self.history, key=lambda m: m.gcells_per_s))


if __name__ == "__main__":
    tools = SimulationTools()
    print(tools.update_simulation_params(block_x=16, block_y=8, block_z=8))
    print(tools.trigger_gpu_run(n_steps=50))
