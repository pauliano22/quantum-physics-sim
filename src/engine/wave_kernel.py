"""
Triton kernel for 3D quantum wave evolution.

Placeholder for split-step Schrödinger evolution:
    i * hbar * d_psi/dt = -(hbar^2 / 2m) * laplacian(psi) + V(x) * psi

This boilerplate implements the kinetic-term stencil update on a 3D grid.
The agent layer (NemoClaw) tunes BLOCK_X/Y/Z, num_warps, num_stages.
"""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _wave_step_kernel(
    psi_re_ptr, psi_im_ptr,
    out_re_ptr, out_im_ptr,
    potential_ptr,
    Nx, Ny, Nz,
    dt, inv_dx2,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
    BLOCK_Z: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)

    offs_x = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    offs_y = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    offs_z = pid_z * BLOCK_Z + tl.arange(0, BLOCK_Z)

    mask_x = offs_x < Nx
    mask_y = offs_y < Ny
    mask_z = offs_z < Nz

    idx = (offs_x[:, None, None] * Ny * Nz
           + offs_y[None, :, None] * Nz
           + offs_z[None, None, :])
    mask = mask_x[:, None, None] & mask_y[None, :, None] & mask_z[None, None, :]

    interior = (
        (offs_x[:, None, None] > 0) & (offs_x[:, None, None] < Nx - 1) &
        (offs_y[None, :, None] > 0) & (offs_y[None, :, None] < Ny - 1) &
        (offs_z[None, None, :] > 0) & (offs_z[None, None, :] < Nz - 1)
    )
    safe = mask & interior

    c = tl.load(psi_re_ptr + idx, mask=mask, other=0.0)
    ci = tl.load(psi_im_ptr + idx, mask=mask, other=0.0)

    xp = tl.load(psi_re_ptr + idx + Ny * Nz, mask=safe, other=0.0)
    xm = tl.load(psi_re_ptr + idx - Ny * Nz, mask=safe, other=0.0)
    yp = tl.load(psi_re_ptr + idx + Nz, mask=safe, other=0.0)
    ym = tl.load(psi_re_ptr + idx - Nz, mask=safe, other=0.0)
    zp = tl.load(psi_re_ptr + idx + 1, mask=safe, other=0.0)
    zm = tl.load(psi_re_ptr + idx - 1, mask=safe, other=0.0)

    xpi = tl.load(psi_im_ptr + idx + Ny * Nz, mask=safe, other=0.0)
    xmi = tl.load(psi_im_ptr + idx - Ny * Nz, mask=safe, other=0.0)
    ypi = tl.load(psi_im_ptr + idx + Nz, mask=safe, other=0.0)
    ymi = tl.load(psi_im_ptr + idx - Nz, mask=safe, other=0.0)
    zpi = tl.load(psi_im_ptr + idx + 1, mask=safe, other=0.0)
    zmi = tl.load(psi_im_ptr + idx - 1, mask=safe, other=0.0)

    lap_re = (xp + xm + yp + ym + zp + zm - 6.0 * c) * inv_dx2
    lap_im = (xpi + xmi + ypi + ymi + zpi + zmi - 6.0 * ci) * inv_dx2

    V = tl.load(potential_ptr + idx, mask=mask, other=0.0)

    # i * d_psi/dt = H psi  =>  d_psi/dt = -i (H psi)
    # H psi = -0.5 * lap + V * psi   (units: hbar=m=1)
    h_re = -0.5 * lap_re + V * c
    h_im = -0.5 * lap_im + V * ci

    # Explicit Euler step (placeholder; real impl uses split-step or RK4)
    new_re = c + dt * h_im
    new_im = ci - dt * h_re

    tl.store(out_re_ptr + idx, new_re, mask=mask)
    tl.store(out_im_ptr + idx, new_im, mask=mask)


@dataclass
class KernelConfig:
    block_x: int = 8
    block_y: int = 8
    block_z: int = 8
    num_warps: int = 4
    num_stages: int = 3


def wave_step(
    psi_re: torch.Tensor,
    psi_im: torch.Tensor,
    potential: torch.Tensor,
    dt: float,
    dx: float,
    cfg: KernelConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert psi_re.is_cuda and psi_re.shape == psi_im.shape == potential.shape
    cfg = cfg or KernelConfig()
    Nx, Ny, Nz = psi_re.shape
    out_re = torch.empty_like(psi_re)
    out_im = torch.empty_like(psi_im)

    grid = (triton.cdiv(Nx, cfg.block_x),
            triton.cdiv(Ny, cfg.block_y),
            triton.cdiv(Nz, cfg.block_z))

    _wave_step_kernel[grid](
        psi_re, psi_im, out_re, out_im, potential,
        Nx, Ny, Nz,
        dt, 1.0 / (dx * dx),
        BLOCK_X=cfg.block_x, BLOCK_Y=cfg.block_y, BLOCK_Z=cfg.block_z,
        num_warps=cfg.num_warps, num_stages=cfg.num_stages,
    )
    return out_re, out_im


class WaveSimulation:
    def __init__(self, shape: tuple[int, int, int], dx: float = 0.1,
                 dt: float = 1e-4, device: str = "cuda"):
        self.shape = shape
        self.dx = dx
        self.dt = dt
        self.device = device
        self.psi_re = torch.zeros(shape, device=device, dtype=torch.float32)
        self.psi_im = torch.zeros(shape, device=device, dtype=torch.float32)
        self.potential = torch.zeros(shape, device=device, dtype=torch.float32)
        self.cfg = KernelConfig()
        self.step_count = 0

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.psi_re, self.psi_im = wave_step(
                self.psi_re, self.psi_im, self.potential,
                self.dt, self.dx, self.cfg,
            )
            self.step_count += 1

    def probability_density(self) -> torch.Tensor:
        return self.psi_re * self.psi_re + self.psi_im * self.psi_im
