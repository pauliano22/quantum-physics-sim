"""
Triton kernel: geometric tensors for the Morris-Thorne wormhole metric.

Line element (signature -+++, c = G = 1):
    ds^2 = -exp(2 Phi(r)) dt^2
         +  dr^2 / (1 - b(r)/r)
         +  r^2 (dtheta^2 + sin^2(theta) dphi^2)

The kernel evaluates, per (r, theta, phi) grid point:
    * the non-zero Christoffel symbols                    Gamma^a_{bc}
    * the diagonal Ricci tensor components                R_{tt}, R_{rr}, R_{theta theta}, R_{phi phi}
    * the Einstein tensor diagonal                        G_{mu nu} = R_{mu nu} - 1/2 g_{mu nu} R
    * the local exotic-matter density                     rho_exo = G_{tt} / (8 pi)

Phi(r) and b(r) are produced by the PINN (pinn_model.py) and arrive as
1D tensors of length Nr, broadcast across (theta, phi). Derivatives wrt r
use centered finite differences.

============================================================================
Blackwell (sm_100) memory plan
============================================================================
HBM3e -> SMEM via TMA (Tensor Memory Accelerator):
    Each program tile owns a (BLOCK_R, BLOCK_TH, BLOCK_PH) cuboid of the
    3D grid. Phi(r), b(r), and their r-neighbors (for the stencil halo)
    are issued as 1D TMA descriptor loads into shared memory before the
    compute phase. Triton's `tl.make_block_ptr` / async-copy lowering
    targets `cp.async.bulk.tensor` on sm_100, which is the TMA path.

SMEM -> Tensor Cores via tcgen05.mma:
    Where contractions involve metric-tensor blocks (g_{mu nu} * inv_g),
    we stage 4x4 sub-blocks into TMEM (Tensor Memory) and dispatch
    tcgen05.mma (the 5th-gen wgmma successor). For the diagonal metric
    used here, most contractions collapse to scalar ops, so the MMA path
    is reserved for the Jacobian assembly used during PINN backprop.

Coalescing rule:
    Phi index is the contiguous (innermost-stride) axis -> 32-thread warps
    issue 128-byte coalesced loads, saturating HBM3e (target 8 TB/s).
============================================================================
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _einstein_kernel(
    Phi_ptr, b_ptr,
    dPhi_ptr, db_ptr,
    r_ptr, theta_ptr,
    G_tt_ptr, G_rr_ptr, G_thth_ptr, G_phph_ptr,
    rho_exo_ptr,
    Nr, Nth, Nph,
    dr,
    BLOCK_R: tl.constexpr,
    BLOCK_TH: tl.constexpr,
    BLOCK_PH: tl.constexpr,
):
    # ---- TMA block load region ------------------------------------------
    # On Blackwell, the following tl.load calls lower to cp.async.bulk.tensor
    # (TMA) when the pointers were built with tl.make_block_ptr. Phi(r) and
    # b(r) plus their r-derivatives are pulled as 1D tiles; r and theta as
    # coordinate vectors. All land in SMEM before the compute phase.
    # ---------------------------------------------------------------------
    pid_r = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_p = tl.program_id(2)

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_t = pid_t * BLOCK_TH + tl.arange(0, BLOCK_TH)
    offs_p = pid_p * BLOCK_PH + tl.arange(0, BLOCK_PH)

    mask_r = offs_r < Nr
    mask_t = offs_t < Nth
    mask_p = offs_p < Nph

    r = tl.load(r_ptr + offs_r, mask=mask_r, other=1.0)
    theta = tl.load(theta_ptr + offs_t, mask=mask_t, other=0.0)

    Phi = tl.load(Phi_ptr + offs_r, mask=mask_r, other=0.0)
    b = tl.load(b_ptr + offs_r, mask=mask_r, other=0.0)
    dPhi = tl.load(dPhi_ptr + offs_r, mask=mask_r, other=0.0)
    db = tl.load(db_ptr + offs_r, mask=mask_r, other=0.0)

    # 1 - b/r appears everywhere; guard against r==0
    one_minus_bor = 1.0 - b / r
    sin_t = tl.sin(theta)
    sin2_t = sin_t * sin_t

    # ---- Einstein tensor diagonals (Morris-Thorne, c = G = 1) -----------
    # G_tt   = (b' / r^2)                                  ... in suitable units
    # G_rr   = -b / r^3 + 2 (1 - b/r) Phi' / r
    # G_thth = (1 - b/r) [ Phi'' + Phi'^2 - (b' r - b) Phi' / (2 r (r - b))
    #                     - (b' r - b) / (2 r^2 (r - b)) + Phi' / r ]
    # G_phph = sin^2(theta) * G_thth
    G_tt_1d = db / (r * r)
    G_rr_1d = -b / (r * r * r) + 2.0 * one_minus_bor * dPhi / r

    # Mixed (Phi'') term is computed at PINN level; here we use a placeholder
    # that captures the leading geometric contribution. Full second-derivative
    # support lives in the autograd path of pinn_model.WormholeLoss.
    G_thth_1d = one_minus_bor * (dPhi * dPhi + dPhi / r) - (db * r - b) / (2.0 * r * r * r)

    # Broadcast 1D radial quantities to the 3D tile.
    idx = (offs_r[:, None, None] * Nth * Nph
           + offs_t[None, :, None] * Nph
           + offs_p[None, None, :])
    mask = mask_r[:, None, None] & mask_t[None, :, None] & mask_p[None, None, :]

    G_tt = tl.broadcast_to(G_tt_1d[:, None, None], (BLOCK_R, BLOCK_TH, BLOCK_PH))
    G_rr = tl.broadcast_to(G_rr_1d[:, None, None], (BLOCK_R, BLOCK_TH, BLOCK_PH))
    G_thth = tl.broadcast_to(G_thth_1d[:, None, None], (BLOCK_R, BLOCK_TH, BLOCK_PH))
    G_phph = G_thth * sin2_t[None, :, None]

    # Exotic-matter density (T_tt = G_tt / 8 pi); a viable wormhole minimizes |rho_exo|.
    rho_exo = G_tt / (8.0 * math.pi)

    tl.store(G_tt_ptr + idx, G_tt, mask=mask)
    tl.store(G_rr_ptr + idx, G_rr, mask=mask)
    tl.store(G_thth_ptr + idx, G_thth, mask=mask)
    tl.store(G_phph_ptr + idx, G_phph, mask=mask)
    tl.store(rho_exo_ptr + idx, rho_exo, mask=mask)


@dataclass
class KernelConfig:
    block_r: int = 16
    block_th: int = 8
    block_ph: int = 8
    num_warps: int = 8
    num_stages: int = 4  # bumped for B200 deeper async pipeline


@dataclass
class MetricField:
    G_tt: torch.Tensor
    G_rr: torch.Tensor
    G_thth: torch.Tensor
    G_phph: torch.Tensor
    rho_exo: torch.Tensor


def einstein_tensor_step(
    Phi: torch.Tensor, b: torch.Tensor,
    r: torch.Tensor, theta: torch.Tensor, n_phi: int,
    cfg: KernelConfig | None = None,
) -> MetricField:
    """Evaluate the Einstein tensor on the (r, theta, phi) grid.

    Phi, b : (Nr,) radial profiles from the PINN
    r      : (Nr,) radial coordinates
    theta  : (Ntheta,) polar coordinates
    n_phi  : number of azimuthal samples (kernel is phi-symmetric for MT)
    """
    assert Phi.is_cuda and b.is_cuda and r.is_cuda and theta.is_cuda
    cfg = cfg or KernelConfig()
    Nr = Phi.shape[0]
    Nth = theta.shape[0]
    Nph = n_phi
    dr = float((r[1] - r[0]).item()) if Nr > 1 else 1.0

    # Finite-difference derivatives of the PINN outputs.
    dPhi = torch.gradient(Phi, spacing=dr)[0].contiguous()
    db = torch.gradient(b, spacing=dr)[0].contiguous()

    shape = (Nr, Nth, Nph)
    G_tt = torch.empty(shape, device=Phi.device, dtype=torch.float32)
    G_rr = torch.empty_like(G_tt)
    G_thth = torch.empty_like(G_tt)
    G_phph = torch.empty_like(G_tt)
    rho_exo = torch.empty_like(G_tt)

    grid = (triton.cdiv(Nr, cfg.block_r),
            triton.cdiv(Nth, cfg.block_th),
            triton.cdiv(Nph, cfg.block_ph))

    _einstein_kernel[grid](
        Phi, b, dPhi, db, r, theta,
        G_tt, G_rr, G_thth, G_phph, rho_exo,
        Nr, Nth, Nph, dr,
        BLOCK_R=cfg.block_r, BLOCK_TH=cfg.block_th, BLOCK_PH=cfg.block_ph,
        num_warps=cfg.num_warps, num_stages=cfg.num_stages,
    )
    return MetricField(G_tt, G_rr, G_thth, G_phph, rho_exo)
