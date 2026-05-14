"""
Physics-Informed Neural Network for Morris-Thorne wormhole discovery.

Outputs:
    Phi(r) : redshift function   -> bounded to avoid horizons (no zeros of g_tt)
    b(r)   : shape function      -> b(r0) = r0 at the throat, b(r)/r -> 0 at infinity

The PINN learns geometries that:
    1) satisfy the Einstein field equations,
    2) honor boundary conditions at the throat r0 and asymptotic infinity,
    3) MINIMIZE the integrated exotic-matter requirement (negative energy density)
       while keeping the throat flare-out condition (b'(r0) < 1) satisfied.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class SineActivation(nn.Module):
    """SIREN-style sine activation; good for representing smooth metrics."""
    def __init__(self, w0: float = 30.0) -> None:
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class WormholePINN(nn.Module):
    """
    Maps r -> (Phi(r), b(r)).

    Final activations enforce coarse physical priors:
        Phi(r) = tanh(...)          -> finite, no horizons
        b(r)   = r * sigmoid(...)   -> 0 <= b(r) <= r everywhere (flare-out admissible)
    """
    def __init__(self, hidden: int = 128, depth: int = 5, w0: float = 30.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, hidden), SineActivation(w0)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), SineActivation(w0)]
        self.trunk = nn.Sequential(*layers)
        self.head_phi = nn.Linear(hidden, 1)
        self.head_b = nn.Linear(hidden, 1)

    def forward(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if r.dim() == 1:
            r = r.unsqueeze(-1)
        h = self.trunk(r)
        Phi = torch.tanh(self.head_phi(h)).squeeze(-1)
        b = (r.squeeze(-1)) * torch.sigmoid(self.head_b(h)).squeeze(-1)
        return Phi, b


@dataclass
class LossWeights:
    boundary: float = 1.0
    einstein: float = 1.0
    exotic: float = 0.5


@dataclass
class LossBreakdown:
    total: torch.Tensor
    boundary: torch.Tensor
    einstein: torch.Tensor
    exotic: torch.Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "boundary": float(self.boundary.detach()),
            "einstein": float(self.einstein.detach()),
            "exotic": float(self.exotic.detach()),
        }


class WormholeLoss(nn.Module):
    """
    L = w_b * MSE_Boundary + w_e * MSE_EinsteinFieldEquations + w_x * MSE_ExoticMatter

    Boundary terms enforce:
        b(r0) = r0           (throat closure)
        b'(r0) < 1           (flare-out)
        Phi finite           (no horizon)
        b(r)/r -> 0          (asymptotic flatness)

    Einstein term measures residual of G_{mu nu} = 8 pi T_{mu nu} for an
    ansatz stress-energy tensor; we sample collocation points along r.

    Exotic-matter term penalizes integrated negative energy density
    rho_exo(r) = (1 / 8 pi) * (b'(r) / r^2).  Minimizing this drives the
    network toward "least pathological" wormhole geometries.
    """
    def __init__(self, r0: float = 1.0, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.r0 = r0
        self.w = weights or LossWeights()

    @staticmethod
    def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(y.sum(), x, create_graph=True)[0]

    def forward(self, model: WormholePINN, r_collocation: torch.Tensor) -> LossBreakdown:
        r = r_collocation.detach().clone().requires_grad_(True)
        Phi, b = model(r)

        dPhi = self._grad(Phi, r)
        db = self._grad(b, r)
        d2Phi = self._grad(dPhi, r)

        # ---- Boundary --------------------------------------------------
        r0_t = torch.tensor([[self.r0]], device=r.device, dtype=r.dtype)
        r0_t.requires_grad_(True)
        Phi0, b0 = model(r0_t)
        db0 = self._grad(b0, r0_t)
        r_far = torch.tensor([[self.r0 * 50.0]], device=r.device, dtype=r.dtype)
        _, b_far = model(r_far)

        mse_throat = (b0 - self.r0).pow(2).mean()
        mse_flare = torch.relu(db0 - 1.0 + 1e-3).pow(2).mean()        # b'(r0) < 1
        mse_asymp = (b_far / r_far.squeeze(-1)).pow(2).mean()         # b/r -> 0
        loss_boundary = mse_throat + mse_flare + mse_asymp

        # ---- Einstein field equations ---------------------------------
        # For Morris-Thorne with diagonal stress-energy:
        #   rho   =  b' / (8 pi r^2)
        #   p_r   = -b / (8 pi r^3) + 2 (1 - b/r) Phi' / (8 pi r)
        #   p_t   = (1 - b/r) * [ Phi'' + Phi'^2 + Phi'/r
        #                         - (b'r - b)/(2 r (r - b)) * (Phi' + 1/r) ] / (8 pi)
        # Conservation: dp_r/dr + (2/r)(p_r - p_t) + (rho + p_r) Phi' = 0
        rho = db / (8.0 * math.pi * r.squeeze(-1).pow(2))
        p_r = -b / (8.0 * math.pi * r.squeeze(-1).pow(3)) \
              + 2.0 * (1.0 - b / r.squeeze(-1)) * dPhi / (8.0 * math.pi * r.squeeze(-1))
        one_minus_bor = 1.0 - b / r.squeeze(-1)
        bracket = d2Phi + dPhi.pow(2) + dPhi / r.squeeze(-1) \
                  - (db * r.squeeze(-1) - b) / (2.0 * r.squeeze(-1) * (r.squeeze(-1) - b + 1e-9)) \
                    * (dPhi + 1.0 / r.squeeze(-1))
        p_t = one_minus_bor * bracket / (8.0 * math.pi)

        dp_r = self._grad(p_r, r)
        residual = dp_r + (2.0 / r.squeeze(-1)) * (p_r - p_t) + (rho + p_r) * dPhi
        loss_einstein = residual.pow(2).mean()

        # ---- Exotic-matter minimization -------------------------------
        # rho_exo = max(0, -(rho + p_r))   (null energy condition violation magnitude)
        nec_violation = torch.relu(-(rho + p_r))
        loss_exotic = nec_violation.pow(2).mean()

        total = (self.w.boundary * loss_boundary
                 + self.w.einstein * loss_einstein
                 + self.w.exotic * loss_exotic)
        return LossBreakdown(total, loss_boundary, loss_einstein, loss_exotic)
