"""PyVista volumetric rendering of |psi|^2."""
from __future__ import annotations

import numpy as np
import pyvista as pv


def render_density(density: np.ndarray, *, opacity: str = "sigmoid",
                   cmap: str = "magma", screenshot: str | None = None) -> None:
    if density.ndim != 3:
        raise ValueError("density must be a 3D array")
    grid = pv.ImageData(dimensions=density.shape)
    grid.point_data["psi2"] = density.flatten(order="F")

    p = pv.Plotter(off_screen=screenshot is not None)
    p.add_volume(grid, scalars="psi2", cmap=cmap, opacity=opacity)
    p.add_axes()
    if screenshot:
        p.show(screenshot=screenshot)
    else:
        p.show()
