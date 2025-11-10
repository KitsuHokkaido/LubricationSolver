import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional

from .flow import FlowModel


class NoCore(FlowModel):
    def __init__(self, U1: float, U2: float, mu: float):
        super().__init__(U1, U2, mu)

    def compute_dp_dx(
        self,
        h: float,
        q: float,
        tau_0: Optional[float] = None,
        reference_flux: Optional[float] = None,
    ) -> float:
        return -12 * self._mu * q / h**3 + 6 * self._mu * (self._U1 + self._U2) / h**2

    def h_a(self, h: float, dp_dx: float, tau_0: float) -> float:
        return h / 2

    def h_b(self, h: float, dp_dx: float, tau_0: float) -> float:
        return h / 2

    def compute_u(
        self,
        h: float,
        q: float,
        tau_0: Optional[float] = None,
        reference_flux: Optional[float] = None,
        nb_points: int = 201,
    ) -> Tuple[NDArray, NDArray]:
        y = np.linspace(0, h, nb_points)

        dp_dx = self.compute_dp_dx(h, q)

        u = (
            self._U1
            + (self._U2 - self._U1) * y / h
            + 1 / (2 * self._mu) * dp_dx * (y**2 - y * h)
        )

        return y, u
