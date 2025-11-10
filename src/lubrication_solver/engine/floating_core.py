import numpy as np
from numpy.polynomial.polynomial import Polynomial
from numpy.typing import NDArray

from typing import Tuple, Optional

from .flow import FlowModel


class FloatingCore(FlowModel):
    def __init__(self, U1: float, U2: float, mu: float, R0: float, Ri: float):
        super().__init__(U1, U2, mu)

        self._R = R0
        self._omega_R = np.abs(-U1)
        self._c = R0 - Ri

        self._tol_complex_number = 1e-10

    def tau_1(self, h: float, dp_dx: float, tau_0: float) -> float:
        pre_factor = -dp_dx / (2 * h * dp_dx + 4 * tau_0)

        parenthesis = (
            h**2 * dp_dx + 2 * h * tau_0 + 2 * self._mu * (self._U1 - self._U2)
        )

        return pre_factor * parenthesis

    def tau_2(self, h: float, dp_dx: float, tau_0: float) -> float:
        return self.tau_1(h, dp_dx, tau_0) + h * dp_dx

    def h_a(self, h: float, dp_dx: float, tau_0: float) -> float:
        return (tau_0 - self.tau_1(h, dp_dx, tau_0)) / dp_dx

    def h_b(self, h: float, dp_dx: float, tau_0: float) -> float:
        return self.h_a(h, dp_dx, tau_0) - 2 * tau_0 / dp_dx

    def _cubic_dp_dx(self, q: float, h: float, tau_0: float):
        coeffs = np.array(
            [
                -4 * tau_0**3,
                0,
                3 * (4 * self._mu * (self._omega_R * h + q) + h**2 * tau_0),
                h**3,
            ]
        )

        return Polynomial(coef=coeffs)

    def compute_dp_dx(
        self,
        h: float,
        q: float,
        tau_0: Optional[float] = None,
        reference_flux: Optional[float] = None,
    ) -> float:
        if tau_0 is None or reference_flux is None:
            raise ValueError("tau_0 and the reference_flux must be specify")

        poly = self._cubic_dp_dx(q, h, tau_0)
        roots = poly.roots()

        real_roots = [
            np.real(root)
            for root in roots
            if np.abs(np.imag(root)) < self._tol_complex_number
        ]

        if len(real_roots) == 0:
            return reference_flux

        same_sign_roots = np.array(
            [root for root in real_roots if np.sign(root) == np.sign(reference_flux)]
        )

        if len(same_sign_roots) == 0:
            return reference_flux

        index_right_dpdx = np.argmin(np.abs(same_sign_roots - reference_flux))

        return same_sign_roots[index_right_dpdx]

    def _compute_u_0_ha(self, y: float, h: float, dp_dx: float, tau_0: float) -> float:
        return self._U1 - (
            (tau_0 - self.tau_1(h, dp_dx, tau_0) / self._mu) * y
            + dp_dx * y**2 / (2 * self._mu)
        )

    def _compute_u_ha_hb(
        self, ha: float, h: float, dp_dx: float, tau_0: float
    ) -> float:
        return (
            self._U1
            - ((tau_0 - self.tau_1(h, dp_dx, tau_0)) / self._mu) * ha
            + dp_dx * ha**2 / 2 * self._mu
        )

    def _compute_u_hb_h(self, y: float, h: float, tau_0: float, dp_dx: float) -> float:
        return (
            self._U2
            - ((tau_0 - self.tau_2(h, dp_dx, tau_0)) / self._mu) * (y - h)
            + dp_dx * (y - h) ** 2 / 2 * self._mu
        )

    def compute_u(
        self,
        h: float,
        q: float,
        tau_0: Optional[float] = None,
        reference_flux: Optional[float] = None,
        nb_points: int = 201,
    ) -> Tuple[NDArray, NDArray]:
        if tau_0 is None or reference_flux is None:
            raise ValueError("tau_0 and the reference_flux must be specify")

        y = np.linspace(0, h, nb_points)
        u = np.empty(nb_points)

        dp_dx = self.compute_dp_dx(h, q, tau_0, reference_flux)

        ha = self.h_a(h, dp_dx, tau_0)
        hb = self.h_b(h, dp_dx, tau_0)

        for i in range(u.shape[0]):
            if 0 < y[i] < ha:
                u[i] = self._compute_u_0_ha(y[i], h, dp_dx, tau_0)
            elif ha < y[i] < hb:
                u[i] = self._compute_u_ha_hb(ha, h, dp_dx, tau_0)
            elif hb < y[i] < h:
                u[i] = self._compute_u_hb_h(y[i], h, dp_dx, tau_0)

        return y, u
