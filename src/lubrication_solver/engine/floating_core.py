import numpy as np
from numpy.polynomial.polynomial import Polynomial

from .flow import FlowModel


class FloatingCore(FlowModel):
    def __init__(self, U1, U2, mu, R0, Ri):
        super().__init__(U1, U2, mu)

        self._R = R0
        self._omega_R = np.abs(-U1)
        self._c = R0 - Ri

        self._tol_complex_number = 1e-10

    def tau_1(self, h, dp_dx, tau_0):
        pre_factor = -dp_dx / (2 * h * dp_dx + 4 * tau_0)

        parenthesis = (
            h**2 * dp_dx + 2 * h * tau_0 + 2 * self._mu * (self._U1 - self._U2)
        )

        return pre_factor * parenthesis

    def tau_2(self, h, dp_dx, tau_0):
        return self.tau_1(h, dp_dx, tau_0) + h * dp_dx

    def h_a(self, h, dp_dx, tau_0):
        return (tau_0 - self.tau_1(h, dp_dx, tau_0)) / dp_dx

    def h_b(self, h, dp_dx, tau_0):
        return self.h_a(h, dp_dx, tau_0) - 2 * tau_0 / dp_dx

    def _cubic_dp_dx(self, q, h, tau_0):
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
        self, h: float, q: float, tau_0: float = 0, reference_flux: float = 0
    ) -> float:
        poly = self._cubic_dp_dx(q, h, tau_0)
        roots = poly.roots()

        real_roots = [
            np.real(root)
            for root in roots
            if np.abs(np.imag(root)) < self._tol_complex_number
        ]

        if len(real_roots) == 0:
            return reference_flux

        same_sign_roots = np.array([
            root 
            for root in real_roots 
            if np.sign(root) == np.sign(reference_flux)
        ])
        
        #print(same_sign_roots)

        if len(same_sign_roots) == 0:
            return reference_flux
        
        index_right_dpdx = np.argmin(np.abs(same_sign_roots - reference_flux))

        return same_sign_roots[index_right_dpdx]

        
