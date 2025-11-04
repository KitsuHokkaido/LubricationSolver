import numpy as np
from numpy.polynomial.polynomial import Polynomial
from numpy.typing import NDArray

class FloatingCore:
    def __init__(self, U1, U2, mu, R0=0, Ri=0):
        self._U1 = U1
        self._U2 = U2
        self._mu = mu
        self._R = R0
        self._omega_R = -U1
        self._c = R0 - Ri

        self._tol_complex_number = 1e-10

    def tau_1(self, h, dp_dx, tau_0):
        pre_factor = - dp_dx / (2 * h * dp_dx + 4 * tau_0)

        parenthesis = (
            h**2 * dp_dx
            + 2 * h * tau_0
            + 2 * self._mu * (self._U1 - self._U2)
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
                3
                * (
                    4 * self._mu * (self._omega_R * h + q)
                    + h**2 * tau_0
                ),
                h**3,
            ]
        )

        return Polynomial(coef=coeffs) 


    def _compute_dp_dx(self, q, h, tau_0, reference_flux) -> NDArray:
        poly = self._cubic_dp_dx(q, h, tau_0)
        roots = poly.roots()

        real_roots = np.array([np.real(root) for root in roots if np.imag(root) < self._tol_complex_number])
        
        true_root = []
        for root in real_roots:
            if np.sign(root) == np.sign(reference_flux):
                ha = self.h_a(h, root, -np.sign(root) * np.abs(tau_0))
                hb = self.h_b(h, root, -np.sign(root) * np.abs(tau_0))

                if 0 < ha < hb < h:
                    true_root.append(root)

        return np.array(true_root)




 
