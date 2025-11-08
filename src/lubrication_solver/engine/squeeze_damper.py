import numpy as np
from numpy.typing import NDArray

from .no_core import NoCore
from .floating_core import FloatingCore

from .cylindrical_shape import CylindricalShape
from .flow import FlowModel


class SqueezeDamper(CylindricalShape):
    def __init__(self, U1, U2, mu, R0, Ri):
        super().__init__(U1, U2, mu, R0, Ri)

        self._no_core = NoCore(U1, U2, mu)
        self._floating_core = FloatingCore(U1, U2, mu, R0, Ri)

    def _compute_h(self, x: float, epsilon: float):
        return super()._compute_h(x, epsilon)

    def _compute_p(self, q: float, epsilon, tau_0) -> NDArray:
        return super()._compute_p(q, epsilon, tau_0)

    def solve(
        self, tau_zero_star: float, epsilon: float, nb_points: int, verbose: bool
    ):
        return super().solve(tau_zero_star, epsilon, nb_points, verbose)

    def _compute_post_processing_datas(self, q, epsilon, tau_0):
        return super()._compute_post_processing_datas(q, epsilon, tau_0)

    @property
    def post_processing_datas(self):
        return super().post_processing_datas

    def _get_flow_type(self, h, q, tau_0) -> FlowModel:
        dp_dx_ref = self._no_core.compute_dp_dx(h, q)
        tau_zero = np.sign(dp_dx_ref) * tau_0

        ha = self._floating_core.h_a(h, dp_dx_ref, tau_zero)
        hb = self._floating_core.h_b(h, dp_dx_ref, tau_zero)

        if 0 < ha < hb < h:
            return self._floating_core

        return self._no_core
