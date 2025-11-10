import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from .no_core import NoCore
from .floating_core import FloatingCore

from .cylindrical_shape import CylindricalShape
from .flow import FlowModel


class SqueezeDamper(CylindricalShape):
    def __init__(self, U1: float, U2: float, mu: float, R0: float, Ri: float):
        super().__init__(U1, U2, mu, R0, Ri)

    def _get_flow_type(
        self, h: float, q: float, tau_0: float
    ) -> Tuple[FlowModel, float]:
        dp_dx_ref = self._no_core.compute_dp_dx(h, q)
        tau_zero = -np.sign(dp_dx_ref) * tau_0

        dp_dx = self._floating_core.compute_dp_dx(h, q, tau_zero, dp_dx_ref)

        tau_zero = -np.sign(dp_dx_ref) * tau_0

        ha = self._floating_core.h_a(h, dp_dx, tau_zero)
        hb = self._floating_core.h_b(h, dp_dx, tau_zero)

        if 0 < ha < hb < h:
            return self._floating_core, dp_dx_ref

        return self._floating_core, dp_dx_ref
