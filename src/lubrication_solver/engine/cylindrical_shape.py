from abc import abstractmethod
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from typing import Tuple, Dict


from .no_core import NoCore
from .floating_core import FloatingCore
from .flow import FlowModel

from ..newton_solver.solver import NewtonSolver


class CylindricalShape:
    def __init__(self, U1: float, U2: float, mu: float, R0: float, Ri: float):
        self._mu = mu

        self._c = R0 - Ri
        self._omega_R = np.abs(-U1)
        self._R = R0

        self._verbose = False
        self._grid = None
        self._datas = None

        self._no_core = NoCore(U1, U2, mu)
        self._floating_core = FloatingCore(U1, U2, mu, R0, Ri)

    def _set_grid(self, nb_points: int) -> None:
        self._grid = np.linspace(0, 2 * np.pi * self._R, nb_points)

    def _compute_h(self, x: float, epsilon: float) -> float:
        return self._c * (1 + epsilon * np.cos(x / self._R))

    @abstractmethod
    def _get_flow_type(
        self, h: float, q: float, tau_0: float
    ) -> Tuple[FlowModel, float]:
        pass

    def _compute_dp_dxs(self, q: float, epsilon: float, tau_0: float) -> NDArray:
        if self._grid is None:
            raise ValueError("Incorrect initialisation of the grid")

        dp_dxs = np.empty(self._grid.shape[0])
        for i in range(self._grid.shape[0]):
            h = self._compute_h(self._grid[i], epsilon)

            flow, ref_flux = self._get_flow_type(h, q, tau_0)

            tau_zero = -np.sign(ref_flux) * tau_0

            dp_dxs[i] = flow.compute_dp_dx(h, q, tau_zero, ref_flux)

        return dp_dxs

    def _compute_all_ha_hb(
        self, q: float, epsilon: float, tau_0: float
    ) -> Tuple[NDArray, NDArray]:
        if self._grid is None:
            raise ValueError("Incorrect initialisation of the grid")

        ha = np.empty(self._grid.shape[0])
        hb = np.empty(self._grid.shape[0])

        for i in range(self._grid.shape[0]):
            h = self._compute_h(self._grid[i], epsilon)

            flow, ref_flux = self._get_flow_type(h, q, tau_0)
            tau_zero = -np.sign(ref_flux) * tau_0

            dp_dx = flow.compute_dp_dx(h, q, tau_zero, ref_flux)

            tau_zero = -np.sign(dp_dx) * tau_0

            ha[i] = flow.h_a(h, dp_dx, tau_zero) / h
            hb[i] = flow.h_b(h, dp_dx, tau_zero) / h

        return ha, hb

    def _compute_p(self, q: float, epsilon: float, tau_0: float) -> NDArray:
        dp_dxs = self._compute_dp_dxs(q, epsilon, tau_0)

        I = cumulative_trapezoid(dp_dxs, self._grid, initial=0.0)
        return I

    def solve(
        self, tau_zero_star: float, epsilon: float, nb_points: int, verbose: bool
    ) -> None:
        self._verbose = verbose

        tau_0 = np.abs((tau_zero_star * self._mu * self._omega_R) / self._c)

        self._set_grid(nb_points)

        q = -2 * self._omega_R * self._c * (1 - epsilon**2) / (2 + epsilon**2)

        q_star = q / (self._omega_R * self._c)
        print(
            f"Initializing flow rate with newtonian fluid hypothesis: q* = {q_star:.3f}, q = {q:0.3f}"
        )
        print()

        def residus(q):
            p = self._compute_p(q, epsilon, tau_0)
            return p[-1] - p[0]

        newton_solver = NewtonSolver([residus], vb=verbose)

        print("Solving the periodicity pressure...")
        try:
            q = newton_solver.solve(np.array([q]), 25, 1e-4).unwrap()
        except Exception as e:
            print(f"Error : {e}")
            print(f"Solution did not converge, τ0*=0 will be used")

        q_star = q / (self._omega_R * self._c)
        print(f"Final solution : q* = {q_star:.3f}")
        print("")

        print("Computing post processing datas...")
        self._compute_post_processing_datas(q, epsilon, tau_0, nb_points)
        print("Done !")
        print("")

    def _compute_post_processing_datas(
        self, q: float, epsilon: float, tau_0: float, nb_points: int
    ) -> None:
        if self._grid is None:
            raise RuntimeError("A problem occurs when setting the grid")

        q = -0.3835 * self._omega_R * self._c

        dp_dxs = self._compute_dp_dxs(q, epsilon, tau_0)
        p = self._compute_p(q, epsilon, tau_0)

        p = ((p - p[0]) * self._c**2) / (self._mu * self._omega_R * self._R)

        ha, hb = self._compute_all_ha_hb(q, epsilon, tau_0)

        q_star = q / (self._omega_R * self._c)
        tau_zero_star = np.abs(tau_0 * self._c / (self._mu * self._omega_R))

        x = 0.5 * 2 * np.pi * self._R
        h = self._compute_h(x, epsilon)
        flow, reference_flux = self._get_flow_type(h, q, tau_0)
        u = flow.compute_u(h, q, tau_0, reference_flux, nb_points)

        flow_type = ""
        if isinstance(flow, FloatingCore):
            flow_type = "floating core"
        else:
            flow_type = "no core"

        self._datas = {
            "theta": self._grid / (np.pi * self._R),
            "p*": p,
            "ha": ha,
            "hb": hb,
            "q*": q_star,
            "epsilon": epsilon,
            "tau_0*": tau_zero_star,
            "dp_dxs": self._R * dp_dxs,
            "u": (u, flow_type),
        }

    @property
    def post_processing_datas(self) -> Dict:
        if self._datas is None:
            raise ValueError(
                "Please compute the solution in order to exploit post processing datas"
            )
        return self._datas
