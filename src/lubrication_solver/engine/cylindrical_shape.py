from abc import abstractmethod
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from typing import Tuple

from ..newton_solver.solver import NewtonSolver
from .flow import FlowModel


class CylindricalShape:
    def __init__(self, U1, U2, mu, R0, Ri):
        self._mu = mu

        self._c = R0 - Ri
        self._omega_R = -U1
        self._R = R0

        self._verbose = False
        self._grid = None
        self._datas = None

    def _set_grid(self, nb_points):
        self._grid = np.linspace(0, 2 * np.pi * self._R, nb_points)

    def _compute_h(self, x: float, epsilon: float):
        return self._c * (1 + epsilon * np.cos(x / self._R))

    @abstractmethod
    def _get_flow_type(self, h, q, tau_0) -> FlowModel:
        pass

    def _compute_dp_dxs(self, q, epsilon, tau_0) -> NDArray:
        if self._grid is None:
            raise ValueError("Incorrect initialisation of the grid")

        dp_dxs = np.empty(self._grid.shape[0])
        for i in range(self._grid.shape[0]):
            h = self._compute_h(self._grid[i], epsilon)

            flow = self._get_flow_type(h, q, tau_0)
            dp_dxs[i] = flow.compute_dp_dx(h, q, tau_0)

        return dp_dxs

    def _compute_all_ha_hb(self, q, epsilon, tau_0) -> Tuple[NDArray, NDArray]:
        if self._grid is None:
            raise ValueError("Incorrect initialisation of the grid")

        ha = np.empty(self._grid.shape[0])
        hb = np.empty(self._grid.shape[0])

        for i in range(self._grid.shape[0]):
            h = self._compute_h(self._grid[i], epsilon)

            flow = self._get_flow_type(h, q, tau_0)
            dp_dx = flow.compute_dp_dx(h, q, tau_0)

            tau_zero = np.sign(dp_dx) * tau_0

            ha[i] = flow.h_a(h, dp_dx, tau_zero) / h
            hb[i] = flow.h_b(h, dp_dx, tau_zero) / h

        return ha, hb

    def _compute_p(self, q: float, epsilon, tau_0) -> NDArray:
        dp_dxs = self._compute_dp_dxs(q, epsilon, tau_0)

        I = cumulative_trapezoid(dp_dxs, self._grid, initial=0.0)
        return I

    def solve(
        self, tau_zero_star: float, epsilon: float, nb_points: int, verbose: bool
    ):
        tau_0 = np.abs((tau_zero_star * self._mu * self._omega_R) / self._c)
        self._verbose = verbose

        self._set_grid(nb_points)

        q = -2 * self._omega_R * self._c * (1 - epsilon**2) / (2 + epsilon**2)

        q_star = q / (self._omega_R * self._c)
        print(
            f"Initialisation débit avec hypothèse fluide newtonien : q* = {q_star:.3f}, q = {q:0.3f}"
        )
        print()

        def residus(q):
            return self._compute_p(q, epsilon, tau_0)[-1]

        newton_solver = NewtonSolver([residus], vb=True)

        print("Solving the periodicity pressure...")
        try:
            q = newton_solver.solve(np.array([q]), 25, 1e-4).unwrap()
        except Exception as e:
            print(f"Error : {e}")

        q_star = q / (self._omega_R * self._c)
        print(f"Final solution : q* = {q_star:.3f}")
        print("")

        print("Computing post processing datas...")
        self._compute_post_processing_datas(q, epsilon, tau_0)
        print("Done !")
        print("")

    def _compute_post_processing_datas(self, q, epsilon, tau_0):
        if self._grid is None:
            return

        q = -0.934 * self._omega_R * self._c

        p = self._compute_p(q, epsilon, tau_0)
        p = ((p - p[0]) * self._c**2)/ (self._mu * self._omega_R * self._R)

        ha, hb = self._compute_all_ha_hb(q, epsilon, tau_0)
        
        q_star = q / (self._omega_R * self._c) 
        tau_zero_star = np.abs(tau_0 * self._c / (self._mu * self._omega_R))

        self._datas = {"x": self._grid / (np.pi * self._R), "p": p, "ha": ha, "hb": hb, "q*": q_star, "epsilon": epsilon, "tau_0*": tau_zero_star}

    @property
    def post_processing_datas(self):
        return self._datas
