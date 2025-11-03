import numpy as np

from .floating_core import FloatingCore
from .no_core import NoCore

from ..newton_solver.solver import NewtonSolver

class SqueezeDamper:
    def __init__(self, U1, U2, mu, R0, Ri): 
        self._mu = mu
        
        self._c = R0 - Ri
        self._omega_R = -U1
        self._R = R0 

        self._floating_core = FloatingCore(U1, U2, mu, R0, Ri)
        self._no_core = NoCore(U1, U2, mu)

    def _compute_h(self, x, epsilon):
        return self._c * (1 + epsilon * np.cos(x / self._R))

    def _compute_p(self, x_value:float, q:float, epsilon, tau_0) -> float:
        x = None
        if x_value > 0:
            x = np.linspace(0, x_value, 100)
        else:
            x = np.array([x_value])

        dp_dxs = np.empty(len(x)) 

        for i, x_i in enumerate(x):
            h = self._compute_h(x_i, epsilon)
            dp_dx = self._no_core.compute_dp_dx(h, q)
            tau_zero = - np.sign(dp_dx) * tau_0
            
            ha = self._floating_core.h_a(h, dp_dx, tau_zero)
            hb = self._floating_core.h_b(h, dp_dx, tau_zero)
            
            print(f"Pour x_i = {x_i:.2f}, theta = {x_i / (self._R * np.pi):.2f}π ")
            print("¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯")
            print(f"dpdx = {dp_dx:.2e}, sign = {np.sign(dp_dx)}, tau_zero = {tau_zero:.2e}")
            print(f"ha = {ha:.2e} (m), hb = {hb:.2e} (m), h = {h:.2e} (m)")

            if 0 < ha < hb < h:
                print(f"Cas {x_i} : floating core")
                dp_dx = self._floating_core._compute_dp_dx(q, h, tau_zero)
            
            print()
            dp_dxs[i] = dp_dx
        
        print(f"dp_dxs : {dp_dxs}")

        I = np.trapezoid(dp_dxs, x)
        print(I)

        return I 

    def solve(self, tau_zero_star, epsilon):
        tau_0 = (tau_zero_star * self._mu * self._omega_R) / self._c

        q = (
            -2
            * self._omega_R
            * self._c
            * (1 - epsilon**2)
            / (2 + epsilon**2)
        )

        q_star = q / (self._omega_R * self._c)
        print(f"q* = {q_star:.2f}")


        def residus(q):
            return self._compute_p(2 * np.pi * self._R, q, epsilon, tau_0) - self._compute_p(0, q, epsilon, tau_0)

        newton_solver = NewtonSolver([residus], vb=True)

        q = newton_solver.solve(np.array([q]), 25, 1e-7).unwrap()
        
        q_star = q / (self._omega_R * self._c)
        print(f"q* = {q_star:.3f}")

        #self._compute_p(2 * np.pi * self._R, q, epsilon, tau_0)

    
