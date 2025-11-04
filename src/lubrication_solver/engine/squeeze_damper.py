import numpy as np
import matplotlib.pyplot as plt

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

        self._nb_points = None
        self._verbose = False
        self._nb_floating_core = 0

        self._all_dpdx = []

    def _compute_h(self, x, epsilon):
        return self._c * (1 + epsilon * np.cos(x / self._R))

    def _compute_p(self, x_value:float, q:float, epsilon, tau_0) -> float:
        x = None
        if self._nb_points is None:
            raise ValueError("The grid has not yet been created !") 

        if x_value > 0:
            x = np.linspace(0, x_value, self._nb_points)
        else:
            x = np.array([x_value])

        dp_dxs = np.empty(len(x)) 
        dp_dx = 0

        for i, x_i in enumerate(x):
            h = self._compute_h(x_i, epsilon)
            dp_dx = self._no_core.compute_dp_dx(h, q)
            tau_zero = -np.sign(dp_dx) * tau_0
            
            ha = self._floating_core.h_a(h, dp_dx, tau_zero)
            hb = self._floating_core.h_b(h, dp_dx, tau_zero) 
            
            situation = ""

            if 0 < ha < hb < h:
                situation = "floating core"

                dp_dxs_cubique = self._floating_core._compute_dp_dx(q, h, tau_zero, dp_dx)
                min = np.argmin(np.abs(dp_dxs_cubique - dp_dxs[i - 1]))
                dp_dx = dp_dxs_cubique[min]
                
                tau_zero = - np.sign(tau_0)
                ha = self._floating_core.h_a(h, dp_dx, tau_zero)
                hb = self._floating_core.h_b(h, dp_dx, tau_zero)

                self._nb_floating_core += 1
            else:
                situation = "no core"
                
            if self._verbose:
                print("=======================")
                print(f"Cas {i} : {situation}")
                print("=======================")
                print(f"Pour x_i = {x_i:.2f}, theta = {x_i / (self._R * np.pi):.2f}π ")
                print("¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯")
                print(f"dpdx = {dp_dx:.2e}, sign = {np.sign(dp_dx)}, tau_zero = {tau_zero:.2e}")
                print(f"ha = {ha:.2e} (m), hb = {hb:.2e} (m), h = {h:.2e} (m)")
                print()

            dp_dxs[i] = dp_dx
        
        if np.abs(q - self._q_prec) > 1e-2:
            self._all_dpdx.append((dp_dxs, q))
            self._q_prec = q

         
        I = np.trapezoid(dp_dxs, x)

        return I 

    def solve(self, tau_zero_star:float, epsilon:float, nb_points:int, verbose: bool):
        tau_0 = np.abs((tau_zero_star * self._mu * self._omega_R) / self._c)
        self._nb_points = nb_points
        self._verbose = verbose

        q = (
            -2
            * self._omega_R
            * self._c
            * (1 - epsilon**2)
            / (2 + epsilon**2)
        )

        q_star = q / (self._omega_R * self._c)
        print(f"q* = {q_star:.3f}")


        self._q_prec = 0

        def residus(q):
            return self._compute_p(2 * np.pi * self._R, q, epsilon, tau_0)

        newton_solver = NewtonSolver([residus], vb=True)
        
        try:
            q = newton_solver.solve(np.array([q]), 25, 1e-4).unwrap()
        except Exception as e:
            print(f"Error : {e}")
        
        q_star = q / (self._omega_R * self._c)
        print(f"q* = {q_star:.3f}")
        print(f"Nb Floating Core : {self._nb_floating_core}")

        #self._compute_p(2 * np.pi * self._R, q, epsilon, tau_0)
        
        x = np.linspace(0, 2 * np.pi * self._R, nb_points)
        plt.figure(figsize=(10, 8))
        for i, (dpdx, q) in enumerate(self._all_dpdx):
            plt.plot(x, dpdx, '+', label=f"Iter {i} : q = {q}")
        
        plt.grid(True, alpha=0.5)
        plt.legend()
        plt.show()

    
