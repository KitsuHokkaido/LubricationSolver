import numpy as np
from typing import Callable, List, Tuple
from numpy.typing import NDArray
from multiprocessing import Pool, cpu_count
import time

from ..error.error import Result

class NewtonSolver:
    def __init__(self, res: List[Callable], vb: bool):
        self._deg = len(res)
        self._res = res
        self._h = 0.0001
        self._verbose = vb

    def _derivative(self, func: Callable, x: float) -> float:
        d = (func(x + self._h * x) - func(x)) / (self._h * x)
        return d

    def _partial_derivative(self, func: Callable, var_deriv: int, x0: NDArray) -> float:
        x = np.copy(x0)
        x[var_deriv - 1] = x[var_deriv - 1] + x[var_deriv - 1] * self._h
        d = (func(x) - func(x0)) / (x[var_deriv - 1] * self._h)
        return d

    def _solve_1d(self, s_init: NDArray, max_iter: int, tol: float) -> Result:
        s = s_init[0]
        func = self._res[0]
        for i in range(max_iter):
            if self._verbose:
                print(f"Iteration {i + 1}")
                print("--------------")
                print(f"s = {s:0.3f}")
                print(f"res = {np.abs(func(s)):.3f}")
                print()

            s_new = s - (func(s) / self._derivative(func, s))

            if np.abs(func(s_new)) < tol:
                print("=============================================")
                print(f"||   Solution converged in {i + 1} iterations.   ||")
                print("=============================================")
                print("Root :")
                print("¯¯¯¯¯¯")
                print(f"s = {s:0.3f}")
                print(f"res = {np.abs(func(s_new)):.2e}")

                return Result(value=s_new)

            s = s_new

        return Result(error="did not converge !")

    def _compute_row(self, args) -> Tuple[int, NDArray]:
        i, s = args

        row = np.empty(self._deg)

        for j in range(self._deg):
            row[j] = self._partial_derivative(self._res[i], j + 1, s)

        return i, row

    def _compute_jacobian_n(self, s: NDArray, pool) -> NDArray:
        jacobian = np.empty((self._deg, self._deg))

        args = [(i, s) for i in range(self._deg)]

        results = pool.map(self._compute_row, args)

        for i, row in results:
            jacobian[i, :] = row

        return jacobian

    def _all_res_inf_to(self, s: NDArray, tol: float) -> bool:
        for i in range(self._deg):
            if np.abs(self._res[i](s)) > tol:
                return False
        return True

    def _solve_nd(self, s_init: NDArray, max_iter: int, tol: float, pool) -> Result:
        s = s_init
        for i in range(max_iter):
            if self._verbose:
                print(f"Iteration {i + 1}")
                print("--------------")
                print(f"s1 = {s[0]:0.3f}           s4 = {s[3]:0.3f}")
                print(f"s2 = {s[1]:0.3f}           s5 = {s[4]:0.3f}")
                print(f"s3 = {s[2]:0.3f}")
                print()

            J = self._compute_jacobian_n(s, pool)
            R = np.array([self._res[j](s) for j in range(self._deg)])
            ds = np.linalg.solve(J, -R)

            s_new = s + ds

            per = (i + 1) * 100 // max_iter
            print(f"\r{per} % of max iterations...", end=" ", flush=True)

            if self._all_res_inf_to(s_new, tol):
                print("")
                print("=============================================")
                print(f"||   Solution converged in {i + 1} iterations.   ||")
                print("=============================================")
                print()
                print("Roots : ")
                print("¯¯¯¯¯¯¯")
                print(f"s1 = {s[0]:0.3f}           s4 = {s[3]:0.3f}")
                print(f"s2 = {s[1]:0.3f}           s5 = {s[4]:0.3f}")
                print(f"s3 = {s[2]:0.3f}")

                return Result(value=s_new)

            s = s_new

        return Result(error="Did not converge")

    def solve(self, s_init: NDArray[np.float64], max_iter: int, tol: float) -> Result:
        if self._deg == 1:
            start_time = time.time()
            result = self._solve_1d(s_init, max_iter, tol)
            deltaT = time.time() - start_time

            print("________________________")
            print(f"Execution time : {deltaT:0.2f} s")
            print("¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯")

            return result

        elif self._deg >= 2:
            nb_process = max(1, cpu_count() - 2)

            start_time = time.time()
            with Pool(processes=nb_process) as pool:
                result = self._solve_nd(s_init, max_iter, tol, pool)
            deltaT = time.time() - start_time

            print("________________________")
            print(f"Execution time : {deltaT:0.2f} s")
            print("¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯")

            return result

        return Result(error="Invalid degree")
