from enum import Enum

from .squeeze_damper import SqueezeDamper

class Geometry(Enum):
    SQUEEZE = 1
    JOURNAL = 2

class LubricationSolver:
    def __init__(self):
        self._squeeze_film_damper = SqueezeDamper(100, 100, 0.2, 0.2, 0.18)

    def solve(self, geometry_type):
        if geometry_type == Geometry.SQUEEZE: 
            self._squeeze_film_damper.solve(tau_zero_star=5, epsilon=0.75, nb_points=201, verbose=False)

