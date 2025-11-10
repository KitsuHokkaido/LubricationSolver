from .cylindrical_shape import CylindricalShape


class JournalBearing(CylindricalShape):
    def __init__(self, U1: float, U2: float, mu: float, R0: float, Ri: float):
        super().__init__(U1, U2, mu, R0, Ri)
