from .cylindrical_shape import CylindricalShape


class JournalBearing(CylindricalShape):
    def __init__(self, U1, U2, mu, R0, Ri):
        super().__init__(U1, U2, mu, R0, Ri)
