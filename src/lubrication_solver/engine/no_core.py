from .flow import FlowModel


class NoCore(FlowModel):
    def __init__(self, U1, U2, mu):
        super().__init__(U1, U2, mu)

    def compute_dp_dx(
        self, h: float, q: float, tau_0: float = 0, reference_flux: float = 0
    ) -> float:
        return -12 * self._mu * q / h**3 + 6 * self._mu * (self._U1 + self._U2) / h**2

    def h_a(self, h, dp_dx, tau_0) -> float:
        return h/2

    def h_b(self, h, dp_dx, tau_0) -> float:
        return h/2

    def compute_u(self, y, h, q):
        dp_dx = self.compute_dp_dx(h, q)

        return (
            self._U1
            + (self._U2 - self._U1) * y / h
            + 1 / (2 * self._mu) * dp_dx * (y**2 - y * h)
        )
