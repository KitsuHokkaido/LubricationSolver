class NoCore:
    def __init__(self, U1, U2, mu):
        self._U1 = U1 
        self._U2 = U2
        self._mu = mu

    def compute_dp_dx(self, h, q):
        return -12*self._mu * q / h**3 + 6*self._mu*(self._U1 + self._U2) / h**2

    def compute_u(self, y, h, q):
        dp_dx = self.compute_dp_dx(h, q)

        return self._U1 + (self._U2 - self._U1) * y / h + 1 / (2 * self._mu) * dp_dx * (y**2 - y*h)

