from abc import ABC, abstractmethod


class FlowModel(ABC):
    def __init__(self, U1: float, U2: float, mu: float):
        self._U1 = U1
        self._U2 = U2
        self._mu = mu

    @abstractmethod
    def h_a(self, h, dp_dx, tau_0) -> float:
        pass

    @abstractmethod
    def h_b(self, h, dp_dx, tau_0) -> float:
        pass

    @abstractmethod
    def compute_dp_dx(
        self, h: float, q: float, tau_0: float = 0, reference_flux: float = 0
    ) -> float:
        pass
