from abc import ABC, abstractmethod

from numpy.typing import NDArray
from typing import Tuple, Optional


class FlowModel(ABC):
    def __init__(self, U1: float, U2: float, mu: float):
        self._U1 = U1
        self._U2 = U2
        self._mu = mu

    @abstractmethod
    def h_a(self, h: float, dp_dx: float, tau_0: float) -> float:
        pass

    @abstractmethod
    def h_b(self, h: float, dp_dx: float, tau_0: float) -> float:
        pass

    @abstractmethod
    def compute_dp_dx(
        self,
        h: float,
        q: float,
        tau_0: Optional[float] = None,
        reference_flux: Optional[float] = None,
    ) -> float:
        pass

    @abstractmethod
    def compute_u(
        self,
        h: float,
        q: float,
        tau_0: Optional[float] = None,
        reference_flux: Optional[float] = None,
        nb_points: int = 201,
    ) -> Tuple[NDArray, NDArray]:
        pass
