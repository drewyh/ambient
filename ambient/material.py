"""Material property classes."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass
class MaterialBase:
    """Base material related information."""

    # XXX: subclasses must provide thermal_resistance attribute

    def calculate_response_matrices(self, frequencies: np.ndarray) -> np.ndarray:
        """Calculate the frequency response for the material."""
        raise NotImplementedError()


@dataclass
class Material(MaterialBase):
    """Material related information."""

    conductivity: ClassVar[float]  #: Material conductivity [W/m.K]
    density: ClassVar[float]  #: Material density [kg/m^3]
    specific_heat: ClassVar[float]  #: Material specific heat capacity [J/kg.K]
    thickness: float  #: Material thickness [m]

    def __post_init__(self) -> None:
        """Set the thermal resistance of the material [m^2.K/W]."""
        self.thermal_resistance = self.thickness / self.conductivity

    def calculate_response_matrices(self, frequencies: np.ndarray) -> np.ndarray:
        """Calculate the frequency response for the material."""
        # common arg for sinh/cosh
        values = (
            self.thickness
            * np.sqrt(self.density * self.specific_heat / self.conductivity)
            * np.sqrt(frequencies * 1.0j),
        )

        # precompute sinh/cosh value
        sinh = np.sinh(values)
        cosh = np.cosh(values)

        # create an empty array of the correct size
        matrices = np.empty((frequencies.size, 2, 2), dtype=np.complex)

        # matrix components
        matrices[:, 0, 0] = cosh
        matrices[:, 0, 1] = self.thermal_resistance * sinh / values
        matrices[:, 1, 0] = values * sinh / self.thermal_resistance
        matrices[:, 1, 1] = cosh

        return matrices


@dataclass
class MaterialResistanceOnly(MaterialBase):
    """Resistance only material related information."""

    thermal_resistance: float  #: Material resistance [m^2.K/W]

    def calculate_response_matrices(self, frequencies: np.ndarray) -> np.ndarray:
        """Calculate the frequency response for the material."""
        # create an empty array of the correct size
        matrices = np.ones((frequencies.size, 2, 2), dtype=np.complex)

        # matrix components
        matrices[:, 0, 1] = self.thermal_resistance
        matrices[:, 1, 0] = 0.0

        return matrices
