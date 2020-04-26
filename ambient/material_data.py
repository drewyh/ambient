"""Material instances for use."""

from ambient.material import Material


class Brickwork(Material):
    """Chen and Wang Table 1 Wall I."""

    conductivity = 0.840
    density = 1700
    specific_heat = 800


class Brickwork2(Material):
    """Chen and Wang Table 1 Wall III."""

    conductivity = 0.81
    density = 1800
    specific_heat = 880


class LimePlaster(Material):
    """Chen and Wang Table 1 Wall I."""

    conductivity = 0.7
    density = 1600
    specific_heat = 880


class HeavyweightConcrete(Material):
    """Chen and Wang Table 1 Wall III."""

    conductivity = 1.63
    density = 2300
    specific_heat = 1000


class Stucco(Material):
    """Chen and Wang Table 1 Wall II."""

    conductivity = 0.6924
    density = 1858
    specific_heat = 836.8  # note the values in the paper are incorrect


class HighDensityConcrete(Material):
    """Chen and Wang Table 1 Wall II."""

    conductivity = 1.731
    density = 2243
    specific_heat = 836.8  # note the values in the paper are incorrect


class Insulation(Material):
    """Chen and Wang Table 1 Wall II."""

    conductivity = 0.0433
    density = 32
    specific_heat = 836.8  # note the values in the paper are incorrect


class Plaster(Material):
    """Chen and Wang Table 1 Wall II."""

    conductivity = 0.7270
    density = 1602
    specific_heat = 836.8  # note the values in the paper are incorrect


class Aluminium(Material):
    """Generic aluminium properties."""

    conductivity = 30.0
    density = 3950
    specific_heat = 192.5
