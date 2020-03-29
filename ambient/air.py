"""Psychrometric functions."""

# This module is based on ASHRAE Fundamentals 2017. All references are to
# Chapter 1 unless noted otherwise.
# Units are as follows:
#   Temperature: Kelvin [K]
#   Pressure: Pascals [Pa]
#   Altitude: meters [m]
#   Humidity ratio: [kg_w / kg_da]
#   Specific enthalpy: [kJ / kg_da]
#   Specific volume: [m^3 / kg_da]
#   Mass flow rate: [kg / s]
#   Volume flow rate: [m3 / s]


from dataclasses import dataclass, field, InitVar
from enum import IntEnum

import numpy as np
import scipy.optimize

# refer to eq. 1
_DRY_AIR_GAS_CONSTANT = 8.314472 / 28.966

# refer to eq. 3
STANDARD_PRESSURE = 101325  #: Standard atmospheric pressure [Pa].

# refer to eq. 8
_MOLAR_MASS_RATIO = 18.015268 / 28.966

# make a special constant for saturated humidity ratio
SATURATED_HUMIDITY_RATIO = np.inf  #: Specifies humidity ratio at saturation.


class FlowType(IntEnum):
    """Type of air flow."""

    MASS = 1  #: Mass flow rate [kg / s].
    VOLUME = 2  #: Volumetric flow rate [m^3 / s].


class PropertyType(IntEnum):
    """Type of psychrometric property."""

    HUMIDITY_RATIO = 1  #: Humidity ratio [kg_w / kg_da].
    RELATIVE_HUMIDITY = 2  #: Relative humidity [%].
    SPECIFIC_ENTHALPY = 3  #: Specific enthalpy [kJ / kg_da].
    SPECIFIC_VOLUME = 4  #: Specific volume [m^3 / kg_da].
    WETBULB = 5  #: Wetbulb temperature [K].
    DEWPOINT = 6  #: Dewpoint temperature [K].


@dataclass
class MoistAir:
    """Class for psychrometric calculations."""

    # pylint: disable=too-many-instance-attributes

    property_type: InitVar[PropertyType]
    property_value: InitVar[np.ndarray]

    drybulb: np.ndarray
    humidity_ratio: np.ndarray = field(init=False, default=np.nan)

    flow_rate_type: InitVar[FlowType] = FlowType.MASS
    flow_rate_value: InitVar[np.ndarray] = None

    mass_flow_rate: np.ndarray = field(init=False, default=np.nan)

    pressure: float = STANDARD_PRESSURE

    def __post_init__(
        self,
        property_type: PropertyType,
        property_value: np.ndarray,
        flow_rate_type: FlowType,
        flow_rate_value: np.ndarray,
    ) -> None:
        """Initialise based on arguments."""
        if property_type is PropertyType.HUMIDITY_RATIO:
            # ensure this is an array
            self.humidity_ratio = np.asarray(property_value)
            self.drybulb = np.asarray(self.drybulb)

            # allow for special saturated state
            saturated_mask = self.humidity_ratio == SATURATED_HUMIDITY_RATIO
            if np.any(saturated_mask):
                self.humidity_ratio[saturated_mask] = humidity_ratio_saturation(
                    self.drybulb[saturated_mask], self.pressure
                )
        elif property_type is PropertyType.RELATIVE_HUMIDITY:
            # see eq. 23
            pressure_ratio = saturation_pressure(self.drybulb) / self.pressure
            degree_of_saturation = (
                property_value
                * (1.0 - pressure_ratio)
                / (1.0 - property_value * pressure_ratio)
            )
            self.humidity_ratio = degree_of_saturation * humidity_ratio_saturation(
                self.drybulb, self.pressure
            )
        elif property_type is PropertyType.SPECIFIC_ENTHALPY:
            # see eq. 30
            drybulb = kelvin_to_celsius(self.drybulb)
            self.humidity_ratio = (
                property_value - 1.006 * drybulb
            ) / enthalpy_water_vapour_saturated(self.drybulb)
        elif property_type is PropertyType.SPECIFIC_VOLUME:
            # see eq. 26
            self.humidity_ratio = (
                (self.pressure / 1000)
                * property_value
                / (_DRY_AIR_GAS_CONSTANT * self.drybulb)
                - 1.0
            ) / 1.607858
        elif property_type is PropertyType.WETBULB:
            # see eq. 31
            wetbulb_saturated_humidity_ratio = humidity_ratio_saturation(
                property_value, self.pressure
            )
            enth_water_sat = enthalpy_water_condensed_saturated(property_value)
            enth_wetbulb_sat = _specific_enthalpy_air(
                wetbulb_saturated_humidity_ratio, property_value
            )
            drybulb = kelvin_to_celsius(self.drybulb)
            self.humidity_ratio = (
                enth_wetbulb_sat
                - 1.006 * drybulb
                - wetbulb_saturated_humidity_ratio * enth_water_sat
            ) / (2501.0 - 1.86 * drybulb - enth_water_sat)
        elif property_type is PropertyType.DEWPOINT:
            # see eq. 36
            dewpoint_sat_pressure = saturation_pressure(property_value)
            self.humidity_ratio = (
                621.945
                * dewpoint_sat_pressure
                / (self.pressure - dewpoint_sat_pressure)
            )
        else:
            assert False, f"invalid property type: {property_type}"

        # set flow rates
        if flow_rate_type is FlowType.MASS:
            self.mass_flow_rate = flow_rate_value
        elif flow_rate_type is FlowType.VOLUME:
            self.mass_flow_rate = flow_rate_value / self.specific_volume

    def __mul__(self, other: "MoistAir") -> "MoistAir":
        """Multiply mass flow by float or array."""
        if isinstance(other, (float, np.array)):
            return MoistAir(
                property_type=PropertyType.HUMIDITY_RATIO,
                property_value=self.humidity_ratio,
                drybulb=self.drybulb,
                flow_rate_type=FlowType.MASS,
                flow_rate_value=other * self.mass_flow_rate,
            )

        raise TypeError(f"must be float or numpy array not: {type(other)}")

    def __add__(self, other: "MoistAir") -> "MoistAir":
        """Add two instances based on mass and energy conservation."""
        if isinstance(other, MoistAir):
            assert self.mass_flow_rate is not None
            assert other.mass_flow_rate is not None
            assert (
                self.pressure == other.pressure
            ), f"pressures are not equal: {self.pressure} != {other.pressure}"

            mass_flow_total = self.mass_flow_rate + other.mass_flow_rate

            enthalpy_total = (
                self.mass_flow_rate * self.specific_enthalpy
                + other.mass_flow_rate * other.specific_enthalpy
            ) / mass_flow_total
            humidity_ratio_total = (
                self.mass_flow_rate * self.humidity_ratio
                + other.mass_flow_rate * other.humidity_ratio
            ) / mass_flow_total

            drybulb = celsius_to_kelvin(
                (enthalpy_total - 2501.0 * humidity_ratio_total)
                / (1.006 + 1.86 * humidity_ratio_total)
            )

            return MoistAir(
                property_type=PropertyType.HUMIDITY_RATIO,
                property_value=humidity_ratio_total,
                drybulb=drybulb,
                flow_rate_type=FlowType.MASS,
                flow_rate_value=mass_flow_total,
                pressure=self.pressure,
            )

        raise TypeError(f"must be MoistAir not: {type(other)}")

    @property
    def degree_of_saturation(self) -> float:
        """Calculate the degree of saturation of moist air."""
        return self.humidity_ratio / humidity_ratio_saturation(
            self.drybulb, self.pressure
        )

    @property
    def relative_humidity(self) -> float:
        """Calculate the relative humidity of moist air."""
        saturation = self.degree_of_saturation
        pressure_ws = saturation_pressure(self.drybulb)
        return saturation / (1.0 - (1.0 - saturation) * pressure_ws / self.pressure)

    @property
    def specific_volume(self) -> float:
        """Calculate the specific volume of moist air.

        References:
            2017 ASHRAE Handbook - Fundamentals, Chapter 1, Equation 26
        """
        return (
            _DRY_AIR_GAS_CONSTANT
            * self.drybulb
            * (1.0 + 1.607858 * self.humidity_ratio)
            / (self.pressure / 1000.0)
        )

    @property
    def specific_density(self) -> float:
        """Calculate the density of moist air."""
        return (1.0 + self.humidity_ratio) / self.specific_volume

    @property
    def specific_enthalpy(self) -> float:
        """Calculate the specific enthalpy of moist air."""
        return _specific_enthalpy_air(self.humidity_ratio, self.drybulb)

    @property
    def wetbulb(self) -> float:
        """Calculate the wet-bulb temperature of moist air."""

        def root_function(wetbulb: np.ndarray) -> np.ndarray:
            # refer to eq. 31
            humidity_ratio_wb_sat = humidity_ratio_saturation(wetbulb, self.pressure)
            return (
                self.specific_enthalpy
                + (humidity_ratio_wb_sat - self.humidity_ratio)
                * enthalpy_water_condensed_saturated(wetbulb)
                - _specific_enthalpy_air(humidity_ratio_wb_sat, wetbulb)
            )

        return scipy.optimize.root(root_function, self.drybulb).x

    @property
    def dewpoint(self) -> float:
        """Calculate the dew-point temperature of moist air."""
        # refer to eq. 36
        partial_pressure_water_dewpoint = partial_pressure_water(
            self.humidity_ratio, self.pressure
        )

        def root_function(dewpoint: np.ndarray) -> np.ndarray:
            return saturation_pressure(dewpoint) - partial_pressure_water_dewpoint

        return scipy.optimize.root(root_function, self.drybulb).x


def celsius_to_kelvin(temperature: np.ndarray) -> np.ndarray:
    """Convert celsius temperature to kelvin."""
    return temperature + 273.15


def kelvin_to_celsius(temperature: np.ndarray) -> np.ndarray:
    """Convert kelvin temperature to celsius."""
    return temperature - 273.15


def standard_pressure(altitude: np.ndarray) -> np.ndarray:
    """Calculate the standard pressure from altitude.

    Args:
        altitude: The altitude in meters.

    Returns:
        float: Standard pressure in Pa.
    """
    return 101325 * np.power(1.0 - 2.25577e-5 * altitude, 5.2559)


def standard_temperature(altitude: np.ndarray) -> np.ndarray:
    """Calculate the standard temperature from altitude.

    Args:
        altitude: The altitude in meters.

    Returns:
        float: Standard temperature in K.
    """
    return celsius_to_kelvin(15.0 - 0.0065 * altitude)


def _saturation_pressure_water(temperature: np.ndarray) -> np.ndarray:
    # Refer to RP-1485 eq. 2.91
    const_n = [
        np.nan,
        0.11670521452767e4,
        -0.72421316703206e6,
        -0.17073846940092e2,
        0.12020824702470e5,
        -0.32325550322333e7,
        0.14915108613530e2,
        -0.48232657361591e4,
        0.40511340542057e6,
        -0.23855557567849,
        0.65017534844798e3,
    ]
    theta = temperature + const_n[9] / (temperature - const_n[10])

    const_a = theta ** 2 + const_n[1] * theta + const_n[2]
    const_b = const_n[3] * theta ** 2 + const_n[4] * theta + const_n[5]
    const_c = const_n[6] * theta ** 2 + const_n[7] * theta + const_n[8]

    return (
        1.0e6
        * (2 * const_c / (-const_b + np.sqrt(const_b ** 2 - 4.0 * const_a * const_c)))
        ** 4.0
    )


def _saturation_pressure_ice(temperature: np.ndarray) -> np.ndarray:
    # Refer to RP-1485 eq. 2.92
    const_a = [np.nan, -0.212144006e2, 0.273203819e2, -0.610598130e1]
    const_b = [np.nan, 0.333333333e-2 - 1.0, 0.120666667e1 - 1.0, 0.170333333e1 - 1.0]

    theta = temperature / 273.16

    return 611.657 * np.exp(
        const_a[1] * theta ** const_b[1]
        + const_a[2] * theta ** const_b[2]
        + const_a[3] * theta ** const_b[3]
    )


def saturation_pressure(temperature: np.ndarray) -> np.ndarray:
    """Calculate the saturation pressure based on temperature.

    Args:
        temperature: Temperature in K.

    Returns:
        float: Saturation pressure in Pa.
    """
    temperature = np.asarray(temperature)
    pressure = np.empty_like(temperature)

    positive = temperature >= 0.0
    negative = temperature < 0.0

    pressure[negative] = _saturation_pressure_ice(temperature[negative])
    pressure[positive] = _saturation_pressure_water(temperature[positive])

    return pressure


def partial_pressure_water(
    humidity_ratio: np.ndarray, pressure: np.ndarray
) -> np.ndarray:
    """Calculate the partial pressure of water vapour."""
    return pressure * humidity_ratio / (_MOLAR_MASS_RATIO + humidity_ratio)


def humidity_ratio_saturation(
    temperature: np.ndarray, pressure: np.ndarray
) -> np.ndarray:
    """Calculate the saturation humidity ratio for a given temperature and pressure."""
    pressure_ws = saturation_pressure(temperature)
    return _MOLAR_MASS_RATIO * pressure_ws / (pressure - pressure_ws)


def _enthalpy_water_saturated_liquid(temperature: np.ndarray) -> np.ndarray:
    """Calculate the enthalpy of water as a saturated liquid.

    See also: eq. 32.
    """
    return 4.186 * kelvin_to_celsius(temperature)


def _enthalpy_water_saturated_solid(temperature: np.ndarray) -> np.ndarray:
    """Calculate the enthalpy of water as a solid (ice).

    See also: eq. 34.
    """
    return -333.4 + 2.1 * kelvin_to_celsius(temperature)


def enthalpy_water_condensed_saturated(temperature: np.ndarray) -> np.ndarray:
    """Calculate the enthalpy of water in saturated liquid or solid form.

    Args:
        temperature: Temperature of the water [K].

    Returns:
        float: Enthalpy of saturated water at the given temperature [kJ / kg_w].
    """
    temperature = np.asarray(temperature)
    enthalpy = np.empty_like(temperature)

    positive = temperature >= 0.0
    negative = temperature < 0.0

    enthalpy[negative] = _enthalpy_water_saturated_solid(temperature[negative])
    enthalpy[positive] = _enthalpy_water_saturated_liquid(temperature[positive])

    return enthalpy


def enthalpy_water_vapour_saturated(temperature: np.ndarray) -> np.ndarray:
    """Calculate the enthalpy of saturated water vapour.

    Args:
        temperature: Temperature of the water [K].

    Returns:
        float: Enthalpy of saturated water vapour at the given temperature [kJ / kg_w].
    """
    temperature = kelvin_to_celsius(temperature)
    return 2501 + 1.86 * temperature


def _specific_enthalpy_dry_air(drybulb: np.ndarray) -> np.ndarray:
    """Calculate the enthalpy from dry-bulb temperature."""
    drybulb = kelvin_to_celsius(drybulb)
    return 1.006 * drybulb


def _specific_enthalpy_air(
    humidity_ratio: np.ndarray, drybulb: np.ndarray
) -> np.ndarray:
    """Calculate the enthalpy from humidity ratio and dry-bulb temperature."""
    drybulb = kelvin_to_celsius(drybulb)
    return 1.006 * drybulb + humidity_ratio * (2501 + 1.86 * drybulb)
