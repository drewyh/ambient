"""Calculate solar data."""

from dataclasses import dataclass

import numpy as np

from ambient.time import TimePeriodBase


@dataclass
class Location:
    """Location related information."""

    latitude: float  #: location latitude [degrees]
    longitude: float  #: location longitude [degrees]
    timezone: float  #: time difference from UTC [hours]
    elevation: float = 0.0  #: elevation above sealevel [m]


@dataclass
class SolarBase:
    """Solar related calculation base class."""

    location: Location  #: Location for the solar calculations
    timeperiod: TimePeriodBase  #: Time period for solar calculations

    @property
    def _extraterrestrial_solar(self) -> np.ndarray:
        # the extraterrestrial solar irradiation from eq. D.15
        # note: days are 1 based for calculations
        return 1367 * (
            1.0 + 0.033 * np.cos(2.0 * np.pi * (self.timeperiod.days - 2) / 365)
        )

    @property
    def _equation_of_time(self) -> np.ndarray:
        # note: days are 1 based for calculations
        gamma = 2 * np.pi * self.timeperiod.days / 365
        return 2.2918 * (
            0.0075
            + 0.1868 * np.cos(gamma)
            - 3.2077 * np.sin(gamma)
            - 1.4615 * np.cos(2 * gamma)
            - 4.089 * np.sin(2 * gamma)
        )

    @property
    def _declination(self) -> np.ndarray:
        # LCAM uses a simplified equation for declination
        # note: days are 1 based for calculations
        return np.radians(
            23.45 * np.sin(2 * np.pi * (self.timeperiod.days + 285) / 365)
        )

    @property
    def _apparent_solar_time(self) -> np.ndarray:
        standard_meridian = np.radians(self.location.timezone * 15)
        longitude = np.radians(self.location.longitude)

        return (
            self.timeperiod.hours[np.newaxis, :]
            + self._equation_of_time[:, np.newaxis] / 60.0
            + (standard_meridian - longitude) / 15.0
        )

    @property
    def _hour_angle(self) -> np.ndarray:
        return np.radians(15.0 * (self._apparent_solar_time - 12.0))

    @property
    def _solar_altitude(self) -> np.ndarray:
        latitude = np.radians(self.location.latitude)
        declination = self._declination

        solar_altitude = np.arcsin(
            np.cos(latitude)
            * np.cos(declination[:, np.newaxis])
            * np.cos(self._hour_angle)
            + np.sin(latitude) * np.sin(declination[:, np.newaxis])
        )
        solar_altitude[solar_altitude < 0.0] = 0.0

        return solar_altitude

    @property
    def _solar_azimuth(self) -> np.ndarray:
        latitude = np.radians(self.location.latitude)
        declination = self._declination
        hour_angle = self._hour_angle

        solar_azimuth = np.arccos(
            (
                np.sin(declination[:, np.newaxis]) * np.cos(latitude)
                - np.cos(declination[:, np.newaxis])
                * np.sin(latitude)
                * np.cos(self._hour_angle)
            )
            / np.cos(self._solar_altitude)
        )
        solar_azimuth[hour_angle > 0.0] *= -1.0
        solar_azimuth[hour_angle > 0.0] += 2.0 * np.pi

        return solar_azimuth

    @property
    def diffuse_horizontal_irradiance_all(self) -> np.ndarray:
        """Return the diffuse horizontal irrandiance for all timesteps."""
        raise NotImplementedError()

    @property
    def beam_normal_irradiance_all(self) -> np.ndarray:
        """Return the beam normal irrandiance for all timesteps."""
        raise NotImplementedError()

    def calculate_incident_diffuse_radiation(
        self, surface_azimuth: np.ndarray, surface_tilt: np.ndarray
    ) -> np.ndarray:
        """Calculate the incident diffuse radiation on surfaces.

        Args:
            surface_azimuth (float): surface azimuth values measured clockwise
                from North [degrees].
            surface_tilt (float): surface tilt values measured from horizontal [degrees]

        Returns:
            float: the incident diffuse solar radiation on each surface, for each day,
            for each hour [W / m^2]
        """
        surface_azimuth = surface_azimuth[np.newaxis, np.newaxis, :]
        surface_tilt = surface_tilt[np.newaxis, np.newaxis, :]
        solar_azimuth = self._solar_azimuth[..., np.newaxis]
        solar_altitude = self._solar_altitude[..., np.newaxis]

        # calculate the angle of incidence
        gamma = solar_azimuth - surface_azimuth
        cos_theta = np.cos(solar_altitude) * np.cos(gamma) * np.sin(
            surface_tilt
        ) + np.sin(solar_altitude) * np.cos(surface_tilt)

        # calculate the Y factor from eq. D.23
        y_factor = np.full_like(cos_theta, 0.45)
        y_factor[cos_theta >= -0.2] = 0.55 + 0.437 * cos_theta + 0.313 * cos_theta ** 2

        # calculate the incident diffuse radiant from eqs. D.24 and D.25
        incident_diffuse = y_factor * np.sin(surface_tilt)
        incident_diffuse[surface_tilt <= np.radians(90)] += np.cos(surface_tilt)
        incident_diffuse *= self.diffuse_horizontal_irradiance_all[..., np.newaxis]

        return incident_diffuse


@dataclass
class SolarDesignDay(SolarBase):
    """Solar related calculation design day."""

    tau_b: np.array  #: clear sky beam optical depth on the 21st of each month []
    tau_d: np.array  #: clear sky diffuse optical depth on the 21st of each month []

    def __post_init__(self) -> None:
        """Pre-compute data for annual calculations."""
        # calculate the beam solar
        m_const = 1.0 / (
            np.sin(self._solar_altitude)
            + 0.50572 * np.power((6.07995 + np.degrees(self._solar_altitude)), -1.6364)
        )
        m_const[self._solar_altitude < 0.0] = 0.0

        # calculate exponents
        ab_const = (
            1.454
            - 0.406 * self.tau_b
            - 0.268 * self.tau_d
            + 0.021 * self.tau_b * self.tau_d
        )
        ad_const = (
            0.507
            + 0.205 * self.tau_b
            - 0.080 * self.tau_d
            - 0.190 * self.tau_b * self.tau_d
        )

        # calculate the irradiance values
        self._beam_normal_irradiance = self._extraterrestrial_solar[
            :, np.newaxis
        ] * np.exp(
            -self.tau_b.take(self.timeperiod.months)[:, np.newaxis]
            * np.power(m_const, ab_const.take(self.timeperiod.months)[:, np.newaxis])
        )
        self._diffuse_horizontal_irradiance = self._extraterrestrial_solar[
            :, np.newaxis
        ] * np.exp(
            -self.tau_d.take(self.timeperiod.months)[:, np.newaxis]
            * np.power(m_const, ad_const.take(self.timeperiod.months)[:, np.newaxis])
        )

    @property
    def diffuse_horizontal_irradiance_all(self) -> np.ndarray:
        """Return the diffuse horizontal irrandiance for all timesteps."""
        return self._diffuse_horizontal_irradiance

    @property
    def beam_normal_irradiance_all(self) -> np.ndarray:
        """Return the beam normal irrandiance for all timesteps."""
        return self._beam_normal_irradiance
