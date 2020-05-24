"""Time based functionality."""

from calendar import isleap
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import ClassVar, Type, TypeVar

import numpy as np

from ambient.core import BaseElement

HOUR_IN_SECONDS = 60 * 60
DAY_IN_HOURS = 24
DAY_IN_SECONDS = DAY_IN_HOURS * HOUR_IN_SECONDS

DEFAULT_TIMESTEPS_PER_HOUR = 1
DEFAULT_YEAR = 1900


T = TypeVar("T", bound="TimePeriodBase")  # pylint: disable=invalid-name


def _month_lengths(is_leap_year: bool = False) -> np.ndarray:
    month_lengths = np.array((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31))

    if is_leap_year:
        month_lengths[1] = 29

    return month_lengths


def _month_lookup(month_lengths_array: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(12), month_lengths_array)


class DayType(IntEnum):
    """Enumerator for day types."""

    NORMAL = 1  #: Treat the days as normal days of the year.

    COOLING_DESIGN = 10  #: Treat the days as cooling design days.
    HEATING_DESIGN = 11  #: Treat the days as heating design days.
    VENTILATION_DESIGN = 12  #: Treat the days as ventilation design days.
    OTHER_DESIGN = 13  #: Treat the days as ventilation design days.


@dataclass
class TimePeriodBase(BaseElement):
    """Class for a simulation time period."""

    day_type: ClassVar[DayType] = DayType.NORMAL

    timesteps_per_hour: int = DEFAULT_TIMESTEPS_PER_HOUR
    days: np.ndarray = field(default_factory=np.array)
    year: int = DEFAULT_YEAR

    @classmethod
    def from_days_and_months(
        cls: Type[T],
        days: np.ndarray,
        months: np.ndarray,
        timesteps_per_hour: int = DEFAULT_TIMESTEPS_PER_HOUR,
        year: int = DEFAULT_YEAR,
    ) -> T:
        """Build a time period from days of the month and the months.

        Args:
            days (array_like): The days of the month.
            months (array_like): The months for each day.
            timesteps_per_hour (int): The number of timesteps per hour.
            year (int): The year for the days.

        Returns:
            TimePeriodBase: The time period with days derived from the input.
        """
        if isinstance(days, int):
            days = np.asarray([days])

        if isinstance(months, int):
            months = np.asarray([months])

        days, months = np.broadcast_arrays(days, months)
        offsets = np.cumsum(
            np.pad(_month_lengths(isleap(year)), (1, 0), constant_values=0)
        )
        days = np.take(offsets, months) + days

        return cls(timesteps_per_hour=timesteps_per_hour, days=days, year=year)

    def __post_init__(self) -> None:
        """Intialise other data."""
        assert self.days is not None

        self._timestep_length = HOUR_IN_SECONDS / self.timesteps_per_hour
        self._timestep_index = 0
        self._day_index = 0
        self._times = (
            np.arange(0, DAY_IN_HOURS * self.timesteps_per_hour)
            / self.timesteps_per_hour
        )
        self._months = np.fromiter(
            (
                datetime.strptime(f"{day + 1} {self.year}", "%j %Y").month - 1
                for day in self.days
            ),
            dtype=np.int,
        )

    @property
    def timestep(self) -> int:
        """Return the current timestep."""
        return self._timestep_index

    @property
    def times_in_hours(self) -> np.ndarray:
        """Return all timesteps for a day in hours."""
        return self._times

    @property
    def time_in_hours(self) -> float:
        """Return the current time of day in hours."""
        return self.times_in_hours[self._timestep_index]

    @property
    def day(self) -> int:
        """Return the current day of the year."""
        assert self.days is not None
        return self.days[self._day_index]

    @property
    def months(self) -> np.ndarray:
        """Return all the months of the year."""
        return self._months

    @property
    def month(self) -> int:
        """Return the current month of the year."""
        return self._months[self._day_index]

    @property
    def is_leap_year(self) -> bool:
        """Return true if the current year is a leap year."""
        return isleap(self.year)

    @property
    def total_timesteps(self) -> int:
        """Return the total number of timesteps from start to end."""
        assert self.days is not None
        return self.days.size * self.times_in_hours.size


@dataclass
class CoolingDesignDay(TimePeriodBase):
    """Cooling design day time period."""

    day_type: ClassVar[DayType] = DayType.COOLING_DESIGN


@dataclass
class HeatingDesignDay(TimePeriodBase):
    """Heating design day time period."""

    day_type: ClassVar[DayType] = DayType.HEATING_DESIGN
