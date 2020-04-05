"""Time based functionality."""

from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar, Type, TypeVar

import numpy as np

HOUR_IN_SECONDS = 60 * 60
DAY_IN_SECONDS = 24 * HOUR_IN_SECONDS

_MONTH_LENGTHS = np.array((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31))
_MONTH_LOOKUP = np.repeat(np.arange(12), _MONTH_LENGTHS)


T = TypeVar("T", bound="TimePeriodBase")  # pylint: disable=invalid-name


class DayType(IntEnum):
    """Enumerator for day types."""

    NORMAL = 1  #: Treat the days as normal days of the year.

    COOLING_DESIGN = 10  #: Treat the days as cooling design days.
    HEATING_DESIGN = 11  #: Treat the days as heating design days.
    VENTILATION_DESIGN = 12  #: Treat the days as ventilation design days.
    OTHER_DESIGN = 13  #: Treat the days as ventilation design days.


@dataclass
class TimePeriodBase:
    """Class for a simulation time period."""

    day_type: ClassVar[DayType] = DayType.NORMAL
    iterations: ClassVar[int] = 1

    days: np.ndarray
    timestep: int = HOUR_IN_SECONDS

    @classmethod
    def from_days_and_months(
        cls: Type[T],
        days: np.ndarray,
        months: np.ndarray,
        timestep: int = HOUR_IN_SECONDS,
    ) -> T:
        """Build a time period from days of the month and the months.

        Args:
            days (array_like): The days of the month.
            months (array_like): The months for each day.
            timestep (int): The timestep to be passed to the underlying constructor.

        Returns:
            TimePeriodBase: The time period with days derived from the input.
        """
        days, months = np.broadcast_arrays(days, months)
        offsets = np.cumsum(np.pad(_MONTH_LENGTHS, (1, 0), constant_values=0))
        days = np.take(offsets, months) + days

        return cls(days, timestep=timestep)

    def __post_init__(self) -> None:
        """Do some initialisation checks."""
        if DAY_IN_SECONDS % self.timestep != 0:
            raise ValueError("Day not evenly divisible by timestep: {self.timestep}s")

    @property
    def hours(self) -> np.ndarray:
        """Return the timesteps for the day in hours."""
        return np.arange(0, DAY_IN_SECONDS, self.timestep) / HOUR_IN_SECONDS

    @property
    def months(self) -> np.ndarray:
        """Return the month associated with each day."""
        return np.take(_MONTH_LOOKUP, self.days)

    @property
    def total_timesteps(self) -> np.ndarray:
        """Return the total number of timesteps from start to end."""
        return self.days.size * self.hours.size


@dataclass
class CoolingDesignDay(TimePeriodBase):
    """Cooling design day time period."""

    day_type: ClassVar[DayType] = DayType.COOLING_DESIGN
    iterations: ClassVar[int] = -1


@dataclass
class HeatingDesignDay(TimePeriodBase):
    """Heating design day time period."""

    day_type: ClassVar[DayType] = DayType.HEATING_DESIGN
    iterations: ClassVar[int] = -1
