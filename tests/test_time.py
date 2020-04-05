from datetime import datetime, timedelta

import numpy as np

from ambient.time import HeatingDesignDay


def test_total_timesteps():
    """Test the calculation of number of timesteps."""
    time = HeatingDesignDay(days=np.array((21,)))

    assert time.total_timesteps == 24


def test_total_day_month_constructor_jan():
    """Test the construction using days and months."""
    time = HeatingDesignDay.from_days_and_months(days=21, months=0)

    assert time.days == 21
    assert time.months == 0
    assert time.total_timesteps == 24


def test_total_day_month_constructor_feb():
    """Test the construction using days and months."""
    time = HeatingDesignDay.from_days_and_months(days=21, months=1)

    assert time.days == 52
    assert time.months == 1
    assert time.total_timesteps == 24
