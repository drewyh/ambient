import numpy as np

from ambient import solar
from ambient import time

# from table D.1, corrected for 1-based
DAYS_OF_YEAR = np.array((21, 52, 80, 111, 141, 172, 202, 233, 264, 294, 325, 355)) - 1

DESIGN_DAYS = time.TimePeriodBase(DAYS_OF_YEAR)

ATLANTA_GA = solar.Location(33.64, 84.43, -5)
SOLAR_ATLANTA_GA = solar.SolarBase(ATLANTA_GA, DESIGN_DAYS)


def test_extraterrestrial_solar():
    """Test calculation the extraterrestrial solar."""
    paper_ets = np.array(
        (1410, 1397, 1378, 1354, 1334, 1323, 1324, 1336, 1357, 1380, 1400, 1411)
    )

    assert np.array_equal(np.round(SOLAR_ATLANTA_GA._extraterrestrial_solar), paper_ets)


def test_equation_of_time():
    """Test the calculation of the equation of time."""
    paper_eot = np.array(
        (-10.6, -14.0, -7.9, 1.2, 3.7, -1.3, -6.4, -3.6, 6.9, 15.5, 13.8, 2.2)
    )

    assert np.array_equal(np.round(SOLAR_ATLANTA_GA._equation_of_time, 1), paper_eot)


def test_declination():
    """Test the calculation of the declination angle."""
    paper_dec = np.array(
        (-20.1, -11.2, -0.4, 11.6, 20.1, 23.4, 20.4, 11.8, -0.2, -11.8, -20.4, -23.4)
    )

    assert np.array_equal(
        np.round(np.degrees(SOLAR_ATLANTA_GA._declination), 1), paper_dec
    )


def test_solar_irradiance_atlanta_georgia():
    """Test the calculation of solar irradiance."""
    loc = solar.Location(33.64, 84.43, -5)
    sol = solar.SolarDesignDay(
        location=loc,
        timeperiod=DESIGN_DAYS,
        tau_b=np.array(
            (
                0.334,
                0.324,
                0.355,
                0.383,
                0.379,
                0.406,
                0.440,
                0.427,
                0.388,
                0.358,
                0.354,
                0.335,
            )
        ),
        tau_d=np.array(
            (
                2.614,
                2.580,
                2.474,
                2.328,
                2.324,
                2.270,
                2.202,
                2.269,
                2.428,
                2.514,
                2.523,
                2.618,
            )
        ),
    )
    print(np.round(sol._beam_normal_irradiance[:, 12]))
    print(np.round(sol._diffuse_horizontal_irradiance[:, 12]))


def test_solar_irradiance_chicago_illinois():
    """Test the calculation of solar irradiance."""
    loc = solar.Location(41.99, 87.91, -6)
    sol = solar.SolarDesignDay(
        location=loc,
        timeperiod=DESIGN_DAYS,
        tau_b=np.array(
            (
                0.288,
                0.305,
                0.325,
                0.359,
                0.369,
                0.383,
                0.415,
                0.416,
                0.392,
                0.344,
                0.306,
                0.294,
            )
        ),
        tau_d=np.array(
            (
                2.524,
                2.474,
                2.473,
                2.343,
                2.310,
                2.285,
                2.209,
                2.242,
                2.343,
                2.487,
                2.629,
                2.584,
            )
        ),
    )
    print(np.round(sol._beam_normal_irradiance[:, 12]))
    print(np.round(sol._diffuse_horizontal_irradiance[:, 12]))
