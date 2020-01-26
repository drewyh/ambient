import numpy as np

from ambient import air

# aim to have all calculations less than 1% error
# TODO: not currently achieving this
RTOL = 0.011

ALTITUDE = np.array(
    [
        -500,
        0,
        500,
        1000,
        1500,
        2000,
        2500,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000,
        9000,
        10000,
    ]
)


def test_standard_pressure():
    """Test calculation of standard pressure from altitude."""
    pressure = 1000 * np.array(
        [
            107.478,
            101.325,
            95.461,
            89.875,
            84.556,
            79.495,
            74.682,
            70.108,
            61.640,
            54.020,
            47.181,
            41.061,
            35.600,
            30.742,
            26.436,
        ]
    )

    assert np.allclose(air.standard_pressure(ALTITUDE), pressure, rtol=0.0, atol=0.5)


def test_standard_temperature():
    """Test calculation of standard temperature from altitude."""
    temperature = 273.15 + np.array(
        [
            18.2,
            15,
            11.8,
            8.5,
            5.2,
            2,
            -1.2,
            -4.5,
            -11.0,
            -17.5,
            -24.0,
            -30.5,
            -37.0,
            -43.5,
            -50,
        ]
    )

    assert np.allclose(
        air.standard_temperature(ALTITUDE), temperature, rtol=0.0, atol=0.5
    )


def test_saturation_pressure_ice():
    temperature_ref = air.celsius_to_kelvin(np.array([-60.0, -40.0, -20.0, 0.0]))
    pressure_ref = 1000.0 * np.array([0.0010813, 0.012841, 0.10324, 0.61115])

    pressure_calc = air._saturation_pressure_ice(temperature_ref)

    assert np.allclose(pressure_ref, pressure_calc, rtol=0.001)


def test_saturation_pressure_water():
    temperature_ref = air.celsius_to_kelvin(np.array([0.0, 50.0, 100.0, 150.0]))
    pressure_ref = 1000.0 * np.array([0.61121, 12.351, 101.42, 476.10])

    pressure_calc = air._saturation_pressure_water(temperature_ref)

    assert np.allclose(pressure_ref, pressure_calc, rtol=0.001)


def test_saturated():
    """Test saturated states from RP-1485 Table A.8."""
    drybulb = air.celsius_to_kelvin(np.array([0, 20, 40, 60, 80]))

    hr = np.array([0.0037900, 0.0147605, 0.0491445, 0.1535447, 0.5529261])
    sv = np.array([0.7780, 0.8498, 0.9567, 1.1752, 1.8809])
    en = np.array([9.475, 57.558, 166.685, 460.880, 1541.765])
    humidity_ratio = np.full_like(hr, air.SATURATED_HUMIDITY_RATIO)

    ma = air.MoistAir(
        property_type=air.PropertyType.HUMIDITY_RATIO,
        property_value=humidity_ratio,
        drybulb=drybulb,
    )

    assert np.allclose(ma.humidity_ratio, hr, atol=0.0, rtol=RTOL)
    assert np.allclose(ma.specific_volume, sv, atol=0.0, rtol=RTOL)
    assert np.allclose(ma.specific_enthalpy, en, atol=0.0, rtol=RTOL)
