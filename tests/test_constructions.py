import numpy as np
import pytest

from ambient.construction import ConstructionLayered
from ambient.material import MaterialResistanceOnly
import ambient.material_data as mats


CTF_EPS = 0.02
HEAT_TRANSFER_EPS = 0.007
HEAT_TRANSFER_SUM_EPS = 1.0e-8

# fmt: off
SOL_AIR = 273.15 + np.array(
    (
        24.4, 24.4, 23.8, 23.3, 23.3, 23.8,
        25.5, 27.2, 29.4, 31.6, 33.8, 36.1,
        43.3, 49.4, 53.8, 55.0, 52.7, 45.5,
        30.5, 29.4, 28.3, 27.2, 26.1, 25.0,
    )
)
# fmt: on


def _calculate_sums(ctfs):
    """Calculate the ratio of sums based on eq. 41."""
    ctf_sum = np.sum(ctfs, axis=1)
    ctf_sum[:-1] /= ctf_sum[-1]

    return ctf_sum[:-1]


def iterate_heat_transfer(con, temp_in, temp_out, qi, qo=None, iterations=100):
    """Iterate the heat transfer calculation."""
    for iteration in range(iterations):
        for hour in range(24):
            qi[hour] = con.calculate_heat_flux_inside(temp_out, temp_in, qi, hour)
            if qo is not None:
                qo[hour] = con.calculate_heat_flux_outside(temp_out, temp_in, qo, hour)


def test_heat_transfer_wang_wall_iii():
    """Calculate properties of wall III from:

        Shengwei Wang, Youming Chen,
        Transient heat flow calculation for multilayer constructions using
            a frequency-domain regression method,
        Building and Environment,
        Volume 38, Issue 1,
        2003,
        Pages 45-61,
        ISSN 0360-1323,
        https://doi.org/10.1016/S0360-1323(02)00024-0.
    """
    con = ConstructionLayered(
        materials=[
            MaterialResistanceOnly(0.0586),
            mats.Stucco(25.39 / 1000),
            mats.HighDensityConcrete(101.59 / 1000),
            mats.Insulation(25.30 / 1000),
            mats.Plaster(19.05 / 1000),
            MaterialResistanceOnly(0.1206),
        ],
        timestep=3600,
    )

    # fmt: off
    qe_paper = np.array(
        (
            11.646,  9.999,  8.517,  7.203,  6.016,  4.960,
             4.081,  3.473,  3.209,  3.304,  3.755,  4.523,
             5.584,  7.162,  9.498, 12.441, 15.591, 18.406,
            20.260, 20.392, 19.019, 17.157, 15.247, 13.403,
        )
    )
    # fmt: on

    # initialise values
    qe = np.zeros(24)
    qo = np.zeros(24)
    temp_in = np.full_like(SOL_AIR, 273.15 + 24)
    temp_out = SOL_AIR

    # the paper values are after the 4th iteration
    iterate_heat_transfer(con, temp_in, temp_out, qe, qo, iterations=4)

    # check the heat transfer into the room
    assert np.allclose(qe, qe_paper, rtol=0.0, atol=0.02)
    assert np.allclose(qe, qe_paper, rtol=0.002, atol=0.0)

    # now do further iterations to check heat transfer convergence
    iterate_heat_transfer(con, temp_in, temp_out, qe, qo, iterations=96)

    # check inside and outside heat transfer
    assert np.allclose(
        np.sum(qe) + np.sum(qo), 0.0, rtol=0.0, atol=HEAT_TRANSFER_SUM_EPS
    )


def test_ctfs_wang_wall_iii():
    """Calculate properties of wall III from:

        Shengwei Wang, Youming Chen,
        Transient heat flow calculation for multilayer constructions using
            a frequency-domain regression method,
        Building and Environment,
        Volume 38, Issue 1,
        2003,
        Pages 45-61,
        ISSN 0360-1323,
        https://doi.org/10.1016/S0360-1323(02)00024-0.
    """
    con = ConstructionLayered(
        materials=[
            MaterialResistanceOnly(0.0586),
            mats.Stucco(25.39 / 1000),
            mats.HighDensityConcrete(101.59 / 1000),
            mats.Insulation(25.30 / 1000),
            mats.Plaster(19.05 / 1000),
            MaterialResistanceOnly(0.1206),
        ],
        timestep=3600,
    )

    # fix the order of the ctfs for comparison with paper values
    con._min_ctf_order = 5
    con._max_ctf_order = 5

    ctf_b = np.flip(np.array([0.002868, 0.053248, 0.060036, 0.007236, 0.000050, 0.0]))
    ctf_d = np.flip(np.array([1.0, -1.175710, 0.300608, -0.015605, 0.000005, 0.0]))

    # check the values of the bk and dk ctfs
    # note: only these values are check as we use the improved ctf method
    # so only the bk and dk values are expected to match
    assert np.allclose(np.round(con._ctfs[1], 6), ctf_b, rtol=CTF_EPS, atol=0.0)
    assert np.allclose(np.round(con._ctfs[3], 6), ctf_d, rtol=CTF_EPS, atol=0.0)

    # check all the ctf sums are close to K
    assert np.allclose(_calculate_sums(con._ctfs), con.thermal_transmittance)


def test_heat_transfer_wang_wall_iv():
    """Calculate properties of wall IV from:

        Shengwei Wang, Youming Chen,
        Transient heat flow calculation for multilayer constructions using
            a frequency-domain regression method,
        Building and Environment,
        Volume 38, Issue 1,
        2003,
        Pages 45-61,
        ISSN 0360-1323,
        https://doi.org/10.1016/S0360-1323(02)00024-0.
    """
    con = ConstructionLayered(
        materials=[
            MaterialResistanceOnly(0.060),
            mats.Brickwork(105 / 1000),
            MaterialResistanceOnly(0.180),
            mats.HeavyweightConcrete(100 / 1000),
            MaterialResistanceOnly(0.120),
        ],
        timestep=3600,
    )

    # fix the order of the ctfs for comparison with paper values
    con._min_ctf_order = 5
    con._max_ctf_order = 5

    # fmt: off
    qe_paper = np.array(
        (
            23.165, 21.194, 19.155, 17.146, 15.212, 13.375,
            11.671, 10.166,  8.955,  8.112,  7.690,  7.709,
             8.165,  9.107, 10.747, 13.210, 16.396, 19.982,
            23.461, 26.138, 27.378, 27.325, 26.407, 24.950,
        )
    )
    # fmt: on

    ctf_b = np.flip(
        np.array([0.000178, 0.013915, 0.043475, 0.018078, 0.001052, 0.000006])
    )
    ctf_d = np.flip(
        np.array([1.0, -1.619841, 0.724516, -0.064305, 0.001542, -0.000006])
    )

    # check the values of the bk and dk ctfs
    # note: only these values are check as we use the improved ctf method
    # so only the bk and dk values are expected to match
    assert np.allclose(np.round(con._ctfs[1], 6), ctf_b, rtol=CTF_EPS, atol=0.0)
    assert np.allclose(np.round(con._ctfs[3], 6), ctf_d, rtol=CTF_EPS, atol=0.0)

    # initialise values
    qe = np.zeros(24)
    qo = np.zeros(24)
    temp_in = np.full_like(SOL_AIR, 273.15 + 24)
    temp_out = SOL_AIR

    # the paper values are after the 4th iteration
    iterate_heat_transfer(con, temp_in, temp_out, qe, qo, iterations=4)

    # check the heat transfer into the room
    assert np.allclose(qe, qe_paper, rtol=0.0, atol=0.01)
    assert np.allclose(qe, qe_paper, rtol=0.002, atol=0.0)

    # now do further iterations to check heat transfer convergence
    iterate_heat_transfer(con, temp_in, temp_out, qe, qo, iterations=96)

    # check inside and outside heat transfer
    assert np.allclose(
        np.sum(qe) + np.sum(qo), 0.0, rtol=0.0, atol=HEAT_TRANSFER_SUM_EPS
    )

    # check all the ctf sums are close to K
    assert np.allclose(_calculate_sums(con._ctfs), con.thermal_transmittance)

    # check thermal transmittance value from paper
    assert np.around(con.thermal_transmittance, 5) == 1.83033


def test_construction_resistance_low():
    """Calculate properties of a low capacitance wall."""
    con = ConstructionLayered(
        materials=[
            MaterialResistanceOnly(0.060),
            mats.Aluminium(1 / 1000),
            MaterialResistanceOnly(0.020),
        ],
        timestep=3600,
    )

    # initialise values
    qe = np.zeros(24)
    qo = np.zeros(24)
    temp_in = np.full_like(SOL_AIR, 273.15 + 24)
    temp_out = SOL_AIR

    # use a large number of iterations for convergence
    iterate_heat_transfer(con, temp_in, temp_out, qe, qo, iterations=100)

    # check inside and outside heat transfer
    assert np.allclose(
        np.sum(qe) + np.sum(qo), 0.0, rtol=0.0, atol=HEAT_TRANSFER_SUM_EPS
    )

    # check all the ctf sums are close to K
    assert np.allclose(_calculate_sums(con._ctfs), con.thermal_transmittance)


def test_heat_transfer_chen_wall_i():
    """Wall I from Chen and Wang (2001) Appl. Math. Modelling 25."""
    con = ConstructionLayered(
        materials=[
            MaterialResistanceOnly(0.0546),
            mats.Brickwork2(240 / 1000),
            mats.LimePlaster(20 / 1000),
            MaterialResistanceOnly(0.1149),
        ],
        timestep=3600,
    )

    # fmt: off
    qe_paper = np.array(
        (
            26.2538, 24.2517, 22.1326, 20.0110, 17.9603, 16.0019,
            14.1543, 12.4624, 11.0116,  9.8995,  9.1978,  8.9494,
             9.1640,  9.8302, 11.0838, 13.1564, 16.1038, 19.7167,
            23.5651, 27.0298, 29.2700, 29.9109, 29.3332, 28.0139,
        )
    )
    # fmt: on

    # initialise values
    qe = np.zeros(24)
    qo = np.zeros(24)
    temp_in = np.full_like(SOL_AIR, 273.15 + 24)
    temp_out = SOL_AIR

    # the paper values are after the 4th iteration
    iterate_heat_transfer(con, temp_in, temp_out, qe, qo, iterations=4)

    # check the heat transfer into the room
    assert np.allclose(qe, qe_paper, rtol=HEAT_TRANSFER_EPS, atol=0.0)

    # now do further iterations to check heat transfer convergence
    iterate_heat_transfer(con, temp_in, temp_out, qe, qo, iterations=96)

    # check inside and outside heat transfer
    assert np.allclose(
        np.sum(qe) + np.sum(qo), 0.0, rtol=0.0, atol=HEAT_TRANSFER_SUM_EPS
    )

    # check all the ctf sums are close to K
    assert np.allclose(_calculate_sums(con._ctfs), con.thermal_transmittance)


def test_heat_transfer_resistance_only():
    """Wall I from Chen and Wang (2001) Appl. Math. Modelling 25."""
    con = ConstructionLayered(materials=[MaterialResistanceOnly(0.5)], timestep=3600)

    # initialise values
    qe = np.zeros(24)
    qo = np.zeros(24)
    temp_in = np.full_like(SOL_AIR, 273.15 + 24)
    temp_out = np.full_like(SOL_AIR, 273.15 + 26)

    # since there is no time history component, one iteration is sufficient
    iterate_heat_transfer(con, temp_in, temp_out, qe, qo, iterations=1)

    # check the heat transfer into the room
    assert np.allclose(qe, 4.0, rtol=0.0, atol=0.0)

    # now do further iterations to check heat transfer convergence
    iterate_heat_transfer(con, temp_in, temp_out, qe, qo, iterations=96)

    # check inside and outside heat transfer
    assert np.allclose(np.sum(qe) + np.sum(qo), 0.0, rtol=0.0, atol=0.0)

    # check all the ctf sums are close to K
    assert np.allclose(con.thermal_transmittance, 2.0)
