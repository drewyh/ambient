"""Constructions for simulation."""

# All references are to the paper:
# Transient heat flow calculation for multilayer constructions
# using a frequency-domain regression method
# Wang and Chen
# Building and Environment 38 (2003)

from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Tuple, Union

import numpy as np

from numpy.polynomial.polynomial import polyzero, polyone

from ambient.material import Material, MaterialResistanceOnly

_EPS = 1.0e-9


def _calculate_frequencies(
    lower_limit_power: int = -8, upper_limit_power: int = -3, num_freqs: int = 50
) -> np.ndarray:
    """Calculate the frequencies for the FDR method.

    References:
        Section 4, steps 1 and 2.
    """
    return np.logspace(lower_limit_power, upper_limit_power, num=num_freqs)


def _calculate_response_functions(response_matrix: np.ndarray) -> np.ndarray:
    """Calculate the response functions for FDR method.

    References:
        Section 2, equations 10, 11, and 12
    """
    response_functions = 1.0 / response_matrix[:, 0, 1]
    response_functions = np.tile(response_functions, (3, 1))
    response_functions[0] *= response_matrix[:, 0, 0]
    response_functions[2] *= response_matrix[:, 1, 1]

    return response_functions


def _approximate_response_function(
    response_function: np.ndarray,
    frequencies: np.ndarray,
    numerator_order: int,
    denominator_order: int,
    denominator_values: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate the transfer functions by rational functions."""
    # calculate two parts of matrix from eq. 21
    max_power = max(numerator_order, denominator_order)
    powers = np.power(1.0j * frequencies[:, np.newaxis], np.arange(max_power + 1))

    h_matrix = np.copy(powers[:, : numerator_order + 1])

    g_vector = response_function

    # account for case of solving for denominator too
    if denominator_values is None:
        d_comp = -1.0 * np.copy(powers[:, 1 : denominator_order + 1])
        h_matrix = np.concatenate((h_matrix, d_comp * g_vector[:, np.newaxis]), axis=1)
    else:
        g_vector = g_vector * denominator_values

    gamma = np.real(np.conj(h_matrix.T) @ h_matrix)
    theta = 0.5 * np.real(
        np.conj(h_matrix.T) @ g_vector + h_matrix.T @ np.conj(g_vector)
    )

    coeffs = np.linalg.solve(gamma, theta)
    coeffs = np.insert(coeffs, numerator_order + 1, 1)

    return (
        np.flip(coeffs[: numerator_order + 1]),
        np.flip(coeffs[numerator_order + 1 :]),
    )


def _calculate_residuals(
    numerator_poly: np.poly, denominator_poly: np.poly, denominator_roots: np.ndarray
) -> np.ndarray:
    """See eq. 36."""
    delta = np.zeros_like(denominator_roots)
    for idx, root in enumerate(denominator_roots):
        root_poly = np.poly1d((1.0, -root))
        alpha_prime_coeff, rem = np.polydiv(denominator_poly, root_poly)

        assert np.allclose(rem, np.poly1d(polyzero), rtol=0.0, atol=1.0e-7)

        alpha_prime = np.poly1d(alpha_prime_coeff)

        delta_n = -1.0 * numerator_poly(root) / (root ** 2 * alpha_prime(root))
        delta[idx] = delta_n

    return delta


@dataclass
class ConstructionBase:
    """Construction base class."""

    @property
    def thermal_resistance(self) -> float:
        """Return the total thermal resistance of the construction."""
        raise NotImplementedError()

    @property
    def thermal_transmittance(self) -> float:
        """Return the total thermal transimittance of the construction."""
        return 1.0 / self.thermal_resistance

    def calculate_heat_flux_inside(
        self,
        outside_temps: np.ndarray,
        inside_temps: np.ndarray,
        inside_heat_fluxes: np.ndarray,
        current_index: int,
    ) -> float:
        """Calculated the inside heat fluxes."""
        raise NotImplementedError()

    def calculate_heat_flux_outside(
        self,
        outside_temps: np.ndarray,
        inside_temps: np.ndarray,
        outside_heat_fluxes: np.ndarray,
        current_index: int,
    ) -> float:
        """Calculated the inside heat fluxes."""
        raise NotImplementedError()


@dataclass
class ConstructionLayered(ConstructionBase):
    """Class for layered constructions.

    The order of layers is outside to inside.
    """

    materials: List[Union[Material, MaterialResistanceOnly]]
    timestep: int = 3600  #: The time step for simulation [s]

    def __post_init__(self) -> None:
        """Set construction data not stored in fields."""
        self._thermal_resistance = np.sum(
            np.fromiter((m.thermal_resistance for m in self.materials), dtype=np.float)
        )

        # allow fixing the minimum and maximum order of ctfs for testing
        self._min_ctf_order = 1
        self._max_ctf_order = 6

        self._ctfs_internal = None
        self._is_resistance_only = all(
            isinstance(m, MaterialResistanceOnly) for m in self.materials
        )

    @property
    def thermal_resistance(self) -> float:
        """Return the thermal transimittance of the construction."""
        return self._thermal_resistance

    @property
    def _ctfs(self) -> np.ndarray:
        """Return the conduction transfer functions."""
        if self._ctfs_internal is None:
            # paper notes typical orders of 3-5, however 6 may be required for heavy
            # constructions. Additionally, 1 allows resistance only (or close to)
            # constructions to be modelled.
            coeffs = self._approximate_response_functions()
            den = np.poly1d(coeffs[-1])
            arr: List[np.ndarray] = []
            for numerator_coeff in coeffs[:-1]:
                num = np.poly1d(numerator_coeff)
                arr.extend(self._calculate_ctf(num, den, self.timestep))

            self._ctfs_internal = np.array(arr[::2] + [arr[-1]])

        return self._ctfs_internal

    def _calculate_response_matrix(self, frequencies: np.ndarray) -> np.ndarray:
        layer_matrices = [
            m.calculate_response_matrices(frequencies) for m in self.materials
        ]

        # calculate the product over all layers
        ms_matrix = reduce(np.matmul, layer_matrices)

        return ms_matrix

    def _approximate_response_functions(self) -> List[np.ndarray]:
        """Approximate the transfer functions by rational functions."""
        frequencies = _calculate_frequencies()
        response_matrix = self._calculate_response_matrix(frequencies)
        response_functions = _calculate_response_functions(response_matrix)

        # calculate the best fit for approximation
        coeff_map = defaultdict(list)
        value_map: Dict = defaultdict(float)
        for numerator_order in range(self._min_ctf_order, self._max_ctf_order + 1):
            for denominator_order in range(numerator_order, self._max_ctf_order + 1):
                key = (numerator_order, denominator_order)

                coeffs = [None] * 4

                # first calculate the cross term (Y) to fix the denominator
                try:
                    coeffs[1], coeffs[3] = _approximate_response_function(
                        response_functions[1],
                        frequencies,
                        denominator_order=denominator_order,
                        numerator_order=numerator_order,
                    )
                except np.linalg.LinAlgError:
                    # catch the case of a singular matrix solution
                    continue

                # calculate the value of the denominator at frequences
                denom_values = np.polyval(coeffs[3], frequencies)

                # we can't except blow up of the solution
                # XXX: since we need roots elsewhere it may be better to return them
                if np.any(np.real(np.roots(coeffs[3])) >= 0.0):
                    continue

                # calculate the diff with the exact
                diff = (
                    response_functions[1]
                    - np.polyval(coeffs[1], frequencies) / denom_values
                )
                value_map[key] += np.real(np.sum(diff * diff.conjugate()))

                # now calculate the other functions
                for idx, response_function in enumerate(response_functions):
                    if idx == 1:
                        continue
                    coeffs[idx], _ = _approximate_response_function(
                        response_function,
                        frequencies,
                        denominator_order=denominator_order,
                        numerator_order=numerator_order,
                        denominator_values=denom_values,
                    )

                    # add to the diff
                    diff = (
                        response_function
                        - np.polyval(coeffs[idx], frequencies) / denom_values
                    )
                    value_map[key] += np.real(np.sum(diff * diff.conjugate()))

                coeff_map[key] = coeffs

        return coeff_map[min(value_map, key=value_map.get)]

    def _calculate_ctf(
        self, numerator_poly: np.poly, denominator_poly: np.poly, timestep: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        # step 4
        roots = denominator_poly.roots
        delta = _calculate_residuals(numerator_poly, denominator_poly, roots)

        # XXX: delta is complex for some constructions, not sure why
        sum_delta = np.sum(delta)
        assert np.imag(sum_delta) == 0.0
        sum_delta = np.real(sum_delta)

        coeffs = np.ones((roots.shape[0], 2), dtype=np.complex)

        # note that there is no -1.0 factor in the exponent since the paper
        # assumes the roots are -si (see eq. 35)
        coeffs[:, 0] = -1.0 * np.exp(roots * timestep)

        # XXX: this solves the issue of nans for very low resistance constructions
        coeffs[np.isinf(coeffs)] = 0.0

        fac = np.array([-1.0, 1.0])

        if coeffs.size > 0:
            # note: convolution is analogous to polynomial multiplication
            polytot = reduce(np.convolve, coeffs)
        else:
            polytot = polyone

        numerator_poly = polytot * sum_delta

        partial_numerator_sum = np.zeros_like(numerator_poly[1:], dtype=np.complex)
        for idx, delta_i in enumerate(delta):
            mask = np.ones_like(roots, dtype=bool)
            mask[idx] = False
            partial_numerator_sum += delta_i * reduce(np.convolve, coeffs[mask], 1.0)

        # XXX: delta is complex for some constructions, not sure why
        assert np.all(
            np.abs(np.imag(partial_numerator_sum[np.isfinite(partial_numerator_sum)]))
            <= _EPS
        )
        partial_numerator_sum = np.real(partial_numerator_sum)

        if partial_numerator_sum.size > 0:
            numerator_poly -= np.convolve(partial_numerator_sum, fac)

        numerator_poly = np.convolve(numerator_poly, fac)

        assert np.abs(numerator_poly[-1]) <= _EPS

        # this is a division by z
        numerator_poly = numerator_poly[:-1]

        numerator_poly += polytot * self.thermal_transmittance * timestep
        numerator_poly /= timestep

        assert np.all(
            np.abs(np.imag(numerator_poly[np.isfinite(numerator_poly)])) <= _EPS
        )
        numerator_poly = np.real(numerator_poly)

        # numerator_poly[np.abs(numerator_poly) <= _EPS] = 0.0

        assert np.all(np.abs(np.imag(polytot[np.isfinite(polytot)])) <= _EPS)
        polytot = np.real(polytot)

        return numerator_poly, polytot

    @property
    def _timeseries_length(self) -> int:
        return self._ctfs.shape[-1]  # pylint: disable=unsubscriptable-object

    def calculate_heat_flux_inside(
        self,
        outside_temps: np.ndarray,
        inside_temps: np.ndarray,
        inside_heat_fluxes: np.ndarray,
        current_index: int,
    ) -> float:
        """Calculate the inside heat flux."""
        # short circuit for resistance only constructions
        if self._is_resistance_only:
            return self.thermal_transmittance * (
                outside_temps[current_index] - inside_temps[current_index]
            )
        # determine the array values
        take = range(current_index - self._timeseries_length + 1, current_index + 1)
        outside_temps = outside_temps[take]
        inside_temps = inside_temps[take]
        inside_heat_fluxes = inside_heat_fluxes[take[:-1]]

        # from eq. 34
        qi_new = (
            np.sum(self._ctfs[1] * outside_temps)
            - np.sum(self._ctfs[3][:-1] * inside_heat_fluxes)
            - np.sum(self._ctfs[2] * inside_temps)
        )

        return qi_new

    def calculate_heat_flux_outside(
        self,
        outside_temps: np.ndarray,
        inside_temps: np.ndarray,
        outside_heat_fluxes: np.ndarray,
        current_index: int,
    ) -> float:
        """Calculate the inside heat flux."""
        # short circuit for resistance only constructions
        if self._is_resistance_only:
            return self.thermal_transmittance * (
                inside_temps[current_index] - outside_temps[current_index]
            )
        # determine the array values
        take = range(current_index - self._timeseries_length + 1, current_index + 1)
        outside_temps = outside_temps[take]
        inside_temps = inside_temps[take]
        outside_heat_fluxes = outside_heat_fluxes[take[:-1]]

        # from eq. 34
        qo_new = (
            -np.sum(self._ctfs[0] * outside_temps)
            - np.sum(self._ctfs[3][:-1] * outside_heat_fluxes)
            + np.sum(self._ctfs[2] * inside_temps)
        )

        return qo_new
