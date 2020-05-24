import json

import pytest

from ambient.core import BaseElement, SensorMISO
from ambient.construction import ConstructionLayered
from ambient.material_data import Brickwork, HeavyweightConcrete
from ambient.simulation import Simulation
from ambient.io import SimulationEncoder, simulation_decoder


def encode_simulation():
    """Test encoding a simulation."""
    sim = Simulation(name="Test sim")

    # register some materials
    mat1 = Brickwork(thickness=0.1)
    mat2 = HeavyweightConcrete(thickness=0.3)
    sensor = SensorMISO(source=[mat1, mat2])

    sim.register_elements([mat1, mat2, sensor])

    # register some layered constructions
    surfs = [ConstructionLayered(materials=sensor)]
    sim.register_elements(surfs)

    output = json.dumps(sim, cls=SimulationEncoder, indent=4, sort_keys=True)

    return sim, output


def test_decode_simulation():
    """Test decoding a simulation."""

    sim_input, output = encode_simulation()

    decoder = json.JSONDecoder(object_hook=simulation_decoder)
    sim = decoder.decode(output)
    sim.resolve_references()

    output = json.dumps(sim, cls=SimulationEncoder, indent=4, sort_keys=True)

    sim = decoder.decode(output)
    sim.resolve_references()

    assert sim_input == sim
