"""Input output module."""

import json
from dataclasses import fields
from typing import Any, Dict
from uuid import UUID

from ambient.core import BaseElement
from ambient.simulation import Simulation


# don't use dataclasses asdict function since we don't want
# to recurse into each instance.
def _field_dict(obj: Any) -> Dict[str, Any]:
    return {
        "type": type(obj).__qualname__,
        "data": {
            f.name: obj.get_sensor(f.name) or getattr(obj, f.name) for f in fields(obj)
        },
    }


class SimulationEncoder(json.JSONEncoder):
    """Encode simulations as JSON."""

    def default(self, o: Any) -> Any:  # pylint: disable=method-hidden
        """Handle simulation specific classes."""
        if isinstance(o, UUID):
            return f"{o.urn}"

        if isinstance(o, Simulation):
            sim = _field_dict(o)
            sim["data"]["elements"] = {
                e.guid.urn: _field_dict(e) for e in o.elements.values()
            }
            return sim

        if isinstance(o, BaseElement):
            return o.guid

        return json.JSONEncoder.default(self, o)


def simulation_decoder(dct: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild simulation classes on import."""
    if "guid" in dct:
        dct["guid"] = UUID(dct["guid"])

    if "type" in dct and "data" in dct:
        factory = BaseElement.factory[dct["type"]]
        return factory(**dct["data"])

    return dct
