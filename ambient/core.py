"""Base classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from uuid import UUID, uuid4
from typing import Any, ClassVar, Dict, List, Optional, Type

import numpy as np


_FUNCTION_MAP = {
    "AVG": np.average,
    "COUNT": np.size,
    "MAX": np.amax,
    "MIN": np.amin,
    "SUM": np.sum,
}


@dataclass
class BaseElement(ABC):
    """Implements required generic functionality for all elements."""

    factory: ClassVar[Dict[str, Type]] = {}

    guid: UUID = field(default_factory=uuid4)

    def __init_subclass__(cls: Type, **kwargs: Any) -> None:
        """Register the subclass for file loading."""
        super().__init_subclass__(**kwargs)  # type: ignore
        BaseElement.factory[cls.__qualname__] = cls

    def __getattribute__(self, name: str) -> None:
        """Allow sensors to be used as input to classes."""
        attr = super().__getattribute__(name)

        if isinstance(attr, SensorBase):
            return attr.value

        return attr

    def resolve_references(self, references: Dict[UUID, "BaseElement"] = None) -> None:
        """Resolve loaded guids within the simulation."""
        if references is None:
            return

        for fld in fields(self):
            if fld.name == "guid":
                continue

            attr = getattr(self, fld.name)

            if not isinstance(attr, str):
                continue

            try:
                attr = UUID(attr)
            except ValueError:
                continue

            setattr(self, fld.name, references[attr])

    def get_sensor(self, name: str) -> Any:
        """Retrieve the sensor connected to an attribute."""
        attr = super().__getattribute__(name)

        if isinstance(attr, SensorBase):
            return attr

        return None

    def get_dependencies(self) -> List["BaseElement"]:
        """Get all sensors which act as inputs to this element."""
        return [f for f in fields(self) if isinstance(f, SensorBase)]


@dataclass  # type: ignore
class SensorBase(BaseElement):
    """Base class for communication between elements."""

    @property
    @abstractmethod
    def value(self) -> Any:
        """Return the value of the sensor."""


@dataclass  # type: ignore
class SensorAttributeBase(SensorBase):
    """Sensor for retrieving a single attribute."""

    attribute: Optional[str] = None
    function: Optional[str] = None

    def __post_init__(self) -> None:
        """Check the input."""
        assert self.attribute or self.attribute is None
        assert self.function is None or self.function in _FUNCTION_MAP


@dataclass
class SensorSISO(SensorAttributeBase):
    """A sensor for extracting a single element attribute."""

    source: Optional[BaseElement] = None

    def __post_init__(self) -> None:
        """Check the input."""
        super().__post_init__()
        assert self.source
        assert isinstance(self.source, BaseElement)
        assert self.attribute is None or hasattr(self.source, self.attribute)

    def get_dependencies(self) -> List[BaseElement]:
        """All source elements are dependencies."""
        if self.source is None:
            return []

        if not isinstance(self.source, BaseElement):
            raise ValueError(
                "source is not a subclass of BaseElement. "
                "Have references been resolved?"
            )

        return [self.source]

    @property
    def value(self) -> Any:
        """Return the extracted value."""
        arg = self.source

        if self.attribute is not None:
            arg = getattr(arg, self.attribute)

        return arg if self.function is None else _FUNCTION_MAP[self.function](arg)


@dataclass
class SensorMISO(SensorAttributeBase):
    """A sensor for extracting and aggregating element attributes."""

    source: List[BaseElement] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Check the input."""
        super().__post_init__()
        assert self.source
        assert isinstance(self.source, (list, tuple))
        assert self.attribute is None or all(
            hasattr(e, self.attribute) for e in self.source
        )

    def resolve_references(self, references: Dict[UUID, BaseElement] = None) -> None:
        """Resolve loaded guids within the simulation."""
        super().resolve_references(references)

        if references is None:
            return

        self.source = [
            element
            if isinstance(element, BaseElement)
            else references[UUID(str(element))]
            for element in self.source
        ]

    def get_dependencies(self) -> List[BaseElement]:
        """All source elements are dependencies."""
        if any(not isinstance(element, BaseElement) for element in self.source):
            raise ValueError(
                "an element of source is not a subclass of BaseElement. "
                "Have references been resolved?"
            )

        return self.source

    @property
    def value(self) -> Any:
        """Return the aggregated value."""
        args = self.source

        if self.attribute is not None:
            args = [getattr(e, self.attribute) for e in args]

        return args if self.function is None else _FUNCTION_MAP[self.function](args)
