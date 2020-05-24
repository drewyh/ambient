"""Simulation classes."""

from dataclasses import dataclass, field
from typing import Dict, Generator, List
from uuid import UUID

import networkx as nx

from ambient.core import BaseElement


@dataclass
class Simulation(BaseElement):
    """Base object holding all information for simulations."""

    name: str = ""
    elements: Dict[UUID, BaseElement] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialise other attributes."""
        self._dependency_graph = nx.DiGraph()

    def _register_element_no_check(self, element: BaseElement) -> None:
        """Include an element in the simulation without checking its type."""
        self.elements[element.guid] = element

    def register_elements(self, elements: List[BaseElement]) -> None:
        """Include a list of elements in a simulation."""
        if not all(isinstance(ele, BaseElement) for ele in elements):
            raise TypeError("all elements must be a subclass of BaseElement")

        for ele in elements:
            self._register_element_no_check(ele)

    def resolve_references(self, references: Dict[UUID, BaseElement] = None) -> None:
        """Resolve all element guids and recurse."""
        if references is not None:
            raise ValueError("references must be None")

        self.elements = {
            UUID(k) if isinstance(k, str) else k: v for k, v in self.elements.items()
        }

        for ele in self.elements.values():
            ele.resolve_references(self.elements)

    def create_dependency_graph(self) -> None:
        """Create the dependency graph."""
        for guid in self.elements:
            self._dependency_graph.add_node(guid)

        for ele in self.elements.values():
            deps = ele.get_dependencies()

            if deps is None:
                continue

            self._dependency_graph.add_edges_from((ele.guid, dep) for dep in deps)

    def evaluation_order(self) -> Generator:
        """Return the order for element evaluation."""
        return nx.topological_sort(self._dependency_graph)
