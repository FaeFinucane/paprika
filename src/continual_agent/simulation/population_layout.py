"""Named, validated neuron population layout for the configured network."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class Population(str, Enum):
    INPUT = "input"
    HIDDEN = "hidden"
    AFFECT = "affect"
    OUTPUT_ACTION = "output_action"
    OUTPUT_CHAR = "output_char"


@dataclass(frozen=True)
class PopulationLayout:
    """Immutable half-open slices for the network's named populations."""

    input_count: int = 48
    hidden_count: int = 48
    affect_count: int = 7 * 4
    action_count: int = 7 * 4
    char_count: int = 12 * 3
    affect_subgroups: Mapping[str, slice] = MappingProxyType({})
    action_subgroups: Mapping[str, slice] = MappingProxyType({})
    char_subgroups: Mapping[str, slice] = MappingProxyType({})
    _slices: Mapping[Population, slice] = None  # type: ignore[assignment]
    _total_count: int = 0

    def __post_init__(self) -> None:
        counts = (self.input_count, self.hidden_count, self.affect_count,
                  self.action_count, self.char_count)
        if any(not isinstance(count, int) or isinstance(count, bool) or count <= 0 for count in counts):
            raise ValueError("population counts must be positive integers")
        current = self.input_count + self.hidden_count
        slices = {
            Population.INPUT: slice(0, self.input_count),
            Population.HIDDEN: slice(self.input_count, current),
        }
        for population, count in (
            (Population.AFFECT, self.affect_count),
            (Population.OUTPUT_ACTION, self.action_count),
            (Population.OUTPUT_CHAR, self.char_count),
        ):
            slices[population] = slice(current, current + count)
            current += count
        object.__setattr__(self, "_slices", MappingProxyType(slices))
        object.__setattr__(self, "_total_count", current)
        for name, groups, population in (
            ("affect_subgroups", self.affect_subgroups, Population.AFFECT),
            ("action_subgroups", self.action_subgroups, Population.OUTPUT_ACTION),
            ("char_subgroups", self.char_subgroups, Population.OUTPUT_CHAR),
        ):
            normalized = dict(groups)
            bounds = self._slices[population]
            for key, subgroup in normalized.items():
                if subgroup.step not in (None, 1) or subgroup.start is None or subgroup.stop is None:
                    raise ValueError(f"{name} must contain contiguous slices")
                if subgroup.start < bounds.start or subgroup.stop > bounds.stop or subgroup.start >= subgroup.stop:
                    raise ValueError(f"{name} contains an out-of-bounds slice")
            object.__setattr__(self, name, MappingProxyType(normalized))

    @property
    def total_count(self) -> int:
        return self._total_count

    def slice(self, population: Population) -> slice:
        return self._slices[population]

    def subgroup(self, population: Population, name: str) -> slice:
        groups = {
            Population.AFFECT: self.affect_subgroups,
            Population.OUTPUT_ACTION: self.action_subgroups,
            Population.OUTPUT_CHAR: self.char_subgroups,
        }.get(population)
        if groups is None:
            raise ValueError(f"{population.value} has no semantic subgroups")
        try:
            return groups[name]
        except KeyError as error:
            raise KeyError(f"unknown {population.value} subgroup: {name}") from error
