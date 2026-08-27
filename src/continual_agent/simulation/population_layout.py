"""Named, validated neuron population layout for the configured network."""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


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

    @classmethod
    def from_dimensions(
        cls,
        *,
        input_count: int = 48,
        hidden_count: int = 48,
        output_tokens: tuple[str, ...] = (
            "<EOS>",
            "m",
            "a",
            "b",
            " ",
            "d",
            "n",
            "o",
            "i",
            ".",
            "?",
            "!",
        ),
        neurons_per_token: int = 3,
        action_names: tuple[str, ...] = (
            "answer",
            "clarify",
            "acknowledge",
            "uncertain",
            "revise",
            "refuse",
            "wait",
        ),
        neurons_per_action: int = 4,
        affect_names: tuple[str, ...] = (
            "valence",
            "arousal",
            "uncertainty",
            "curiosity",
            "threat",
            "competence",
            "social_affiliation",
        ),
        neurons_per_affect: int = 4,
    ) -> "PopulationLayout":
        """Create the canonical contiguous layout from semantic dimensions."""
        if any(
            not isinstance(value, tuple) for value in (output_tokens, action_names, affect_names)
        ):
            raise ValueError("population names must be tuples")
        if any(not names for names in (output_tokens, action_names, affect_names)):
            raise ValueError("semantic populations must not be empty")
        offset = input_count + hidden_count
        affect = {
            name: slice(offset + i * neurons_per_affect, offset + (i + 1) * neurons_per_affect)
            for i, name in enumerate(affect_names)
        }
        offset += len(affect_names) * neurons_per_affect
        action = {
            name: slice(offset + i * neurons_per_action, offset + (i + 1) * neurons_per_action)
            for i, name in enumerate(action_names)
        }
        offset += len(action_names) * neurons_per_action
        char = {
            name: slice(offset + i * neurons_per_token, offset + (i + 1) * neurons_per_token)
            for i, name in enumerate(output_tokens)
        }
        return cls(
            input_count,
            hidden_count,
            len(affect_names) * neurons_per_affect,
            len(action_names) * neurons_per_action,
            len(output_tokens) * neurons_per_token,
            affect,
            action,
            char,
        )

    def __post_init__(self) -> None:
        counts = (
            self.input_count,
            self.hidden_count,
            self.affect_count,
            self.action_count,
            self.char_count,
        )
        if any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts
        ):
            raise ValueError("population counts must be non-negative integers")
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
                if (
                    subgroup.step not in (None, 1)
                    or subgroup.start is None
                    or subgroup.stop is None
                ):
                    raise ValueError(f"{name} must contain contiguous slices")
                if (
                    subgroup.start < bounds.start
                    or subgroup.stop > bounds.stop
                    or subgroup.start >= subgroup.stop
                ):
                    raise ValueError(f"{name} contains an out-of-bounds slice")
            if normalized:
                ordered = sorted(normalized.values(), key=lambda subgroup: subgroup.start)
                if ordered[0].start != bounds.start or ordered[-1].stop != bounds.stop:
                    raise ValueError(f"{name} must cover its population")
                if any(left.stop != right.start for left, right in zip(ordered, ordered[1:])):
                    raise ValueError(f"{name} subgroups must not overlap or leave gaps")
                widths = {subgroup.stop - subgroup.start for subgroup in ordered}
                if len(widths) != 1:
                    raise ValueError(f"{name} subgroup widths must be consistent")
            elif bounds.start != bounds.stop:
                raise ValueError(f"{name} must cover its non-empty population")
            object.__setattr__(self, name, MappingProxyType(normalized))

    def __deepcopy__(self, memo: dict[int, object]) -> "PopulationLayout":
        memo[id(self)] = self
        return self

    @property
    def total_count(self) -> int:
        return self._total_count

    def slice(self, population: Population) -> builtins.slice:
        return self._slices[population]

    def subgroup(self, population: Population, name: str) -> builtins.slice:
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

    def subgroup_names(self, population: Population) -> tuple[str, ...]:
        return tuple(self._groups(population))

    def subgroup_width(self, population: Population) -> int:
        names = self.subgroup_names(population)
        return self.subgroup(population, names[0]).stop - self.subgroup(population, names[0]).start

    def _groups(self, population: Population) -> Mapping[str, builtins.slice]:
        return {
            Population.AFFECT: self.affect_subgroups,
            Population.OUTPUT_ACTION: self.action_subgroups,
            Population.OUTPUT_CHAR: self.char_subgroups,
        }.get(population, {})


def make_population_layout(**dimensions: Any) -> PopulationLayout:
    """Public factory for constructing a finalized population layout."""
    return PopulationLayout.from_dimensions(**dimensions)
