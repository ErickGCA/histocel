"""Class names: the official list, the aliases found in exports, and named groupings.

No class is built into histocel. Each project declares its own.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


class UnknownLabelError(KeyError):
    """Raised when an annotation uses a name the project never declared."""

    def __init__(self, raw: str, known: Sequence[str]):
        self.raw = raw
        self.known = list(known)
        super().__init__(raw)

    def __str__(self) -> str:
        return (
            f"unknown class {self.raw!r}. Declared classes: {', '.join(self.known)}. "
            "Add it as a new class or as an alias of an existing one."
        )


class TaxonomyError(ValueError):
    """Raised when the class declaration itself is inconsistent."""


def normalize(name: str) -> str:
    """Fold case, accents and repeated whitespace so near-identical names match."""
    decomposed = unicodedata.normalize("NFKD", name)
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(without_accents.casefold().split())


@dataclass(frozen=True)
class Taxonomy:
    """The classes of one project.

    ``classes`` maps each official name to the aliases seen in annotation exports.
    ``groupings`` maps a grouping name to ``{group_name: [official names]}``; a
    grouping lets a dataset merge rare classes without touching the annotations.
    """

    classes: Mapping[str, Sequence[str]]
    groupings: Mapping[str, Mapping[str, Sequence[str]]] = field(default_factory=dict)
    _lookup: dict[str, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.classes:
            raise TaxonomyError("a project needs at least one class")
        lookup: dict[str, str] = {}
        for official, aliases in self.classes.items():
            for name in (official, *aliases):
                key = normalize(name)
                owner = lookup.setdefault(key, official)
                if owner != official:
                    raise TaxonomyError(f"{name!r} is claimed by both {owner!r} and {official!r}")
        for grouping, groups in self.groupings.items():
            seen: dict[str, str] = {}
            for group, members in groups.items():
                for member in members:
                    if member not in self.classes:
                        raise TaxonomyError(
                            f"grouping {grouping!r}: {member!r} is not a declared class"
                        )
                    if member in seen:
                        raise TaxonomyError(
                            f"grouping {grouping!r}: {member!r} is in both "
                            f"{seen[member]!r} and {group!r}"
                        )
                    seen[member] = group
            missing = set(self.classes) - set(seen)
            if missing:
                raise TaxonomyError(
                    f"grouping {grouping!r} leaves out: {', '.join(sorted(missing))}"
                )
        object.__setattr__(self, "_lookup", lookup)

    def resolve(self, raw: str) -> str:
        """Official name for a name found in an export. Unknown names are an error."""
        try:
            return self._lookup[normalize(raw)]
        except KeyError:
            raise UnknownLabelError(raw, list(self.classes)) from None

    def names(self, grouping: str | None = None) -> list[str]:
        """Class names in a stable order; this order defines the model's class ids."""
        if grouping is None:
            return list(self.classes)
        return list(self._groups(grouping))

    def group_of(self, official: str, grouping: str | None = None) -> str:
        """Name a class takes under a grouping (itself when no grouping is given)."""
        if official not in self.classes:
            raise UnknownLabelError(official, list(self.classes))
        if grouping is None:
            return official
        for group, members in self._groups(grouping).items():
            if official in members:
                return group
        raise AssertionError("unreachable: groupings are validated to cover every class")

    def _groups(self, grouping: str) -> Mapping[str, Sequence[str]]:
        try:
            return self.groupings[grouping]
        except KeyError:
            known = ", ".join(self.groupings) or "none declared"
            raise TaxonomyError(f"unknown grouping {grouping!r} (available: {known})") from None
