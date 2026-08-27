"""Small immutable types for the persisted telemetry dataset contract.

This module deliberately has no knowledge of producers, storage, validation, or
merge behaviour.  It is safe for those layers to import without creating a
registry through import side effects.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath


class ColumnAvailability(StrEnum):
    REQUIRED = "required"
    OPTIONAL = "optional"


class MergeMode(StrEnum):
    KEYED = "keyed"
    STATIC_EQUAL = "static_equal"
    REBUILD = "rebuild"


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    name: str
    # Whether this column must be present in the table schema.
    availability: ColumnAvailability
    # Whether values in this column may be null when the column is present.
    nullable: bool
    description: str


@dataclass(frozen=True, slots=True)
class ForeignKey:
    source_columns: tuple[str, ...]
    target_table: str
    target_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TableSpec:
    name: str
    columns: tuple[ColumnSpec, ...]
    stable_key: tuple[str, ...] | None
    required_in_dataset: bool
    merge_mode: MergeMode
    foreign_keys: tuple[ForeignKey, ...] = ()
    ignored_identity_columns: frozenset[str] = frozenset()
    allows_untyped_empty_frame: bool = False
    rebuild_from: tuple[str, ...] = ()
    empty_frame_columns: tuple[str, ...] | None = None
    omit_if_empty: bool = False

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(
            column.name
            for column in self.columns
            if column.availability is ColumnAvailability.REQUIRED
        )


@dataclass(frozen=True, slots=True)
class DatasetContract:
    tables: tuple[TableSpec, ...]

    def table(self, name: str) -> TableSpec | None:
        return next((table for table in self.tables if table.name == name), None)

    def validate_definition(self) -> None:
        names = [table.name for table in self.tables]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate table declarations: {duplicates}")
        for table in self.tables:
            path = PurePosixPath(table.name)
            if (
                not table.name
                or path.is_absolute()
                or path.as_posix() != table.name
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise ValueError(f"Non-canonical logical table name: {table.name!r}")
            column_names = [column.name for column in table.columns]
            duplicate_columns = sorted(
                {name for name in column_names if column_names.count(name) > 1}
            )
            if duplicate_columns:
                raise ValueError(
                    f"Table {table.name!r} declares duplicate columns: {duplicate_columns}"
                )
            if table.merge_mode is MergeMode.KEYED and not table.stable_key:
                raise ValueError(
                    f"Keyed table {table.name!r} must declare a stable key"
                )
            if table.merge_mode is MergeMode.REBUILD and table.required_in_dataset:
                raise ValueError(
                    f"Rebuilt table {table.name!r} cannot be required in input datasets"
                )
            if table.stable_key and not set(table.stable_key) <= set(column_names):
                raise ValueError(
                    f"Table {table.name!r} has a stable key outside its columns"
                )
            if not table.ignored_identity_columns <= set(column_names):
                raise ValueError(
                    f"Table {table.name!r} ignores undeclared identity columns"
                )
            if table.ignored_identity_columns & set(table.stable_key or ()):
                raise ValueError(
                    f"Table {table.name!r} cannot ignore stable-key columns"
                )
            if table.rebuild_from and table.merge_mode is not MergeMode.REBUILD:
                raise ValueError(
                    f"Only rebuilt table {table.name!r} may declare rebuild prerequisites"
                )
            if table.empty_frame_columns is not None and not set(
                table.empty_frame_columns
            ) <= set(column_names):
                raise ValueError(
                    f"Table {table.name!r} has an empty-frame layout outside its columns"
                )
        declared = {table.name: table for table in self.tables}
        for table in self.tables:
            for foreign_key in table.foreign_keys:
                if not foreign_key.source_columns or (
                    len(foreign_key.source_columns) != len(foreign_key.target_columns)
                ):
                    raise ValueError(f"Invalid foreign key on table {table.name!r}")
                if not set(foreign_key.source_columns) <= {
                    column.name for column in table.columns
                }:
                    raise ValueError(
                        f"Foreign key on {table.name!r} uses undeclared source columns"
                    )
                target = declared.get(foreign_key.target_table)
                if target is None:
                    raise ValueError(
                        f"Foreign key on {table.name!r} targets unknown table "
                        f"{foreign_key.target_table!r}"
                    )
                if not set(foreign_key.target_columns) <= {
                    column.name for column in target.columns
                }:
                    raise ValueError(
                        f"Foreign key on {table.name!r} targets undeclared columns"
                    )
            for prerequisite in table.rebuild_from:
                if prerequisite not in declared:
                    raise ValueError(
                        f"Rebuilt table {table.name!r} requires unknown table "
                        f"{prerequisite!r}"
                    )

    def merge_plan(self, names: Iterable[str]) -> tuple[TableSpec, ...]:
        requested = set(names)
        known = {table.name for table in self.tables}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unsupported merge tables: {unknown}")

        # Input datasets may contain stale derived tables, but those tables are
        # never merge inputs.  Include a rebuilt table only when all of its
        # declared prerequisites are available.  The fixed-point loop also
        # supports a rebuilt table depending on another rebuilt table.
        planned = {
            name
            for name in requested
            if (table := self.table(name)) is not None
            and table.merge_mode is not MergeMode.REBUILD
        }
        changed = True
        while changed:
            changed = False
            for table in self.tables:
                if (
                    table.merge_mode is MergeMode.REBUILD
                    and table.name not in planned
                    and set(table.rebuild_from) <= planned
                ):
                    planned.add(table.name)
                    changed = True
        selected = [table for table in self.tables if table.name in planned]
        ordered: list[TableSpec] = []
        while selected:
            ready = [
                table
                for table in selected
                if all(
                    prerequisite in {item.name for item in ordered}
                    for prerequisite in table.rebuild_from
                )
            ]
            if not ready:
                raise ValueError("Cyclic rebuild prerequisites in dataset contract")
            ordered.extend(ready)
            selected = [table for table in selected if table not in ready]
        return tuple(ordered)
