#!/usr/bin/env python3
"""
gem_query.py — Query the GEM read-only Postgres database and export to CSV.

Reads the connection URL from the GEM_READONLY_DB_URL environment variable.
Never hardcodes credentials.

USAGE
-----
    # One-time setup (in your shell, NOT in this file):
    export GEM_READONLY_DB_URL='postgres://readonly:PASSWORD@HOST:5432/DBNAME'

    # Default mode: export EVERY table related to a project type.
    # Discovers tables with a `project_type` column and tables with a
    # foreign key into the projects table. Writes one CSV per table.
    python gem_query.py --project-type lng -o ./gem_export_lng
    python gem_query.py --project-type 8   -o ./gem_export_lng

    # Single-table mode (the old behavior). Useful for one-offs.
    python gem_query.py --table projects --project-type lng -o lng_projects.csv

    # Combustion needs a tracker_type sub-filter (project_type=1 covers ALL
    # combustion subtrackers). The same --where applies to every discovered
    # table that has a `tracker_type` column; tables without it are filtered
    # by project_type / FK only.
    python gem_query.py --project-type combustion \\
        --where "tracker_type = 'gcpt'" \\
        -o ./gem_export_coal

    # Schema introspection (no output file required):
    python gem_query.py --list-tables
    python gem_query.py --describe projects
    python gem_query.py --discover --project-type 8

    # Custom SELECT (read-only enforced; goes to a single CSV):
    python gem_query.py --sql "SELECT id, name FROM projects WHERE project_type = 8 LIMIT 10" \\
        -o sample.csv

NOTES
-----
* The role connected as should be `readonly` at the Postgres level. The script
  additionally sets the session to read-only as a defense-in-depth measure.
* Results stream in chunks, so the script is safe on large tables.
* Multi-table mode writes one CSV per table into a directory you pass via -o.
* Single-table mode writes one CSV file (the path you pass via -o).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import Engine
except ImportError:
    sys.stderr.write(
        "ERROR: sqlalchemy is not installed. Install with:\n"
        "    pip install 'sqlalchemy>=2.0' psycopg2-binary\n"
    )
    sys.exit(1)


# Maps human-readable names to the project_type integer codes used in the DB.
# Source: GEM internal PROJECT_TYPES tuple.
PROJECT_TYPES: dict[str, int] = {
    "combustion": 1,
    "solar": 2,
    "wind": 3,
    "nuclear": 4,
    "geothermal": 5,
    "steel": 6,
    "hydro": 7,
    "lng": 8,
    "goget": 9,
}
PROJECT_TYPE_NAMES: dict[int, str] = {v: k for k, v in PROJECT_TYPES.items()}

# Bundle config: which core tables make up each project type, and the column
# name that other tables use to link to them. Used when the DB does NOT enforce
# FK constraints (e.g., Django ORM-only relations) — we fall back to matching
# by column name instead. The link_column is a GLOBAL id shared across project
# types (typical Django multi-table-inheritance pattern), distinct from each
# table's local primary key.
PROJECT_TYPE_BUNDLES: dict[str, list[tuple[str, str]]] = {
    "lng":        [("lng_project", "project_id"), ("lng_unit", "unit_id")],
    "steel":      [("steel_project", "project_id"), ("steel_unit", "unit_id")],
    "goget":      [("goget_project", "project_id")],
    "combustion": [("plant", "plant_id"), ("powerplant_unit", "powerplant_unit_id")],
}

ENV_VAR = "GEM_READONLY_DB_URL"
DEFAULT_CHUNKSIZE = 1000
DEFAULT_STATEMENT_TIMEOUT_MS = 5 * 60 * 1000  # 5 minutes
PROJECT_TYPE_COL = "project_type"  # the column name we filter on
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _all_typed_tables() -> set[str]:
    """Set of every core table across all project-type bundles."""
    return {tbl for bundle in PROJECT_TYPE_BUNDLES.values() for tbl, _ in bundle}


# Tables we always skip in bundle/multi-table mode: Django infrastructure
# (auth, admin, sessions), allauth/socialaccount, and user-settings tables.
# None of them carry tracker data; including them in a research export would
# leak user PII and bloat the bundle.
EXCLUDED_TABLES: frozenset[str] = frozenset({
    "account_emailaddress",
    "account_emailconfirmation",
    "auth_group",
    "auth_group_permissions",
    "auth_permission",
    "auth_user",
    "auth_user_groups",
    "auth_user_user_permissions",
    "django_admin_log",
    "django_content_type",
    "django_migrations",
    "django_session",
    "django_site",
    "socialaccount_socialaccount",
    "socialaccount_socialapp",
    "socialaccount_socialapp_sites",
    "socialaccount_socialtoken",
    "user_settings",
})


# ---------------------------------------------------------------------------
# Connection setup
# ---------------------------------------------------------------------------

def get_database_url() -> str:
    """Pull the DB URL from the environment, with the Heroku scheme fix applied."""
    url = os.environ.get(ENV_VAR)
    if not url:
        sys.stderr.write(
            f"ERROR: environment variable {ENV_VAR} is not set.\n\n"
            "Set it in your shell (do NOT commit it to git):\n"
            f"    export {ENV_VAR}='postgres://readonly:PASSWORD@HOST:5432/DBNAME'\n\n"
            "Or load it from a gitignored .env file before running.\n"
        )
        sys.exit(2)
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


def build_engine(url: str, statement_timeout_ms: int) -> Engine:
    """Engine where every session defaults to read-only."""
    return create_engine(
        url,
        connect_args={
            "options": (
                f"-c default_transaction_read_only=on "
                f"-c statement_timeout={statement_timeout_ms}"
            )
        },
        pool_pre_ping=True,
    )


def resolve_project_type(value: str) -> int:
    """Accept either a name ('lng') or a numeric code ('8')."""
    v = value.strip().lower()
    if v.isdigit():
        code = int(v)
        if code not in PROJECT_TYPE_NAMES:
            valid = ", ".join(f"{n}={k}" for k, n in PROJECT_TYPES.items())
            sys.stderr.write(
                f"WARNING: project_type={code} is not in the known list ({valid}). "
                "Continuing anyway in case the schema has been extended.\n"
            )
        return code
    if v in PROJECT_TYPES:
        return PROJECT_TYPES[v]
    valid = ", ".join(sorted(PROJECT_TYPES))
    sys.stderr.write(
        f"ERROR: unknown project type '{value}'. Valid names: {valid}.\n"
        "Or pass a numeric code (e.g., 8 for LNG).\n"
    )
    sys.exit(2)


# ---------------------------------------------------------------------------
# Schema introspection helpers
# ---------------------------------------------------------------------------

def quote_ident(name: str) -> str:
    """Double-quote an identifier safely for inlining into SQL."""
    return '"' + name.replace('"', '""') + '"'


def list_relations(engine: Engine) -> list[tuple[str, str, str, bool]]:
    """Return (schema, name, kind, can_select) for every user relation."""
    sql = text(
        """
        SELECT
            n.nspname AS schema,
            c.relname AS name,
            CASE c.relkind
                WHEN 'r' THEN 'table'
                WHEN 'p' THEN 'partitioned table'
                WHEN 'v' THEN 'view'
                WHEN 'm' THEN 'matview'
                WHEN 'f' THEN 'foreign table'
                ELSE c.relkind::text
            END AS kind,
            has_table_privilege(c.oid, 'SELECT') AS can_select
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON c.relnamespace = n.oid
        WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
          AND n.nspname NOT LIKE 'pg_%'
          AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
        ORDER BY n.nspname, c.relname
        """
    )
    with engine.connect() as conn:
        return [(s, n, k, bool(cs)) for (s, n, k, cs) in conn.execute(sql)]


def describe_table(engine: Engine, table: str) -> None:
    schema, tname = ("public", table) if "." not in table else table.split(".", 1)
    sql = text(
        """
        SELECT
            a.attname AS column_name,
            pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type,
            NOT a.attnotnull AS is_nullable,
            has_column_privilege(c.oid, a.attname, 'SELECT') AS can_select
        FROM pg_catalog.pg_attribute a
        JOIN pg_catalog.pg_class c ON a.attrelid = c.oid
        JOIN pg_catalog.pg_namespace n ON c.relnamespace = n.oid
        WHERE n.nspname = :schema
          AND c.relname = :tname
          AND a.attnum > 0
          AND NOT a.attisdropped
        ORDER BY a.attnum
        """
    )
    with engine.connect() as conn:
        rows = list(conn.execute(sql, {"schema": schema, "tname": tname}))
    if not rows:
        sys.stderr.write(f"No columns found for {schema}.{tname}\n")
        sys.exit(1)
    width = max(len(r[0]) for r in rows)
    for col, dtype, nullable, can_select in rows:
        null = "NULL" if nullable else "NOT NULL"
        marker = "" if can_select else "  [no SELECT]"
        print(f"{col.ljust(width)}  {dtype}  {null}{marker}")


def list_tables_command(engine: Engine) -> None:
    rels = list_relations(engine)
    if not rels:
        sys.stderr.write("No relations found.\n")
        return
    width_name = max(len(f"{s}.{n}") for s, n, _, _ in rels)
    width_kind = max(len(k) for _, _, k, _ in rels)
    for schema, name, kind, can_select in rels:
        full = f"{schema}.{name}"
        marker = "" if can_select else "  [no SELECT]"
        print(f"{full.ljust(width_name)}  {kind.ljust(width_kind)}{marker}")


# ---------------------------------------------------------------------------
# Multi-table discovery
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TablePlan:
    """How to extract rows for a single table for a given project_type."""
    schema: str
    name: str
    reason: str
    filter_sql: str
    filter_params: dict
    has_tracker_type: bool
    pk_column: str = "id"  # used when this plan is referenced via IN subquery

    @property
    def fqname(self) -> str:
        return f"{quote_ident(self.schema)}.{quote_ident(self.name)}"

    @property
    def display(self) -> str:
        return f"{self.schema}.{self.name}"

    def as_id_subquery(self) -> str:
        """SQL fragment selecting this plan's PKs for use inside an IN(...)."""
        base = f"SELECT {quote_ident(self.pk_column)} FROM {self.fqname}"
        if self.filter_sql and self.filter_sql != "TRUE":
            base += f" WHERE {self.filter_sql}"
        return base


PROJECT_TYPE_COL_CANDIDATES: tuple[str, ...] = ("projectType", "project_type")
PROJECTS_TABLE_CANDIDATES: tuple[str, ...] = ("plant", "projects")


def _find_discriminator_column(
    engine: Engine, schema: str, name: str
) -> str | None:
    """Return whichever of {projectType, project_type} exists on this table, or None."""
    for col in PROJECT_TYPE_COL_CANDIDATES:
        if table_has_column(engine, schema, name, col):
            return col
    return None


def _find_pk_column(engine: Engine, schema: str, name: str) -> str | None:
    """Return the first primary-key column for a table, or None if it has none."""
    sql = text(
        """
        SELECT a.attname
        FROM pg_catalog.pg_index i
        JOIN pg_catalog.pg_class c ON c.oid = i.indrelid
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY(i.indkey)
        WHERE i.indisprimary
          AND n.nspname = :schema
          AND c.relname = :name
        ORDER BY array_position(i.indkey, a.attnum)
        """
    )
    with engine.connect() as conn:
        rows = [r[0] for r in conn.execute(sql, {"schema": schema, "name": name})]
    return rows[0] if rows else None


def find_projects_table(
    engine: Engine,
    override: str | None,
    project_type_name: str | None = None,
) -> tuple[str, str, str, str | None]:
    """Identify the primary projects table for a given project type.

    Returns (schema, name, primary_key_column, discriminator_column_or_None).

    Resolution order:
      1. --projects-table override, if supplied.
      2. The GEM convention: a table named `plant` with `projectType` column
         (real column name in this schema is camelCase).
      3. Legacy: a `projects` table with `project_type` column.
    """
    schema: str | None = None
    name: str | None = None

    if override:
        schema, name = ("public", override) if "." not in override else override.split(".", 1)
    else:
        # Find any candidate (plant/projects) that has a known discriminator
        # column. We iterate in priority order — the first match wins.
        sql = text(
            """
            SELECT n.nspname, c.relname
            FROM pg_catalog.pg_class c
            JOIN pg_catalog.pg_namespace n ON c.relnamespace = n.oid
            JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid
            WHERE c.relkind IN ('r', 'p')
              AND a.attname = :col
              AND NOT a.attisdropped
              AND c.relname = :name
              AND n.nspname NOT IN ('pg_catalog', 'information_schema')
            LIMIT 1
            """
        )
        with engine.connect() as conn:
            for cand_name in PROJECTS_TABLE_CANDIDATES:
                for col in PROJECT_TYPE_COL_CANDIDATES:
                    row = conn.execute(
                        sql, {"name": cand_name, "col": col}
                    ).fetchone()
                    if row:
                        schema, name = row[0], row[1]
                        break
                if schema:
                    break

        if not schema:
            sys.stderr.write(
                "ERROR: could not find a projects table.\n"
                f"  Looked for relations named {PROJECTS_TABLE_CANDIDATES!r} with a "
                f"discriminator column from {PROJECT_TYPE_COL_CANDIDATES!r}.\n"
                "  Pass --projects-table schema.name to override.\n"
            )
            sys.exit(2)

    discriminator_col = _find_discriminator_column(engine, schema, name)

    pk = _find_pk_column(engine, schema, name)
    if not pk:
        sys.stderr.write(
            f"ERROR: {schema}.{name} has no primary key — cannot follow FKs into it.\n"
        )
        sys.exit(2)
    return schema, name, pk, discriminator_col


def find_tables_with_column(engine: Engine, column: str) -> list[tuple[str, str]]:
    sql = text(
        """
        SELECT n.nspname, c.relname
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON c.relnamespace = n.oid
        JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid
        WHERE c.relkind IN ('r', 'p')
          AND a.attname = :col
          AND NOT a.attisdropped
          AND has_table_privilege(c.oid, 'SELECT')
          AND has_column_privilege(c.oid, a.attname, 'SELECT')
          AND n.nspname NOT IN ('pg_catalog', 'information_schema')
          AND n.nspname NOT LIKE 'pg_%'
        ORDER BY n.nspname, c.relname
        """
    )
    with engine.connect() as conn:
        return [(s, n) for (s, n) in conn.execute(sql, {"col": column})]


def find_tables_with_any_column(
    engine: Engine, columns: list[str]
) -> list[tuple[str, str, list[str]]]:
    """Find tables containing at least one of the given column names.

    Returns (schema, table, [matching columns sorted]).
    Used for ORM-style relations that are not enforced as DB-level FKs.
    """
    if not columns:
        return []
    sql = text(
        """
        SELECT
            n.nspname AS schema,
            c.relname AS name,
            ARRAY_AGG(a.attname ORDER BY a.attname) AS cols
        FROM pg_catalog.pg_attribute a
        JOIN pg_catalog.pg_class c ON a.attrelid = c.oid
        JOIN pg_catalog.pg_namespace n ON c.relnamespace = n.oid
        WHERE c.relkind IN ('r', 'p')
          AND NOT a.attisdropped
          AND a.attnum > 0
          AND a.attname = ANY(:cols)
          AND n.nspname NOT IN ('pg_catalog', 'information_schema')
          AND n.nspname NOT LIKE 'pg_%'
          AND has_table_privilege(c.oid, 'SELECT')
        GROUP BY n.nspname, c.relname
        ORDER BY n.nspname, c.relname
        """
    )
    with engine.connect() as conn:
        return [
            (r[0], r[1], list(r[2]))
            for r in conn.execute(sql, {"cols": list(columns)})
        ]


def find_fk_referencers(
    engine: Engine, target_schema: str, target_table: str
) -> list[tuple[str, str, str]]:
    """Find tables with a single-column FK pointing into target_schema.target_table.

    Returns a list of (schema, name, fk_column).
    """
    sql = text(
        """
        SELECT
            sn.nspname AS src_schema,
            sc.relname AS src_table,
            sa.attname AS src_column
        FROM pg_catalog.pg_constraint con
        JOIN pg_catalog.pg_class sc ON sc.oid = con.conrelid
        JOIN pg_catalog.pg_namespace sn ON sn.oid = sc.relnamespace
        JOIN pg_catalog.pg_class tc ON tc.oid = con.confrelid
        JOIN pg_catalog.pg_namespace tn ON tn.oid = tc.relnamespace
        JOIN pg_catalog.pg_attribute sa
            ON sa.attrelid = sc.oid AND sa.attnum = con.conkey[1]
        WHERE con.contype = 'f'
          AND tn.nspname = :tschema
          AND tc.relname = :tname
          AND array_length(con.conkey, 1) = 1
          AND has_table_privilege(sc.oid, 'SELECT')
          AND has_column_privilege(sc.oid, sa.attname, 'SELECT')
        ORDER BY sn.nspname, sc.relname
        """
    )
    with engine.connect() as conn:
        return [
            (s, t, c)
            for (s, t, c) in conn.execute(sql, {"tschema": target_schema, "tname": target_table})
        ]


def table_has_column(engine: Engine, schema: str, name: str, column: str) -> bool:
    sql = text(
        """
        SELECT 1
        FROM pg_catalog.pg_attribute a
        JOIN pg_catalog.pg_class c ON a.attrelid = c.oid
        JOIN pg_catalog.pg_namespace n ON c.relnamespace = n.oid
        WHERE n.nspname = :schema
          AND c.relname = :name
          AND a.attname = :col
          AND NOT a.attisdropped
        LIMIT 1
        """
    )
    with engine.connect() as conn:
        return conn.execute(sql, {"schema": schema, "name": name, "col": column}).fetchone() is not None


def discover_plans(
    engine: Engine,
    project_type: int,
    project_type_name: str | None,
    projects_table_override: str | None,
    extra_where: str | None,
) -> list[TablePlan]:
    """Build the export plan via BFS over the FK graph from the projects table.

    Step 1 — seed: filter the projects table (e.g., `plant`) by its
    discriminator column (e.g., `projectType = :ptype`).

    Step 2 — closure: walk FK referencers breadth-first. Each child plan
    filters via `<fk_col> IN (SELECT <parent_pk> FROM parent WHERE
    <parent_filter>)`, inheriting params so multi-hop chains stay bound.

    Step 3 — also include any other table that has a discriminator column
    of its own (typically just the projects table, but defensive).

    Tables in EXCLUDED_TABLES (Django/auth/social) and per-type tables
    belonging to OTHER project types are skipped.
    """
    proj_schema, proj_table, proj_pk, discriminator_col = find_projects_table(
        engine, projects_table_override, project_type_name
    )
    sys.stderr.write(
        f"Projects table: {proj_schema}.{proj_table} "
        f"(pk: {proj_pk}, discriminator: {discriminator_col or 'none'})\n"
    )

    excluded = set(EXCLUDED_TABLES)
    # Skip per-type tables that belong to OTHER project types. We compute this
    # via PROJECT_TYPE_BUNDLES as a convenience set; if a name doesn't appear
    # there it's still allowed.
    if project_type_name and project_type_name in PROJECT_TYPE_BUNDLES:
        my_cores = {tbl for tbl, _ in PROJECT_TYPE_BUNDLES[project_type_name]}
    else:
        my_cores = set()
    other_typed = _all_typed_tables() - my_cores
    excluded |= other_typed

    plans: list[TablePlan] = []
    seen: set[tuple[str, str]] = set()

    # Seed: projects table.
    if discriminator_col:
        seed = _make_direct_plan(
            engine, proj_schema, proj_table, project_type, extra_where,
            discriminator_col=discriminator_col,
        )
    else:
        seed = _make_full_plan(
            engine, proj_schema, proj_table, extra_where,
            reason_prefix="projects table (no discriminator column; full dump)",
        )
    plans.append(seed)
    seen.add((proj_schema, proj_table))

    # Also add any OTHER tables that carry their own discriminator column.
    # (Defensive — usually just user_settings, which is excluded anyway.)
    if discriminator_col:
        for s, t in find_tables_with_column(engine, discriminator_col):
            if (s, t) in seen or t in excluded:
                continue
            plans.append(
                _make_direct_plan(
                    engine, s, t, project_type, extra_where,
                    discriminator_col=discriminator_col,
                )
            )
            seen.add((s, t))

    # BFS FK closure.
    frontier: list[TablePlan] = [seed]
    depth = 1
    while frontier:
        next_frontier: list[TablePlan] = []
        for parent in frontier:
            refs = find_fk_referencers(engine, parent.schema, parent.name)
            for s, t, fk_col in refs:
                if (s, t) in seen or t in excluded:
                    continue
                child = _make_child_plan(engine, s, t, fk_col, parent, depth, extra_where)
                plans.append(child)
                seen.add((s, t))
                next_frontier.append(child)
        frontier = next_frontier
        depth += 1

    return plans


def _make_direct_plan(
    engine: Engine,
    schema: str,
    name: str,
    project_type: int,
    extra_where: str | None,
    discriminator_col: str = PROJECT_TYPE_COL,
) -> TablePlan:
    has_tt = table_has_column(engine, schema, name, "tracker_type")
    clauses = [f"{quote_ident(discriminator_col)} = :ptype"]
    params = {"ptype": project_type}
    reason = f"has `{discriminator_col}` column"
    if extra_where and has_tt:
        clauses.append(f"({extra_where})")
        reason += "; --where applied"
    elif extra_where and not has_tt:
        reason += "; --where skipped (no tracker_type)"
    pk = _find_pk_column(engine, schema, name) or "id"
    return TablePlan(
        schema=schema,
        name=name,
        reason=reason,
        filter_sql=" AND ".join(clauses),
        filter_params=params,
        has_tracker_type=has_tt,
        pk_column=pk,
    )


def _make_fk_plan(
    engine: Engine,
    schema: str,
    name: str,
    fk_col: str,
    proj_table_q: str,
    proj_pk: str,
    project_type: int,
    extra_where: str | None,
    projects_has_ptype: bool,
) -> TablePlan:
    has_tt = table_has_column(engine, schema, name, "tracker_type")
    # The inner subquery selects matching project IDs. If the projects table
    # has a project_type column we filter by it; otherwise the table IS the
    # project type and no extra filter is needed. The --where clause is
    # applied to the inner subquery so child rows inherit it via the FK.
    inner_clauses: list[str] = []
    params: dict = {}
    if projects_has_ptype:
        inner_clauses.append(f"{quote_ident(PROJECT_TYPE_COL)} = :ptype")
        params["ptype"] = project_type
    if extra_where:
        inner_clauses.append(f"({extra_where})")
    inner = f"SELECT {quote_ident(proj_pk)} FROM {proj_table_q}"
    if inner_clauses:
        inner += f" WHERE {' AND '.join(inner_clauses)}"
    clauses = [f"{quote_ident(fk_col)} IN ({inner})"]
    reason = f"FK `{fk_col}` -> {proj_table_q}.{quote_ident(proj_pk)}"
    if extra_where:
        # Also apply directly to the child if the column exists locally,
        # which avoids extra rows when a child also has tracker_type.
        if has_tt:
            clauses.append(f"({extra_where})")
            reason += "; --where applied to child + inherited via FK"
        else:
            reason += "; --where inherited via FK"
    return TablePlan(
        schema=schema,
        name=name,
        reason=reason,
        filter_sql=" AND ".join(clauses),
        filter_params=params,
        has_tracker_type=has_tt,
    )


def _make_full_plan(
    engine: Engine,
    schema: str,
    name: str,
    extra_where: str | None,
    reason_prefix: str = "per-type projects table (no project_type column; full dump)",
) -> TablePlan:
    """Plan that dumps the entire table (no project_type filter)."""
    has_tt = table_has_column(engine, schema, name, "tracker_type")
    clauses: list[str] = []
    reason = reason_prefix
    if extra_where and has_tt:
        clauses.append(f"({extra_where})")
        reason += "; --where applied"
    elif extra_where:
        reason += "; --where skipped (no tracker_type column)"
    pk = _find_pk_column(engine, schema, name) or "id"
    return TablePlan(
        schema=schema,
        name=name,
        reason=reason,
        filter_sql=" AND ".join(clauses) if clauses else "TRUE",
        filter_params={},
        has_tracker_type=has_tt,
        pk_column=pk,
    )


def _make_child_plan(
    engine: Engine,
    schema: str,
    name: str,
    fk_col: str,
    parent: TablePlan,
    depth: int,
    extra_where: str | None,
) -> TablePlan:
    """Plan that filters rows via FK: fk_col IN (SELECT parent.pk FROM parent ...).

    Inherits parent's filter parameters so multi-hop chains continue to bind
    the same :ptype param.
    """
    has_tt = table_has_column(engine, schema, name, "tracker_type")
    clauses = [f"{quote_ident(fk_col)} IN ({parent.as_id_subquery()})"]
    params = dict(parent.filter_params)
    reason = f"FK chain depth {depth}: `{fk_col}` -> {parent.display}.{parent.pk_column}"
    if extra_where and has_tt:
        clauses.append(f"({extra_where})")
        reason += "; --where applied"
    elif extra_where:
        reason += "; --where inherited via FK"
    pk = _find_pk_column(engine, schema, name) or "id"
    return TablePlan(
        schema=schema,
        name=name,
        reason=reason,
        filter_sql=" AND ".join(clauses),
        filter_params=params,
        has_tracker_type=has_tt,
        pk_column=pk,
    )


# ---------------------------------------------------------------------------
# Streaming export
# ---------------------------------------------------------------------------

def stream_to_csv(
    engine: Engine,
    sql: str,
    params: dict,
    out_path: str,
    chunksize: int,
    progress_label: str = "",
) -> int:
    """Execute sql and stream rows to a CSV file. Returns rows written."""
    written = 0
    with engine.connect().execution_options(stream_results=True) as conn:
        result = conn.execute(text(sql), params)
        columns = list(result.keys())
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            while True:
                batch = result.fetchmany(chunksize)
                if not batch:
                    break
                for row in batch:
                    writer.writerow(row)
                written += len(batch)
                if progress_label:
                    sys.stderr.write(f"\r  {progress_label}: {written:,} rows")
                    sys.stderr.flush()
    if progress_label:
        sys.stderr.write("\n")
    return written


def export_plan(
    engine: Engine,
    plan: TablePlan,
    out_dir: str,
    chunksize: int,
    limit: int | None,
) -> tuple[str, int]:
    """Run one plan and write to {out_dir}/{schema}.{name}.csv. Returns (path, rows)."""
    sql = f"SELECT * FROM {plan.fqname} WHERE {plan.filter_sql}"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    safe = SAFE_NAME_RE.sub("_", plan.display)
    out_path = os.path.join(out_dir, f"{safe}.csv")
    n = stream_to_csv(engine, sql, plan.filter_params, out_path, chunksize, plan.display)
    return out_path, n


# ---------------------------------------------------------------------------
# Single-table mode
# ---------------------------------------------------------------------------

def build_single_table_query(
    table: str,
    project_type: int | None,
    where: str | None,
    limit: int | None,
) -> tuple[str, dict]:
    schema, tname = ("public", table) if "." not in table else table.split(".", 1)
    fq = f"{quote_ident(schema)}.{quote_ident(tname)}"
    clauses: list[str] = []
    params: dict = {}
    if project_type is not None:
        clauses.append(f"{quote_ident(PROJECT_TYPE_COL)} = :ptype")
        params["ptype"] = project_type
    if where:
        clauses.append(f"({where})")
    sql = f"SELECT * FROM {fq}"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return sql, params


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export GEM read-only Postgres data to CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Project type codes:\n  "
            + "\n  ".join(f"{code} = {name}" for name, code in PROJECT_TYPES.items())
        ),
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path. In multi-table mode this is a DIRECTORY (one CSV per "
             "table). In single-table mode (--table) or --sql mode this is a FILE.",
    )
    parser.add_argument(
        "-p", "--project-type",
        help="Project type name (lng, combustion, ...) or numeric code (1-9).",
    )
    parser.add_argument(
        "-t", "--table",
        help="Single-table mode: dump one specific table only. "
             "Use schema.table for non-public schemas. "
             "Without this flag, all related tables are exported.",
    )
    parser.add_argument(
        "--projects-table",
        help="Override the primary projects table (default: 'projects'). "
             "Used as the FK target for related-table discovery.",
    )
    parser.add_argument(
        "-w", "--where",
        help="Extra SQL WHERE clause. In multi-table mode it is applied only to "
             "tables that have a `tracker_type` column (typically the projects "
             "table itself). Useful for combustion subtrackers, e.g. "
             "--where \"tracker_type = 'gcpt'\".",
    )
    parser.add_argument(
        "--limit", type=int,
        help="Cap row count per table (handy for testing).",
    )
    parser.add_argument(
        "--sql",
        help="Run an arbitrary SELECT statement instead of the project-type "
             "shorthand. Read-only is enforced. Output is a single CSV file.",
    )
    parser.add_argument(
        "--chunksize", type=int, default=DEFAULT_CHUNKSIZE,
        help=f"Rows per fetch batch (default: {DEFAULT_CHUNKSIZE}).",
    )
    parser.add_argument(
        "--timeout-ms", type=int, default=DEFAULT_STATEMENT_TIMEOUT_MS,
        help=f"Postgres statement_timeout in ms (default: {DEFAULT_STATEMENT_TIMEOUT_MS}).",
    )
    parser.add_argument(
        "--list-tables", action="store_true",
        help="List all non-system tables and exit.",
    )
    parser.add_argument(
        "--describe", metavar="TABLE",
        help="Print columns + types for a table and exit.",
    )
    parser.add_argument(
        "--discover", action="store_true",
        help="Print the multi-table export plan for --project-type and exit "
             "(no rows fetched). Use to preview what would be exported.",
    )

    args = parser.parse_args(argv)

    introspect = bool(args.list_tables or args.describe or args.discover)

    if args.sql:
        if args.project_type or args.where or args.table or args.discover:
            parser.error("--sql cannot be combined with --project-type/--where/--table/--discover.")
        stripped = args.sql.strip().rstrip(";").lstrip().lower()
        if not stripped.startswith(("select", "with")):
            parser.error("--sql must be a SELECT (or WITH ... SELECT) statement.")

    if args.discover and not args.project_type:
        parser.error("--discover requires --project-type.")

    if not introspect:
        if not args.output:
            parser.error("--output is required (unless using --list-tables, --describe, or --discover).")
        if not args.sql:
            if not args.project_type and not args.where and not args.table:
                parser.error(
                    "Provide --project-type (and optionally --where), "
                    "--table + filter, or --sql. "
                    "Refusing to dump everything by accident."
                )
            if args.table and not args.project_type and not args.where:
                parser.error(
                    "--table requires either --project-type or --where so we don't "
                    "dump the entire table by accident."
                )

    url = get_database_url()
    engine = build_engine(url, args.timeout_ms)

    if args.list_tables:
        list_tables_command(engine)
        return 0
    if args.describe:
        describe_table(engine, args.describe)
        return 0

    if args.discover:
        ptype = resolve_project_type(args.project_type)
        ptype_name = PROJECT_TYPE_NAMES.get(ptype)
        plans = discover_plans(engine, ptype, ptype_name, args.projects_table, args.where)
        if not plans:
            sys.stderr.write("No tables matched.\n")
            return 0
        width = max(len(p.display) for p in plans)
        for p in plans:
            print(f"{p.display.ljust(width)}  {p.reason}")
        return 0

    if args.sql:
        sys.stderr.write(f"Query: {args.sql}\n")
        n = stream_to_csv(engine, args.sql, {}, args.output, args.chunksize, "rows")
        sys.stderr.write(f"Wrote {n:,} rows to {args.output}\n")
        return 0

    if args.table:
        ptype = resolve_project_type(args.project_type) if args.project_type else None
        sql, params = build_single_table_query(args.table, ptype, args.where, args.limit)
        sys.stderr.write(f"Query: {sql}\n")
        if params:
            sys.stderr.write(f"Params: {params}\n")
        n = stream_to_csv(engine, sql, params, args.output, args.chunksize, args.table)
        sys.stderr.write(f"Wrote {n:,} rows to {args.output}\n")
        return 0

    # Multi-table mode (default).
    ptype = resolve_project_type(args.project_type)
    ptype_name = PROJECT_TYPE_NAMES.get(ptype)
    plans = discover_plans(engine, ptype, ptype_name, args.projects_table, args.where)
    if not plans:
        sys.stderr.write("No tables matched. Nothing to export.\n")
        return 1

    os.makedirs(args.output, exist_ok=True)
    sys.stderr.write(
        f"Exporting {len(plans)} table(s) for project_type={ptype} "
        f"({PROJECT_TYPE_NAMES.get(ptype, '?')}) -> {args.output}/\n"
    )

    summary: list[tuple[str, int, str]] = []
    total_rows = 0
    for plan in plans:
        sys.stderr.write(f"\n[{plan.display}] {plan.reason}\n")
        try:
            path, n = export_plan(engine, plan, args.output, args.chunksize, args.limit)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"  FAILED: {e!r}\n")
            summary.append((plan.display, 0, f"FAILED: {e}"))
            continue
        summary.append((plan.display, n, os.path.basename(path)))
        total_rows += n

    manifest_path = os.path.join(args.output, "_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["table", "rows", "file_or_status"])
        for row in summary:
            w.writerow(row)

    sys.stderr.write(
        f"\nDone. {total_rows:,} total rows across {len(plans)} table(s). "
        f"Manifest: {manifest_path}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())