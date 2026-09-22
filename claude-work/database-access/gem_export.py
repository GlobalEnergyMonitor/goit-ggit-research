#!/usr/bin/env python3
"""
gem_export.py — Export rows from the GEM read-only Postgres DB to CSV.

The database URL is read from the GEM_READONLY_DB_URL environment variable.
Nothing sensitive is hardcoded.

QUICK START
-----------
1. Set the connection URL in your shell (do NOT paste it into this file):

       export GEM_READONLY_DB_URL='postgres://readonly:PASSWORD@HOST:5432/DBNAME'

   Or put it in a .env file that's gitignored:

       echo "GEM_READONLY_DB_URL='postgres://...'" > .env
       echo ".env" >> .gitignore
       set -a; source .env; set +a

2. Install deps once:

       pip install 'sqlalchemy>=2.0' psycopg2-binary

3. Run it:

       # Export all LNG projects
       python gem_export.py --project-type lng -o lng_projects.csv

       # Same thing by numeric ID
       python gem_export.py --project-type 8 -o lng_projects.csv

       # Combustion: project_type=1 returns ALL combustion subtrackers, so add
       # a sub-filter on the tracker_type column to narrow it down (e.g., GCPT
       # for the Global Coal Plant Tracker — confirm the actual value with
       # --peek first; the column may use codes you don't expect):
       python gem_export.py --project-type combustion \\
           --where "tracker_type = 'gcpt'" \\
           -o coal_plants.csv

       # See what tables exist
       python gem_export.py --list-tables

       # Peek at a few rows of the projects table to see the column names
       python gem_export.py --peek projects

PROJECT_TYPE codes (from GEM):
    1 = combustion       6 = steel
    2 = solar            7 = hydro
    3 = wind             8 = lng
    4 = nuclear          9 = goget
    5 = geothermal
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import Engine
except ImportError:
    sys.stderr.write(
        "ERROR: sqlalchemy is not installed.\n"
        "Install with: pip install 'sqlalchemy>=2.0' psycopg2-binary\n"
    )
    sys.exit(1)


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

ENV_VAR = "GEM_READONLY_DB_URL"
DEFAULT_TABLE = "projects"          # the main projects table
PROJECT_TYPE_COL = "project_type"   # column we filter on
CHUNKSIZE = 1000


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #

def get_engine() -> Engine:
    """Build a SQLAlchemy engine from the env var, with safety rails."""
    url = os.environ.get(ENV_VAR)
    if not url:
        sys.stderr.write(
            f"ERROR: environment variable {ENV_VAR} is not set.\n"
            f"Set it in your shell:\n"
            f"    export {ENV_VAR}='postgres://readonly:PASSWORD@HOST:5432/DBNAME'\n"
        )
        sys.exit(2)

    # Heroku still hands out the deprecated postgres:// scheme; SQLAlchemy 1.4+
    # wants postgresql://.
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    engine = create_engine(url, pool_pre_ping=True)

    # Defense in depth: even if the role wasn't truly read-only, this would
    # block writes for this session.
    with engine.connect() as conn:
        conn.execute(text("SET default_transaction_read_only = on"))
        conn.execute(text("SET statement_timeout = '5min'"))
        conn.commit()

    return engine


# --------------------------------------------------------------------------- #
# Read-only query guard
# --------------------------------------------------------------------------- #

def assert_readonly_sql(sql: str) -> None:
    """Crude check that custom SQL is a SELECT, not a write."""
    stripped = sql.strip().lower()
    if not (stripped.startswith("select") or stripped.startswith("with")):
        raise ValueError(
            "Only SELECT / WITH queries are allowed. Refused: "
            + sql.split()[0]
        )
    forbidden = ("insert", "update", "delete", "drop", "truncate",
                 "alter", "create", "grant", "revoke")
    # Check first token only — words like 'updated_at' in column names
    # are fine.
    first = stripped.split()[0]
    if first in forbidden:
        raise ValueError(f"Refused write keyword: {first}")


# --------------------------------------------------------------------------- #
# Streaming CSV writer
# --------------------------------------------------------------------------- #

def stream_query_to_csv(engine: Engine, sql: str, params: dict,
                        out_path: str) -> int:
    """Run sql against engine, stream rows into out_path. Returns row count."""
    assert_readonly_sql(sql)
    rows_written = 0
    with engine.connect() as conn:
        # execution_options(stream_results=True) tells psycopg2 to use a
        # server-side cursor, so memory stays flat even for millions of rows.
        result = conn.execution_options(stream_results=True).execute(
            text(sql), params
        )
        cols = list(result.keys())

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(cols)

            while True:
                batch = result.fetchmany(CHUNKSIZE)
                if not batch:
                    break
                for row in batch:
                    writer.writerow(row)
                rows_written += len(batch)

    return rows_written


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #

def cmd_export(args: argparse.Namespace) -> int:
    """The main path: filter `projects` (or another table) by project_type."""
    # Resolve project_type to an integer
    pt_arg = args.project_type
    if pt_arg.isdigit():
        pt_int = int(pt_arg)
        if pt_int not in PROJECT_TYPES.values():
            sys.stderr.write(
                f"WARNING: {pt_int} is not a known PROJECT_TYPE. "
                f"Known: {sorted(PROJECT_TYPES.values())}\n"
            )
    else:
        key = pt_arg.lower()
        if key not in PROJECT_TYPES:
            sys.stderr.write(
                f"ERROR: unknown project type '{pt_arg}'. "
                f"Known: {sorted(PROJECT_TYPES)}\n"
            )
            return 2
        pt_int = PROJECT_TYPES[key]

    table = args.table
    sql = f"SELECT * FROM {table} WHERE {PROJECT_TYPE_COL} = :pt"
    if args.where:
        sql += f" AND ({args.where})"
    if args.limit:
        sql += f" LIMIT {int(args.limit)}"

    print(f"[gem_export] table = {table}")
    print(f"[gem_export] project_type = {pt_int} "
          f"({_name_for(pt_int)})")
    if args.where:
        print(f"[gem_export] extra filter: {args.where}")
    print(f"[gem_export] output = {args.output}")

    engine = get_engine()
    n = stream_query_to_csv(engine, sql, {"pt": pt_int}, args.output)
    print(f"[gem_export] wrote {n:,} rows -> {args.output}")
    return 0


def cmd_list_tables(args: argparse.Namespace) -> int:
    engine = get_engine()
    sql = """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql)).fetchall()
    for schema, name in rows:
        prefix = "" if schema == "public" else f"{schema}."
        print(f"  {prefix}{name}")
    print(f"\n{len(rows)} tables.")
    return 0


def cmd_peek(args: argparse.Namespace) -> int:
    """Show column names + first 3 rows of a table."""
    engine = get_engine()
    sql = f"SELECT * FROM {args.peek} LIMIT 3"
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        cols = list(result.keys())
        rows = result.fetchall()
    print(f"Columns ({len(cols)}):")
    for c in cols:
        print(f"  - {c}")
    print(f"\nFirst {len(rows)} row(s):")
    for r in rows:
        print(" ", dict(zip(cols, r)))
    return 0


def cmd_sql(args: argparse.Namespace) -> int:
    if not args.output:
        sys.stderr.write("ERROR: --sql requires -o/--output\n")
        return 2
    engine = get_engine()
    n = stream_query_to_csv(engine, args.sql, {}, args.output)
    print(f"[gem_export] wrote {n:,} rows -> {args.output}")
    return 0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _name_for(pt_int: int) -> str:
    for name, num in PROJECT_TYPES.items():
        if num == pt_int:
            return name
    return "unknown"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gem_export",
        description="Export rows from the GEM read-only Postgres DB to CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # The four modes are mutually exclusive at the top level.
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--project-type",
        help="Project type to filter on. Either a name (lng, combustion, ...) "
             "or the integer code (1-9).",
    )
    mode.add_argument(
        "--list-tables",
        action="store_true",
        help="List all tables in the database and exit.",
    )
    mode.add_argument(
        "--peek",
        metavar="TABLE",
        help="Show columns and first 3 rows of a table, then exit.",
    )
    mode.add_argument(
        "--sql",
        help="Run an arbitrary SELECT (read-only enforced). Requires -o.",
    )

    p.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help=f"Table to query in --project-type mode (default: {DEFAULT_TABLE}).",
    )
    p.add_argument(
        "--where",
        help="Extra SQL WHERE clause (no leading AND). E.g. "
             "\"tracker_type = 'gcpt'\" for combustion subtracker filtering.",
    )
    p.add_argument(
        "--limit",
        type=int,
        help="Optional LIMIT on the output (handy for sanity checks).",
    )
    p.add_argument(
        "-o", "--output",
        help="Output CSV path. Required for --project-type and --sql modes.",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_tables:
        return cmd_list_tables(args)
    if args.peek:
        return cmd_peek(args)
    if args.sql:
        return cmd_sql(args)
    if args.project_type:
        if not args.output:
            sys.stderr.write("ERROR: --project-type requires -o/--output\n")
            return 2
        return cmd_export(args)

    build_parser().print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())