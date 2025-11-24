"""
SQL DDL Parser for Schema Management.

This module provides a lightweight utility for:
  1. Parsing SQL DDL files and extracting structural statements
     (CREATE TABLE, DROP TABLE, CREATE INDEX).
  2. Executing those statements against a PostgreSQL database.
  3. Inspecting the current schema to discover existing tables.
  4. Validating that a DDL file exists and is non-empty.

It is intended for simple schema management in development / benchmarking
workflows, not as a full SQL parser. The implementation relies on
regex-based extraction and makes a few assumptions about formatting
(e.g. statements are separated by semicolons, no semicolons inside
identifiers, etc.).
"""

import re
import psycopg2
from typing import List, Set, Optional, Any, Tuple


def parse_ddl_file(filepath: str) -> List[str]:
    """
    Parse a DDL file and extract relevant DDL statements.

    Currently, this focuses on:
      - CREATE TABLE
      - DROP TABLE
      - CREATE INDEX

    Comments are stripped out and the file is split on semicolons to
    approximate statement boundaries.

    Args:
        filepath: Path to the DDL file.

    Returns:
        List of DDL statements as strings, each ending with a semicolon.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For other I/O or decoding issues.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        # Explicitly re-raise with a clearer message so callers
        # can show helpful feedback to the user / CLI.
        raise FileNotFoundError(f"DDL file not found: {filepath}")
    except Exception as e:
        # Wrap any other file-related errors (permissions, encoding, etc.).
        raise Exception(f"Error reading DDL file: {e}")
    
    # Remove single-line comments:  -- comment
    content = re.sub(r'--.*?$', '', content, flags=re.MULTILINE)
    # Remove block comments:  /* comment */
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    
    # Split by semicolons to approximate SQL statement boundaries.
    # Note: this is a simplification and will break if semicolons appear
    # inside strings or other constructs.
    statements: List[str] = []
    parts = re.split(r';\s*', content)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Only keep statements we explicitly know how to handle.
        # We also append the semicolon back because PostgreSQL expects it
        # for multi-statement execution contexts and for clarity/logging.
        if re.match(r'CREATE\s+TABLE', part, re.IGNORECASE):
            statements.append(part + ';')
        elif re.match(r'DROP\s+TABLE', part, re.IGNORECASE):
            statements.append(part + ';')
        elif re.match(r'CREATE\s+INDEX', part, re.IGNORECASE):
            statements.append(part + ';')
    
    return statements


def execute_ddl(conn: Any, ddl_statements: List[str], drop_existing: bool = True) -> List[str]:
    """
    Execute a list of DDL statements against PostgreSQL.

    The execution order is:
      1. DROP TABLE statements (to clear out any existing tables).
      2. CREATE TABLE statements (to create the schema).
      3. CREATE INDEX statements (to add indexes on top).

    This ordering minimizes dependency issues (e.g. re-creating tables
    with indexes that reference them).

    Args:
        conn: An open psycopg2 PostgreSQL connection.
        ddl_statements: List of SQL statements to execute.
        drop_existing: If True, execute DROP TABLE statements first.
                       (Currently, this flag controls whether DROP TABLE
                       statements in `ddl_statements` are executed at all.)

    Returns:
        List of table names that were successfully created.

    Raises:
        Exception: If any psycopg2 error occurs while executing DDL.
    """
    created_tables: List[str] = []
    
    with conn.cursor() as cur:
        try:
            # Optionally execute DROP TABLE statements first to ensure a
            # clean slate when re-running migrations in a dev environment.
            if drop_existing:
                for stmt in ddl_statements:
                    if re.match(r'DROP\s+TABLE', stmt, re.IGNORECASE):
                        cur.execute(stmt)
                        conn.commit()
            
            # Execute CREATE TABLE and CREATE INDEX statements.
            for stmt in ddl_statements:
                if re.match(r'CREATE\s+TABLE', stmt, re.IGNORECASE):
                    cur.execute(stmt)

                    # Extract the table name from the CREATE TABLE statement.
                    # This assumes relatively simple table identifiers (no schema
                    # qualification or quoted identifiers).
                    table_match = re.search(
                        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)',
                        stmt,
                        re.IGNORECASE
                    )
                    if table_match:
                        created_tables.append(table_match.group(1))
                    conn.commit()

                elif re.match(r'CREATE\s+INDEX', stmt, re.IGNORECASE):
                    # Index creation is independent from table list tracking,
                    # so we just execute and commit.
                    cur.execute(stmt)
                    conn.commit()
            
        except psycopg2.Error as e:
            # Roll back the entire transaction block on any DDL error so the
            # connection is left in a clean state for subsequent operations.
            conn.rollback()
            raise Exception(f"Error executing DDL: {e}")
    
    return created_tables


def get_tables_in_schema(conn: Any) -> Set[str]:
    """
    Retrieve the set of all base tables in the 'public' schema.

    This is useful for:
      - Sanity checks after running migrations.
      - Comparing expected vs. actual tables during tests.
      - Light introspection of the database state.

    Args:
        conn: An open psycopg2 PostgreSQL connection.

    Returns:
        A set of table names (strings) found in the public schema.
    """
    tables: Set[str] = set()
    
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
              AND table_type = 'BASE TABLE'
        """)
        rows = cur.fetchall()
        tables = {row[0] for row in rows}
    
    return tables


def validate_ddl_file(filepath: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a DDL file exists and is non-empty.

    This function is intended as a lightweight pre-check before attempting
    to parse or execute a DDL file. It does *not* validate the SQL syntax
    itself, only the file's presence and basic content.

    Args:
        filepath: Path to the DDL file.

    Returns:
        A tuple of:
          - is_valid: True if the file exists and contains non-whitespace text.
          - error_message: None if valid, otherwise a human-readable error.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                # File exists but contains no meaningful content.
                return False, "DDL file is empty"
        return True, None
    except FileNotFoundError:
        # Mirror parse_ddl_file() messaging style so callers see consistent errors.
        return False, f"DDL file not found: {filepath}"
    except Exception as e:
        # Any other file I/O related error.
        return False, f"Error reading DDL file: {e}"
