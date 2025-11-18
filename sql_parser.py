"""
SQL DDL Parser for Schema Management

Parses SQL DDL files and executes them in PostgreSQL to create tables.
"""

import re
import psycopg2
from typing import List, Set, Optional, Any, Tuple


def parse_ddl_file(filepath: str) -> List[str]:
    """
    Parse a DDL file and extract CREATE TABLE statements.
    
    Args:
        filepath: Path to the DDL file
        
    Returns:
        List of CREATE TABLE statements as strings
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"DDL file not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error reading DDL file: {e}")
    
    # Split by semicolons and extract CREATE TABLE statements
    statements = []
    
    # Remove comments (-- style and /* */ style)
    content = re.sub(r'--.*?$', '', content, flags=re.MULTILINE)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    
    # Split by semicolons
    parts = re.split(r';\s*', content)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Check if it's a CREATE TABLE statement
        if re.match(r'CREATE\s+TABLE', part, re.IGNORECASE):
            # Add semicolon back
            statements.append(part + ';')
        elif re.match(r'DROP\s+TABLE', part, re.IGNORECASE):
            # Also include DROP TABLE statements
            statements.append(part + ';')
        elif re.match(r'CREATE\s+INDEX', part, re.IGNORECASE):
            # Include CREATE INDEX statements
            statements.append(part + ';')
    
    return statements


def execute_ddl(conn: Any, ddl_statements: List[str], drop_existing: bool = True) -> List[str]:
    """
    Execute DDL statements in PostgreSQL.
    
    Args:
        conn: PostgreSQL connection
        ddl_statements: List of DDL statements to execute
        drop_existing: If True, drop tables before creating (based on DROP TABLE statements)
        
    Returns:
        List of table names that were created
    """
    created_tables = []
    
    with conn.cursor() as cur:
        try:
            # First, execute any DROP TABLE statements
            for stmt in ddl_statements:
                if re.match(r'DROP\s+TABLE', stmt, re.IGNORECASE):
                    cur.execute(stmt)
                    conn.commit()
            
            # Then execute CREATE TABLE statements
            for stmt in ddl_statements:
                if re.match(r'CREATE\s+TABLE', stmt, re.IGNORECASE):
                    cur.execute(stmt)
                    # Extract table name
                    table_match = re.search(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)', stmt, re.IGNORECASE)
                    if table_match:
                        created_tables.append(table_match.group(1))
                    conn.commit()
                elif re.match(r'CREATE\s+INDEX', stmt, re.IGNORECASE):
                    cur.execute(stmt)
                    conn.commit()
            
        except psycopg2.Error as e:
            conn.rollback()
            raise Exception(f"Error executing DDL: {e}")
    
    return created_tables


def get_tables_in_schema(conn: Any) -> Set[str]:
    """
    Get list of all tables in the current database schema.
    
    Returns:
        Set of table names
    """
    tables = set()
    
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
    Validate that a DDL file exists and is readable.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return False, "DDL file is empty"
        return True, None
    except FileNotFoundError:
        return False, f"DDL file not found: {filepath}"
    except Exception as e:
        return False, f"Error reading DDL file: {e}"

