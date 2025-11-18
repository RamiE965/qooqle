"""
SQL Query Parser for Join Optimization Benchmark

Parses SQL SELECT queries to extract:
- Table aliases and actual table names
- Join conditions
- Query parts (SELECT, FROM, WHERE, GROUP BY, ORDER BY, LIMIT)
"""

import re
from typing import Dict, List, Set, Tuple, Optional


def parse_sql_query(query: str) -> Dict:
    """
    Parse a SQL SELECT query and extract all information needed for benchmarking.
    
    Returns:
        dict with keys:
        - relations_map: dict mapping alias -> table_name
        - relations: list of relation aliases in order
        - join_conditions: dict mapping frozenset([alias1, alias2]) -> join_condition_string
        - query_parts: dict with 'select', 'from', 'where', 'group_by', 'order_by', 'limit'
    """
    # Extract query parts first (before normalization to preserve formatting)
    query_parts = extract_query_parts(query)
    
    # Normalize FROM clause for parsing (but keep original in query_parts)
    from_clause = query_parts.get('from', '')
    if not from_clause:
        raise ValueError("Query must contain a FROM clause")
    
    # Extract tables and joins from the original FROM clause
    relations_map, relations, join_conditions = extract_tables_and_joins(from_clause)
    
    return {
        'relations_map': relations_map,
        'relations': relations,
        'join_conditions': join_conditions,
        'query_parts': query_parts
    }


def extract_query_parts(query: str) -> Dict[str, str]:
    """
    Extract different parts of a SQL query (SELECT, FROM, WHERE, etc.)
    Preserves original case and formatting.
    """
    # Pattern to match SQL query structure
    # SELECT ... FROM ... [WHERE ...] [GROUP BY ...] [ORDER BY ...] [LIMIT ...]
    
    parts = {
        'select': '',
        'from': '',
        'where': '',
        'group_by': '',
        'order_by': '',
        'limit': ''
    }
    
    # Extract SELECT clause - preserve original case
    select_match = re.search(r'SELECT\s+(.+?)(?=\s+FROM)', query, re.IGNORECASE | re.DOTALL)
    if select_match:
        parts['select'] = query[select_match.start():select_match.end()].strip()
    
    # Extract FROM clause (everything from FROM to WHERE/GROUP BY/ORDER BY/LIMIT or end)
    from_match = re.search(r'FROM\s+(.+?)(?=\s+(?:WHERE|GROUP\s+BY|ORDER\s+BY|LIMIT|$))', query, re.IGNORECASE | re.DOTALL)
    if from_match:
        # Find the actual FROM keyword position to preserve case
        from_keyword_pos = query.upper().find('FROM', from_match.start())
        if from_keyword_pos >= 0:
            parts['from'] = query[from_keyword_pos:from_match.end()].strip()
        else:
            parts['from'] = 'FROM ' + from_match.group(1).strip()
    
    # Extract WHERE clause
    where_match = re.search(r'WHERE\s+(.+?)(?=\s+(?:GROUP\s+BY|ORDER\s+BY|LIMIT|$))', query, re.IGNORECASE | re.DOTALL)
    if where_match:
        where_keyword_pos = query.upper().find('WHERE', where_match.start())
        if where_keyword_pos >= 0:
            parts['where'] = query[where_keyword_pos:where_match.end()].strip()
        else:
            parts['where'] = 'WHERE ' + where_match.group(1).strip()
    
    # Extract GROUP BY clause
    group_by_match = re.search(r'GROUP\s+BY\s+(.+?)(?=\s+(?:ORDER\s+BY|LIMIT|$))', query, re.IGNORECASE | re.DOTALL)
    if group_by_match:
        group_by_keyword_pos = query.upper().find('GROUP', group_by_match.start())
        if group_by_keyword_pos >= 0:
            # Find the end of GROUP BY clause
            end_pos = group_by_match.end()
            parts['group_by'] = query[group_by_keyword_pos:end_pos].strip()
        else:
            parts['group_by'] = 'GROUP BY ' + group_by_match.group(1).strip()
    
    # Extract ORDER BY clause
    order_by_match = re.search(r'ORDER\s+BY\s+(.+?)(?=\s+LIMIT|$)', query, re.IGNORECASE | re.DOTALL)
    if order_by_match:
        order_by_keyword_pos = query.upper().find('ORDER', order_by_match.start())
        if order_by_keyword_pos >= 0:
            end_pos = order_by_match.end()
            parts['order_by'] = query[order_by_keyword_pos:end_pos].strip()
        else:
            parts['order_by'] = 'ORDER BY ' + order_by_match.group(1).strip()
    
    # Extract LIMIT clause
    limit_match = re.search(r'LIMIT\s+(\d+)', query, re.IGNORECASE)
    if limit_match:
        limit_keyword_pos = query.upper().find('LIMIT', limit_match.start())
        if limit_keyword_pos >= 0:
            end_pos = limit_match.end()
            parts['limit'] = query[limit_keyword_pos:end_pos].strip()
        else:
            parts['limit'] = 'LIMIT ' + limit_match.group(1)
    
    return parts


def extract_tables_and_joins(from_clause: str) -> Tuple[Dict[str, str], List[str], Dict]:
    """
    Extract table names, aliases, and join conditions from FROM clause.
    
    Returns:
        (relations_map, relations, join_conditions)
        - relations_map: dict mapping alias -> table_name
        - relations: list of aliases in order they appear
        - join_conditions: dict mapping frozenset([alias1, alias2]) -> join_condition_string
    """
    relations_map = {}
    relations = []
    join_conditions = {}
    
    # Remove 'FROM' keyword if present
    from_clause = re.sub(r'^\s*FROM\s+', '', from_clause, flags=re.IGNORECASE).strip()
    
    # Split by JOIN keywords (handles INNER JOIN, LEFT JOIN, RIGHT JOIN, etc.)
    # Pattern: table_name [AS] alias JOIN ...
    join_pattern = r'\b(?:INNER\s+)?JOIN|LEFT\s+(?:OUTER\s+)?JOIN|RIGHT\s+(?:OUTER\s+)?JOIN|FULL\s+(?:OUTER\s+)?JOIN'
    
    # Split the FROM clause into parts
    parts = re.split(join_pattern, from_clause, flags=re.IGNORECASE)
    
    if not parts:
        return relations_map, relations, join_conditions
    
    # Process first table (before any JOIN)
    first_table_part = parts[0].strip()
    table_name, alias = parse_table_alias(first_table_part)
    if table_name:
        relations_map[alias] = table_name
        relations.append(alias)
    
    # Process JOIN clauses
    join_matches = list(re.finditer(join_pattern, from_clause, re.IGNORECASE))
    
    for i, join_match in enumerate(join_matches):
        # Get the part after this JOIN keyword
        if i + 1 < len(parts):
            join_table_part = parts[i + 1].strip()
            
            # Split by ON to get table and join condition
            on_match = re.search(r'\s+ON\s+(.+)$', join_table_part, re.IGNORECASE)
            if on_match:
                join_condition = on_match.group(1).strip()
                table_part = join_table_part[:on_match.start()].strip()
            else:
                # No ON clause - this shouldn't happen in valid queries, but handle it
                table_part = join_table_part
                join_condition = None
            
            # Parse the table and alias
            table_name, alias = parse_table_alias(table_part)
            if table_name:
                relations_map[alias] = table_name
                relations.append(alias)
                
                # Extract join condition and determine which two tables it connects
                if join_condition:
                    # Find which two tables are involved in the join condition
                    # Look for patterns like alias1.column = alias2.column
                    table_pair = find_join_table_pair(join_condition, relations_map)
                    if table_pair:
                        join_conditions[table_pair] = join_condition
    
    return relations_map, relations, join_conditions


def parse_table_alias(table_part: str) -> Tuple[Optional[str], str]:
    """
    Parse a table reference like 'table_name AS alias' or 'table_name alias' or just 'table_name'.
    
    Returns:
        (table_name, alias) - if no alias, alias = table_name
    """
    table_part = table_part.strip()
    if not table_part:
        return None, ''
    
    # Pattern: table_name [AS] alias
    # Handle: "table AS alias", "table alias", or just "table"
    as_pattern = r'^(.+?)\s+AS\s+(\w+)$'
    as_match = re.match(as_pattern, table_part, re.IGNORECASE)
    if as_match:
        return as_match.group(1).strip(), as_match.group(2).strip()
    
    # Pattern: table_name alias (without AS)
    # Try to split by whitespace - last word might be alias
    words = table_part.split()
    if len(words) >= 2:
        # Check if last word looks like an alias (single word, no dots, no parentheses)
        last_word = words[-1]
        if re.match(r'^\w+$', last_word) and '.' not in last_word and '(' not in last_word:
            # Check if the part before is a valid table name (not a JOIN keyword)
            table_name = ' '.join(words[:-1])
            if not re.match(r'^(INNER|LEFT|RIGHT|FULL|CROSS)', table_name, re.IGNORECASE):
                return table_name, last_word
    
    # No alias found, use table name as alias
    # Remove any leading/trailing whitespace and return
    clean_table = table_part.strip()
    return clean_table, clean_table


def find_join_table_pair(join_condition: str, relations_map: Dict[str, str]) -> Optional[frozenset]:
    """
    Find which two tables (aliases) are involved in a join condition.
    
    Example: 'l.l_orderkey = o.o_orderkey' -> frozenset(['l', 'o'])
    """
    # Extract all aliases mentioned in the join condition
    # Pattern: alias.column
    aliases_found = set()
    
    for alias in relations_map.keys():
        # Look for alias.column pattern in the join condition
        pattern = r'\b' + re.escape(alias) + r'\.\w+'
        if re.search(pattern, join_condition, re.IGNORECASE):
            aliases_found.add(alias)
    
    if len(aliases_found) == 2:
        return frozenset(aliases_found)
    elif len(aliases_found) > 2:
        # Complex join condition involving more than 2 tables
        # For now, return None - this case needs special handling
        return None
    
    return None


def validate_query(parsed_query: Dict, available_tables: Set[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate that all tables in the query exist in the schema.
    
    Returns:
        (is_valid, error_message)
    """
    relations_map = parsed_query.get('relations_map', {})
    
    for alias, table_name in relations_map.items():
        if table_name not in available_tables:
            return False, f"Table '{table_name}' (alias '{alias}') not found in schema"
    
    return True, None

