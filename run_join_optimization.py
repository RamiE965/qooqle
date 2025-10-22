import duckdb
import pandas as pd
import numpy as np
import time
import re
import sys
from io import StringIO
import sqlite3
import warnings
from scipy.sparse import SparseEfficiencyWarning

# Suppress scipy sparse matrix efficiency warnings
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

from SSS_QUBO import QUBO_formulation, QUBO_Split_Optimization_func, Helping_functions, SolverType

def setup_database(db_type, connection, scale_factor=0.1):
    """
    Sets up the database for either DuckDB or SQLite.
    """
    print(f"--- Setting up {db_type} database and generating data... ---")
    
    # Generate data with Pandas
    nations_df = pd.DataFrame({
        'n_nationkey': range(25), 
        'n_name': [f'NATION_{i}' for i in range(25)], 
        'n_regionkey': np.random.randint(0, 5, size=25)
    })
    num_customers = int(15000 * scale_factor)
    customers_df = pd.DataFrame({
        'c_custkey': range(num_customers), 
        'c_name': [f'Customer#{i}' for i in range(num_customers)], 
        'c_nationkey': np.random.randint(0, 25, size=num_customers)
    })
    num_orders = int(150000 * scale_factor)
    orders_df = pd.DataFrame({
        'o_orderkey': range(num_orders), 
        'o_custkey': np.random.randint(0, num_customers, size=num_orders), 
        'o_totalprice': np.random.uniform(100, 5000, size=num_orders)
    })
    num_lineitems = int(600000 * scale_factor)
    lineitems_df = pd.DataFrame({
        'l_orderkey': np.random.randint(0, num_orders, size=num_lineitems), 
        'l_extendedprice': np.random.uniform(50, 2000, size=num_lineitems)
    })

    if db_type == 'duckdb':
        # DuckDB table creation and loading
        connection.execute("CREATE TABLE nation (n_nationkey INTEGER, n_name VARCHAR, n_regionkey INTEGER);")
        connection.execute("CREATE TABLE customer (c_custkey INTEGER, c_name VARCHAR, c_nationkey INTEGER);")
        connection.execute("CREATE TABLE orders (o_orderkey INTEGER, o_custkey INTEGER, o_totalprice DECIMAL(10, 2));")
        connection.execute("CREATE TABLE lineitem (l_orderkey INTEGER, l_extendedprice DECIMAL(10, 2));")
        
        # Insert data
        connection.execute("INSERT INTO nation SELECT * FROM nations_df")
        connection.execute("INSERT INTO customer SELECT * FROM customers_df")
        connection.execute("INSERT INTO orders SELECT * FROM orders_df")
        connection.execute("INSERT INTO lineitem SELECT * FROM lineitems_df")
    
    elif db_type == 'sqlite':
        # SQLite table creation and loading using Pandas' to_sql
        nations_df.to_sql("nation", connection, if_exists="replace", index=False)
        customers_df.to_sql("customer", connection, if_exists="replace", index=False)
        orders_df.to_sql("orders", connection, if_exists="replace", index=False)
        lineitems_df.to_sql("lineitem", connection, if_exists="replace", index=False)
        
    print("--- Database setup complete. ---\n")

def get_join_costs(conn, relations_map):
    print("--- Calculating join costs using DuckDB's EXPLAIN... ---")
    relations, costs = list(relations_map.keys()), []
    join_clauses = {
        ('c', 'n'): 'c.c_nationkey = n.n_nationkey', 
        ('o', 'c'): 'o.o_custkey = c.c_custkey', 
        ('l', 'o'): 'l.l_orderkey = o.o_orderkey'
    }
    
    for combo in QUBO_formulation.relation_sublists(relations):
        if len(combo) < 2: 
            continue
            
        from_clause, current_joins, temp_combo = f"FROM {relations_map[combo[0]]} AS {combo[0]}", {combo[0]}, combo[1:]
        
        while len(temp_combo) > 0:
            found_join = False
            for r_idx, r in enumerate(temp_combo):
                for joined_table in current_joins:
                    key1 = tuple(sorted((r, joined_table)))
                    if key1 in join_clauses:
                        from_clause += f" JOIN {relations_map[r]} AS {r} ON {join_clauses[key1]}"
                        current_joins.add(r)
                        temp_combo.pop(r_idx)
                        found_join = True
                        break
                if found_join: 
                    break
            if not found_join: 
                from_clause += f", {relations_map[temp_combo.pop(0)]} AS r"
                
        try:
            explain_result = conn.execute(f"EXPLAIN SELECT COUNT(*) {from_clause};").fetchone()[1]
            match = re.search(r'\(Estimated Cardinality: (\d+\.?\d*)\)', explain_result)
            if match:
                costs.append(int(float(match.group(1))))
            else:
                costs.append(1_000_000_000)
        except Exception as e: 
            print(f"Warning: Could not get cost for {combo}: {e}")
            costs.append(1_000_000_000)
            
    print(f"Calculated {len(costs)} join costs")
    print("--- Cost calculation complete. ---\n")
    return costs

def parse_join_order_from_duckdb(explain_plan):
    try:
        conditions = re.findall(r'(\w)_\w+\s*=\s*(\w+)_\w+', explain_plan)
        if not conditions: 
            return "Could not find join conditions based on column name prefixes."
        conditions.reverse()
        order, seen = [], set()
        for t1, t2 in conditions:
            if not seen: 
                order.extend([t1, t2])
                seen.update([t1, t2])
            else:
                if t1 in seen and t2 not in seen: 
                    order.append(t2)
                    seen.add(t2)
                elif t2 in seen and t1 not in seen: 
                    order.append(t1)
                    seen.add(t1)
        return " -> ".join(order)
    except Exception as e: 
        return f"An unexpected error occurred while parsing the join order: {e}"

def parse_join_order_from_sqlite(explain_plan_rows):
    """
    [V3 - Final] Parses the output of SQLite's EXPLAIN QUERY PLAN.
    This version is tailored to the observed output format.
    """
    try:
        order = []
        regex = r'(?:SCAN|SEARCH|BLOOM FILTER ON)\s+(\w+)'
        
        for row in explain_plan_rows:
            detail = row[3]
            match = re.search(regex, detail)
            if match:
                alias = match.group(1)
                if alias not in order:
                    order.append(alias)
                    
        return " -> ".join(order) if order else "Could not determine join order."
    except Exception as e:
        return f"An error occurred while parsing SQLite plan: {e}"

def build_from_clause_recursively(tree, relations_map, join_conditions):
    if isinstance(tree, str): 
        return f"{relations_map[tree]} AS {tree}", {tree}
    if len(tree) != 2: 
        raise ValueError(f"Invalid tree structure for join: {tree}")
        
    left_sql, left_relations = build_from_clause_recursively(tree[0], relations_map, join_conditions)
    right_sql, right_relations = build_from_clause_recursively(tree[1], relations_map, join_conditions)
    
    on_clause = None
    for r1 in left_relations:
        for r2 in right_relations:
            key = frozenset([r1, r2])
            if key in join_conditions:
                on_clause = join_conditions[key]
                break
        if on_clause:
            break
            
    if on_clause: 
        join_type, join_sql = "JOIN", f"ON {on_clause}"
    else: 
        join_type, join_sql = "CROSS JOIN", ""
        print(f"Warning: No direct join condition between {left_relations} and {right_relations}. Using CROSS JOIN.")
        
    return f"({left_sql} {join_type} {right_sql} {join_sql})", left_relations.union(right_relations)

def run_benchmark(num_tables, relations_map, relations, join_conditions, query_parts, duckdb_conn, sqlite_conn):
    """
    Run benchmark for a specific number of tables.
    """
    print(f"\n" + "="*50 + f"\n### STARTING {num_tables}-TABLE BENCHMARK ###\n" + "="*50 + "\n")
    
    # Build query
    query = f"{query_parts['select']} {query_parts['from']}"
    if 'where' in query_parts:
        query += f" {query_parts['where']}"
    query += f" {query_parts['group_by']} {query_parts['order_by']} {query_parts['limit']};"

    # 1. DuckDB Benchmark
    print("Running DuckDB default optimization...")
    start_time = time.time()
    duckdb_conn.execute(query).fetchall()
    duration_duckdb = time.time() - start_time
    explain_duckdb = duckdb_conn.execute(f"EXPLAIN {query}").fetchone()[1]
    duckdb_order = parse_join_order_from_duckdb(explain_duckdb)
    
    # 2. SQLite Benchmark
    print("Running SQLite default optimization...")
    start_time = time.time()
    sqlite_conn.execute(query).fetchall()
    duration_sqlite = time.time() - start_time
    explain_sqlite = sqlite_conn.execute(f"EXPLAIN QUERY PLAN {query}").fetchall()
    sqlite_order = parse_join_order_from_sqlite(explain_sqlite)
    
    # 3. QUBO Benchmark with QAOA
    print("Running QUBO optimization with QAOA...")
    weights = get_join_costs(duckdb_conn, relations_map)
    
    # Capture QUBO output to avoid cluttering console
    original_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        optimizer = QUBO_Split_Optimization_func(f"join_log_{num_tables}")
        qubo_tree, selected_joins, error_msg = optimizer.finding_opt_jo(relations, weights, SolverType.QAOA)
    except Exception as e:
        print(f"QUBO optimization failed: {e}")
        qubo_tree = None
        error_msg = str(e)
    finally:
        sys.stdout = original_stdout
    
    duration_qubo = -1
    qubo_order_str = "No valid tree"
    
    if qubo_tree and not error_msg:
        qubo_order_str = str(qubo_tree)
        try:
            forced_from, _ = build_from_clause_recursively(qubo_tree, relations_map, join_conditions)
            forced_query = f"{query_parts['select']} FROM {forced_from}"
            if 'where' in query_parts:
                forced_query += f" {query_parts['where']}"
            forced_query += f" {query_parts['group_by']} {query_parts['order_by']} {query_parts['limit']};"
            
            print(f"Forcing QUBO join order: {forced_query}")
            duckdb_conn.execute("SET disabled_optimizers='join_order';")
            start_time = time.time()
            duckdb_conn.execute(forced_query).fetchall()
            duration_qubo = time.time() - start_time
            duckdb_conn.execute("SET disabled_optimizers='';")
        except Exception as e:
            print(f"Error executing forced QUBO query: {e}")
            duration_qubo = -1
    else:
        print(f"QUBO optimization failed: {error_msg}")
    
    # Print results
    print(f"\n--- Final Results ({num_tables} Tables) ---")
    print(f"DuckDB Default Time: {duration_duckdb * 1000:.2f} ms (Order: {duckdb_order})")
    print(f"SQLite Default Time: {duration_sqlite * 1000:.2f} ms (Order: {sqlite_order})")
    if duration_qubo >= 0:
        print(f"QUBO QAOA Time:     {duration_qubo * 1000:.2f} ms (Order: {qubo_order_str})")
    else:
        print(f"QUBO QAOA Time:     Failed to execute")
    
    return duration_duckdb, duration_sqlite, duration_qubo

if __name__ == "__main__":
    # Initialize databases
    duckdb_conn = duckdb.connect(database=':memory:')
    sqlite_conn = sqlite3.connect(':memory:')
    
    setup_database('duckdb', duckdb_conn, scale_factor=0.1)
    setup_database('sqlite', sqlite_conn, scale_factor=0.1)

    # 3-Table Benchmark
    q_parts_3 = {
        "select": "SELECT c.c_name, SUM(l.l_extendedprice) AS total_revenue",
        "from": "FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey JOIN customer c ON o.o_custkey = c.c_custkey",
        "group_by": "GROUP BY c.c_name", 
        "order_by": "ORDER BY total_revenue DESC", 
        "limit": "LIMIT 10"
    }
    
    relations_map_3 = {'l': 'lineitem', 'o': 'orders', 'c': 'customer'}
    relations_3 = ['l', 'o', 'c']
    join_conditions_3 = {
        frozenset(['l', 'o']): 'l.l_orderkey = o.o_orderkey', 
        frozenset(['o', 'c']): 'o.o_custkey = c.c_custkey'
    }
    
    run_benchmark(3, relations_map_3, relations_3, join_conditions_3, q_parts_3, duckdb_conn, sqlite_conn)

    # 4-Table Benchmark  
    q_parts_4 = {
        "select": "SELECT n.n_name, c.c_name, SUM(l.l_extendedprice) AS total_revenue",
        "from": "FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey JOIN customer c ON o.o_custkey = c.c_custkey JOIN nation n ON c.c_nationkey = n.n_nationkey",
        "where": "WHERE n.n_name = 'NATION_5'",
        "group_by": "GROUP BY n.n_name, c.c_name",
        "order_by": "ORDER BY total_revenue DESC", 
        "limit": "LIMIT 10"
    }
    
    relations_map_4 = {'l': 'lineitem', 'o': 'orders', 'c': 'customer', 'n': 'nation'}
    relations_4 = ['l', 'o', 'c', 'n']
    join_conditions_4 = {
        frozenset(['l', 'o']): 'l.l_orderkey = o.o_orderkey', 
        frozenset(['o', 'c']): 'o.o_custkey = c.c_custkey',
        frozenset(['c', 'n']): 'c.c_nationkey = n.n_nationkey'
    }
    
    run_benchmark(4, relations_map_4, relations_4, join_conditions_4, q_parts_4, duckdb_conn, sqlite_conn)
    
    duckdb_conn.close()
    sqlite_conn.close()