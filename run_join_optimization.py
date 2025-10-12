import duckdb
import pandas as pd
import numpy as np
import time
import re
import sys
from io import StringIO
import sqlite3 # Import sqlite3 library

from SSS_QUBO import QUBO_formulation, QUBO_Split_Optimization_func

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
        connection.execute("CREATE TABLE nation (n_nationkey INTEGER, n_name VARCHAR, n_regionkey INTEGER);").from_df(nations_df).insert_into("nation")
        connection.execute("CREATE TABLE customer (c_custkey INTEGER, c_name VARCHAR, c_nationkey INTEGER);").from_df(customers_df).insert_into("customer")
        connection.execute("CREATE TABLE orders (o_orderkey INTEGER, o_custkey INTEGER, o_totalprice DECIMAL(10, 2));").from_df(orders_df).insert_into("orders")
        connection.execute("CREATE TABLE lineitem (l_orderkey INTEGER, l_extendedprice DECIMAL(10, 2));").from_df(lineitems_df).insert_into("lineitem")
    
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
    join_clauses = {('c', 'n'): 'c.c_nationkey = n.n_nationkey', ('o', 'c'): 'o.o_custkey = c.c_custkey', ('l', 'o'): 'l.l_orderkey = o.o_orderkey'}
    for combo in QUBO_formulation.relation_sublists(relations):
        if len(combo) < 2: continue
        from_clause, current_joins, temp_combo = f"FROM {relations_map[combo[0]]} AS {combo[0]}", {combo[0]}, combo[1:]
        while len(temp_combo) > 0:
            found_join = False
            for r_idx, r in enumerate(temp_combo):
                for joined_table in current_joins:
                    if (key1 := tuple(sorted((r, joined_table)))) in join_clauses:
                        from_clause += f" JOIN {relations_map[r]} AS {r} ON {join_clauses[key1]}"; current_joins.add(r); temp_combo.pop(r_idx); found_join = True; break
                if found_join: break
            if not found_join: from_clause += f", {relations_map[temp_combo.pop(0)]} AS r"
        try:
            explain_result = conn.execute(f"EXPLAIN SELECT COUNT(*) {from_clause};").fetchone()[1]
            costs.append(int(float(match.group(1))) if (match := re.search(r'\(Estimated Cardinality: (\d+\.?\d*)\)', explain_result)) else 1_000_000_000)
        except Exception: costs.append(1_000_000_000)
    print("--- Cost calculation complete. ---\n"); return costs

def parse_join_order_from_duckdb(explain_plan):
    try:
        conditions = re.findall(r'(\w)_\w+\s*=\s*(\w+)_\w+', explain_plan)
        if not conditions: return "Could not find join conditions based on column name prefixes."
        conditions.reverse(); order, seen = [], set()
        for t1, t2 in conditions:
            if not seen: order.extend([t1, t2]); seen.update([t1, t2])
            else:
                if t1 in seen and t2 not in seen: order.append(t2); seen.add(t2)
                elif t2 in seen and t1 not in seen: order.append(t1); seen.add(t1)
        return " -> ".join(order)
    except Exception as e: return f"An unexpected error occurred while parsing the join order: {e}"

# In run_join_optimization.py

# In run_join_optimization.py

def parse_join_order_from_sqlite(explain_plan_rows):
    """
    [V3 - Final] Parses the output of SQLite's EXPLAIN QUERY PLAN.
    This version is tailored to the observed output format.
    """
    try:
        order = []
        # This new regex looks for SCAN, SEARCH, or BLOOM FILTER ON,
        # and then captures the table alias that follows.
        regex = r'(?:SCAN|SEARCH|BLOOM FILTER ON)\s+(\w+)'
        
        for row in explain_plan_rows:
            # The plan detail is in the 4th column (index 3).
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
    if isinstance(tree, str): return f"{relations_map[tree]} AS {tree}", {tree}
    if len(tree) != 2: raise ValueError(f"Invalid tree structure for join: {tree}")
    left_sql, left_relations = build_from_clause_recursively(tree[0], relations_map, join_conditions)
    right_sql, right_relations = build_from_clause_recursively(tree[1], relations_map, join_conditions)
    on_clause = next((join_conditions[key] for r1 in left_relations for r2 in right_relations if (key := frozenset([r1, r2])) in join_conditions), None)
    if on_clause: join_type, join_sql = "JOIN", f"ON {on_clause}"
    else: join_type, join_sql = "CROSS JOIN", ""; print(f"Warning: No direct join condition between {left_relations} and {right_relations}. Using CROSS JOIN.")
    return f"({left_sql} {join_type} {right_sql} {join_sql})", left_relations.union(right_relations)

if __name__ == "__main__":
    duckdb_conn = duckdb.connect(database=':memory:')
    sqlite_conn = sqlite3.connect(':memory:') # In-memory SQLite database
    
    setup_database('duckdb', duckdb_conn, scale_factor=0.1)
    setup_database('sqlite', sqlite_conn, scale_factor=0.1)

    ############################################################################
    # BENCHMARK WITH 3 TABLES
    ############################################################################
    print("\n" + "="*50 + "\n### STARTING 3-TABLE BENCHMARK ###\n" + "="*50 + "\n")
    
    q_parts_3 = {"select": "SELECT c.c_name, SUM(l.l_extendedprice) AS total_revenue", "from": "FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey JOIN customer c ON o.o_custkey = c.c_custkey", "group_by": "GROUP BY c.c_name", "order_by": "ORDER BY total_revenue DESC", "limit": "LIMIT 10"}
    query_3 = f"{q_parts_3['select']} {q_parts_3['from']} {q_parts_3['group_by']} {q_parts_3['order_by']} {q_parts_3['limit']};"

    # 1. DuckDB Benchmark
    start_time = time.time(); duckdb_conn.execute(query_3).fetchall(); duration_duckdb_3 = time.time() - start_time
    explain_duckdb_3 = duckdb_conn.execute(f"EXPLAIN {query_3}").fetchone()[1]; duckdb_order_3 = parse_join_order_from_duckdb(explain_duckdb_3)
    
    # 2. SQLite Benchmark
    start_time = time.time(); sqlite_conn.execute(query_3).fetchall(); duration_sqlite_3 = time.time() - start_time
    explain_sqlite_3 = sqlite_conn.execute(f"EXPLAIN QUERY PLAN {query_3}").fetchall(); sqlite_order_3 = parse_join_order_from_sqlite(explain_sqlite_3)
    
    # 3. QUBO Benchmark
    relations_map_3, relations_3 = {'l': 'lineitem', 'o': 'orders', 'c': 'customer'}, ['l', 'o', 'c']
    join_conditions_3 = {frozenset(['l', 'o']): 'l.l_orderkey = o.o_orderkey', frozenset(['o', 'c']): 'o.o_custkey = c.c_custkey'}
    weights_3 = get_join_costs(duckdb_conn, relations_map_3)
    original_stdout, sys.stdout = sys.stdout, StringIO()
    try: qubo_tree_3, _, _ = QUBO_Split_Optimization_func("join_log_3").finding_opt_jo(relations_3, weights_3, 'simulated_annealing')
    finally: sys.stdout = original_stdout
    
    duration_qubo_3 = -1
    if qubo_tree_3:
        forced_from_3, _ = build_from_clause_recursively(qubo_tree_3, relations_map_3, join_conditions_3)
        forced_query_3 = f"{q_parts_3['select']} FROM {forced_from_3} {q_parts_3['group_by']} {q_parts_3['order_by']} {q_parts_3['limit']};"
        duckdb_conn.execute("SET disabled_optimizers='join_order';")
        start_time = time.time(); duckdb_conn.execute(forced_query_3).fetchall(); duration_qubo_3 = time.time() - start_time
        duckdb_conn.execute("SET disabled_optimizers='';")

    # Final Results for 3 Tables
    print("--- Final Results (3 Tables) ---")
    print(f"DuckDB Default Time: {duration_duckdb_3 * 1000:.2f} ms (Order: {duckdb_order_3})")
    print(f"SQLite Default Time: {duration_sqlite_3 * 1000:.2f} ms (Order: {sqlite_order_3})")
    if qubo_tree_3: print(f"QUBO Forced Time:  {duration_qubo_3 * 1000:.2f} ms (Order: {qubo_tree_3})")
    else: print("QUBO solver did not return a valid tree for 3 tables.")

    ############################################################################
    # BENCHMARK WITH 4 TABLES
    ############################################################################
    print("\n" + "="*50 + "\n### STARTING 4-TABLE BENCHMARK ###\n" + "="*50 + "\n")

    q_parts_4 = {"select": "SELECT n.n_name, c.c_name, SUM(l.l_extendedprice) AS total_revenue", "from": "FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey JOIN customer c ON o.o_custkey = c.c_custkey JOIN nation n ON c.c_nationkey = n.n_nationkey", "where": "WHERE n.n_name = 'NATION_5'", "group_by": "GROUP BY n.n_name, c.c_name", "order_by": "ORDER BY total_revenue DESC", "limit": "LIMIT 10"}
    query_4 = f"{q_parts_4['select']} {q_parts_4['from']} {q_parts_4['where']} {q_parts_4['group_by']} {q_parts_4['order_by']} {q_parts_4['limit']};"

    # 1. DuckDB Benchmark
    start_time = time.time(); duckdb_conn.execute(query_4).fetchall(); duration_duckdb_4 = time.time() - start_time
    explain_duckdb_4 = duckdb_conn.execute(f"EXPLAIN {query_4}").fetchone()[1]; duckdb_order_4 = parse_join_order_from_duckdb(explain_duckdb_4)

    # 2. SQLite Benchmark
    start_time = time.time(); sqlite_conn.execute(query_4).fetchall(); duration_sqlite_4 = time.time() - start_time
    explain_sqlite_4 = sqlite_conn.execute(f"EXPLAIN QUERY PLAN {query_4}").fetchall(); sqlite_order_4 = parse_join_order_from_sqlite(explain_sqlite_4)
    
    # 3. QUBO Benchmark
    relations_map_4, relations_4 = {'l': 'lineitem', 'o': 'orders', 'c': 'customer', 'n': 'nation'}, ['l', 'o', 'c', 'n']
    join_conditions_4 = {frozenset(['l', 'o']): 'l.l_orderkey = o.o_orderkey', frozenset(['o', 'c']): 'o.o_custkey = c.c_custkey', frozenset(['c', 'n']): 'c.c_nationkey = n.n_nationkey'}
    weights_4 = get_join_costs(duckdb_conn, relations_map_4)
    original_stdout, sys.stdout = sys.stdout, StringIO()
    try: qubo_tree_4, _, _ = QUBO_Split_Optimization_func("join_log_4").finding_opt_jo(relations_4, weights_4, 'simulated_annealing')
    finally: sys.stdout = original_stdout

    duration_qubo_4 = -1
    if qubo_tree_4:
        forced_from_4, _ = build_from_clause_recursively(qubo_tree_4, relations_map_4, join_conditions_4)
        forced_query_4 = f"{q_parts_4['select']} FROM {forced_from_4} {q_parts_4['where']} {q_parts_4['group_by']} {q_parts_4['order_by']} {q_parts_4['limit']};"
        duckdb_conn.execute("SET disabled_optimizers='join_order';")
        start_time = time.time(); duckdb_conn.execute(forced_query_4).fetchall(); duration_qubo_4 = time.time() - start_time
        duckdb_conn.execute("SET disabled_optimizers='';")
        
    # Final Results for 4 Tables
    print("\n--- Final Results (4 Tables) ---")
    print(f"DuckDB Default Time: {duration_duckdb_4 * 1000:.2f} ms (Order: {duckdb_order_4})")
    print(f"SQLite Default Time: {duration_sqlite_4 * 1000:.2f} ms (Order: {sqlite_order_4})")
    if qubo_tree_4: print(f"QUBO Forced Time:  {duration_qubo_4 * 1000:.2f} ms (Order: {qubo_tree_4})")
    else: print("QUBO solver did not return a valid tree for 4 tables.")
    
    duckdb_conn.close()
    sqlite_conn.close()