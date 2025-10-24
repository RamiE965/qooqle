import duckdb
import pandas as pd
import numpy as np
import time
import re
import sys
from io import StringIO
import sqlite3
import warnings
import argparse
import random
from scipy.sparse import SparseEfficiencyWarning

# Suppress scipy sparse matrix efficiency warnings
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

from SSS_QUBO import QUBO_formulation, QUBO_Split_Optimization_func, Helping_functions

# Conditional import for plotting
try:
    import matplotlib.pyplot as plt
    PLOTTING_ENABLED = True
except ImportError:
    PLOTTING_ENABLED = False
    print("\nWarning: matplotlib not found. Plotting will be disabled. To enable, run: pip install matplotlib")

def parse_cardinality(s):
    """Parses cardinality strings like '100', '10.5k', '2.1M', '1e+6', or '25,143'."""
    s = str(s).lower().strip().replace(',', '')  # Remove commas
    if not s: return 0
    
    if 'e+' in s:
        return int(float(s))
        
    s_num = s
    multiplier = 1
    if s.endswith('k'):
        multiplier = 1000
        s_num = s[:-1]
    elif s.endswith('m'):
        multiplier = 1000000
        s_num = s[:-1]
    elif s.endswith('b') or s.endswith('g'):
        multiplier = 1000000000
        s_num = s[:-1]
        
    try:
        return int(float(s_num) * multiplier)
    except ValueError:
        print(f"Warning: Could not parse cardinality string '{s}'")
        return 0

def setup_database(db_type, connection, scale_factor=0.1, seed=None, scenario='default'):
    """
    Sets up the database for either DuckDB or SQLite.
    If seed is provided, uses it for reproducible random data generation.
    Scenario determines table size ratios and data patterns.
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"--- Setting up {db_type} database (scenario={scenario}, seed={seed})... ---")
    
    # Different scenarios create different table size ratios and selectivities
    # This makes each run test QAOA on fundamentally different join optimization problems
    scenarios = {
        'default': {
            'nations': 25,
            'customers_mult': 1.0,    # 1,500 customers
            'orders_mult': 1.0,       # 15,000 orders
            'lineitems_mult': 1.0,    # 60,000 lineitems
            'order_cust_ratio': 1.0,  # Normal distribution
            'lineitem_order_ratio': 1.0
        },
        'many_customers': {
            'nations': 25,
            'customers_mult': 3.0,    # 4,500 customers (3x more)
            'orders_mult': 1.0,       # 15,000 orders (same)
            'lineitems_mult': 0.8,    # 48,000 lineitems (fewer)
            'order_cust_ratio': 0.5,  # Fewer orders per customer
            'lineitem_order_ratio': 0.8
        },
        'large_orders': {
            'nations': 25,
            'customers_mult': 0.6,    # 900 customers (fewer)
            'orders_mult': 2.0,       # 30,000 orders (2x more)
            'lineitems_mult': 1.5,    # 90,000 lineitems (more)
            'order_cust_ratio': 2.0,  # More orders per customer
            'lineitem_order_ratio': 1.2
        },
        'heavy_lineitems': {
            'nations': 25,
            'customers_mult': 0.8,    # 1,200 customers
            'orders_mult': 0.7,       # 10,500 orders (fewer)
            'lineitems_mult': 2.5,    # 150,000 lineitems (huge!)
            'order_cust_ratio': 1.2,
            'lineitem_order_ratio': 3.0  # Many items per order
        },
        'balanced_small': {
            'nations': 25,
            'customers_mult': 0.5,    # 750 customers
            'orders_mult': 0.5,       # 7,500 orders
            'lineitems_mult': 0.5,    # 30,000 lineitems
            'order_cust_ratio': 1.0,
            'lineitem_order_ratio': 1.0
        }
    }
    
    config = scenarios.get(scenario, scenarios['default'])
    
    # Generate data with varying sizes based on scenario
    num_regions = 5
    regions_df = pd.DataFrame({
        'r_regionkey': range(num_regions),
        'r_name': [f'REGION_{i}' for i in range(num_regions)]
    })
    
    nations_df = pd.DataFrame({
        'n_nationkey': range(config['nations']), 
        'n_name': [f'NATION_{i}' for i in range(config['nations'])], 
        'n_regionkey': np.random.randint(0, num_regions, size=config['nations'])
    })
    
    num_suppliers = int(10000 * scale_factor * config.get('suppliers_mult', 1.0))
    suppliers_df = pd.DataFrame({
        's_suppkey': range(num_suppliers),
        's_name': [f'Supplier#{i}' for i in range(num_suppliers)],
        's_nationkey': np.random.randint(0, config['nations'], size=num_suppliers)
    })
    
    num_customers = int(15000 * scale_factor * config['customers_mult'])
    customers_df = pd.DataFrame({
        'c_custkey': range(num_customers), 
        'c_name': [f'Customer#{i}' for i in range(num_customers)], 
        'c_nationkey': np.random.randint(0, config['nations'], size=num_customers)
    })
    
    num_orders = int(150000 * scale_factor * config['orders_mult'])
    orders_df = pd.DataFrame({
        'o_orderkey': range(num_orders), 
        'o_custkey': np.random.randint(0, num_customers, size=num_orders), 
        'o_totalprice': np.random.uniform(100, 5000, size=num_orders)
    })
    
    num_lineitems = int(600000 * scale_factor * config['lineitems_mult'])
    lineitems_df = pd.DataFrame({
        'l_orderkey': np.random.randint(0, num_orders, size=num_lineitems),
        'l_suppkey': np.random.randint(0, num_suppliers, size=num_lineitems),
        'l_extendedprice': np.random.uniform(50, 2000, size=num_lineitems)
    })
    
    print(f"   Tables: {num_customers} customers, {num_orders} orders, {num_lineitems} lineitems")

    if db_type == 'duckdb':
        connection.execute("CREATE TABLE region (r_regionkey INTEGER, r_name VARCHAR);")
        connection.execute("CREATE TABLE nation (n_nationkey INTEGER, n_name VARCHAR, n_regionkey INTEGER);")
        connection.execute("CREATE TABLE supplier (s_suppkey INTEGER, s_name VARCHAR, s_nationkey INTEGER);")
        connection.execute("CREATE TABLE customer (c_custkey INTEGER, c_name VARCHAR, c_nationkey INTEGER);")
        connection.execute("CREATE TABLE orders (o_orderkey INTEGER, o_custkey INTEGER, o_totalprice DECIMAL(10, 2));")
        connection.execute("CREATE TABLE lineitem (l_orderkey INTEGER, l_suppkey INTEGER, l_extendedprice DECIMAL(10, 2));")
        # Use execute with SELECT * FROM df_name, which is the correct way to load from a local DataFrame
        connection.execute("INSERT INTO region SELECT * FROM regions_df")
        connection.execute("INSERT INTO nation SELECT * FROM nations_df")
        connection.execute("INSERT INTO supplier SELECT * FROM suppliers_df")
        connection.execute("INSERT INTO customer SELECT * FROM customers_df")
        connection.execute("INSERT INTO orders SELECT * FROM orders_df")
        connection.execute("INSERT INTO lineitem SELECT * FROM lineitems_df")
    
    elif db_type == 'sqlite':
        regions_df.to_sql("region", connection, if_exists="replace", index=False)
        nations_df.to_sql("nation", connection, if_exists="replace", index=False)
        suppliers_df.to_sql("supplier", connection, if_exists="replace", index=False)
        customers_df.to_sql("customer", connection, if_exists="replace", index=False)
        orders_df.to_sql("orders", connection, if_exists="replace", index=False)
        lineitems_df.to_sql("lineitem", connection, if_exists="replace", index=False)
        
    print("--- Database setup complete. ---\n")


def get_join_costs_random(relations, seed=None):
    """
    Generate RANDOM join costs (like research papers do).
    This creates a true NP-hard optimization problem where the optimal solution
    is not obvious from the problem structure.
    
    WHY RANDOM WEIGHTS:
    - Tests QAOA's pure optimization ability on hard combinatorial problems
    - Different runs produce different optimal solutions (more interesting!)
    - Matches how quantum computing papers test optimization algorithms
    - Shows when QAOA finds suboptimal solutions (quantum noise/approximation)
    
    Cost model: Random integers between 1-100 for each join subset
    """
    if seed is not None:
        random.seed(seed)
    
    print(f"--- Generating RANDOM join costs (seed={seed}) ---")
    
    costs_map = {}
    weights = []
    sublists = QUBO_formulation.relation_sublists(relations)
    
    print("Random weights assigned:")
    for combo in sublists:
        if len(combo) < 2:
            continue
        
        # Random cost between 1-100 (like the research papers)
        cost = random.randint(1, 100)
        costs_map[frozenset(combo)] = cost
        weights.append(cost)
        print(f"  {combo}: {cost}")
    
    print(f"Generated {len(weights)} random join costs (1-100)")
    print("--- Cost generation complete. ---\n")
    return costs_map, weights


def get_join_costs_simple(conn, relations_map, relations, join_conditions):
    """
    Calculate join costs using cumulative cardinality model with Dynamic Programming.
    This ensures QAOA optimizes for the SAME objective we use for evaluation.
    
    For each subset S, we calculate the optimal cumulative cost using DP:
    - cost(single table) = 0
    - cost(S) = min over all partitions A,B: cost(A) + cost(B) + |A JOIN B|
    
    This matches our evaluate_tree_cost function and aligns QAOA's optimization
    with our evaluation metric.
    """
    print("--- Calculating join costs using cumulative cardinality (DP) ---")
    
    # Get table cardinalities
    table_sizes = {}
    for rel, table_name in relations_map.items():
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        table_sizes[rel] = count
        print(f"  {rel} ({table_name}): {count} rows")
    
    # Helper: estimate join result size
    def estimate_join_size(left_rels, right_rels, left_size, right_size):
        """
        Estimate the result size of joining two relation sets.
        Uses a more realistic selectivity model that accounts for actual join behavior.
        """
        # Check if there's a direct join condition
        has_join = False
        for l in left_rels:
            for r in right_rels:
                if frozenset([l, r]) in join_conditions:
                    has_join = True
                    break
            if has_join:
                break
        
        if has_join:
            # FK-like join with more realistic selectivity
            # Instead of perfect selectivity (1/max), use a model that considers:
            # - If both are small (<1000): tight FK relationship → max(left, right)
            # - Otherwise: looser relationship → geometric mean with dampening
            
            if max(left_size, right_size) < 1000:
                # Small tables: tight FK relationship
                result = max(left_size, right_size)
            else:
                # Larger tables: use geometric mean as estimate
                # This is more realistic than min() but less than full product
                import math
                result = int(math.sqrt(left_size * right_size))
            
            return max(result, 1) if left_size > 0 and right_size > 0 else result
        else:
            # Cross product
            return left_size * right_size
    
    # Dynamic programming to calculate optimal cumulative cost for each subset
    dp = {}  # frozenset -> (optimal_cost, result_size)
    
    # Base case: single tables
    for rel in relations:
        dp[frozenset([rel])] = (0, table_sizes[rel])
    
    print("\nCalculating optimal cumulative costs with DP:")
    
    # Build up subsets by size
    sublists = QUBO_formulation.relation_sublists(relations)
    for combo in sublists:
        if len(combo) < 2:
            continue
        
        combo_set = frozenset(combo)
        best_cost = float('inf')
        best_result_size = 0
        
        # Try all ways to partition this subset into two parts
        from itertools import combinations
        for k in range(1, len(combo) // 2 + 1):
            for left_tuple in combinations(combo, k):
                left_set = frozenset(left_tuple)
                right_set = combo_set - left_set
                
                if len(right_set) == 0:
                    continue
                
                # Get optimal costs and sizes for both parts
                left_cost, left_size = dp[left_set]
                right_cost, right_size = dp[right_set]
                
                # Estimate the join result size
                join_result_size = estimate_join_size(left_set, right_set, left_size, right_size)
                
                # Total cumulative cost
                total_cost = left_cost + right_cost + join_result_size
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_result_size = join_result_size
        
        dp[combo_set] = (best_cost, best_result_size)
        print(f"  {list(combo)}: optimal_cost={best_cost}, result_size={best_result_size}")
    
    # Extract weights for QAOA (use optimal cumulative costs)
    costs_map = {}
    weights = []
    
    for combo in sublists:
        if len(combo) < 2:
            continue
        combo_set = frozenset(combo)
        optimal_cost, _ = dp[combo_set]
        costs_map[combo_set] = optimal_cost
        weights.append(optimal_cost)
    
    print(f"\nCalculated {len(weights)} join costs using DP cumulative cardinality model")
    print("--- Cost calculation complete. ---\n")
    return costs_map, weights, table_sizes



def calculate_tree_cost(tree, costs_map, table_sizes, join_conditions, debug=False, indent=0):
    """
    Recursively calculates the total cost of a join tree using cardinality estimation with selectivity.
    Returns (total_cost, result_size) as a tuple.
    
    Cost accumulates: cost = cost_left + cost_right + result_size_of_this_join
    Result size uses selectivity estimation for realistic join cardinality.
    """
    prefix = "  " * indent
    
    if isinstance(tree, str):
        # Leaf node: cost = 0 (no work yet), result_size = table size
        size = table_sizes.get(tree, 0)
        if debug:
            print(f"{prefix}Leaf '{tree}': cost=0, size={size:,}")
        return 0, size
    
    left_subtree, right_subtree = tree[0], tree[1]
    
    # Recursively get costs and result sizes of subtrees
    if debug:
        print(f"{prefix}Join {tree}:")
        print(f"{prefix}  Left subtree:")
    left_cost, left_result_size = calculate_tree_cost(left_subtree, costs_map, table_sizes, join_conditions, debug, indent+2)
    if debug:
        print(f"{prefix}  Right subtree:")
    right_cost, right_result_size = calculate_tree_cost(right_subtree, costs_map, table_sizes, join_conditions, debug, indent+2)
    
    def get_relations(subtree):
        if isinstance(subtree, str): return {subtree}
        return get_relations(subtree[0]).union(get_relations(subtree[1]))
    
    left_rels = get_relations(left_subtree)
    right_rels = get_relations(right_subtree)
    all_rels = left_rels.union(right_rels)
    
    # Check if there's a join condition between left and right
    has_join_condition = False
    for l in left_rels:
        for r in right_rels:
            if frozenset([l, r]) in join_conditions:
                has_join_condition = True
                break
        if has_join_condition:
            break
    
    if has_join_condition:
        # With join condition: use realistic selectivity estimation
        # - Small tables (<1000): tight FK → max(left, right)
        # - Larger tables: geometric mean (more realistic than min, less than product)
        
        if max(left_result_size, right_result_size) < 1000:
            # Small tables: tight FK relationship
            result_size = max(left_result_size, right_result_size)
        else:
            # Larger tables: geometric mean
            import math
            result_size = int(math.sqrt(left_result_size * right_result_size))
        
        # Ensure result is at least 1 if both inputs are non-empty
        if left_result_size > 0 and right_result_size > 0 and result_size == 0:
            result_size = 1
    else:
        # Cross product: full Cartesian product
        result_size = left_result_size * right_result_size
    
    # Total cost = cost of building left + cost of building right + work to join them
    # Work to join = result_size (proportional to rows processed)
    total_cost = left_cost + right_cost + result_size
    
    if debug:
        print(f"{prefix}  Join cost: {left_cost:,} + {right_cost:,} + {result_size:,} = {total_cost:,}")
        print(f"{prefix}  Result size: {result_size:,}")
    
    return total_cost, result_size

def linear_order_to_left_deep_tree(tables):
    """
    Convert a linear join order like ['n', 'c', 'o', 'l'] into a left-deep tree: [[[n, c], o], l]
    This represents the join structure: (((n JOIN c) JOIN o) JOIN l)
    """
    if len(tables) == 0:
        return None
    if len(tables) == 1:
        return tables[0]
    
    # Start with first two tables
    tree = [tables[0], tables[1]]
    
    # Add remaining tables one by one to the left side
    for table in tables[2:]:
        tree = [tree, table]
    
    return tree

def parse_join_order_from_duckdb(explain_plan):
    try:
        conditions = re.findall(r'(\w)_\w+\s*=\s*(\w+)_\w+', explain_plan)
        if not conditions: return "Could not find join conditions based on column name prefixes."
        conditions.reverse()
        order, seen = [], set()
        for t1, t2 in conditions:
            if not seen: order.extend([t1, t2]); seen.update([t1, t2])
            elif t1 in seen and t2 not in seen: order.append(t2); seen.add(t2)
            elif t2 in seen and t1 not in seen: order.append(t1); seen.add(t1)
        return " -> ".join(order)
    except Exception as e: return f"An unexpected error occurred: {e}"

def parse_join_order_from_sqlite(explain_plan_rows):
    try:
        order = []
        for row in explain_plan_rows:
            match = re.search(r'(?:SCAN|SEARCH|BLOOM FILTER ON)\s+(\w+)', row[3])
            if match and match.group(1) not in order:
                order.append(match.group(1))
        return " -> ".join(order) if order else "Could not determine join order."
    except Exception as e: return f"An error occurred: {e}"

def build_from_clause_flat(tree, relations_map, join_conditions, reverse_qaoa=False, experimental_lr_hybrid=False):
    """
    Build a FLAT FROM clause from join tree without nested parentheses.
    This allows DuckDB to use proper cardinality estimates while still forcing join order.
    
    WHY FLAT INSTEAD OF NESTED:
    - Nested parentheses: FROM ((a JOIN b) JOIN c) causes DuckDB to make naive estimates
    - Flat joins: FROM a JOIN b JOIN c allows proper cardinality estimation and filter pushdown
    - DuckDB will still execute in the specified order when join_order optimizer is disabled
    
    Parameters:
    - experimental_lr_hybrid: If True, read join structure right-to-left BUT table pairs left-to-right
      Example: Tree ['l', ['o', 'c']] normally gives: c->o->l
               With hybrid: o->c->l (reversed pairs)
    """
    # Extract join order from tree
    def extract_join_order(t, reverse=False, hybrid=False):
        """Extract the join order as a list of table aliases."""
        if isinstance(t, str):
            return [t]
        
        if reverse:
            if hybrid:
                # EXPERIMENTAL: Process structure right-to-left, but reverse ONLY immediate pairs
                # Tree ['l', ['o', 'c']] becomes: ['o', 'c'] then 'l'
                # Then reverse ONLY if it's a direct pair: [c, o] → [o, c]
                # Final order: o, c, l (instead of c, o, l)
                left_order = extract_join_order(t[0], reverse, hybrid)
                right_order = extract_join_order(t[1], reverse, hybrid)
                
                # Only reverse if right_order is EXACTLY 2 elements (a direct pair)
                # This prevents breaking larger join graphs
                if len(right_order) == 2:
                    return right_order[::-1] + left_order
                else:
                    # For larger subtrees, just concatenate normally (right then left)
                    return right_order + left_order
            else:
                # Original: right-to-left, so process right first, then left
                return extract_join_order(t[1], reverse, hybrid) + extract_join_order(t[0], reverse, hybrid)
        else:
            # Normal: left-to-right
            return extract_join_order(t[0], reverse, hybrid) + extract_join_order(t[1], reverse, hybrid)
    
    # Get the join order
    join_order = extract_join_order(tree, reverse_qaoa, experimental_lr_hybrid)
    
    print(f"[DEBUG] Extracted join order: {join_order} (reverse_qaoa={reverse_qaoa}, hybrid={experimental_lr_hybrid})")
    
    if len(join_order) == 0:
        raise ValueError("Empty join order")
    
    # Build flat FROM clause: start with first table
    first_table = join_order[0]
    from_clause = f"{relations_map[first_table]} AS {first_table}"
    joined_tables = {first_table}
    all_relations = {first_table}
    
    # Add remaining tables in order
    for table in join_order[1:]:
        # Find join condition with any already-joined table
        on_clause = None
        for joined in joined_tables:
            key = frozenset([table, joined])
            if key in join_conditions:
                on_clause = join_conditions[key]
                break
        
        if on_clause:
            from_clause += f" JOIN {relations_map[table]} AS {table} ON {on_clause}"
        else:
            # No direct condition found - cross join
            from_clause += f" CROSS JOIN {relations_map[table]} AS {table}"
            print(f"Warning: No direct join condition found for {table}")
        
        joined_tables.add(table)
        all_relations.add(table)
    
    return from_clause, all_relations


def build_from_clause_recursively(tree, relations_map, join_conditions, reverse_qaoa=False):
    """
    Build FROM clause from join tree using nested parentheses.
    
    For QAOA trees (reverse_qaoa=True):
    - Within each immediate pair [A, B] where both are leaves: read left-to-right → "A JOIN B"  
    - For larger structures: right-to-left traversal → process right subtree first
    
    Example: ['l', ['o', 'c']] becomes:
    - ['o', 'c'] is an immediate pair → "o JOIN c" (left-to-right)
    - Top level has right subtree ['o', 'c'] → execute it first: "((o JOIN c) JOIN l)"
    """
    if isinstance(tree, str): 
        return f"{relations_map[tree]} AS {tree}", {tree}
    if len(tree) != 2: 
        raise ValueError(f"Invalid tree structure: {tree}")
    
    # ALWAYS read pairs left-to-right: tree[0] is left, tree[1] is right
    left_tree, right_tree = tree[0], tree[1]
        
    # Recursively build both subtrees
    left_sql, left_relations = build_from_clause_recursively(left_tree, relations_map, join_conditions, reverse_qaoa)
    right_sql, right_relations = build_from_clause_recursively(right_tree, relations_map, join_conditions, reverse_qaoa)
    
    # Find join condition between the two relation sets
    on_clause = None
    for r1 in left_relations:
        for r2 in right_relations:
            key = frozenset([r1, r2])
            if key in join_conditions:
                on_clause = join_conditions[key]
                break
        if on_clause:
            break
    
    join_type, join_sql = ("JOIN", f"ON {on_clause}") if on_clause else ("CROSS JOIN", "")
    if not on_clause: 
        print(f"Warning: No direct join condition between {left_relations} and {right_relations}. Using CROSS JOIN.")
    
    # Check if this is an immediate pair (both children are leaves)
    both_leaves = isinstance(left_tree, str) and isinstance(right_tree, str)
    
    # Build the SQL JOIN expression
    if reverse_qaoa and not both_leaves:
        # For QAOA with nested structures: put right subtree first (executes first)
        # This implements "right-to-left" traversal
        return f"({right_sql} {join_type} {left_sql} {join_sql})", left_relations.union(right_relations)
    else:
        # For immediate pairs or normal mode: always left-to-right
        # This implements "left-to-right within pairs"
        return f"({left_sql} {join_type} {right_sql} {join_sql})", left_relations.union(right_relations)

def run_benchmark(num_tables, relations_map, relations, join_conditions, query_parts, duckdb_conn, sqlite_conn, weight_method='cardinality', run_seed=None):
    print(f"\n" + "="*50 + f"\n### STARTING {num_tables}-TABLE BENCHMARK ###\n" + "="*50 + "\n")
    
    query = f"{query_parts['select']} {query_parts['from']}"
    if 'where' in query_parts: query += f" {query_parts['where']}"
    query += f" {query_parts['group_by']} {query_parts['order_by']} {query_parts['limit']};"

    # 1. DuckDB Benchmark (separating planning and execution)
    print("--- Running DuckDB Benchmark ---")
    explain_duckdb = duckdb_conn.execute(f"EXPLAIN {query}").fetchone()[1]
    duckdb_order = parse_join_order_from_duckdb(explain_duckdb)
    duckdb_cost = -2  # Sentinel: calculate later using cardinality model

    print("Executing DuckDB with its optimized plan (excluding planning time)...")
    start_time = time.time()
    duckdb_conn.execute(query).fetchall()
    duration_duckdb = time.time() - start_time
    
    # 2. SQLite Benchmark
    print("\n--- Running SQLite Benchmark ---")
    start_time = time.time()
    sqlite_conn.execute(query).fetchall()
    duration_sqlite = time.time() - start_time
    explain_sqlite = sqlite_conn.execute(f"EXPLAIN QUERY PLAN {query}").fetchall()
    sqlite_order = parse_join_order_from_sqlite(explain_sqlite)
    sqlite_cost = -2  # Sentinel: calculate later using cardinality model
    
    # 3. QUBO Benchmark with QAOA
    print("\n--- Running QUBO Benchmark (QAOA) ---")
    
    # Generate join costs for QAOA optimization
    if weight_method == 'random':
        costs_map, weights = get_join_costs_random(relations, seed=run_seed)
        table_sizes = {rel: duckdb_conn.execute(f"SELECT COUNT(*) FROM {relations_map[rel]}").fetchone()[0] 
                      for rel in relations_map}
    else:  # cardinality (default)
        costs_map, weights, table_sizes = get_join_costs_simple(duckdb_conn, relations_map, relations, join_conditions)
    
    # Now calculate DuckDB's cost using the cardinality model
    if duckdb_cost == -2:
        try:
            print(f"\n=== DuckDB Cost Calculation (Cardinality Model) ===")
            duckdb_tables = duckdb_order.split(" -> ")
            if len(duckdb_tables) >= 2 and all(t in relations_map for t in duckdb_tables):
                duckdb_tree = linear_order_to_left_deep_tree(duckdb_tables)
                print(f"DuckDB join order: {duckdb_order}")
                print(f"DuckDB tree: {duckdb_tree}")
                print(f"\n--- Detailed DuckDB Cost Calculation ---")
                duckdb_cost, _ = calculate_tree_cost(duckdb_tree, costs_map, table_sizes, join_conditions, debug=True)
                print(f"--- End Detailed Calculation ---\n")
                print(f"DuckDB cost: {duckdb_cost}")
            else:
                print(f"Warning: Could not parse DuckDB join order: {duckdb_order}")
                duckdb_cost = -1
        except Exception as e:
            print(f"Error calculating DuckDB cost: {e}")
            import traceback
            traceback.print_exc()
            duckdb_cost = -1
    
    # Now calculate SQLite's cost using the cardinality model
    if sqlite_cost == -2:
        try:
            print(f"\n=== SQLite Cost Calculation (Cardinality Model) ===")
            sqlite_tables = sqlite_order.split(" -> ")
            if len(sqlite_tables) >= 2 and all(t in relations_map for t in sqlite_tables):
                sqlite_tree = linear_order_to_left_deep_tree(sqlite_tables)
                sqlite_cost, _ = calculate_tree_cost(sqlite_tree, costs_map, table_sizes, join_conditions)
                print(f"SQLite join order: {sqlite_order}")
                print(f"SQLite tree: {sqlite_tree}")
                print(f"SQLite cost: {sqlite_cost}")
            else:
                print(f"Warning: Could not parse SQLite join order: {sqlite_order}")
                sqlite_cost = -1
        except Exception as e:
            print(f"Error calculating SQLite cost: {e}")
            import traceback
            traceback.print_exc()
            sqlite_cost = -1
    
    # Track QAOA optimization time
    qaoa_planning_start = time.time()
    original_stdout, sys.stdout = sys.stdout, StringIO()
    try:
        optimizer = QUBO_Split_Optimization_func(f"join_log_{num_tables}")
        qubo_tree, _, error_msg = optimizer.finding_opt_jo(relations, weights, 'qaoa')
    except Exception as e:
        qubo_tree, error_msg = None, str(e)
    finally:
        sys.stdout = original_stdout
    qaoa_planning_time = time.time() - qaoa_planning_start
    
    duration_qubo, qubo_cost, qubo_order_str = -1, -1, "No valid tree"
    duration_qubo_total = -1  # Total time including planning
    
    if qubo_tree and not error_msg:
        qubo_order_str = str(qubo_tree)
        
        # Calculate QAOA cost using proper cardinality model
        tree_cost = -1
        try:
            print(f"\n=== QAOA Join Plan ===")
            print(f"QAOA tree: {qubo_tree}")
            print(f"\n--- Detailed QAOA Cost Calculation ---")
            tree_cost, _ = calculate_tree_cost(qubo_tree, costs_map, table_sizes, join_conditions, debug=True)
            print(f"--- End Detailed Calculation ---\n")
            print(f"QAOA cost (cardinality model): {tree_cost}")
            
        except Exception as e:
            print(f"Error calculating QAOA cost: {e}")
            import traceback
            traceback.print_exc()
        
        # Build and execute forced query with QAOA's join order
        try:
            # Detect join type and build appropriate FROM clause
            is_bushy = isinstance(qubo_tree[0], list) and isinstance(qubo_tree[1], list)
            
            if is_bushy:
                forced_from, _ = build_from_clause_recursively(qubo_tree, relations_map, join_conditions, reverse_qaoa=True)
                if forced_from.startswith('(') and forced_from.endswith(')'):
                    forced_from = forced_from[1:-1]
            else:
                forced_from, _ = build_from_clause_flat(qubo_tree, relations_map, join_conditions, reverse_qaoa=True, experimental_lr_hybrid=True)
            
            forced_query = f"{query_parts['select']} FROM {forced_from}"
            if 'where' in query_parts: 
                forced_query += f" {query_parts['where']}"
            forced_query += f" {query_parts['group_by']} {query_parts['order_by']} {query_parts['limit']};"
            
            qubo_cost = tree_cost
            
            # Cost comparison
            if duckdb_cost > 0 and sqlite_cost > 0 and qubo_cost > 0:
                costs = [('DuckDB', duckdb_cost), ('SQLite', sqlite_cost), ('QAOA', qubo_cost)]
                costs_sorted = sorted(costs, key=lambda x: x[1])
                print(f"\n📊 Cost Comparison (lower is better):")
                print(f"  1st: {costs_sorted[0][0]:<10} - {costs_sorted[0][1]:,}")
                print(f"  2nd: {costs_sorted[1][0]:<10} - {costs_sorted[1][1]:,}")
                print(f"  3rd: {costs_sorted[2][0]:<10} - {costs_sorted[2][1]:,}")
                
        except Exception as e:
            print(f"Error getting plan cost: {e}")
            import traceback
            traceback.print_exc()
            qubo_cost = tree_cost if tree_cost > 0 else -1
        
        # Execute forced query to measure actual execution time
        try:
            duckdb_conn.execute("SET disabled_optimizers TO 'join_order';")
            start_time = time.time()
            duckdb_conn.execute(forced_query).fetchall()
            duration_qubo = time.time() - start_time
            duration_qubo_total = qaoa_planning_time + duration_qubo
            duckdb_conn.execute("SET disabled_optimizers TO '';")
        except Exception as e:
            print(f"Error executing QAOA query: {e}")
            duckdb_conn.execute("SET disabled_optimizers TO '';")
            duration_qubo, duration_qubo_total = -1, -1
    else:
        print(f"QUBO optimization failed: {error_msg}")
        duration_qubo_total = -1
    
    print(f"\n--- Final Results ({num_tables} Tables) ---")
    print(f"DuckDB | Time: {duration_duckdb * 1000:.2f} ms | Cost: {duckdb_cost if duckdb_cost >= 0 else 'N/A':<10} | Order: {duckdb_order}")
    print(f"SQLite | Time: {duration_sqlite * 1000:.2f} ms | Cost: {sqlite_cost if sqlite_cost >= 0 else 'N/A':<10} | Order: {sqlite_order}")
    if duration_qubo >= 0:
        print(f"QAOA   | Exec: {duration_qubo * 1000:.2f} ms | Total: {duration_qubo_total * 1000:.2f} ms | Cost: {qubo_cost if qubo_cost >= 0 else 'N/A':<10} | Order: {qubo_order_str}")
    else:
        print(f"QAOA   | Execution Failed")
    
    print(f"\nNote: Costs calculated using cumulative cardinality model")
    
    return duration_duckdb, duckdb_cost, duration_sqlite, duration_qubo, qubo_cost, duration_qubo_total, sqlite_cost

def plot_results(results_dict):
    """Generates and displays plots comparing benchmark results.
    
    Results format: (duration_duckdb, duckdb_cost, duration_sqlite, duration_qubo, qubo_cost, duration_qubo_total, sqlite_cost)
    """
    for num_tables, results_list in results_dict.items():
        if not results_list: continue

        runs = range(1, len(results_list) + 1)
        
        # Extract times (convert to ms) - ALL THREE SYSTEMS
        duckdb_times = [r[0] * 1000 for r in results_list]
        sqlite_times = [r[2] * 1000 for r in results_list]
        qaoa_total_times = [r[5] * 1000 for r in results_list if r[5] >= 0]
        
        # Extract costs - ALL THREE SYSTEMS (cardinality model)
        duckdb_costs = [r[1] for r in results_list if r[1] >= 0]
        sqlite_costs = [r[6] for r in results_list if len(r) > 6 and r[6] >= 0]
        qaoa_costs = [r[4] for r in results_list if r[4] >= 0]
        
        # Create runs indices for valid data
        qaoa_valid_runs = [i for i, r in enumerate(results_list, 1) if r[5] >= 0]
        sqlite_cost_valid_runs = [i for i, r in enumerate(results_list, 1) if len(r) > 6 and r[6] >= 0]
        cost_valid_runs = range(1, len(results_list) + 1)  # All runs should have costs
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"Benchmark Comparison for {num_tables}-Table Join ({len(runs)} runs)", fontsize=16)

        # Plot 1: Overall Runtime (including planning time for QAOA)
        ax1.plot(runs, duckdb_times, 'o-', label='DuckDB', linewidth=2)
        ax1.plot(runs, sqlite_times, '^-', label='SQLite', linewidth=2)
        if qaoa_total_times: 
            ax1.plot(qaoa_valid_runs, qaoa_total_times, 's-', label='QAOA (incl. planning)', linewidth=2)
        ax1.set_xlabel("Run Number", fontsize=12)
        ax1.set_ylabel("Total Time (ms)", fontsize=12)
        ax1.set_title("Overall Runtime Comparison", fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Plot 2: Estimated Costs - All three using cardinality model
        if duckdb_costs:
            ax2.plot(cost_valid_runs, duckdb_costs, 'o-', label='DuckDB', linewidth=2.5, 
                    markersize=9, alpha=0.85, linestyle='-')
            if sqlite_costs:
                ax2.plot(sqlite_cost_valid_runs, sqlite_costs, '^--', label='SQLite', linewidth=2.5, 
                        markersize=9, alpha=0.85, linestyle='--')
            if qaoa_costs:
                qaoa_cost_runs = [i for i, r in enumerate(results_list, 1) if r[4] >= 0]
                ax2.plot(qaoa_cost_runs, qaoa_costs, 's:', label='QAOA', linewidth=2.5, 
                        markersize=9, alpha=0.85, linestyle=':')
            ax2.set_xlabel("Run Number", fontsize=12)
            ax2.set_ylabel("Estimated Cost (Cardinality)", fontsize=12)
            ax2.set_title("Cost Comparison: All Three Optimizers\n(Cardinality))", fontsize=13, fontweight='bold')
            ax2.set_yscale('log')
            ax2.legend(fontsize=11)
            ax2.grid(True, linestyle='--', alpha=0.6)
        else:
            ax2.text(0.5, 0.5, 'No cost data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title("Cost Comparison", fontsize=14, fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Join Order Benchmark using DuckDB, SQLite, and QAOA.")
    parser.add_argument('--loop', type=int, default=1, metavar='N', help='Number of times to run each benchmark (default: 1).')
    parser.add_argument('--tables', type=int, default=0, metavar='N', help='Run only N-table benchmark (3, 4, 5, or 6). Default: run 3 and 4.')
    parser.add_argument('--weights', type=str, default='cardinality', choices=['cardinality', 'random'], 
                        help='Weight generation method: "cardinality" (realistic, default) or "random" (research paper style, 1-100)')
    args = parser.parse_args()
    num_runs = args.loop
    tables_filter = args.tables
    weight_method = args.weights

    # Print benchmark configuration
    print("\n" + "="*70)
    print("  🚀 JOIN ORDER OPTIMIZATION BENCHMARK")
    print("="*70)
    print(f"  Number of runs:       {num_runs}")
    if tables_filter == 0:
        print(f"  Tables to test:       3 and 4 (default)")
    else:
        print(f"  Tables to test:       {tables_filter}")
    print(f"  Weight method:        {weight_method}")
    print(f"  Scale factor:         10.0 (100x larger)")
    print(f"  Optimizers:           DuckDB, SQLite, QAOA (Qiskit)")
    print("="*70 + "\n")

    results_3_tables, results_4_tables, results_5_tables, results_6_tables = [], [], [], []
    
    # Different scenarios to test various table size ratios
    scenarios = ['default', 'many_customers', 'large_orders', 'heavy_lineitems', 'balanced_small']

    for i in range(num_runs):
        # Cycle through different scenarios for variety
        scenario = scenarios[i % len(scenarios)] if num_runs > 1 else 'default'
        
        print(f"\n{'='*25} RUN {i + 1}/{num_runs} - Scenario: {scenario.upper()} {'='*25}\n")
        
        duckdb_conn = duckdb.connect(database=':memory:')
        sqlite_conn = sqlite3.connect(':memory:')
        
        # Use different seed AND scenario for each iteration
        # This creates fundamentally different optimization problems
        run_seed = 42 + i if num_runs > 1 else None
        setup_database('duckdb', duckdb_conn, scale_factor=10.0, seed=run_seed, scenario=scenario)
        setup_database('sqlite', sqlite_conn, scale_factor=10.0, seed=run_seed, scenario=scenario)

        # --- 3-Table Benchmark ---
        if tables_filter == 0 or tables_filter == 3:
            q_parts_3 = {
                "select": "SELECT c.c_name, SUM(l.l_extendedprice) AS total_revenue",
                "from": "FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey JOIN customer c ON o.o_custkey = c.c_custkey",
                "group_by": "GROUP BY c.c_name", "order_by": "ORDER BY total_revenue DESC", "limit": "LIMIT 10"
            }
            relations_map_3 = {'l': 'lineitem', 'o': 'orders', 'c': 'customer'}
            relations_3 = ['l', 'o', 'c']
            join_conditions_3 = {
                frozenset(['l', 'o']): 'l.l_orderkey = o.o_orderkey', 
                frozenset(['o', 'c']): 'o.o_custkey = c.c_custkey'
            }
            res3 = run_benchmark(3, relations_map_3, relations_3, join_conditions_3, q_parts_3, duckdb_conn, sqlite_conn, weight_method, run_seed)
            results_3_tables.append(res3)

        # --- 4-Table Benchmark ---
        if tables_filter == 0 or tables_filter == 4:
            q_parts_4 = {
                "select": "SELECT n.n_name, c.c_name, SUM(l.l_extendedprice) AS total_revenue",
                "from": "FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey JOIN customer c ON o.o_custkey = c.c_custkey JOIN nation n ON c.c_nationkey = n.n_nationkey",
                "where": "WHERE n.n_name = 'NATION_5'",
                "group_by": "GROUP BY n.n_name, c.c_name", "order_by": "ORDER BY total_revenue DESC", "limit": "LIMIT 10"
            }
            relations_map_4 = {'l': 'lineitem', 'o': 'orders', 'c': 'customer', 'n': 'nation'}
            relations_4 = ['l', 'o', 'c', 'n']
            join_conditions_4 = {
                frozenset(['l', 'o']): 'l.l_orderkey = o.o_orderkey', 
                frozenset(['o', 'c']): 'o.o_custkey = c.c_custkey',
                frozenset(['c', 'n']): 'c.c_nationkey = n.n_nationkey'
            }
            res4 = run_benchmark(4, relations_map_4, relations_4, join_conditions_4, q_parts_4, duckdb_conn, sqlite_conn, weight_method, run_seed)
            results_4_tables.append(res4)
        
        # --- 5-Table Benchmark ---
        if tables_filter == 5:
            q_parts_5 = {
                "select": "SELECT r.r_name, n.n_name, c.c_name, SUM(l.l_extendedprice) AS total_revenue",
                "from": "FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey JOIN customer c ON o.o_custkey = c.c_custkey JOIN nation n ON c.c_nationkey = n.n_nationkey JOIN region r ON n.n_regionkey = r.r_regionkey",
                "where": "WHERE r.r_name = 'REGION_2'",
                "group_by": "GROUP BY r.r_name, n.n_name, c.c_name", "order_by": "ORDER BY total_revenue DESC", "limit": "LIMIT 10"
            }
            relations_map_5 = {'l': 'lineitem', 'o': 'orders', 'c': 'customer', 'n': 'nation', 'r': 'region'}
            relations_5 = ['l', 'o', 'c', 'n', 'r']
            join_conditions_5 = {
                frozenset(['l', 'o']): 'l.l_orderkey = o.o_orderkey', 
                frozenset(['o', 'c']): 'o.o_custkey = c.c_custkey',
                frozenset(['c', 'n']): 'c.c_nationkey = n.n_nationkey',
                frozenset(['n', 'r']): 'n.n_regionkey = r.r_regionkey'
            }
            res5 = run_benchmark(5, relations_map_5, relations_5, join_conditions_5, q_parts_5, duckdb_conn, sqlite_conn, weight_method, run_seed)
            results_5_tables.append(res5)
        
        # --- 6-Table Benchmark ---
        if tables_filter == 6:
            q_parts_6 = {
                "select": "SELECT r.r_name, n.n_name, s.s_name, SUM(l.l_extendedprice) AS total_revenue",
                "from": "FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey JOIN customer c ON o.o_custkey = c.c_custkey JOIN nation n ON c.c_nationkey = n.n_nationkey JOIN region r ON n.n_regionkey = r.r_regionkey JOIN supplier s ON l.l_suppkey = s.s_suppkey",
                "where": "WHERE r.r_name = 'REGION_2'",
                "group_by": "GROUP BY r.r_name, n.n_name, s.s_name", "order_by": "ORDER BY total_revenue DESC", "limit": "LIMIT 10"
            }
            relations_map_6 = {'l': 'lineitem', 'o': 'orders', 'c': 'customer', 'n': 'nation', 'r': 'region', 's': 'supplier'}
            relations_6 = ['l', 'o', 'c', 'n', 'r', 's']
            join_conditions_6 = {
                frozenset(['l', 'o']): 'l.l_orderkey = o.o_orderkey', 
                frozenset(['o', 'c']): 'o.o_custkey = c.c_custkey',
                frozenset(['c', 'n']): 'c.c_nationkey = n.n_nationkey',
                frozenset(['n', 'r']): 'n.n_regionkey = r.r_regionkey',
                frozenset(['l', 's']): 'l.l_suppkey = s.s_suppkey',
                frozenset(['s', 'n']): 's.s_nationkey = n.n_nationkey'
            }
            res6 = run_benchmark(6, relations_map_6, relations_6, join_conditions_6, q_parts_6, duckdb_conn, sqlite_conn, weight_method, run_seed)
            results_6_tables.append(res6)
        
        duckdb_conn.close()
        sqlite_conn.close()

    if num_runs > 1 and PLOTTING_ENABLED:
        print("\n--- Generating plots... ---")
        results_to_plot = {}
        if results_3_tables:
            results_to_plot[3] = results_3_tables
        if results_4_tables:
            results_to_plot[4] = results_4_tables
        if results_5_tables:
            results_to_plot[5] = results_5_tables
        if results_6_tables:
            results_to_plot[6] = results_6_tables
        if results_to_plot:
            plot_results(results_to_plot)
    elif num_runs > 1:
        print("\nPlotting skipped because matplotlib is not installed.")

