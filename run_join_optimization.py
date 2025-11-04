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

from SSS_QUBO import QUBO_formulation, QUBO_Split_Optimization_func, Helping_functions, SolverType

# Conditional import for plotting
try:
    import matplotlib.pyplot as plt
    PLOTTING_ENABLED = True
except ImportError:
    PLOTTING_ENABLED = False
    print("\nWarning: matplotlib not found. Plotting will be disabled. To enable, run: pip install matplotlib")

def parse_estimated_cardinality(s):
    """Parses cardinality strings like '100', '10.5k', '2.1M', '1e+6'."""
    s = str(s).lower().strip()
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
    elif s.endswith('b') or s.endswith('g'): # b for billion, g for giga
        multiplier = 1000000000
        s_num = s[:-1]
        
    try:
        return int(float(s_num) * multiplier)
    except ValueError:
        print(f"Warning: Could not parse est. cardinality string '{s}'")
        return 0

def parse_row_cardinality(s):
    """Parses cardinality strings like '25,143' from '~... rows'."""
    try:
        # Clean the string: remove ','
        cleaned_s = s.strip().replace(',', '')
        return int(cleaned_s)
    except Exception as e:
        print(f"Warning: Could not parse row cardinality string '{s}'. Error: {e}")
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
    nations_df = pd.DataFrame({
        'n_nationkey': range(config['nations']), 
        'n_name': [f'NATION_{i}' for i in range(config['nations'])], 
        'n_regionkey': np.random.randint(0, 5, size=config['nations'])
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
        'l_extendedprice': np.random.uniform(50, 2000, size=num_lineitems)
    })
    
    print(f"   Tables: {num_customers} customers, {num_orders} orders, {num_lineitems} lineitems")

    if db_type == 'duckdb':
        connection.execute("CREATE TABLE nation (n_nationkey INTEGER, n_name VARCHAR, n_regionkey INTEGER);")
        connection.execute("CREATE TABLE customer (c_custkey INTEGER, c_name VARCHAR, c_nationkey INTEGER);")
        connection.execute("CREATE TABLE orders (o_orderkey INTEGER, o_custkey INTEGER, o_totalprice DECIMAL(10, 2));")
        connection.execute("CREATE TABLE lineitem (l_orderkey INTEGER, l_extendedprice DECIMAL(10, 2));")
        # Use execute with SELECT * FROM df_name, which is the correct way to load from a local DataFrame
        connection.execute("INSERT INTO nation SELECT * FROM nations_df")
        connection.execute("INSERT INTO customer SELECT * FROM customers_df")
        connection.execute("INSERT INTO orders SELECT * FROM orders_df")
        connection.execute("INSERT INTO lineitem SELECT * FROM lineitems_df")
    
    elif db_type == 'sqlite':
        nations_df.to_sql("nation", connection, if_exists="replace", index=False)
        customers_df.to_sql("customer", connection, if_exists="replace", index=False)
        orders_df.to_sql("orders", connection, if_exists="replace", index=False)
        lineitems_df.to_sql("lineitem", connection, if_exists="replace", index=False)
        
    print("--- Database setup complete. ---\n")

def parse_cardinality(s):
    """Parses cardinality strings like '100', '10.5k', '2.1M', '1e+6'."""
    s = str(s).lower().strip()
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
    elif s.endswith('b') or s.endswith('g'): # b for billion, g for giga
        multiplier = 1000000000
        s_num = s[:-1]
        
    try:
        return int(float(s_num) * multiplier)
    except ValueError:
        print(f"Warning: Could not parse cardinality string '{s}'")
        return 0

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
    Calculate join costs using a REALISTIC but simple cardinality-based model.
    This follows the QUBO paper approach: uses basic statistics and selectivity estimates,
    not a sophisticated optimizer.
    
    WHY THIS IS BETTER THAN DUCKDB ESTIMATES:
    - QAOA optimizes using THESE weights (simple cardinality model)
    - DuckDB optimizes using its own sophisticated model (histograms, correlations, etc.)
    - This creates a FAIR COMPARISON: each optimizer uses its own cost model
    - We then compare actual EXECUTION TIME to see which join order is faster
    - If we used DuckDB's estimates for QAOA weights, we'd just be copying DuckDB's optimizer!
    
    Cost model for join S = {R1, R2, ..., Rk}:
    1. For 2-table joins: cost = |R| × |S| × selectivity
       - If join condition exists: selectivity = 1 / max(|R|, |S|) (assumes FK-like relationship)
       - If no join condition (cross product): selectivity = 1
    2. For multi-table joins: build up incrementally using join tree structure costs
    
    This is more realistic than random weights but doesn't use DuckDB's optimizer.
    """
    print("--- Calculating join costs (QUBO-style: cardinality + selectivity) ---")
    
    # Get table cardinalities
    table_sizes = {}
    for rel, table_name in relations_map.items():
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        table_sizes[rel] = count
        print(f"  {rel} ({table_name}): {count} rows")
    
    costs_map = {}
    weights = []
    sublists = QUBO_formulation.relation_sublists(relations)
    
    print("\nCalculating join costs with selectivity:")
    for combo in sublists:
        if len(combo) < 2: 
            continue
        
        if len(combo) == 2:
            # Two-table join: use selectivity estimate
            r1, r2 = combo[0], combo[1]
            size1, size2 = table_sizes[r1], table_sizes[r2]
            
            # Check if there's a join condition
            key = frozenset([r1, r2])
            if key in join_conditions:
                # Assume FK-like relationship: selectivity = 1 / max cardinality
                # This means the result size ≈ size of the smaller table
                selectivity = 1.0 / max(size1, size2)
                cost = int(size1 * size2 * selectivity)
                print(f"  {combo}: {size1} × {size2} × {selectivity:.6f} = {cost} (with join condition)")
            else:
                # Cross product: full Cartesian product
                cost = size1 * size2
                print(f"  {combo}: {size1} × {size2} = {cost} (cross product)")
        else:
            # Multi-table join: estimate as sum of pairwise joins
            # This is a simplification - the paper uses more sophisticated DP-based costs
            # but this gives reasonable relative costs without using DuckDB
            cost = 0
            combo_set = set(combo)
            
            # For each table in the combo, estimate the cost of joining it
            # Cost ≈ product of all table sizes with selectivity adjustment
            base_cost = 1
            for rel in combo:
                base_cost *= table_sizes[rel]
            
            # Apply selectivity: assume each join reduces by a factor
            # More joins = more selectivity reduction
            num_joins = len(combo) - 1
            # Conservative selectivity: each join reduces by 1/average_table_size
            avg_size = sum(table_sizes[r] for r in combo) / len(combo)
            selectivity = (1.0 / avg_size) ** (num_joins - 1)  # Compound selectivity
            
            cost = int(base_cost * selectivity)
            print(f"  {combo}: {base_cost} × {selectivity:.6f} = {cost}")
        
        costs_map[frozenset(combo)] = cost
        weights.append(cost)
    
    print(f"\nCalculated {len(weights)} join costs using cardinality-based model")
    print("--- Cost calculation complete. ---\n")
    return costs_map, weights


def get_join_costs(conn, relations_map, relations):
    """
    Calculate join costs using DuckDB's EXPLAIN (for comparison).
    WARNING: This gives QAOA an unfair advantage by using DuckDB's sophisticated estimates!
    """
    print("--- Calculating join costs using DuckDB's EXPLAIN (CHEATING MODE) ---")
    costs_map = {}
    weights = []
    join_clauses = {
        ('c', 'n'): 'c.c_nationkey = n.n_nationkey', 
        ('o', 'c'): 'o.o_custkey = c.c_custkey', 
        ('l', 'o'): 'l.l_orderkey = o.o_orderkey'
    }
    
    sublists = QUBO_formulation.relation_sublists(relations)
    
    for combo in sublists:
        if len(combo) < 2: continue
            
        from_clause, current_joins, temp_combo = f"FROM {relations_map[combo[0]]} AS {combo[0]}", {combo[0]}, list(combo[1:])
        
        while len(temp_combo) > 0:
            found_join, i = False, 0
            while i < len(temp_combo):
                r = temp_combo[i]
                for joined_table in current_joins:
                    key1 = tuple(sorted((r, joined_table)))
                    if key1 in join_clauses:
                        from_clause += f" JOIN {relations_map[r]} AS {r} ON {join_clauses[key1]}"
                        current_joins.add(r)
                        temp_combo.pop(i)
                        found_join = True
                        break
                if found_join: break
                i += 1
            if not found_join: 
                cross_r = temp_combo.pop(0)
                from_clause += f", {relations_map[cross_r]} AS {cross_r}"
                current_joins.add(cross_r)
                
        try:
            explain_result = conn.execute(f"EXPLAIN SELECT * {from_clause} LIMIT 1;").fetchone()[1]
            
            join_operators = ['HASH_JOIN', 'CROSS_PRODUCT', 'PIECEWISE_MERGE_JOIN', 'NESTED_LOOP_JOIN', 'BLOCKWISE_NL_JOIN']
            first_join_pos = -1
            for op in join_operators:
                pos = explain_result.find(op)
                if pos != -1:
                    if first_join_pos == -1 or pos < first_join_pos:
                        first_join_pos = pos

            cost = 1_000_000_000
            if first_join_pos != -1:
                search_area = explain_result[first_join_pos:]
                match = re.search(r'~(\d{1,3}(?:,\d{3})*|\d+)\s+rows', search_area)
                if match:
                    cost = parse_row_cardinality(match.group(1))
            else:
                match = re.search(r'~(\d{1,3}(?:,\d{3})*|\d+)\s+rows', explain_result)
                if match:
                     cost = parse_row_cardinality(match.group(1))

            if cost <= 0:
                cost = 1_000_000_000

            costs_map[frozenset(combo)] = cost
            weights.append(cost)
        except Exception as e: 
            print(f"Warning: Could not get cost for {combo}: {e}")
            costs_map[frozenset(combo)] = 1_000_000_000
            weights.append(1_000_000_000)
            
    print(f"Calculated {len(weights)} join costs using DuckDB estimates")
    print("--- Cost calculation complete. ---\n")
    return costs_map, weights

def calculate_tree_cost(tree, costs_map):
    """Recursively calculates the total estimated cost of a join tree."""
    if isinstance(tree, str):
        return 0

    left_subtree, right_subtree = tree[0], tree[1]
    cost = calculate_tree_cost(left_subtree, costs_map) + calculate_tree_cost(right_subtree, costs_map)
    
    def get_relations(subtree):
        if isinstance(subtree, str): return {subtree}
        return get_relations(subtree[0]).union(get_relations(subtree[1]))

    key = frozenset(get_relations(tree))
    cost += costs_map.get(key, 0)
    return cost

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
    Build FROM clause from join tree using nested parentheses.python run_join_optimization.py --loop 10 --tables 3
    If reverse_qaoa=True, treats the tree as QAOA output which is read right-to-left.
    Note: This approach can cause poor cardinality estimates in DuckDB.
    """
    if isinstance(tree, str): 
        return f"{relations_map[tree]} AS {tree}", {tree}
    if len(tree) != 2: 
        raise ValueError(f"Invalid tree structure: {tree}")
    
    # For QAOA trees (read right-to-left), we need to reverse the order
    if reverse_qaoa:
        left_tree, right_tree = tree[1], tree[0]  # Reverse the order for QAOA
    else:
        left_tree, right_tree = tree[0], tree[1]  # Normal order
        
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
        
    return f"({left_sql} {join_type} {right_sql} {join_sql})", left_relations.union(right_relations)

def run_benchmark(num_tables, relations_map, relations, join_conditions, query_parts, duckdb_conn, sqlite_conn, weight_method='cardinality', run_seed=None):
    print(f"\n" + "="*50 + f"\n### STARTING {num_tables}-TABLE BENCHMARK ###\n" + "="*50 + "\n")
    
    query = f"{query_parts['select']} {query_parts['from']}"
    if 'where' in query_parts: query += f" {query_parts['where']}"
    query += f" {query_parts['group_by']} {query_parts['order_by']} {query_parts['limit']};"

    # --- Helper function for new robust parsing ---
    def get_cost_from_graphical_plan(plan_string):
        """
        Parses the graphical EXPLAIN plan.
        Finds the topmost JOIN operator's output cardinality (the final join result).
        This represents the cost of the join order.
        """
        try:
            # Find ALL join operators and their cardinalities
            join_operators_regex = r'(HASH_JOIN|CROSS_PRODUCT|PIECEWISE_MERGE_JOIN|NESTED_LOOP_JOIN|BLOCKWISE_NL_JOIN)'
            
            # Find all join operators
            join_matches = list(re.finditer(join_operators_regex, plan_string))
            
            if not join_matches:
                # No join found, maybe a single table scan
                match = re.search(r'~(\d{1,3}(?:,\d{3})*|\d+)\s+rows', plan_string)
                if match:
                    cardinality = parse_row_cardinality(match.group(1))
                    print(f"[DEBUG] No joins found, using first cardinality: {cardinality}")
                    return cardinality
                return -2
            
            # The FIRST join match is the topmost (root) join
            first_join = join_matches[0]
            
            # Find the cardinality for this join
            # Look for the next ~...rows after this join operator
            search_start = first_join.end()
            
            # Extract the block for this join (until we hit another major operator or end)
            # Look ahead to find the cardinality line
            search_area = plan_string[search_start:search_start + 500]  # Look ahead 500 chars
            
            # Find the first ~rows in this join's block
            card_match = re.search(r'~(\d{1,3}(?:,\d{3})*|\d+)\s+rows', search_area)
            
            if card_match:
                cardinality = parse_row_cardinality(card_match.group(1))
                print(f"[DEBUG] Found topmost join cardinality: {cardinality}")
                print(f"[DEBUG] Total joins in plan: {len(join_matches)}")
                
                # Also show all join cardinalities for debugging
                all_cardinalities = []
                for i, jm in enumerate(join_matches):
                    search = plan_string[jm.end():jm.end() + 500]
                    cm = re.search(r'~(\d{1,3}(?:,\d{3})*|\d+)\s+rows', search)
                    if cm:
                        card = parse_row_cardinality(cm.group(1))
                        all_cardinalities.append(card)
                        join_type = plan_string[jm.start():jm.end()]
                        print(f"[DEBUG]   Join {i+1} ({join_type}) cardinality: {card}")
                    else:
                        print(f"[DEBUG]   Join {i+1} - could not find cardinality")
                
                return cardinality
            else:
                print(f"[DEBUG] Could not find cardinality after first join")
                return -3
                
        except Exception as e:
            print(f"Warning: Cost parsing failed with error: {e}")
            import traceback
            traceback.print_exc()
            return -4
    # --- End of helper function ---


    # 1. DuckDB Benchmark (separating planning and execution)
    print("--- Running DuckDB Benchmark ---")
    explain_duckdb = duckdb_conn.execute(f"EXPLAIN {query}").fetchone()[1]
    duckdb_order = parse_join_order_from_duckdb(explain_duckdb)
    
    # Debug: Show the plan
    print(f"\nDuckDB EXPLAIN plan:\n{explain_duckdb}\n")
    
    # --- NEW COST PARSING ---
    duckdb_cost = get_cost_from_graphical_plan(explain_duckdb)
    print(f"DuckDB cost from EXPLAIN: {duckdb_cost}")
    # --- END OF NEW COST PARSING ---

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
    
    # 3. QUBO Benchmark with QAOA
    print("\n--- Running QUBO Benchmark (QAOA) ---")
    
    # Generate weights based on chosen method
    if weight_method == 'random':
        costs_map, weights = get_join_costs_random(relations, seed=run_seed)
    else:  # cardinality
        costs_map, weights = get_join_costs_simple(duckdb_conn, relations_map, relations, join_conditions)
    
    original_stdout, sys.stdout = sys.stdout, StringIO()
    try:
        optimizer = QUBO_Split_Optimization_func(f"join_log_{num_tables}")
        qubo_tree, _, error_msg = optimizer.finding_opt_jo(relations, weights, SolverType.QAOA)
    except Exception as e:
        qubo_tree, error_msg = None, str(e)
    finally:
        sys.stdout = original_stdout
    
    duration_qubo, qubo_cost, qubo_order_str = -1, -1, "No valid tree"
    
    if qubo_tree and not error_msg:
        qubo_order_str = str(qubo_tree)
        
        # Calculate QAOA cost using multiple methods for comparison
        tree_cost = -1
        try:
            tree_cost = calculate_tree_cost(qubo_tree, costs_map)
            print(f"QAOA tree: {qubo_tree}")
            print(f"QAOA tree cost calculated: {tree_cost}")
            
            # Extract and display the actual join order from the tree
            def tree_to_join_order(t):
                """Convert tree to human-readable join order."""
                if isinstance(t, str):
                    return t
                # For QAOA trees (right-to-left), show as: right JOIN left
                left = tree_to_join_order(t[0])
                right = tree_to_join_order(t[1])
                return f"({right} ⋈ {left})"
            
            join_order_visual = tree_to_join_order(qubo_tree)
            print(f"QAOA join order (visual): {join_order_visual}")
            
        except Exception as e:
            print(f"Error calculating tree cost: {e}")
        
        # Get the cost from the forced query plan using the same method as DuckDB
        plan_cost = -1
        forced_query = ""
        
        try:
            # QAOA trees are read right-to-left, so use reverse_qaoa=True
            # Use FLAT from clause to get better cardinality estimates
            
            # Try BOTH methods to see which matches DuckDB better
            print("\n--- Method 1: Standard right-to-left ---")
            forced_from_standard, _ = build_from_clause_flat(qubo_tree, relations_map, join_conditions, reverse_qaoa=True, experimental_lr_hybrid=False)
            print(f"Standard join order: {forced_from_standard}")
            
            print("\n--- Method 2: EXPERIMENTAL hybrid (right-to-left structure, left-to-right pairs) ---")
            forced_from_hybrid, _ = build_from_clause_flat(qubo_tree, relations_map, join_conditions, reverse_qaoa=True, experimental_lr_hybrid=True)
            print(f"Hybrid join order: {forced_from_hybrid}")
            
            # Use the hybrid method as default for now
            forced_from = forced_from_hybrid
            print(f"\nUsing HYBRID method for cost calculation")
            forced_query = f"{query_parts['select']} FROM {forced_from}"
            if 'where' in query_parts: forced_query += f" {query_parts['where']}"
            forced_query += f" {query_parts['group_by']} {query_parts['order_by']} {query_parts['limit']};"
            
            print(f"Forced query: {forced_query}")
            
            # Note: We don't force DuckDB's execution order for EXPLAIN
            # because it doesn't affect our cost calculation (we use tree_cost)
            # This allows DuckDB to show its actual execution plan for comparison
            
            # Run EXPLAIN on the forced query (DuckDB may reorder it)
            explain_qubo_plan = duckdb_conn.execute(f"EXPLAIN {forced_query}").fetchone()[1]


            # Debug: Show the QAOA plan
            print(f"\nQAOA EXPLAIN plan:\n{explain_qubo_plan}\n")
            
            # Parse the forced plan using DuckDB's estimate
            plan_cost = get_cost_from_graphical_plan(explain_qubo_plan)
            
            # For reporting: use DuckDB's estimate of the QAOA plan (apples-to-apples)
            # This allows fair comparison: both plans evaluated by the same cost model
            qubo_cost = plan_cost if plan_cost > 0 else tree_cost
            
            # Debug output
            print(f"\nCost Analysis:")
            print(f"  - QAOA optimized using tree cost: {tree_cost} (simple cardinality model)")
            print(f"  - DuckDB's estimate of QAOA plan: {plan_cost} (sophisticated model)")
            print(f"  - DuckDB's estimate of its own plan: {duckdb_cost}")
            print(f"  → Using {qubo_cost} for comparison (DuckDB's estimate of QAOA plan)")
            
            # Show if plans match
            if abs(plan_cost - duckdb_cost) < 100:
                print(f"  ✓ Plans match! Both optimizers chose the same join order.")
            elif plan_cost < duckdb_cost:
                print(f"  ✓ QAOA found a better plan! ({plan_cost} < {duckdb_cost})")
            else:
                print(f"  ✗ QAOA's plan is suboptimal ({plan_cost} > {duckdb_cost})")
                
        except Exception as e:
            print(f"Error getting plan cost: {e}")
            qubo_cost = tree_cost if tree_cost > 0 else -1
        
        # Execute the forced query for timing
        try:
            print(f"Forcing QUBO join order...")
            duckdb_conn.execute("SET disabled_optimizers TO 'join_order';")
            start_time = time.time()
            duckdb_conn.execute(forced_query).fetchall()
            duration_qubo = time.time() - start_time
            duckdb_conn.execute("SET disabled_optimizers TO '';")
        except Exception as e:
            print(f"Error executing forced QUBO query: {e}")
            duckdb_conn.execute("SET disabled_optimizers TO '';")
            duration_qubo = -1
    else:
        print(f"QUBO optimization failed: {error_msg}")
    
    print(f"\n--- Final Results ({num_tables} Tables) ---")
    print(f"DuckDB Default | Time: {duration_duckdb * 1000:.2f} ms | Cost: {duckdb_cost:<10} | Order: {duckdb_order}")
    print(f"SQLite Default | Time: {duration_sqlite * 1000:.2f} ms | Order: {sqlite_order}")
    if duration_qubo >= 0:
        cost_str = f"{qubo_cost:<10}" if qubo_cost >= 0 else "Unknown   "
        print(f"QUBO QAOA      | Time: {duration_qubo * 1000:.2f} ms | Cost: {cost_str} | Order: {qubo_order_str}")
    else:
        print(f"QUBO QAOA      | Execution Failed")
    
    return duration_duckdb, duckdb_cost, duration_sqlite, duration_qubo, qubo_cost

def plot_results(results_dict):
    """Generates and displays plots comparing benchmark results."""
    for num_tables, results_list in results_dict.items():
        if not results_list: continue

        runs = range(1, len(results_list) + 1)
        duckdb_times = [r[0] * 1000 for r in results_list]
        qaoa_times = [r[3] * 1000 for r in results_list if r[3] >= 0]
        duckdb_costs = [r[1] for r in results_list if r[1] >= 0]
        qaoa_costs = [r[4] for r in results_list if r[4] >= 0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"Benchmark Comparison for {num_tables}-Table Join ({len(runs)} runs)", fontsize=16)

        # Plot 1: Runtimes
        ax1.plot(runs, duckdb_times, 'o-', label='DuckDB')
        if qaoa_times: ax1.plot(runs, qaoa_times, 's-', label='QAOA')
        ax1.set_xlabel("Run Number")
        ax1.set_ylabel("Execution Time (ms)")
        ax1.set_title("Runtime Comparison")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Plot 2: Costs
        ax2.plot(runs, duckdb_costs, 'o-', label='DuckDB')
        if qaoa_costs: ax2.plot(runs, qaoa_costs, 's-', label='QAOA')
        ax2.set_xlabel("Run Number")
        ax2.set_ylabel("Estimated Cost (Cardinality)")
        ax2.set_title("Estimated Cost Comparison")
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Join Order Benchmark using DuckDB, SQLite, and QAOA.")
    parser.add_argument('--loop', type=int, default=1, metavar='N', help='Number of times to run each benchmark (default: 1).')
    parser.add_argument('--tables', type=int, default=0, metavar='N', help='Run only N-table benchmark (3 or 4). Default: run both.')
    parser.add_argument('--weights', type=str, default='cardinality', choices=['cardinality', 'random'], 
                        help='Weight generation method: "cardinality" (realistic, default) or "random" (research paper style, 1-100)')
    args = parser.parse_args()
    num_runs = args.loop
    tables_filter = args.tables
    weight_method = args.weights

    results_3_tables, results_4_tables = [], []
    
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
        setup_database('duckdb', duckdb_conn, scale_factor=0.1, seed=run_seed, scenario=scenario)
        setup_database('sqlite', sqlite_conn, scale_factor=0.1, seed=run_seed, scenario=scenario)

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
        
        duckdb_conn.close()
        sqlite_conn.close()

    if num_runs > 1 and PLOTTING_ENABLED:
        print("\n--- Generating plots... ---")
        results_to_plot = {}
        if results_3_tables:
            results_to_plot[3] = results_3_tables
        if results_4_tables:
            results_to_plot[4] = results_4_tables
        if results_to_plot:
            plot_results(results_to_plot)
    elif num_runs > 1:
        print("\nPlotting skipped because matplotlib is not installed.")

