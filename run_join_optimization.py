import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import numpy as np
import time
import re
import sys
from io import StringIO
import warnings
import argparse
import random
from scipy.sparse import SparseEfficiencyWarning

# Suppress scipy sparse matrix efficiency warnings
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

# SSS_QUBO import assumes it's in the same directory or python path
from SSS_QUBO import QUBO_formulation, QUBO_Split_Optimization_func, Helping_functions, SolverType

# Conditional import for plotting
try:
    import matplotlib.pyplot as plt
    PLOTTING_ENABLED = True
except ImportError:
    PLOTTING_ENABLED = False
    print("\nWarning: matplotlib not found. Plotting will be disabled. To enable, run: pip install matplotlib")

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

def parse_row_cardinality(s):
    """Parses cardinality strings like '25,143' from '~... rows'."""
    try:
        # Clean the string: remove ','
        cleaned_s = s.strip().replace(',', '')
        return int(cleaned_s)
    except Exception as e:
        print(f"Warning: Could not parse row cardinality string '{s}'. Error: {e}")
        return 0

def is_connected(tables_subset, join_conditions):
    """
    Checks if a subset of tables forms a connected graph based on join conditions.
    Uses a simple Breadth-First Search (BFS).
    """
    if not tables_subset:
        return True
    
    subset = set(tables_subset)
    if len(subset) <= 1:
        return True

    start_node = list(subset)[0]
    
    visited = {start_node}
    queue = [start_node]
    
    while queue:
        current_node = queue.pop(0)
        
        # Find neighbors within the subset
        for other_node in subset:
            if other_node not in visited:
                # Check if a join condition exists between them
                if frozenset([current_node, other_node]) in join_conditions:
                    visited.add(other_node)
                    queue.append(other_node)
                    
    # The graph is connected if BFS visited all nodes in the subset
    return visited == subset

def setup_database(connection, scale_factor=0.1, seed=None, scenario='default'):
    """
    Sets up the PostgreSQL database.
    If seed is provided, uses it for reproducible random data generation.
    Scenario determines table size ratios and data patterns.
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"--- Setting up PostgreSQL database (scenario={scenario}, seed={seed})... ---")
    
    # Different scenarios create different table size ratios and selectivities
    # This makes each run test QAOA on fundamentally different join optimization problems
    scenarios = {
        'default': {
            'nations': 25,
            'customers_mult': 1.0,      # 1,500 customers
            'orders_mult': 1.0,         # 15,000 orders
            'lineitems_mult': 1.0,      # 60,000 lineitems
            'order_cust_ratio': 1.0,  # Normal distribution
            'lineitem_order_ratio': 1.0
        },
        'many_customers': {
            'nations': 25,
            'customers_mult': 3.0,      # 4,500 customers (3x more)
            'orders_mult': 1.0,         # 15,000 orders (same)
            'lineitems_mult': 0.8,      # 48,000 lineitems (fewer)
            'order_cust_ratio': 0.5,  # Fewer orders per customer
            'lineitem_order_ratio': 0.8
        },
        'large_orders': {
            'nations': 25,
            'customers_mult': 0.6,      # 900 customers (fewer)
            'orders_mult': 2.0,         # 30,000 orders (2x more)
            'lineitems_mult': 1.5,      # 90,000 lineitems (more)
            'order_cust_ratio': 2.0,  # More orders per customer
            'lineitem_order_ratio': 1.2
        },
        'heavy_lineitems': {
            'nations': 25,
            'customers_mult': 0.8,      # 1,200 customers
            'orders_mult': 0.7,         # 10,500 orders (fewer)
            'lineitems_mult': 2.5,      # 150,000 lineitems (huge!)
            'order_cust_ratio': 1.2,
            'lineitem_order_ratio': 3.0  # Many items per order
        },
        'balanced_small': {
            'nations': 25,
            'customers_mult': 0.5,      # 750 customers
            'orders_mult': 0.5,         # 7,500 orders
            'lineitems_mult': 0.5,      # 30,000 lineitems
            'order_cust_ratio': 1.0,
            'lineitem_order_ratio': 1.0
        }
    }
    
    config = scenarios.get(scenario, scenarios['default'])
    
    # Generate data with varying sizes based on scenario
    # Regions (for 5+ table joins)
    num_regions = 5
    regions_df = pd.DataFrame({
        'r_regionkey': range(num_regions),
        'r_name': [f'REGION_{i}' for i in range(num_regions)],
        'r_comment': [f'Comment for region {i}' for i in range(num_regions)]
    })
    
    nations_df = pd.DataFrame({
        'n_nationkey': range(config['nations']), 
        'n_name': [f'NATION_{i}' for i in range(config['nations'])], 
        'n_regionkey': np.random.randint(0, num_regions, size=config['nations'])
    })
    
    num_suppliers = int(1000 * scale_factor)
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

    # Create tables
    with connection.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS lineitem CASCADE;")
        cur.execute("DROP TABLE IF EXISTS orders CASCADE;")
        cur.execute("DROP TABLE IF EXISTS customer CASCADE;")
        cur.execute("DROP TABLE IF EXISTS supplier CASCADE;")
        cur.execute("DROP TABLE IF EXISTS nation CASCADE;")
        cur.execute("DROP TABLE IF EXISTS region CASCADE;")
        
        cur.execute("""
            CREATE TABLE region (
                r_regionkey INTEGER PRIMARY KEY,
                r_name VARCHAR,
                r_comment VARCHAR
            );
        """)
        
        cur.execute("""
            CREATE TABLE nation (
                n_nationkey INTEGER PRIMARY KEY,
                n_name VARCHAR,
                n_regionkey INTEGER
            );
        """)
        
        cur.execute("""
            CREATE TABLE supplier (
                s_suppkey INTEGER PRIMARY KEY,
                s_name VARCHAR,
                s_nationkey INTEGER
            );
        """)
        
        cur.execute("""
            CREATE TABLE customer (
                c_custkey INTEGER PRIMARY KEY,
                c_name VARCHAR,
                c_nationkey INTEGER
            );
        """)
        
        cur.execute("""
            CREATE TABLE orders (
                o_orderkey INTEGER PRIMARY KEY,
                o_custkey INTEGER,
                o_totalprice DECIMAL(10, 2)
            );
        """)
        
        cur.execute("""
            CREATE TABLE lineitem (
                l_orderkey INTEGER,
                l_suppkey INTEGER,
                l_extendedprice DECIMAL(10, 2)
            );
        """)
        
        # Insert data using execute_values for efficiency
        # Convert numpy types to Python native types to avoid PostgreSQL schema errors
        def convert_to_python_types(row):
            """Convert numpy types in a row to Python native types."""
            return tuple(
                int(val) if isinstance(val, (np.integer, np.int64, np.int32)) else
                float(val) if isinstance(val, (np.floating, np.float64, np.float32)) else
                str(val) if not isinstance(val, (int, float, str)) else val
                for val in row
            )
        
        execute_values(cur, """
            INSERT INTO region (r_regionkey, r_name, r_comment) VALUES %s
        """, [convert_to_python_types(row) for row in regions_df.values])
        
        execute_values(cur, """
            INSERT INTO nation (n_nationkey, n_name, n_regionkey) VALUES %s
        """, [convert_to_python_types(row) for row in nations_df.values])
        
        execute_values(cur, """
            INSERT INTO supplier (s_suppkey, s_name, s_nationkey) VALUES %s
        """, [convert_to_python_types(row) for row in suppliers_df.values])
        
        execute_values(cur, """
            INSERT INTO customer (c_custkey, c_name, c_nationkey) VALUES %s
        """, [convert_to_python_types(row) for row in customers_df.values])
        
        execute_values(cur, """
            INSERT INTO orders (o_orderkey, o_custkey, o_totalprice) VALUES %s
        """, [convert_to_python_types(row) for row in orders_df.values])
        
        execute_values(cur, """
            INSERT INTO lineitem (l_orderkey, l_suppkey, l_extendedprice) VALUES %s
        """, [convert_to_python_types(row) for row in lineitems_df.values])
        
        # Create indexes for better join performance
        cur.execute("CREATE INDEX idx_orders_custkey ON orders(o_custkey);")
        cur.execute("CREATE INDEX idx_lineitem_orderkey ON lineitem(l_orderkey);")
        cur.execute("CREATE INDEX idx_lineitem_suppkey ON lineitem(l_suppkey);")
        cur.execute("CREATE INDEX idx_customer_nationkey ON customer(c_nationkey);")
        cur.execute("CREATE INDEX idx_supplier_nationkey ON supplier(s_nationkey);")
        cur.execute("CREATE INDEX idx_nation_regionkey ON nation(n_regionkey);")
        
        connection.commit()
        
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
    Calculate join costs using a REALISTIC but simple cardinality-based model.
    This follows the QUBO paper approach: uses basic statistics and selectivity estimates,
    not a sophisticated optimizer.
    
    WHY THIS IS BETTER THAN POSTGRES ESTIMATES:
    - QAOA optimizes using THESE weights (simple cardinality model)
    - PostgreSQL optimizes using its own sophisticated model (histograms, correlations, etc.)
    - This creates a FAIR COMPARISON: each optimizer uses its own cost model
    - We then compare actual EXECUTION TIME to see which join order is faster
    - If we used PostgreSQL's estimates for QAOA weights, we'd just be copying PostgreSQL's optimizer!
    
    Cost model for join S = {R1, R2, ..., Rk}:
    1. For 2-table joins: cost = |R| × |S| × selectivity
       - If join condition exists: selectivity = 1 / max(|R|, |S|) (assumes FK-like relationship)
       - If no join condition (cross product): selectivity = 1
    2. For multi-table joins: build up incrementally using join tree structure costs
    
    This is more realistic than random weights but doesn't use PostgreSQL's optimizer.
    """
    print("--- Calculating join costs (QUBO-style: cardinality + selectivity) ---")
    
    # Get table cardinalities
    table_sizes = {}
    with conn.cursor() as cur:
        for rel, table_name in relations_map.items():
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cur.fetchone()[0]
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
            # Multi-table join: sum the valid pairwise join costs
            # A join tree never does N-way Cartesian products - it's a series of pairwise joins
            # So we estimate as the sum of valid (connected) pairwise joins
            pairwise_costs = []
            for i in range(len(combo)):
                for j in range(i+1, len(combo)):
                    key = frozenset([combo[i], combo[j]])
                    if key in join_conditions:  # Only include valid joins with join conditions
                        pairwise_cost = costs_map.get(key, 0)
                        if pairwise_cost > 0 and pairwise_cost < 9000000000:  # Exclude cross products
                            pairwise_costs.append(pairwise_cost)
            
            cost = sum(pairwise_costs) if pairwise_costs else 1000
            print(f"  {combo}: {cost} (sum of {len(pairwise_costs)} valid pairwise joins)")
        
        costs_map[frozenset(combo)] = cost
        weights.append(cost)
    
    print(f"\nCalculated {len(weights)} join costs using cardinality-based model")
    print("--- Cost calculation complete. ---\n")
    return costs_map, weights


def get_join_costs_postgres(conn, relations_map, relations, join_conditions, query_parts):
    """
    Calculate join costs by asking PostgreSQL to estimate the cost of each possible join.
    This uses PostgreSQL's sophisticated cost model (I/O, CPU, indexes, statistics)
    to provide accurate weights for QAOA optimization.
    
    WHY THIS IS INTERESTING:
    - QAOA now optimizes using the SAME cost model as PostgreSQL
    - We can compare QAOA's optimization algorithm vs PostgreSQL's dynamic programming
    - If QAOA still loses, it's the optimization algorithm, not the cost model
    - If QAOA wins, it found a better join order than PostgreSQL's greedy search!
    
    For each possible join subset, we construct a query with that specific join order
    and use EXPLAIN to get PostgreSQL's cost estimate without executing the query.
    """
    print("--- Calculating join costs (PostgreSQL EXPLAIN estimates) ---")
    
    costs_map = {}
    weights = []
    sublists = QUBO_formulation.relation_sublists(relations)
    
    # NEW: Find all aliases referenced in the WHERE clause
    where_clause = query_parts.get('where', '')
    all_aliases = relations_map.keys()
    # This regex finds any alias (e.g., 'l', 'o', 'c') followed by a dot
    alias_pattern = r'\b(' + '|'.join(re.escape(alias) for alias in all_aliases) + r')\.'
    where_aliases = set(re.findall(alias_pattern, where_clause))
    
    print("\nQuerying PostgreSQL for join cost estimates:")
    
    with conn.cursor() as cur:
        for combo in sublists:
            if len(combo) < 2:
                continue
            
            # --- FIX: This entire block is new/modified ---
            # For multi-table joins (3+), check connectivity
            if len(combo) > 2:
                
                # NEW CONNECTIVITY CHECK
                if not is_connected(combo, join_conditions):
                    # If the subset is NOT connected, it's an invalid plan.
                    # Assign the massive cross-product penalty.
                    cost = 999999999999  # Massive penalty
                    print(f"  {combo}: {cost:,} (cross product penalty - disconnected subset)")
                
                else:
                    # OLD LOGIC (now in an 'else' block)
                    # The subset is connected, so its cost is the sum of its parts.
                    pairwise_costs = []
                    for i in range(len(combo)):
                        for j in range(i+1, len(combo)):
                            key = frozenset([combo[i], combo[j]])
                            if key in join_conditions: 
                                cost_val = costs_map.get(key, 1000) # Use cost_val to avoid shadowing
                                if cost_val < 999999999:
                                    pairwise_costs.append(cost_val)
                    
                    cost = sum(pairwise_costs) if pairwise_costs else 1000
                    print(f"  {combo}: {cost} (sum of valid pairwise joins)")
                
                costs_map[frozenset(combo)] = cost
                weights.append(cost)
                continue
            # --- END FIX ---
            
            try:
                # Use savepoint for each query so failures don't abort the whole transaction
                cur.execute("SAVEPOINT cost_estimate;")
                
                # Temporarily set join settings to allow estimation
                cur.execute("SET LOCAL join_collapse_limit = 1;")
                cur.execute("SET LOCAL from_collapse_limit = 1;")
                cur.execute("SET LOCAL geqo = off;")
                # Build a simple query that performs this specific join
                if len(combo) == 2:
                    # Two-table join
                    r1, r2 = combo[0], combo[1]
                    t1, t2 = relations_map[r1], relations_map[r2]
                    
                    # Find join condition
                    key = frozenset([r1, r2])
                    if key in join_conditions:
                        join_cond = join_conditions[key]
                        test_query = f"SELECT * FROM {t1} AS {r1} JOIN {t2} AS {r2} ON {join_cond}"
                    else:
                        # No join condition = cross product. Use penalty cost to avoid these
                        print(f"  {combo}: 999999999 (cross product penalty - no join condition)")
                        costs_map[frozenset(combo)] = 999999999
                        weights.append(999999999)
                        cur.execute("RELEASE SAVEPOINT cost_estimate;")
                        continue
                
                # --- FIX (This is the fix from the *previous* turn) ---
                # MODIFIED: Only add the WHERE clause if all its referenced tables are in the current combo
                combo_set = set(combo)
                if where_aliases.issubset(combo_set):
                    test_query += f" {query_parts['where']}"
                # --- FIX END ---
                
                # Use EXPLAIN to get cost estimate
                cur.execute(f"EXPLAIN {test_query}")
                explain_output = "\n".join([row[0] for row in cur.fetchall()])
                
                # Extract cost from EXPLAIN output
                cost_match = re.search(r'cost=[\d.]+\.\.([\d.]+)', explain_output)
                if cost_match:
                    cost = int(float(cost_match.group(1)))
                    costs_map[frozenset(combo)] = cost
                    weights.append(cost)
                    print(f"  {combo}: {cost}")
                    cur.execute("RELEASE SAVEPOINT cost_estimate;")
                else:
                    print(f"  {combo}: Could not parse cost, using fallback")
                    # Fallback to simple estimate
                    cost = 1000 * len(combo)
                    costs_map[frozenset(combo)] = cost
                    weights.append(cost)
                    cur.execute("ROLLBACK TO SAVEPOINT cost_estimate;")
                    
            except Exception as e:
                print(f"  {combo}: Error estimating cost ({e}), using fallback")
                # Rollback to savepoint to continue with other queries
                cur.execute("ROLLBACK TO SAVEPOINT cost_estimate;")
                # Fallback to simple estimate
                cost = 1000 * len(combo)
                costs_map[frozenset(combo)] = cost
                weights.append(cost)
        
        # Commit to clear any remaining savepoints
        conn.commit()
    
    print(f"\nCalculated {len(weights)} join costs using PostgreSQL EXPLAIN")
    print("--- Cost calculation complete. ---\n")
    return costs_map, weights


def calculate_tree_cost(tree, costs_map, join_conditions=None):
    """
    Recursively calculates the total estimated cost of a join tree.
    If join_conditions is provided, validates that each join has a valid path.
    """
    if isinstance(tree, str):
        return 0

    left_subtree, right_subtree = tree[0], tree[1]
    cost = calculate_tree_cost(left_subtree, costs_map, join_conditions) + calculate_tree_cost(right_subtree, costs_map, join_conditions)
    
    def get_relations(subtree):
        if isinstance(subtree, str): return {subtree}
        return get_relations(subtree[0]).union(get_relations(subtree[1]))

    left_relations = get_relations(left_subtree)
    right_relations = get_relations(right_subtree)
    all_relations = left_relations.union(right_relations)
    
    # Check if there's a valid join path between left and right subtrees
    if join_conditions is not None:
        has_direct_join = False
        for l in left_relations:
            for r in right_relations:
                if frozenset([l, r]) in join_conditions:
                    has_direct_join = True
                    break
            if has_direct_join:
                break
        
        # If no direct join condition, this is a cross product - use massive penalty
        if not has_direct_join:
            # Use the cross product penalty (should be 9 billion range)
            cross_product_cost = 999999999999  # Massive penalty to avoid this plan
            print(f"  [CROSS PRODUCT DETECTED] No join condition between {left_relations} and {right_relations} → cost: {cross_product_cost:,}")
            cost += cross_product_cost
            return cost
    
    key = frozenset(all_relations)
    join_cost = costs_map.get(key, 0)
    cost += join_cost
    return cost

def parse_postgres_join_tree(explain_output, relations_map):
    """
    Parse PostgreSQL's join tree structure from EXPLAIN output and convert to our tree format.
    Returns a tree structure compatible with calculate_tree_cost.
    """
    try:
        # Find all table scans with their aliases
        table_pattern = r'(?:Seq Scan|Index Scan|Bitmap Heap Scan|Index Only Scan|Parallel Seq Scan).*?on\s+(\w+)\s+(\w+)'
        matches = re.findall(table_pattern, explain_output, re.IGNORECASE)
        
        # Create reverse mapping: table_name -> alias
        table_to_alias = {table: alias for table, alias in matches}
        
        # For simple left-deep trees, just return the scan order
        # This is a simplified parser - real implementation would need full tree parsing
        scan_order = []
        for line in explain_output.split('\n'):
            scan_match = re.search(r'(?:Seq Scan|Index Scan|Bitmap Heap Scan|Index Only Scan|Parallel Seq Scan).*?on\s+(\w+)\s+(\w+)', line, re.IGNORECASE)
            if scan_match:
                table_name, alias = scan_match.groups()
                if alias not in scan_order:
                    scan_order.append(alias)
        
        # Build left-deep tree from scan order
        if len(scan_order) < 2:
            return None
        
        tree = [scan_order[0], scan_order[1]]
        for i in range(2, len(scan_order)):
            tree = [tree, scan_order[i]]
        
        return tree
    except Exception as e:
        print(f"Warning: Could not parse join tree: {e}")
        return None

def parse_join_order_from_postgres(explain_output):
    """
    Parse join order from PostgreSQL EXPLAIN output.
    PostgreSQL's EXPLAIN shows join order in the plan tree structure.
    The order of table scans in the output reflects the join execution order.
    """
    try:
        # Extract table names (aliases) from the plan in the order they appear
        # Look for patterns like "Seq Scan on customer c" or "Index Scan using..."
        table_pattern = r'(?:Seq Scan|Index Scan|Bitmap Heap Scan|Index Only Scan).*?on\s+(\w+)\s+(\w+)'
        matches = re.findall(table_pattern, explain_output, re.IGNORECASE)
        
        # Build order based on the sequence of table scans (preserving first occurrence)
        order = []
        seen = set()
        
        for table_name, alias in matches:
            if alias not in seen:
                order.append(alias)
                seen.add(alias)
        
        return " -> ".join(order) if order else "Could not determine join order."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"An error occurred parsing join order: {e}"

def get_cost_from_postgres_explain(explain_output):
    """
    Extract total cost from PostgreSQL EXPLAIN output.
    PostgreSQL shows costs as "cost=0.00..123.45 rows=1000"
    
    COST CALCULATION METHOD:
    - We extract the total cost from the root node (first line of EXPLAIN output)
    - The cost is in PostgreSQL's arbitrary units (includes I/O, CPU, etc.)
    - Format: cost=startup_cost..total_cost where total_cost is the full query cost
    - This represents PostgreSQL's estimate of the query execution cost
    
    Returns: Total cost estimate (as integer) or -1 if parsing fails
    """
    try:
        # Find the root node's cost (first cost in the plan - top of the tree)
        # Pattern: "cost=0.00..123.45 rows=1000"
        cost_pattern = r'cost=([\d.]+)\.\.([\d.]+)\s+rows=(\d+)'
        matches = re.findall(cost_pattern, explain_output)
        
        if matches:
            # Get the first (root) node's cost - this is the top-level operation
            first_match = matches[0]
            total_cost = float(first_match[1])  # Second number is total cost
            rows = int(first_match[2])
            print(f"[DEBUG] PostgreSQL cost: {total_cost}, rows: {rows}")
            return int(total_cost)  # Return total cost (not rows)
        
        return -1
    except Exception as e:
        print(f"Warning: Could not parse PostgreSQL cost: {e}")
        return -1

def build_from_clause_forced(tree, relations_map, join_conditions, reverse_qaoa=False):
    """
    Build FROM clause using nested parentheses to force join order.
    With join_collapse_limit=1, from_collapse_limit=1, and geqo=off,
    PostgreSQL will respect the join order specified by explicit JOIN syntax.
    If reverse_qaoa=True, treats the tree as QAOA output which is read right-to-left.
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
        
    left_sql, left_relations = build_from_clause_forced(left_tree, relations_map, join_conditions, reverse_qaoa)
    right_sql, right_relations = build_from_clause_forced(right_tree, relations_map, join_conditions, reverse_qaoa)
    
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
    
    # Use nested parentheses to force join order
    # With join_collapse_limit=1, PostgreSQL will process joins left-to-right as specified
    return f"({left_sql} {join_type} {right_sql} {join_sql})", left_relations.union(right_relations)

def run_benchmark(num_tables, relations_map, relations, join_conditions, query_parts, postgres_conn, weight_method='cardinality', run_seed=None):
    print(f"\n" + "="*50 + f"\n### STARTING {num_tables}-TABLE BENCHMARK ###\n" + "="*50 + "\n")
    
    query = f"{query_parts['select']} {query_parts['from']}"
    if 'where' in query_parts: query += f" {query_parts['where']}"
    query += f" {query_parts['group_by']} {query_parts['order_by']} {query_parts['limit']};"

    # 1. PostgreSQL Benchmark (default optimizer)
    print("--- Running PostgreSQL Benchmark (Default Optimizer) ---")
    with postgres_conn.cursor() as cur:
        # Clear cache to ensure fair comparison
        cur.execute("DISCARD PLANS;")
        
        # Warm up: run query once to populate cache (simulating real-world scenario)
        cur.execute(query)
        cur.fetchall()
        
        # Get EXPLAIN output
        cur.execute(f"EXPLAIN ANALYZE {query}")
        explain_rows = cur.fetchall()
        explain_output = "\n".join([row[0] for row in explain_rows])
        
        print(f"\nPostgreSQL EXPLAIN plan:\n{explain_output}\n")
        
        # Parse join order and cost
        postgres_order = parse_join_order_from_postgres(explain_output)
        postgres_cost = get_cost_from_postgres_explain(explain_output)
        
        # Also parse PostgreSQL's tree structure for pairwise cost comparison
        postgres_tree = parse_postgres_join_tree(explain_output, relations_map)
        
        # Execute query for timing (separate from EXPLAIN ANALYZE to get pure execution time)
        # Run multiple times and take average to reduce variance
        print("Executing PostgreSQL with its optimized plan (3 runs for average)...")
        times = []
        for _ in range(3):
            cur.execute("DISCARD PLANS;")  # Clear plans between runs
            start_time = time.time()
            cur.execute(query)
            cur.fetchall()
            times.append(time.time() - start_time)
        duration_postgres = sum(times) / len(times)
    
    # 2. QUBO Benchmark with QAOA
    print("\n--- Running QUBO Benchmark (QAOA) ---")
    
    # Generate weights based on chosen method
    if weight_method == 'random':
        costs_map, weights = get_join_costs_random(relations, seed=run_seed)
    elif weight_method == 'postgres':
        costs_map, weights = get_join_costs_postgres(postgres_conn, relations_map, relations, join_conditions, query_parts)
    else:  # cardinality
        costs_map, weights = get_join_costs_simple(postgres_conn, relations_map, relations, join_conditions)
    
    # Use the same cost model for comparing both plans (consistent metric)
    # The costs_map from above already contains the selected weight_method
    comparison_costs_map = costs_map
    cost_model_name = {'postgres': 'PostgreSQL Estimates', 'cardinality': 'Cardinality', 'random': 'Random'}[weight_method]
    
    postgres_tree_cost_comparison = None
    if postgres_tree:
        try:
            postgres_tree_cost_comparison = calculate_tree_cost(postgres_tree, comparison_costs_map, join_conditions)
            print(f"[DEBUG] PostgreSQL's tree using {weight_method} model: {postgres_tree} → cost: {postgres_tree_cost_comparison}")
        except Exception as e:
            print(f"[DEBUG] Could not calculate PostgreSQL tree cost: {e}")
    
    # Time the QAOA optimization itself
    print("Running QAOA optimization...")
    qaoa_start_time = time.time()
    
    original_stdout, sys.stdout = sys.stdout, StringIO()
    try:
        optimizer = QUBO_Split_Optimization_func(f"join_log_{num_tables}")
        qubo_tree, _, error_msg = optimizer.finding_opt_jo(relations, weights, SolverType.QAOA)
    except Exception as e:
        qubo_tree, error_msg = None, str(e)
        import traceback
        print(f"QAOA optimization exception: {e}")
        traceback.print_exc()
    finally:
        sys.stdout = original_stdout
    
    qaoa_optimization_time = time.time() - qaoa_start_time
    print(f"QAOA optimization completed in {qaoa_optimization_time * 1000:.2f} ms")
    
    duration_qubo, qubo_cost, qubo_order_str = -1, -1, "No valid tree"
    tree_cost_optimization = -1  # Cost using the optimization model
    tree_cost_comparison = -1   # Cost using the selected weight model (for comparison)
    
    # Note: DP removed for performance - QAOA will always return the best solution found
    
    if qubo_tree and not error_msg:
        qubo_order_str = str(qubo_tree)
        
        # Calculate QAOA cost using the selected weight model
        try:
            tree_cost_optimization = calculate_tree_cost(qubo_tree, costs_map, join_conditions)
            tree_cost_comparison = tree_cost_optimization  # Same as optimization model now
            print(f"QAOA tree: {qubo_tree}")
            print(f"QAOA tree cost ({weight_method} model): {tree_cost_optimization}")
            
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
            tree_cost_optimization = 0
            tree_cost_comparison = 0
        
        # Build forced query with nested parentheses to enforce join order
        plan_cost = -1
        forced_query = ""
        
        try:
            # QAOA trees are read right-to-left, so use reverse_qaoa=True
            # Use nested parentheses to force join order
            forced_from, _ = build_from_clause_forced(qubo_tree, relations_map, join_conditions, reverse_qaoa=True)
            
            forced_query = f"{query_parts['select']} FROM {forced_from}"
            if 'where' in query_parts: forced_query += f" {query_parts['where']}"
            forced_query += f" {query_parts['group_by']} {query_parts['order_by']} {query_parts['limit']};"
            
            print(f"\nForced QAOA query:\n{forced_query}")
            
            # Execute with join order forced using join_collapse_limit
            with postgres_conn.cursor() as cur:
                # Force join order by preventing join reordering
                # join_collapse_limit=1 prevents the planner from reordering explicit JOINs
                # from_collapse_limit=1 prevents the planner from collapsing subqueries
                cur.execute("SET join_collapse_limit = 1;")
                cur.execute("SET from_collapse_limit = 1;")
                # Also disable GEQO (genetic query optimizer) which can reorder joins
                cur.execute("SET geqo = off;")
                
                # Verify settings
                cur.execute("SHOW join_collapse_limit;")
                join_limit = cur.fetchone()[0]
                cur.execute("SHOW from_collapse_limit;")
                from_limit = cur.fetchone()[0]
                cur.execute("SHOW geqo;")
                geqo_setting = cur.fetchone()[0]
                print(f"[DEBUG] join_collapse_limit={join_limit}, from_collapse_limit={from_limit}, geqo={geqo_setting}")
                
                # Get EXPLAIN output for the forced query
                cur.execute(f"EXPLAIN ANALYZE {forced_query}")
                explain_rows = cur.fetchall()
                explain_output = "\n".join([row[0] for row in explain_rows])
                
                print(f"\nQAOA EXPLAIN plan:\n{explain_output}\n")
                
                # Parse the forced plan using PostgreSQL's estimate
                plan_cost = get_cost_from_postgres_explain(explain_output)
                
                # For reporting: use PostgreSQL's estimate of the QAOA plan (apples-to-apples)
                qubo_cost = plan_cost if plan_cost > 0 else tree_cost_comparison
                
                # Show cost comparison using actual EXPLAIN costs
                print(f"\n=== Cost Comparison (PostgreSQL EXPLAIN Costs) ===")
                print(f"  - QAOA's plan (forced):        {plan_cost:.2f}")
                print(f"  - PostgreSQL's plan (default): {postgres_cost:.2f}")
                
                if plan_cost < postgres_cost:
                    improvement = ((postgres_cost - plan_cost) / postgres_cost) * 100
                    print(f"  ✓ QAOA found better join order! ({improvement:.1f}% lower cost)")
                else:
                    penalty = ((plan_cost - postgres_cost) / postgres_cost) * 100
                    print(f"  ✗ PostgreSQL prefers its own plan ({penalty:.1f}% lower cost)")
                
                weight_model_name = {
                    'random': 'random weights (1-100)',
                    'postgres': 'PostgreSQL EXPLAIN estimates (pairwise joins)',
                    'cardinality': 'cardinality estimates'
                }.get(weight_method, weight_method)
                print(f"\n  Note: QAOA optimized using {weight_model_name}.")
                print(f"        Costs shown are PostgreSQL's actual EXPLAIN estimates for full plans.")
                
                # Execute the forced query for timing
                # Run multiple times and take average to reduce variance
                print(f"\nExecuting forced QAOA join order (3 runs for average)...")
                query_times = []
                for _ in range(3):
                    cur.execute("DISCARD PLANS;")  # Clear plans between runs
                    start_time = time.time()
                    cur.execute(forced_query)
                    cur.fetchall()
                    query_times.append(time.time() - start_time)
                query_execution_time = sum(query_times) / len(query_times)
                
                # Total time includes both QAOA optimization and query execution
                duration_qubo = qaoa_optimization_time + query_execution_time
                print(f"Query execution time: {query_execution_time * 1000:.2f} ms")
                print(f"Total time (QAOA + execution): {duration_qubo * 1000:.2f} ms")
                
                # Reset join collapse limits and GEQO
                cur.execute("SET join_collapse_limit = DEFAULT;")
                cur.execute("SET from_collapse_limit = DEFAULT;")
                cur.execute("SET geqo = DEFAULT;")
                
        except Exception as e:
            print(f"Error executing forced QUBO query: {e}")
            import traceback
            traceback.print_exc()
            # Reset join collapse limits and GEQO on error
            try:
                with postgres_conn.cursor() as cur:
                    cur.execute("SET join_collapse_limit = DEFAULT;")
                    cur.execute("SET from_collapse_limit = DEFAULT;")
                    cur.execute("SET geqo = DEFAULT;")
            except:
                    pass
            duration_qubo = -1
            qubo_cost = tree_cost_comparison if tree_cost_comparison > 0 else -1
    else:
        print(f"QUBO optimization failed: {error_msg}")
    
    print(f"\n--- Final Results ({num_tables} Tables) ---")
    # Use actual EXPLAIN costs for final comparison
    cost_str = f"{postgres_cost:.2f}" if postgres_cost >= 0 else "Unknown"
    print(f"PostgreSQL Default | Time: {duration_postgres * 1000:.2f} ms | EXPLAIN Cost: {cost_str:<15} | Order: {postgres_order}")
    if duration_qubo >= 0:
        # qubo_cost is already set to the actual EXPLAIN cost (plan_cost) from line 935
        cost_str = f"{qubo_cost:.2f}" if qubo_cost >= 0 else "Unknown"
        print(f"QUBO QAOA          | Time: {duration_qubo * 1000:.2f} ms | EXPLAIN Cost: {cost_str:<15} | Order: {qubo_order_str}")
    else:
        print(f"QUBO QAOA          | Execution Failed")
    
    return duration_postgres, postgres_cost, duration_qubo, qubo_cost

def plot_results(results_dict):
    """Generates and displays plots comparing benchmark results."""
    for num_tables, results_list in results_dict.items():
        if not results_list: continue

        runs = range(1, len(results_list) + 1)
        postgres_times = [r[0] * 1000 for r in results_list]
        qaoa_times = [r[2] * 1000 for r in results_list if r[2] >= 0]
        postgres_costs = [r[1] for r in results_list if r[1] >= 0]
        qaoa_costs = [r[3] for r in results_list if r[3] >= 0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"Benchmark Comparison for {num_tables}-Table Join ({len(runs)} runs)", fontsize=16)

        # Plot 1: Runtimes
        ax1.plot(runs, postgres_times, 'o-', label='PostgreSQL')
        if qaoa_times: ax1.plot(runs, qaoa_times, 's-', label='QAOA')
        ax1.set_xlabel("Run Number")
        ax1.set_ylabel("Execution Time (ms)")
        ax1.set_title("Runtime Comparison")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Plot 2: Costs
        ax2.plot(runs, postgres_costs, 'o-', label='PostgreSQL')
        if qaoa_costs: ax2.plot(runs, qaoa_costs, 's-', label='QAOA')
        ax2.set_xlabel("Run Number")
        ax2.set_ylabel("PostgreSQL EXPLAIN Cost")
        ax2.set_title("Cost Comparison (EXPLAIN Estimates)")
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Join Order Benchmark using PostgreSQL and QAOA.")
    parser.add_argument('--loop', type=int, default=1, metavar='N', help='Number of times to run each benchmark (default: 1).')
    parser.add_argument('--tables', type=int, default=0, metavar='N', help='Run only N-table benchmark (3, 4, 5, or 6). Default: run all.')
    parser.add_argument('--weights', type=str, default='postgres', choices=['cardinality', 'random', 'postgres'], 
                        help='Weight generation method: "postgres" (PostgreSQL EXPLAIN estimates, default), "cardinality" (simple model), or "random" (1-100)')
    parser.add_argument('--scale', type=float, default=1.0, help='Database scale factor (default: 1.0 = 15k customers, 150k orders, 600k lineitems)')
    parser.add_argument('--host', type=str, default='localhost', help='PostgreSQL host (default: localhost)')
    parser.add_argument('--port', type=int, default=5432, help='PostgreSQL port (default: 5432)')
    parser.add_argument('--user', type=str, default='postgres', help='PostgreSQL user (default: postgres)')
    parser.add_argument('--password', type=str, default='postgres', help='PostgreSQL password (default: postgres)')
    parser.add_argument('--database', type=str, default='db', help='PostgreSQL database (default: db)')
    args = parser.parse_args()
    num_runs = args.loop
    tables_filter = args.tables
    weight_method = args.weights
    scale_factor = args.scale

    # Connect to PostgreSQL
    try:
        postgres_conn = psycopg2.connect(
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
            database=args.database
        )
        print(f"Connected to PostgreSQL at {args.host}:{args.port}")
    except psycopg2.OperationalError as e:
        print(f"Error connecting to PostgreSQL: {e}")
        print("\nMake sure PostgreSQL is running. You can start it with Docker:")
        print("  docker-compose up -d")
        sys.exit(1)

    results_3_tables, results_4_tables, results_5_tables, results_6_tables = [], [], [], []
    
    # Different scenarios to test various table size ratios
    scenarios = ['default', 'many_customers', 'large_orders', 'heavy_lineitems', 'balanced_small']

    try:
        for i in range(num_runs):
            # Cycle through different scenarios for variety
            scenario = scenarios[i % len(scenarios)] if num_runs > 1 else 'default'
            
            print(f"\n{'='*25} RUN {i + 1}/{num_runs} - Scenario: {scenario.upper()} {'='*25}\n")
            
            # Use different seed AND scenario for each iteration
            # This creates fundamentally different optimization problems
            run_seed = 42 + i if num_runs > 1 else None
            setup_database(postgres_conn, scale_factor=scale_factor, seed=run_seed, scenario=scenario)

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
                res3 = run_benchmark(3, relations_map_3, relations_3, join_conditions_3, q_parts_3, postgres_conn, weight_method, run_seed)
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
                res4 = run_benchmark(4, relations_map_4, relations_4, join_conditions_4, q_parts_4, postgres_conn, weight_method, run_seed)
                results_4_tables.append(res4)

            # --- 5-Table Benchmark ---
            if tables_filter == 0 or tables_filter == 5:
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
                res5 = run_benchmark(5, relations_map_5, relations_5, join_conditions_5, q_parts_5, postgres_conn, weight_method, run_seed)
                results_5_tables.append(res5)

            # --- 6-Table Benchmark ---
            if tables_filter == 0 or tables_filter == 6:
                q_parts_6 = {
                    "select": "SELECT r.r_name, n.n_name, s.s_name, c.c_name, SUM(l.l_extendedprice) AS total_revenue",
                    "from": "FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey JOIN customer c ON o.o_custkey = c.c_custkey JOIN supplier s ON l.l_suppkey = s.s_suppkey JOIN nation n ON c.c_nationkey = n.n_nationkey JOIN region r ON n.n_regionkey = r.r_key",
                    "where": "WHERE r.r_name = 'REGION_2'",
                    "group_by": "GROUP BY r.r_name, n.n_name, s.s_name, c.c_name", "order_by": "ORDER BY total_revenue DESC", "limit": "LIMIT 10"
                }
                relations_map_6 = {'l': 'lineitem', 'o': 'orders', 'c': 'customer', 's': 'supplier', 'n': 'nation', 'r': 'region'}
                relations_6 = ['l', 'o', 'c', 's', 'n', 'r']
                join_conditions_6 = {
                    frozenset(['l', 'o']): 'l.l_orderkey = o.o_orderkey',
                    frozenset(['o', 'c']): 'o.o_custkey = c.c_custkey',
                    frozenset(['l', 's']): 'l.l_suppkey = s.s_suppkey',
                    frozenset(['c', 'n']): 'c.c_nationkey = n.n_nationkey',
                    frozenset(['s', 'n']): 's.s_nationkey = n.n_nationkey',
                    frozenset(['n', 'r']): 'n.n_regionkey = r.r_regionkey'
                }
                res6 = run_benchmark(6, relations_map_6, relations_6, join_conditions_6, q_parts_6, postgres_conn, weight_method, run_seed)
                results_6_tables.append(res6)

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
    
    finally:
        postgres_conn.close()
        print("\nPostgreSQL connection closed.")