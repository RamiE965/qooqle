"""
Interactive CLI for Join Optimization Benchmarking

Allows users to:
- Connect to existing PostgreSQL database
- Auto-detect schema from existing tables
- Enter SQL queries interactively
- Run benchmarks comparing PostgreSQL vs QAOA (using PostgreSQL EXPLAIN estimates)
"""

import sys
import os
import psycopg2
from typing import Optional, Dict, List, Any, Tuple, Set

# Import parsers
from sql_parser import get_tables_in_schema
from query_parser import parse_sql_query, validate_query

# Lazy import of run_benchmark to avoid importing Qiskit dependencies until needed
_run_benchmark = None

def get_run_benchmark():
    """Lazy import of run_benchmark to avoid loading Qiskit dependencies at startup"""
    global _run_benchmark
    if _run_benchmark is None:
        from run_join_optimization import run_benchmark
        _run_benchmark = run_benchmark
    return _run_benchmark


class CLISession:
    """Manages CLI session state"""
    
    def __init__(self):
        self.conn: Optional[Any] = None
        self.available_tables: Set[str] = set()
        self.current_query: Optional[str] = None
        self.parsed_query: Optional[Dict] = None
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'user': 'postgres',
            'password': 'postgres',
            'database': 'db'
        }
    
    def connect_db(self) -> bool:
        """Connect to PostgreSQL database and auto-detect schema"""
        try:
            self.conn = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database']
            )
            print(f"✓ Connected to PostgreSQL at {self.db_config['host']}:{self.db_config['port']}")
            
            # Auto-detect schema
            self.available_tables = get_tables_in_schema(self.conn)
            if self.available_tables:
                print(f"✓ Detected {len(self.available_tables)} table(s) in database")
            else:
                print("⚠ No tables found in database")
            
            return True
        except psycopg2.OperationalError as e:
            print(f"✗ Error connecting to PostgreSQL: {e}")
            print("\nMake sure PostgreSQL is running. You can start it with Docker:")
            print("  docker-compose up -d")
            return False
    
    def disconnect_db(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None


def display_menu():
    """Display main menu options"""
    print("\n" + "="*60)
    print("  Join Optimization Benchmark CLI")
    print("="*60)
    print("  1. View Existing Tables")
    print("  2. Enter SQL Query")
    print("  3. Run Benchmark")
    print("  4. View Connection Settings")
    print("  5. Exit")
    print("="*60)


def view_tables_menu(session: CLISession) -> bool:
    """Display existing tables in the database"""
    if not session.conn:
        print("✗ Please connect to database first")
        return False
    
    # Refresh table list
    session.available_tables = get_tables_in_schema(session.conn)
    
    if not session.available_tables:
        print("\n--- Existing Tables ---")
        print("No tables found in the database.")
        return True
    
    print("\n--- Existing Tables ---")
    print(f"Found {len(session.available_tables)} table(s):\n")
    
    # Get table information
    with session.conn.cursor() as cur:
        for table_name in sorted(session.available_tables):
            # Get row count
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cur.fetchone()[0]
            
            # Get column names
            cur.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}' 
                ORDER BY ordinal_position
            """)
            columns = cur.fetchall()
            
            print(f"  {table_name} ({row_count} rows)")
            if columns:
                col_info = ", ".join([f"{col[0]} ({col[1]})" for col in columns[:5]])
                if len(columns) > 5:
                    col_info += f", ... ({len(columns) - 5} more)"
                print(f"    Columns: {col_info}")
            print()
    
    return True


def enter_query_menu(session: CLISession) -> bool:
    """Handle interactive SQL query input"""
    if not session.conn:
        print("✗ Please connect to database first")
        return False
    
    if not session.available_tables:
        print("⚠ Warning: No tables detected in database.")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return False
    
    print("\n--- Enter SQL Query ---")
    print("Enter your SQL SELECT query (end with a blank line or ';' on its own line):")
    print("(You can enter multiple lines)")
    
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == '' or line.strip() == ';':
                break
            lines.append(line)
        except EOFError:
            break
    
    query = ' '.join(lines).strip()
    
    if not query:
        print("✗ No query provided")
        return False
    
    # Remove trailing semicolon if present
    if query.endswith(';'):
        query = query[:-1].strip()
    
    try:
        # Parse query
        print("\nParsing query...")
        parsed_query = parse_sql_query(query)
        
        # Validate query tables exist
        is_valid, error_msg = validate_query(parsed_query, session.available_tables)
        
        if not is_valid:
            print(f"✗ Query validation failed: {error_msg}")
            return False
        
        # Check if query has joins (at least 2 tables)
        relations = parsed_query.get('relations', [])
        if len(relations) < 2:
            print("✗ Query must involve at least 2 tables for join optimization")
            return False
        
        if len(relations) > 6:
            print(f"⚠ Warning: Query involves {len(relations)} tables. Maximum supported is 6 tables.")
            response = input("Continue anyway? (y/n): ").strip().lower()
            if response != 'y':
                return False
        
        # Store query
        session.current_query = query
        session.parsed_query = parsed_query
        
        print(f"✓ Query parsed successfully")
        print(f"  Tables: {', '.join(relations)}")
        print(f"  Join conditions: {len(parsed_query.get('join_conditions', {}))}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error parsing query: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_postgres_benchmark_only(postgres_conn, num_tables, relations_map, relations, join_conditions, query_parts):
    """
    Run only PostgreSQL benchmark (default optimizer).
    """
    from run_join_optimization import (
        parse_postgres_join_tree,
        get_cost_from_postgres_explain,
        parse_join_order_from_postgres
    )
    import time
    
    print(f"\n" + "="*50 + f"\n### STARTING {num_tables}-TABLE BENCHMARK (PostgreSQL Only) ###\n" + "="*50 + "\n")
    
    query = f"{query_parts['select']} {query_parts['from']}"
    if 'where' in query_parts: query += f" {query_parts['where']}"
    query += f" {query_parts['group_by']} {query_parts['order_by']} {query_parts['limit']};"

    print("--- Running PostgreSQL Benchmark (Default Optimizer) ---")
    with postgres_conn.cursor() as cur:
        cur.execute("DISCARD PLANS;")
        cur.execute(query)
        cur.fetchall()
        
        cur.execute(f"EXPLAIN ANALYZE {query}")
        explain_rows = cur.fetchall()
        explain_output = "\n".join([row[0] for row in explain_rows])
        
        print(f"\nPostgreSQL EXPLAIN plan:\n{explain_output}\n")
        
        postgres_order = parse_join_order_from_postgres(explain_output)
        postgres_cost = get_cost_from_postgres_explain(explain_output)
        postgres_tree = parse_postgres_join_tree(explain_output, relations_map)
        
        print("Executing PostgreSQL with its optimized plan (3 runs for average)...")
        times = []
        for _ in range(3):
            cur.execute("DISCARD PLANS;")
            start_time = time.time()
            cur.execute(query)
            cur.fetchall()
            times.append(time.time() - start_time)
        duration_postgres = sum(times) / len(times)
    
    print(f"\n--- Final Results ({num_tables} Tables) ---")
    cost_str = f"{postgres_cost:.2f}" if postgres_cost >= 0 else "Unknown"
    print(f"PostgreSQL Default | Time: {duration_postgres * 1000:.2f} ms | EXPLAIN Cost: {cost_str:<15} | Order: {postgres_order}")
    
    return duration_postgres, postgres_cost


def run_qaoa_benchmark_only(postgres_conn, num_tables, relations_map, relations, join_conditions, query_parts):
    """
    Run only QAOA benchmark.
    """
    from run_join_optimization import (
        get_join_costs_postgres, 
        get_cost_from_postgres_explain,
        calculate_tree_cost,
        build_from_clause_forced,
        QUBO_Split_Optimization_func,
        SolverType,
        QUBO_formulation
    )
    import time
    import sys
    from io import StringIO
    
    print(f"\n" + "="*50 + f"\n### STARTING {num_tables}-TABLE BENCHMARK (QAOA Only) ###\n" + "="*50 + "\n")
    
    query = f"{query_parts['select']} {query_parts['from']}"
    if 'where' in query_parts: query += f" {query_parts['where']}"
    query += f" {query_parts['group_by']} {query_parts['order_by']} {query_parts['limit']};"

    print("--- Running QUBO Benchmark (QAOA) ---")
    
    # Calculate costs using PostgreSQL EXPLAIN for QAOA optimization
    print("Calculating join costs using PostgreSQL EXPLAIN estimates...")
    costs_map, weights = get_join_costs_postgres(postgres_conn, relations_map, relations, join_conditions, query_parts)
    
    # Use actual PostgreSQL EXPLAIN costs as weights for fair comparison
    print(f"Using PostgreSQL EXPLAIN costs as weights for QAOA optimization ({len(weights)} weights)")
    print(f"  Weight range: {min(weights):,} to {max(weights):,}")
    
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
    
    if qubo_tree and not error_msg:
        qubo_order_str = str(qubo_tree)
        
        # Calculate QAOA cost using PostgreSQL's cost model for comparison
        try:
            tree_cost = calculate_tree_cost(qubo_tree, costs_map, join_conditions)
            print(f"QAOA tree: {qubo_tree}")
            print(f"QAOA tree cost (PostgreSQL EXPLAIN model): {tree_cost}")
        except Exception as e:
            print(f"Error calculating tree cost: {e}")
            tree_cost = 0
        
        # Build forced query with nested parentheses to enforce join order
        try:
            forced_from, _ = build_from_clause_forced(qubo_tree, relations_map, join_conditions, reverse_qaoa=True)
            
            forced_query = f"{query_parts['select']} FROM {forced_from}"
            if 'where' in query_parts: forced_query += f" {query_parts['where']}"
            forced_query += f" {query_parts['group_by']} {query_parts['order_by']} {query_parts['limit']};"
            
            print(f"\nForced QAOA query:\n{forced_query}")
            
            with postgres_conn.cursor() as cur:
                cur.execute("SET join_collapse_limit = 1;")
                cur.execute("SET from_collapse_limit = 1;")
                cur.execute("SET geqo = off;")
                
                cur.execute(f"EXPLAIN ANALYZE {forced_query}")
                explain_rows = cur.fetchall()
                explain_output = "\n".join([row[0] for row in explain_rows])
                
                print(f"\nQAOA EXPLAIN plan:\n{explain_output}\n")
                
                plan_cost = get_cost_from_postgres_explain(explain_output)
                qubo_cost = plan_cost if plan_cost > 0 else tree_cost
                
                print(f"\n  Note: QAOA optimized using PostgreSQL EXPLAIN cost estimates as weights.")
                print(f"        Costs shown are PostgreSQL's actual EXPLAIN estimates for full plans.")
                print(f"        Both optimizers use the same cost model for fair comparison.")
                
                print(f"\nExecuting forced QAOA join order (3 runs for average)...")
                query_times = []
                for _ in range(3):
                    cur.execute("DISCARD PLANS;")
                    start_time = time.time()
                    cur.execute(forced_query)
                    cur.fetchall()
                    query_times.append(time.time() - start_time)
                query_execution_time = sum(query_times) / len(query_times)
                
                duration_qubo = qaoa_optimization_time + query_execution_time
                print(f"Query execution time: {query_execution_time * 1000:.2f} ms")
                print(f"Total time (QAOA + execution): {duration_qubo * 1000:.2f} ms")
                
                cur.execute("SET join_collapse_limit = DEFAULT;")
                cur.execute("SET from_collapse_limit = DEFAULT;")
                cur.execute("SET geqo = DEFAULT;")
                
        except Exception as e:
            print(f"Error executing forced QUBO query: {e}")
            import traceback
            traceback.print_exc()
            try:
                with postgres_conn.cursor() as cur:
                    cur.execute("SET join_collapse_limit = DEFAULT;")
                    cur.execute("SET from_collapse_limit = DEFAULT;")
                    cur.execute("SET geqo = DEFAULT;")
            except:
                pass
            duration_qubo = -1
            qubo_cost = tree_cost if tree_cost > 0 else -1
    else:
        print(f"QUBO optimization failed: {error_msg}")
    
    print(f"\n--- Final Results ({num_tables} Tables) ---")
    if duration_qubo >= 0:
        cost_str = f"{qubo_cost:.2f}" if qubo_cost >= 0 else "Unknown"
        print(f"QUBO QAOA          | Time: {duration_qubo * 1000:.2f} ms | EXPLAIN Cost: {cost_str:<15} | Order: {qubo_order_str}")
    else:
        print(f"QUBO QAOA          | Execution Failed")
    
    return duration_qubo, qubo_cost


def run_benchmark_with_weights_of_one(postgres_conn, num_tables, relations_map, relations, join_conditions, query_parts):
    """
    Run benchmark using PostgreSQL EXPLAIN estimates for cost comparison.
    Uses actual PostgreSQL EXPLAIN costs as weights for QAOA optimizer for fair comparison.
    Runs both PostgreSQL and QAOA benchmarks.
    """
    from run_join_optimization import (
        get_join_costs_postgres, 
        parse_postgres_join_tree,
        get_cost_from_postgres_explain,
        parse_join_order_from_postgres,
        calculate_tree_cost,
        build_from_clause_forced,
        QUBO_Split_Optimization_func,
        SolverType,
        QUBO_formulation
    )
    import time
    import sys
    from io import StringIO
    
    print(f"\n" + "="*50 + f"\n### STARTING {num_tables}-TABLE BENCHMARK (Both) ###\n" + "="*50 + "\n")
    
    query = f"{query_parts['select']} {query_parts['from']}"
    if 'where' in query_parts: query += f" {query_parts['where']}"
    query += f" {query_parts['group_by']} {query_parts['order_by']} {query_parts['limit']};"

    # 1. PostgreSQL Benchmark (default optimizer)
    print("--- Running PostgreSQL Benchmark (Default Optimizer) ---")
    with postgres_conn.cursor() as cur:
        cur.execute("DISCARD PLANS;")
        cur.execute(query)
        cur.fetchall()
        
        cur.execute(f"EXPLAIN ANALYZE {query}")
        explain_rows = cur.fetchall()
        explain_output = "\n".join([row[0] for row in explain_rows])
        
        print(f"\nPostgreSQL EXPLAIN plan:\n{explain_output}\n")
        
        postgres_order = parse_join_order_from_postgres(explain_output)
        postgres_cost = get_cost_from_postgres_explain(explain_output)
        postgres_tree = parse_postgres_join_tree(explain_output, relations_map)
        
        print("Executing PostgreSQL with its optimized plan (3 runs for average)...")
        times = []
        for _ in range(3):
            cur.execute("DISCARD PLANS;")
            start_time = time.time()
            cur.execute(query)
            cur.fetchall()
            times.append(time.time() - start_time)
        duration_postgres = sum(times) / len(times)
    
    # 2. QUBO Benchmark with QAOA
    print("\n--- Running QUBO Benchmark (QAOA) ---")
    
    # Calculate costs using PostgreSQL EXPLAIN for QAOA optimization
    print("Calculating join costs using PostgreSQL EXPLAIN estimates...")
    costs_map, weights = get_join_costs_postgres(postgres_conn, relations_map, relations, join_conditions, query_parts)
    
    # Use actual PostgreSQL EXPLAIN costs as weights for fair comparison
    print(f"Using PostgreSQL EXPLAIN costs as weights for QAOA optimization ({len(weights)} weights)")
    print(f"  Weight range: {min(weights):,} to {max(weights):,}")
    
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
    
    if qubo_tree and not error_msg:
        qubo_order_str = str(qubo_tree)
        
        # Calculate QAOA cost using PostgreSQL's cost model for comparison
        try:
            tree_cost = calculate_tree_cost(qubo_tree, costs_map, join_conditions)
            print(f"QAOA tree: {qubo_tree}")
            print(f"QAOA tree cost (PostgreSQL EXPLAIN model): {tree_cost}")
        except Exception as e:
            print(f"Error calculating tree cost: {e}")
            tree_cost = 0
        
        # Build forced query with nested parentheses to enforce join order
        try:
            forced_from, _ = build_from_clause_forced(qubo_tree, relations_map, join_conditions, reverse_qaoa=True)
            
            forced_query = f"{query_parts['select']} FROM {forced_from}"
            if 'where' in query_parts: forced_query += f" {query_parts['where']}"
            forced_query += f" {query_parts['group_by']} {query_parts['order_by']} {query_parts['limit']};"
            
            print(f"\nForced QAOA query:\n{forced_query}")
            
            with postgres_conn.cursor() as cur:
                cur.execute("SET join_collapse_limit = 1;")
                cur.execute("SET from_collapse_limit = 1;")
                cur.execute("SET geqo = off;")
                
                cur.execute(f"EXPLAIN ANALYZE {forced_query}")
                explain_rows = cur.fetchall()
                explain_output = "\n".join([row[0] for row in explain_rows])
                
                print(f"\nQAOA EXPLAIN plan:\n{explain_output}\n")
                
                plan_cost = get_cost_from_postgres_explain(explain_output)
                qubo_cost = plan_cost if plan_cost > 0 else tree_cost
                
                print(f"\n=== Cost Comparison (PostgreSQL EXPLAIN Costs) ===")
                print(f"  - QAOA's plan (forced):        {plan_cost:.2f}")
                print(f"  - PostgreSQL's plan (default): {postgres_cost:.2f}")
                
                if plan_cost < postgres_cost:
                    improvement = ((postgres_cost - plan_cost) / postgres_cost) * 100
                    print(f"  ✓ QAOA found better join order! ({improvement:.1f}% lower cost)")
                else:
                    penalty = ((plan_cost - postgres_cost) / postgres_cost) * 100
                    print(f"  ✗ PostgreSQL prefers its own plan ({penalty:.1f}% lower cost)")
                
                print(f"\n  Note: QAOA optimized using PostgreSQL EXPLAIN cost estimates as weights.")
                print(f"        Costs shown are PostgreSQL's actual EXPLAIN estimates for full plans.")
                print(f"        Both optimizers use the same cost model for fair comparison.")
                
                print(f"\nExecuting forced QAOA join order (3 runs for average)...")
                query_times = []
                for _ in range(3):
                    cur.execute("DISCARD PLANS;")
                    start_time = time.time()
                    cur.execute(forced_query)
                    cur.fetchall()
                    query_times.append(time.time() - start_time)
                query_execution_time = sum(query_times) / len(query_times)
                
                duration_qubo = qaoa_optimization_time + query_execution_time
                print(f"Query execution time: {query_execution_time * 1000:.2f} ms")
                print(f"Total time (QAOA + execution): {duration_qubo * 1000:.2f} ms")
                
                cur.execute("SET join_collapse_limit = DEFAULT;")
                cur.execute("SET from_collapse_limit = DEFAULT;")
                cur.execute("SET geqo = DEFAULT;")
                
        except Exception as e:
            print(f"Error executing forced QUBO query: {e}")
            import traceback
            traceback.print_exc()
            try:
                with postgres_conn.cursor() as cur:
                    cur.execute("SET join_collapse_limit = DEFAULT;")
                    cur.execute("SET from_collapse_limit = DEFAULT;")
                    cur.execute("SET geqo = DEFAULT;")
            except:
                pass
            duration_qubo = -1
            qubo_cost = tree_cost if tree_cost > 0 else -1
    else:
        print(f"QUBO optimization failed: {error_msg}")
    
    print(f"\n--- Final Results ({num_tables} Tables) ---")
    cost_str = f"{postgres_cost:.2f}" if postgres_cost >= 0 else "Unknown"
    print(f"PostgreSQL Default | Time: {duration_postgres * 1000:.2f} ms | EXPLAIN Cost: {cost_str:<15} | Order: {postgres_order}")
    if duration_qubo >= 0:
        cost_str = f"{qubo_cost:.2f}" if qubo_cost >= 0 else "Unknown"
        print(f"QUBO QAOA          | Time: {duration_qubo * 1000:.2f} ms | EXPLAIN Cost: {cost_str:<15} | Order: {qubo_order_str}")
    else:
        print(f"QUBO QAOA          | Execution Failed")
    
    return duration_postgres, postgres_cost, duration_qubo, qubo_cost


def display_benchmark_submenu():
    """Display benchmark submenu options"""
    print("\n" + "="*60)
    print("  Benchmark Options")
    print("="*60)
    print("  1. Run QAOA only")
    print("  2. Run PostgreSQL only")
    print("  3. Run both (QAOA + PostgreSQL)")
    print("  4. Back to main menu")
    print("="*60)


def run_benchmark_menu(session: CLISession) -> bool:
    """Run benchmark with current query (always uses PostgreSQL EXPLAIN estimates)"""
    if not session.conn:
        print("✗ Please connect to database first")
        return False
    
    if not session.parsed_query:
        print("✗ No query loaded. Please enter a query first (option 2)")
        return False
    
    parsed_query = session.parsed_query
    relations = parsed_query.get('relations', [])
    num_tables = len(relations)
    
    if num_tables < 2:
        print("✗ Query must involve at least 2 tables")
        return False
    
    if num_tables > 6:
        print(f"⚠ Warning: Query involves {num_tables} tables. Maximum supported is 6 tables.")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return False
    
    # Display benchmark submenu
    while True:
        display_benchmark_submenu()
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '4':
            return True  # Back to main menu
        
        try:
            # Extract data for benchmark
            relations_map = parsed_query['relations_map']
            relations_list = parsed_query['relations']
            join_conditions = parsed_query['join_conditions']
            query_parts = parsed_query['query_parts']
            
            # Ensure all required query_parts keys exist
            required_keys = ['select', 'from', 'where', 'group_by', 'order_by', 'limit']
            for key in required_keys:
                if key not in query_parts:
                    query_parts[key] = ''
            
            if choice == '1':
                # Run QAOA only
                print(f"\n{'='*60}")
                print(f"Running QAOA benchmark for {num_tables}-table join...")
                print(f"Using PostgreSQL EXPLAIN estimates for cost comparison")
                print(f"Using PostgreSQL EXPLAIN costs as weights for QAOA optimization")
                print(f"{'='*60}\n")
                
                run_qaoa_benchmark_only(
                    postgres_conn=session.conn,
                    num_tables=num_tables,
                    relations_map=relations_map,
                    relations=relations_list,
                    join_conditions=join_conditions,
                    query_parts=query_parts
                )
                return True
                
            elif choice == '2':
                # Run PostgreSQL only
                print(f"\n{'='*60}")
                print(f"Running PostgreSQL benchmark for {num_tables}-table join...")
                print(f"{'='*60}\n")
                
                run_postgres_benchmark_only(
                    postgres_conn=session.conn,
                    num_tables=num_tables,
                    relations_map=relations_map,
                    relations=relations_list,
                    join_conditions=join_conditions,
                    query_parts=query_parts
                )
                return True
                
            elif choice == '3':
                # Run both
                print(f"\n{'='*60}")
                print(f"Running benchmark for {num_tables}-table join...")
                print(f"Using PostgreSQL EXPLAIN estimates for cost comparison")
                print(f"Using PostgreSQL EXPLAIN costs as weights for QAOA optimization")
                print(f"{'='*60}\n")
                
                run_benchmark_with_weights_of_one(
                    postgres_conn=session.conn,
                    num_tables=num_tables,
                    relations_map=relations_map,
                    relations=relations_list,
                    join_conditions=join_conditions,
                    query_parts=query_parts
                )
                return True
            else:
                print("✗ Invalid choice. Please enter 1-4.")
                continue
                
        except Exception as e:
            print(f"✗ Error running benchmark: {e}")
            import traceback
            traceback.print_exc()
            return False


def view_connection_menu(session: CLISession):
    """Display current connection settings"""
    print("\n--- Connection Settings ---")
    print(f"  Host: {session.db_config['host']}")
    print(f"  Port: {session.db_config['port']}")
    print(f"  User: {session.db_config['user']}")
    print(f"  Database: {session.db_config['database']}")
    print(f"  Connected: {'Yes' if session.conn else 'No'}")
    print(f"  Tables detected: {len(session.available_tables)}")
    if session.current_query:
        print(f"  Current query: {len(session.current_query)} characters")
    else:
        print(f"  Current query: None")


def main():
    """Main CLI entry point"""
    session = CLISession()
    
    # Connect to database
    print("Connecting to PostgreSQL...")
    if not session.connect_db():
        print("Failed to connect. Exiting.")
        sys.exit(1)
    
    try:
        while True:
            display_menu()
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                view_tables_menu(session)
            elif choice == '2':
                enter_query_menu(session)
            elif choice == '3':
                run_benchmark_menu(session)
            elif choice == '4':
                view_connection_menu(session)
            elif choice == '5':
                print("\nExiting...")
                break
            else:
                print("✗ Invalid choice. Please enter 1-5.")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.disconnect_db()
        print("Database connection closed.")


if __name__ == "__main__":
    main()
