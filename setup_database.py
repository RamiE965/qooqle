"""
Setup script to populate database tables with TPC-H–style test data.

This script intentionally reuses the `setup_database()` function from
`run_join_optimization.py` to guarantee identical data generation logic
between benchmarking runs. Ensuring both scripts use the same data
builder prevents discrepancies in table schemas, row counts, and
randomized content, which is essential for reproducible performance
comparisons.
"""

import psycopg2
import sys
from run_join_optimization import setup_database

def main():
    """Connect to PostgreSQL and populate benchmark tables with test data."""
    print("Setting up database tables with test data...")
    print(
        "This will create and populate the following tables:\n"
        "  region, nation, supplier, customer, orders, lineitem\n"
    )
    
    try:
        # Establish the database connection.
        # This script assumes a local dev environment or a Dockerized DB.
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            user='postgres',
            password='postgres',
            database='db'
        )
        print("✓ Connected to PostgreSQL")
    except psycopg2.OperationalError as e:
        # Connection failures are common when using Docker and forgetting to start services.
        print(f"✗ Error connecting to PostgreSQL: {e}")
        print("\nMake sure PostgreSQL is running. You can start it with Docker:")
        print("  docker-compose up -d")
        sys.exit(1)
    
    try:
        # Use the shared TPC-H data generator from the join-optimization runner.
        # This keeps benchmarks consistent and avoids accidental drift in schemas
        # or row-generation logic between scripts.
        #
        # scale_factor=1.0 → small but realistic dataset
        # seed=42 → deterministic output for reproducible testing
        # scenario='default' → baseline data generation rules
        setup_database(conn, scale_factor=1.0, seed=42, scenario='default')
        
        # After bulk loading, run ANALYZE to refresh PostgreSQL’s statistics.
        # Without this step, EXPLAIN plans may be inaccurate because the planner
        # uses stale or empty histograms, leading to misleading benchmark results.
        print("\nUpdating PostgreSQL statistics (ANALYZE)...")
        with conn.cursor() as cur:
            cur.execute("ANALYZE customer, orders, lineitem, nation, region, supplier;")
            conn.commit()
        print("✓ Statistics updated")
        
        print("\n✓ Database setup complete!")
        print("Tables are now populated and ready for benchmarking.")
        print("\nYou can now use the CLI to run queries on these tables.")
        
    except Exception as e:
        # Catch any errors from table creation or data generation.
        print(f"✗ Error setting up database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Ensure DB connection is always closed.
        conn.close()

if __name__ == "__main__":
    main()
