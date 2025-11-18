"""
Setup script to populate database tables.
Uses the same setup_database() function from run_join_optimization.py
to ensure consistency.
"""

import psycopg2
import sys
from run_join_optimization import setup_database

def main():
    """Populate database tables with test data"""
    print("Setting up database tables with test data...")
    print("This will create and populate: region, nation, supplier, customer, orders, lineitem\n")
    
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            user='postgres',
            password='postgres',
            database='db'
        )
        print("✓ Connected to PostgreSQL")
    except psycopg2.OperationalError as e:
        print(f"✗ Error connecting to PostgreSQL: {e}")
        print("\nMake sure PostgreSQL is running. You can start it with Docker:")
        print("  docker-compose up -d")
        sys.exit(1)
    
    try:
        # Use the same setup_database function as run_join_optimization.py
        # This ensures consistency - same data generation logic
        # Using scale_factor=1.0 to match default in run_join_optimization.py
        setup_database(conn, scale_factor=1.0, seed=42, scenario='default')
        
        # Update PostgreSQL statistics for accurate EXPLAIN costs
        print("\nUpdating PostgreSQL statistics (ANALYZE)...")
        with conn.cursor() as cur:
            cur.execute("ANALYZE customer, orders, lineitem, nation, region, supplier;")
            conn.commit()
        print("✓ Statistics updated")
        
        print("\n✓ Database setup complete!")
        print("Tables are now populated and ready for benchmarking.")
        print("\nYou can now use the CLI to run queries on these tables.")
        
    except Exception as e:
        print(f"✗ Error setting up database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    main()



