import sys
print("Python executable:", sys.executable)
print("Python path:", sys.path)
try:
    import duckdb
    print("✓ duckdb imported")
except Exception as e:
    print("✗ duckdb failed:", e)
try:
    from run_join_optimization import setup_database
    print("✓ run_join_optimization imported")
except Exception as e:
    print("✗ run_join_optimization failed:", e)
print("Test complete")

