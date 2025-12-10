# Qooqle

Qooqle is a research project that implements a Quantum Approximate Optimization Algorithm (QAOA) to solve the Join Order Optimization (JOO) problem in relational databases. It translates the problem of choosing an optimal SQL join sequence into a Quadratic Unconstrained Binary Optimization (QUBO) model.

The optimizer compares its found plans against the native PostgreSQL query planner. We make use of the PostgreSQL's EXPLAIN cost estimates as weights to ensure a fair comparison: both the classical and quantum optimizers compete to find the lowest-cost plan based on the same cost model.

---

## Repository Structure
```
.
├── cli.py                      # CLI entry point for join-order benchmarks
├── run_cli.sh                  # Wrapper for macOS psycopg2 library paths
├── run_join_optimization.py    # Core join-order optimizer and QAOA pipeline
├── setup_database.py           # Benchmark database generator and table loader
├── setup_db.sh                 # Shell wrapper for DB setup on macOS
├── sql_parser.py               # SQL SELECT/FROM/WHERE parser
├── query_parser.py             # Query parsing and validation logic
├── SSS_QUBO.py                 # QUBO formulations and solver strategies
├── docker-compose.yml          # PostgreSQL environment
└── README.md                   # Project documentation
```

---

## Installation
Prerequisites:
- Docker
- Python 3.11+ 

Install dependencies:
```bash
pip install psycopg2-binary pandas numpy scipy dimod neal
pip install qiskit qiskit-ibm-runtime qiskit-optimization qiskit-algorithms
```
---

## Database Setup

### Start PostgreSQL
```bash
docker-compose up -d
```

### Populate benchmark tables
```bash
# Standard setup
./setup_db.sh

# Or with specific scaling (e.g., 0.1 for small, 1.0 for standard)
python setup_database.py --scale 1.0
```

This creates tables: `region`, `nation`, `supplier`, `customer`, `orders`, `lineitem`.

---

## QUBO Join Optimization

Join order search is encoded as a QUBO (Quadratic Unconstrained Binary Optimization) problem.

### Key Concepts

- Each possible join subset is represented as a binary variable.
- Constraints penalize incompatible subsets.
- The objective reflects total join plan cost.
- Supported formulations include basic QUBO, reduced universes, and split based decompositions.

### Solvers Implemented

- QAOA via Qiskit
- Exact eigensolver
- Simulated annealing (Neal)
- Hybrid classical-quantum techniques

---

## Benchmarking Framework

The benchmarking engine:

1. Extracts join cost weights based on:
   - Random generation
   - Cardinality estimates
   - PostgreSQL EXPLAIN cost model (primary mode)

2. Runs both QAOA and PostgreSQL optimizers

3. Forces PostgreSQL to use the QAOA join order using nested parentheses

4. Measures:
   - Query execution time
   - EXPLAIN total cost
   - QAOA optimization time
   - Final join trees
   - Relative performance of QAOA vs PostgreSQL

---

## Running Benchmarks
There are two ways to interact with the system: the **Interactive CLI** or the **Headless Script**.

#### Option A: Interactive CLI (`cli.py`)

This option allows for more interactivity and inputting manual SQL queries
```
python cli.py
```


**Features**:
1. View Tables: See row counts and schemas for all tables.
2. Enter Query: Type raw SQL (e.g., `SELECT * FROM customer c JOIN orders o ...`).
3. Run Benchmark: Executes QAOA vs. Postgres on the query you just entered.

### Option B: Headless Script (`run_join_optimization.py`)

Best for running repeated experiments or automated benchmarks. The queries are hardcoded in this option.

#### Example: Run a 5-table benchmark
```bash
python run_join_optimization.py --loop 5 --tables 5 
```

Arguments:
- --loop <N>: Run the benchmark N times to average out runtime noise.
- --tables <N>: Run a specific pre-defined benchmark query joining N tables (supports 3, 4, 5, or 6 tables).
- --scale <float>: Database scale factor (default 1.0)
    
---

## Forcing QAOA Join Order in PostgreSQL

To ensure PostgreSQL executes the QAOA join tree exactly, the system sets:
```sql
SET join_collapse_limit = 1;
SET from_collapse_limit = 1;
SET geqo = off;
```

Then constructs a fully nested FROM clause such as:
```
(((l JOIN o ON ...) JOIN c ON ...) JOIN n ON ...)
```

This disables PostgreSQL's internal reordering and enforces the computed plan.

---

## Example Output
```
=== Cost Comparison (PostgreSQL EXPLAIN Costs) ===
  - QAOA's plan:        18492.33
  - PostgreSQL's plan:  21500.12
  QAOA found a better join order (13.9% lower cost)
```
---
## Status and Limitations
What Works
- Full Pipeline: Parsing SQL, generating QUBO weights, and converting bitstring results back to valid SQL join trees.
- V2 Primitives: The code is updated to use modern `qiskit 1.x` and `qiskit_algorithms` (V2 primitives) with `StatevectorSampler`.
- Split Optimization: Implemented "Bushy" and "Left-Deep" split strategies in `SSS_QUBO.py` to handle larger join spaces by breaking them into smaller sub-problems.
- Benchmarking: Accurate timing and cost comparison using EXPLAIN ANALYZE.

Limitations
- Scaling: The QUBO formulation scales exponentially with the number of tables. Joins larger than 5 tables become computationally expensive for the classical simulation of the quantum state.
---

## Future Work
- Support for 10+ table joins via recursive QUBO decomposition
- Additional classical baseline optimizers
- Benchmark execution on real quantum hardware for larger than 3 tables
- Learned cost models integrated with QUBO formulation

---

## Team
- Abdelrahman Mohammad
- Alexandria Prostko
- Benjamin Wiggenhorn
- Nikhil Sethuram
- Sungwoon Park
- Rami Elsayed

---
