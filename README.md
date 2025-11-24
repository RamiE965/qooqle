# Qooqle

Qooqle is a research project for UW–Madison CS 620 Capstone, developed under mentorship from Google.  
It explores whether quantum-inspired optimization methods—specifically QAOA (Quantum Approximate Optimization Algorithm)—can improve join order selection in relational database query planning.

The repository includes a full pipeline for benchmarking quantum-based join ordering against PostgreSQL's optimizer.

---

## Project Overview

Qooqle implements an end-to-end join-order optimization system:

1. Synthetic benchmark database generator for PostgreSQL  
2. QUBO formulation of the join-order search problem  
3. QAOA-based and classical solvers for join optimization  
4. Benchmarking against PostgreSQL using EXPLAIN and actual runtimes  
5. Interactive CLI for running experiments and visualizing results  

The project evaluates whether quantum-inspired techniques can outperform a sophisticated classical optimizer by exploring join order search spaces efficiently.

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
├── get_result.py               # IBM Quantum runtime job fetch utility
├── SSS_QUBO.py                 # QUBO formulations and solver strategies
├── docker-compose.yml          # PostgreSQL environment
└── README.md                   # Project documentation
```

---

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

Key libraries include `psycopg2`, `numpy`, `qiskit`, `dimod`, `neal`, `pandas`, and `scipy`.

---

## Database Setup

### Start PostgreSQL
```bash
docker-compose up -d
```

### Populate benchmark tables
```bash
./setup_db.sh --scale 1.0
```

This creates and loads the classic benchmarking tables:

- `region`
- `nation`
- `supplier`
- `customer`
- `orders`
- `lineitem`

The system supports multiple data distribution scenarios such as:

- `default`
- `many_customers`
- `large_orders`
- `heavy_lineitems`
- `balanced_small`

---

## QUBO Join Optimization

Join order search is encoded as a QUBO (Quadratic Unconstrained Binary Optimization) problem.

### Key Concepts

- Each possible join subset is represented as a binary variable.
- Constraints penalize incompatible subsets.
- The objective reflects total join plan cost.
- Supported formulations include basic QUBO, reduced universes, and split-based decompositions.

### Solvers Implemented

- QAOA via Qiskit
- Exact eigensolver
- Simulated annealing (Neal)
- Hybrid classical–quantum techniques

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

### Example: Run a 4-table benchmark
```bash
./run_cli.sh --tables 4 --weights postgres
```

### Multiple iterations
```bash
./run_cli.sh --loop 10 --weights cardinality
```

### Random-weight benchmarks
```bash
./run_cli.sh --weights random
```

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

## Future Work

- Support for 10+ table joins via recursive QUBO decomposition
- Additional classical baseline optimizers
- Benchmark execution on real quantum hardware
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