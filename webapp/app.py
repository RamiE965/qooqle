import os
import time
from flask import Flask, render_template, request, flash
import traceback
import math
from itertools import combinations
import sys
from io import StringIO

try:
    from qubo_optimizer import (
        QUBO_Split_Optimization_func,
        QUBO_formulation,
        Helping_functions,
        SolverType,
        Experiments_class,
        QISKIT_AVAILABLE
    )
except ImportError as e:
    print(f"Error importing from qubo_optimizer.py: {e}")
    print("Make sure qubo_optimizer.py is in the same directory as app.py")
    exit()

app = Flask(__name__)
app.secret_key = os.urandom(24)

def calculate_costs_from_stats_benchmark_logic(relations, table_stats, join_key_map):
    all_combinations = QUBO_formulation.relation_sublists(relations)
    weights = []
    costs_map = {}

    table_sizes = {rel: stats.get('rows', 1) for rel, stats in table_stats.items()}
    for rel, size in table_sizes.items():
        if size <= 0: table_sizes[rel] = 1

    print("Table Sizes Used:")
    for rel, size in table_sizes.items():
         print(f"  {rel}: {size}\n")

    print("Calculating Join Costs:")
    for subset in all_combinations:
        if len(subset) < 2:
            continue

        subset_frozenset = frozenset(subset)
        cost = float('inf')
        is_joinable = False

        if len(subset) > 1:
            nodes_to_connect = set(subset)
            start_node = subset[0]
            connected_component = {start_node}
            nodes_to_connect.remove(start_node)
            queue = [start_node]
            processed_nodes_in_queue = {start_node}
            while queue:
                current_node = queue.pop(0)
                neighbors_to_add = []
                for potential_neighbor in list(nodes_to_connect):
                    if join_key_map.get((current_node, potential_neighbor)) or join_key_map.get((potential_neighbor, current_node)):
                         if potential_neighbor not in connected_component:
                              neighbors_to_add.append(potential_neighbor)
                for neighbor in neighbors_to_add:
                    if neighbor in nodes_to_connect:
                        connected_component.add(neighbor)
                        nodes_to_connect.remove(neighbor)
                        if neighbor not in processed_nodes_in_queue:
                            queue.append(neighbor)
                            processed_nodes_in_queue.add(neighbor)
            if not nodes_to_connect:
                is_joinable = True

        if not is_joinable:
             cost = float('inf')
             print(f"  {subset}: Unjoinable (Connectivity Check Failed)")
        elif len(subset) == 2:
            r1, r2 = subset[0], subset[1]
            size1, size2 = table_sizes.get(r1, 1), table_sizes.get(r2, 1)
            has_join_condition = join_key_map.get((r1, r2)) or join_key_map.get((r2, r1))
            if has_join_condition:
                if size1 > 0 and size2 > 0:
                     selectivity = 1.0 / max(size1, size2)
                     cost = max(1, int(size1 * size2 * selectivity))
                     print(f"  {subset}: {size1}x{size2}*({selectivity:.2g}) = {cost} (Join)")
                else: cost = 1
            else:
                 cost = size1 * size2
                 print(f"  {subset}: {size1}x{size2} = {cost} (Cross Product)")
        else:
             base_cost_product = 1
             for rel in subset: base_cost_product *= table_sizes.get(rel, 1)
             num_joins_needed = len(subset) - 1
             avg_size = max(1, sum(table_sizes.get(r, 1) for r in subset) / len(subset))
             selectivity_factor = (1.0 / avg_size) ** max(0, num_joins_needed - 1)
             estimated_cost = base_cost_product * selectivity_factor
             cost = max(1, int(estimated_cost))
             print(f"  {subset}: Product({base_cost_product})*Sel({selectivity_factor:.2g}) = {cost} (Multi-Join Est.)")

        final_cost = cost if cost != float('inf') else 1e12
        weights.append(final_cost)
        costs_map[subset_frozenset] = final_cost

    print(f"\nCalculated {len(weights)} weights using simple cardinality model.")
    return weights, costs_map


def get_cost_of_tree(tree_node, costs_map):
    if isinstance(tree_node, str):
        return 0, {tree_node}

    if not isinstance(tree_node, list) or len(tree_node) != 2:
        print(f"Error: Invalid tree node structure: {tree_node}")
        return float('inf'), set()

    left_subtree_total_cost, left_tables = get_cost_of_tree(tree_node[0], costs_map)
    right_subtree_total_cost, right_tables = get_cost_of_tree(tree_node[1], costs_map)

    if left_subtree_total_cost == float('inf') or right_subtree_total_cost == float('inf'):
        return float('inf'), set()

    current_tables_set = left_tables.union(right_tables)
    current_join_frozenset = frozenset(current_tables_set)

    current_step_cost = costs_map.get(current_join_frozenset, float('inf'))

    if current_step_cost >= 1e12:
         print(f"Warning: Join plan includes step marked 'unjoinable' or cost not found: {current_join_frozenset}")
         total_cost_at_node = float('inf')
    else:
         safe_left_cost = left_subtree_total_cost if left_subtree_total_cost != float('inf') else 0
         safe_right_cost = right_subtree_total_cost if right_subtree_total_cost != float('inf') else 0
         total_cost_at_node = safe_left_cost + safe_right_cost + current_step_cost

    return total_cost_at_node, current_tables_set

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', form_data={})

@app.route('/optimize', methods=['POST'])
def optimize():
    relations = []
    table_stats = {}
    join_key_map = {}
    form_data = request.form
    all_combinations = []
    costs_map = {}

    # --- 1. Parse Form Data ---
    for i in range(1, 5):
        table_name = form_data.get(f'table_name_{i}')
        if table_name:
            relations.append(table_name)
            try:
                 rows = int(form_data.get(f'rows_{i}', 1))
                 if rows <= 0: rows = 1; flash(f"Warning: Non-positive row count for {table_name}, using 1.", "warning")
            except ValueError:
                 rows = 1; flash(f"Warning: Invalid row count for {table_name}, using 1.", "warning")
            table_stats[table_name] = {'rows': rows}

    join_pairs_str = form_data.get('join_pairs', '').strip().split('\n')
    for pair_str in join_pairs_str:
        pair_str = pair_str.strip()
        parts = pair_str.replace(' ', '').split('=')
        if len(parts) == 2:
            key1_parts = parts[0].split('.'); key2_parts = parts[1].split('.')
            if len(key1_parts) == 2 and len(key2_parts) == 2:
                t1, c1 = key1_parts; t2, c2 = key2_parts
                join_key_map[(t1, t2)] = (f"{t1}.{c1}", f"{t2}.{c2}")
                join_key_map[(t2, t1)] = (f"{t2}.{c2}", f"{t1}.{c1}")
            elif pair_str: flash(f"Warning: Could not parse join pair '{pair_str}'.", "warning")
        elif pair_str: flash(f"Warning: Could not parse join pair '{pair_str}'.", "warning")

    selected_solver = SolverType.QAOA
    print(f"Using hardcoded solver: {selected_solver.name}")

    if not QISKIT_AVAILABLE:
        flash(f"{selected_solver.name} solver requires Qiskit...", "error"); return render_template('index.html', form_data=form_data)
    if len(relations) < 2:
        flash("Please define at least two tables.", "error"); return render_template('index.html', form_data=form_data)

    # --- 2. Calculate Costs ---
    weights = []
    try:
        # Get both weights list and costs map
        weights, costs_map = calculate_costs_from_stats_benchmark_logic(relations, table_stats, join_key_map)
        all_combinations = QUBO_formulation.relation_sublists(relations) # Needed for get_cost_of_tree context if using indices
        if len(weights) != len(all_combinations):
             flash("Warning: Mismatch between calculated weights and combinations.", "warning")

    except Exception as e:
        flash(f"Error calculating costs: {e}", "error"); print(f"Cost calculation error:\n{traceback.format_exc()}"); return render_template('index.html', form_data=form_data)

    # --- 3. Run Optimizer ---
    print(f"\nRunning optimizer with '{selected_solver.name}' solver...")
    optimizer = QUBO_Split_Optimization_func(filename=f"frontend_{selected_solver.name.lower()}_test_benchmark_model")
    join_tree = None; error_msg = None; elapsed_time = "N/A"; total_plan_cost = None
    start_time = time.time()
    original_stdout = sys.stdout

    try:
        join_tree_result, selected_joins, error_msg_opt = optimizer.finding_opt_jo(relations, weights, solver=selected_solver)

        if isinstance(join_tree_result, list) and join_tree_result:
            join_tree = join_tree_result
            if costs_map:
                 try:
                      plan_cost_result, final_tables = get_cost_of_tree(join_tree, costs_map)
                      if plan_cost_result != float('inf'):
                           total_plan_cost = plan_cost_result
                      else:
                           flash("Warning: Cost calculation indicates the chosen plan involves an 'unjoinable' step or cost lookup failed.", "warning")
                           error_msg = error_msg or "Chosen plan cost calculation failed (infinite cost detected)."
                 except Exception as cost_err:
                      print(f"Error calculating plan cost: {cost_err}\n{traceback.format_exc()}")
                      error_msg = error_msg or "Could not calculate total cost for the chosen plan due to an error."
        elif error_msg_opt: error_msg = error_msg_opt
        else:
             if not error_msg_opt: error_msg = "Optimizer returned no solution ('No solution found!!!' case)."

    except Exception as e:
        error_msg = f"An unexpected error occurred during optimization: {e}"; print(f"Optimizer error:\n{traceback.format_exc()}")
    finally:
        sys.stdout = original_stdout; end_time = time.time(); elapsed_time = f"{end_time - start_time:.2f}"

    # --- 4. Render Results ---
    if error_msg:
        flash_category = "warning" if "No solution found" in str(error_msg) else "error"
        flash(f"Optimization Info/Error: {error_msg}", flash_category)

    return render_template('index.html',
                           join_tree=join_tree,
                           error=error_msg,
                           elapsed_time=elapsed_time,
                           solver_used=selected_solver.name,
                           form_data=form_data,
                           total_plan_cost=total_plan_cost
                           )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)