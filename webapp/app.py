import os
import time
from flask import Flask, render_template, request, flash
import traceback # Added for more detailed error logging

# Import your optimizer classes and the Enum
# Make sure qubo_optimizer.py is in the same directory
try:
    from qubo_optimizer import (
        QUBO_Split_Optimization_func,
        QUBO_formulation,
        Helping_functions,
        SolverType, # Import the Enum
        Experiments_class, # Needed for available solvers check
        QISKIT_AVAILABLE # Import the flag
    )
except ImportError as e:
    print(f"Error importing from qubo_optimizer.py: {e}")
    print("Make sure qubo_optimizer.py is in the same directory as app.py")
    exit()

# Initialize the Flask application
app = Flask(__name__)
# Required for flashing messages (like errors)
app.secret_key = os.urandom(24)

# --- Cost Calculation Placeholder ---
# This is a VERY simplified cost model for the PoC.
# It uses basic stats instead of real dataframes.
def calculate_costs_from_stats(relations, table_stats, join_key_map):
    """
    Placeholder cost calculator based on user-provided stats.
    Handles unjoinable subsets gracefully by assigning a large cost.
    Returns a list of weights aligned with QUBO_formulation.relation_sublists.
    """
    print("Calculating estimated costs from stats (using simple sum-of-rows model)...")
    all_combinations = QUBO_formulation.relation_sublists(relations)
    weights = []
    
    for subset in all_combinations:
        if len(subset) < 2: 
            continue 

        cost = 0
        is_joinable = True # Assume joinable unless proven otherwise
        
        # --- Basic Connectivity Check ---
        # Check if we can form a spanning tree using the defined joins
        if len(subset) > 1:
            nodes_to_connect = set(subset)
            connected_component = {subset[0]}
            nodes_to_connect.remove(subset[0])
            edges_found = 0
            
            # Simple breadth-first search style check
            queue = [subset[0]]
            while queue:
                current_node = queue.pop(0)
                # Find neighbors not yet connected
                neighbors_to_add = []
                for potential_neighbor in list(nodes_to_connect): # Iterate over a copy
                    if join_key_map.get((current_node, potential_neighbor)) or join_key_map.get((potential_neighbor, current_node)):
                         if potential_neighbor not in connected_component:
                              neighbors_to_add.append(potential_neighbor)
                              edges_found += 1
                
                for neighbor in neighbors_to_add:
                    connected_component.add(neighbor)
                    nodes_to_connect.remove(neighbor)
                    queue.append(neighbor)

            # If not all nodes are connected, it's unjoinable
            if nodes_to_connect: 
                is_joinable = False

        if not is_joinable:
             cost = float('inf') # Assign infinite cost if clearly unjoinable
        else:
            # Simple PoC cost: Sum of rows of tables in the subset
            for table_name in subset:
                cost += table_stats.get(table_name, {}).get('rows', 1000) # Default if missing
        
        # Use a large number instead of actual infinity for QUBO compatibility
        weights.append(cost if cost != float('inf') else 1e12) 

    print(f"Calculated {len(weights)} weights.")
    return weights

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main input form."""
    # Pass empty dictionary for form_data on initial load
    return render_template('index.html', form_data={})

@app.route('/optimize', methods=['POST'])
def optimize():
    """Handles form submission, runs optimization, and shows results."""
    relations = []
    table_stats = {}
    join_key_map = {}
    form_data = request.form
    cost_data_for_display = [] # Initialize here

    # --- 1. Parse Form Data ---
    for i in range(1, 5): # Assuming max 4 tables for simplicity
        table_name = form_data.get(f'table_name_{i}')
        if table_name:
            relations.append(table_name)
            try:
                 rows = int(form_data.get(f'rows_{i}', 0))
                 if rows <= 0:
                     rows = 1000 # Default if non-positive
                     flash(f"Warning: Non-positive row count for {table_name}, using default 1000.", "warning")
            except ValueError:
                 rows = 1000 # Default if invalid input
                 flash(f"Warning: Invalid row count for {table_name}, using default 1000.", "warning")
            # In a real app, you'd parse distinct counts for join keys here too
            table_stats[table_name] = {'rows': rows}

    join_pairs_str = form_data.get('join_pairs', '').strip().split('\n')
    for pair_str in join_pairs_str:
        pair_str = pair_str.strip() # Remove leading/trailing whitespace
        parts = pair_str.replace(' ', '').split('=')
        if len(parts) == 2:
            key1_parts = parts[0].split('.')
            key2_parts = parts[1].split('.')
            if len(key1_parts) == 2 and len(key2_parts) == 2:
                t1, c1 = key1_parts
                t2, c2 = key2_parts
                # Store both directions for easier lookup
                join_key_map[(t1, t2)] = (f"{t1}.{c1}", f"{t2}.{c2}")
                join_key_map[(t2, t1)] = (f"{t2}.{c2}", f"{t1}.{c1}")
            elif pair_str: # Only flash if it wasn't an empty line
                flash(f"Warning: Could not parse join pair '{pair_str}'. Expected format: table1.key1 = table2.key2", "warning")
        elif pair_str:
             flash(f"Warning: Could not parse join pair '{pair_str}'. Expected format: table1.key1 = table2.key2", "warning")


    # --- Hardcode the solver ---
    selected_solver = SolverType.QAOA
    print(f"Using hardcoded solver: {selected_solver.name}")

    # --- Check Qiskit Availability ---
    if not QISKIT_AVAILABLE:
        flash(f"{selected_solver.name} solver requires Qiskit, but it's not available. Check installation.", "error")
        return render_template('index.html', form_data=form_data) # Return with error

    if len(relations) < 2:
        flash("Please define at least two tables.", "error")
        return render_template('index.html', form_data=form_data)

    # --- 2. Calculate Costs ---
    weights = []
    try:
        weights = calculate_costs_from_stats(relations, table_stats, join_key_map)
        # Get combinations corresponding to weights for display
        all_combinations = QUBO_formulation.relation_sublists(relations)
        if len(weights) == len(all_combinations):
             cost_data_for_display = list(zip(all_combinations, weights))
        else:
             flash("Warning: Mismatch between calculated weights and combinations.", "warning")
             cost_data_for_display = [ ([], w) for w in weights ] # Pass weights alone

    except Exception as e:
        flash(f"Error calculating costs: {e}", "error")
        print(f"Cost calculation error traceback:\n{traceback.format_exc()}") # Log detailed error
        return render_template('index.html', form_data=form_data)

    # --- 3. Run Optimizer ---
    print(f"\nRunning optimizer with '{selected_solver.name}' solver...")
    optimizer = QUBO_Split_Optimization_func(filename=f"frontend_{selected_solver.name.lower()}_test")
    
    join_tree = None
    error_msg = None
    elapsed_time = "N/A"
    start_time = time.time()
    try:
        # finding_opt_jo expects the Enum member
        join_tree_result, selected_joins, error_msg_opt = optimizer.finding_opt_jo(
            relations,
            weights,
            solver=selected_solver 
        )
        # Handle potential return of non-tree results or specific errors
        if isinstance(join_tree_result, list): # Check if it looks like a tree
             join_tree = join_tree_result
        elif error_msg_opt: # Use error message from optimizer if provided
             error_msg = error_msg_opt
        else: # Generic error if no tree and no specific message
             error_msg = "Optimizer did not return a valid join tree structure."

    except Exception as e:
        error_msg = f"An unexpected error occurred during optimization: {e}"
        print(f"Optimizer error traceback:\n{traceback.format_exc()}") # Log detailed error

    finally:
        end_time = time.time()
        elapsed_time = f"{end_time - start_time:.2f}"

    # --- 4. Render Results ---
    if error_msg:
        flash(f"Optimization Info/Error: {error_msg}", "warning" if "No solution found" in error_msg else "error")
        
    return render_template('index.html', 
                           join_tree=join_tree, 
                           error=error_msg,
                           elapsed_time=elapsed_time,
                           solver_used=selected_solver.name, # Show QAOA ran
                           form_data=form_data, # Pass back form data to repopulate
                           cost_data=cost_data_for_display # Pass cost data
                           )

# --- Run the App ---
if __name__ == '__main__':
    # Makes the app accessible on your local network
    # Use debug=True only for development (auto-reloads, insecure)
    app.run(host='0.0.0.0', port=5000, debug=True)