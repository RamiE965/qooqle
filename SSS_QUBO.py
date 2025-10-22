# General-purpose
import random
import itertools
import time
from itertools import combinations
from math import comb
import numpy as np
import pandas as pd
import csv
import warnings
from scipy.sparse import SparseEfficiencyWarning

# Suppress scipy sparse matrix efficiency warnings
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

# Qiskit (Quantum & Classical Optimization) - Updated for your specific environment
try:
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_algorithms import NumPyMinimumEigensolver, QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.primitives import StatevectorSampler  # Use what's actually available
    
    QISKIT_AVAILABLE = True
    print("Qiskit packages available - QAOA solver enabled")
    
except ImportError as e:
    QISKIT_AVAILABLE = False
    print(f"Qiskit import failed: {e}")

# Classical Optimization / Modeling (for compatibility)
try:
    import dimod
    from dimod import BinaryQuadraticModel
    from neal import SimulatedAnnealingSampler
    CLASSICAL_OPT_AVAILABLE = True
except ImportError:
    CLASSICAL_OPT_AVAILABLE = False
    print("Warning: Classical optimization packages (dimod, neal) not available.")


class QUBO_formulation:
    """
    Build variable sets (S1) and incompatibility constraints (S2/S3) for QUBO-based join-order search.
    All methods are static; no instance state is required.
    """

    def __init__(self, relations):
        self.relations = relations  # unused; kept for backward compatibility

    @staticmethod
    def relation_sublists(relations):
        """
        Return all non-trivial relation subsets (cardinality >= 2) in deterministic order:
        by size 2..n, then lexicographic within each size.
        """
        return [list(s) for k in range(2, len(relations) + 1) for s in combinations(relations, k)]

    @staticmethod
    def construct_QUBO(relations):
        """
        Build:
          - S1: list of variables (all subsets of size >= 2)
          - S2: list of index pairs (i,j) that cannot be selected together
          - n : number of variables
        """
        relations_sublists = QUBO_formulation.relation_sublists(relations)
        n = len(relations_sublists)
        s_1_helper = relations_sublists

        s_2_helper = []
        for i in range(n):
            Si = set(relations_sublists[i])
            for j in range(i + 1, n):
                Sj = set(relations_sublists[j])
                if Si.intersection(Sj):
                    # Only penalize when the smaller (or equal) is NOT a subset of the larger
                    if (len(Si) <= len(Sj) and not Si.issubset(Sj)) or (len(Sj) < len(Si) and not Sj.issubset(Si)):
                        s_2_helper.append([i, j])

        return s_1_helper, s_2_helper, n

    @staticmethod
    def construct_QUBO_opt_jo_higher_vars(lst):
        """
        Same as construct_QUBO, but the universe of variables is provided directly as `lst`
        (each element of `lst` is a subset/list of relations).
        """
        n = len(lst)
        s_1_helper = lst

        s_2_helper = []
        for i in range(n):
            Si = set(lst[i])
            for j in range(i + 1, n):
                Sj = set(lst[j])
                if Si.intersection(Sj):
                    if (len(Si) <= len(Sj) and not Si.issubset(Sj)) or (len(Sj) < len(Si) and not Sj.issubset(Si)):
                        s_2_helper.append([i, j])

        return s_1_helper, s_2_helper, n

    @staticmethod
    def construct_qubo_for_inner_split(lst, l1, l2):
        """
        Inner-split formulation (e.g., n = (m+p)+1+1+... with m=l1, p=l2).
        Inputs:
          - lst: list of variables (each a subset/list)
          - l1 : count of total-cost variables in the first block
          - l2 : count of total-cost variables in the second block
        Returns S1, S2, S3, n.
        """
        n = len(lst)
        s_1_helper = lst

        # S2: general incompatibility as before
        s_2_helper = []
        for i in range(n):
            Si = set(lst[i])
            for j in range(i + 1, n):
                Sj = set(lst[j])
                if Si.intersection(Sj):
                    if (len(Si) <= len(Sj) and not Si.issubset(Sj)) or (len(Sj) < len(Si) and not Sj.issubset(Si)):
                        s_2_helper.append([i, j])

        # S3: extra constraints across inner split
        l3 = l1 + l2
        s_3_helper = []

        # (a) between first block [0..l1-1] and second block [l1..l3-1]:
        #     penalize if they are NOT disjoint (force disjointness across the two blocks)
        for i in range(l1):
            Si = set(lst[i])
            for j in range(l1, l3):
                Sj = set(lst[j])
                if not Si.isdisjoint(Sj):
                    s_3_helper.append([i, j])

        # (b) within first block: penalize pairs that ARE disjoint (enforce overlap coupling)
        for i in range(l1):
            Si = set(lst[i])
            for j in range(i + 1, l1):
                Sj = set(lst[j])
                if Si.isdisjoint(Sj):
                    s_3_helper.append([i, j])

        return s_1_helper, s_2_helper, s_3_helper, n

    
class Solvers_qiskit:
    """
    QUBO solving via Qiskit (NumPyMinimumEigensolver or QAOA) and classical SA (neal).
    Returns dimod.SampleSet for compatibility with existing code paths.
    """

    @staticmethod
    def _penalty(weights_lst):
        return max(weights_lst) * 2

    @staticmethod
    def prepare_model(s_1_helper, s_2_helper, n, weights_lst):
        """
        Build both Qiskit QuadraticProgram and dimod BinaryQuadraticModel.
        """
        P = Solvers_qiskit._penalty(weights_lst)

        # Qiskit QuadraticProgram
        if QISKIT_AVAILABLE:
            qp = QuadraticProgram()
            for i in range(n):
                qp.binary_var(f"x_{i}")

            linear = {f"x_{i}": (weights_lst[i] - P) for i in range(len(s_1_helper))}
            quadratic = {}
            for i, j in s_2_helper:
                key = (f"x_{i}", f"x_{j}")
                quadratic[key] = quadratic.get(key, 0.0) + P

            qp.minimize(linear=linear, quadratic=quadratic)
        else:
            qp = None

        # dimod BQM for compatibility
        if CLASSICAL_OPT_AVAILABLE:
            bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, vartype=dimod.BINARY)
            for i in range(n):
                bqm.add_variable(i)
            for i in range(len(s_1_helper)):
                bqm.set_linear(i, (weights_lst[i] - P))
            for i, j in s_2_helper:
                bqm.set_quadratic(i, j, P)
        else:
            bqm = None

        return qp, bqm

    @staticmethod
    def _qp_solution_to_sampleset(qp, bqm, sol_x):
        """
        Convert Qiskit solution vector -> dimod.SampleSet.
        """
        if str(type(sol_x)).find('SampleView') != -1:
            sample_dict = dict(sol_x)
            return dimod.SampleSet.from_samples_bqm([sample_dict], bqm)
        
        bits = [int(round(v)) for v in sol_x]
        sample = {i: bits[i] for i in range(len(bits))}
        return dimod.SampleSet.from_samples_bqm([sample], bqm)

    # @staticmethod
    # def exact_result(qp_bqm_tuple):
    #     print("HERE")
    #     """
    #     NumPyMinimumEigensolver -> SampleSet.
    #     """
    #     if not QISKIT_AVAILABLE:
    #         raise ValueError("Qiskit packages not available. Cannot use exact_result solver.")
    #     qp, bqm = qp_bqm_tuple
    #     res = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qp)
    #     return Solvers_qiskit._qp_solution_to_sampleset(qp, bqm, res.x)

    @staticmethod
    def qaoa(qp_bqm_tuple, reps=3, seed=30):
        """
        QAOA -> SampleSet.
        """
        if not QISKIT_AVAILABLE:
            raise ValueError("Qiskit packages not available. Cannot use qaoa solver.")
        qp, bqm = qp_bqm_tuple
        
        # Use StatevectorSampler which is available in your environment
        sampler = StatevectorSampler()
        qaoa_mes = QAOA(sampler=sampler, optimizer=COBYLA(), reps=reps)
        res = MinimumEigenOptimizer(qaoa_mes).solve(qp)
        return Solvers_qiskit._qp_solution_to_sampleset(qp, bqm, res.x)

    # @staticmethod
    # def simulated_annealing(qp_bqm_tuple):
    #     """
    #     Classical neal SA on BQM -> SampleSet.
    #     """
    #     if not CLASSICAL_OPT_AVAILABLE:
    #         raise ValueError("Classical optimization packages not available. Cannot use simulated_annealing solver.")
    #     _qp, bqm = qp_bqm_tuple
    #     return SimulatedAnnealingSampler().sample(bqm, num_reads=100000)

class Helping_functions:
    """
    Helper utilities for generating weights, validating via DP, decoding bitstrings to join trees,
    and manipulating variable/cost lists. Pure statics for side-effect minimization.
    """

    # ---------- Random weights ----------

    @staticmethod
    def init_weights(relations):
        """
        Initialize random join costs for all non-trivial subsets (|S| >= 2),
        in the same order as QUBO_formulation.relation_sublists(relations).
        """
        subs = QUBO_formulation.relation_sublists(relations)
        return [random.randint(1, 100) for _ in range(len(subs))]

    # ---------- I/O / inspection ----------

    @staticmethod
    def print_relations_and_weights(relations, weights, logfile="Qiskit-Tests.txt"):
        """
        Write all subset variables and their weights to console and file.
        """
        relation_combinations = QUBO_formulation.relation_sublists(relations)
        with open(logfile, "a", encoding="utf-8") as f:
            for i, combi in enumerate(relation_combinations):
                line = f"Variable:-{i} (weight:-{weights[i]}) -> {list(combi)}"
                print(line)
                f.write(line + "\n")
            f.write("\n")

    # ---------- DP oracle (ground truth) ----------

    @staticmethod
    def dynamic_programming(relations, weights):
        """
        DP over subset partitions:
        cost(S) = min_{A ⊂ S, A≠∅} cost(A)+cost(S\\A)+w(S), with base cost({r})=0.
        Returns (best_tree, best_cost) for the full set of relations.
        """
        rel_sublists = QUBO_formulation.relation_sublists(relations)
        idx = {frozenset(s): i for i, s in enumerate(rel_sublists)}

        # base
        t = {frozenset([r]): ([r], 0) for r in relations}

        # iterate by increasing subset size (>=2)
        for S in rel_sublists:
            S_fs = frozenset(S)
            best = None
            # enumerate bipartitions via choosing A ⊂ S, A non-empty, A != S
            S_elems = list(S)
            m = len(S_elems)
            for k in range(1, m // 2 + 1):
                for A_tuple in combinations(S_elems, k):
                    A = frozenset(A_tuple)
                    B = S_fs - A
                    if len(B) == 0:
                        continue
                    # compute cost if both parts known
                    left_tree, left_cost = t[A] if len(A) == 1 else t[A]
                    right_tree, right_cost = t[B] if len(B) == 1 else t[B]
                    wS = weights[idx[S_fs]]  # weight for this subset S
                    cand_cost = left_cost + right_cost + wS
                    cand_tree = [left_tree, right_tree]
                    if (best is None) or (cand_cost < best[1]):
                        best = (cand_tree, cand_cost)
            t[S_fs] = best

        return t[frozenset(relations)]

    # ---------- Decode selected variables to a binary join tree ----------

    @staticmethod
    def _to_bits_array(possible_state):
        """
        Normalize various result formats to a Python list of ints {0,1}.
        """
        # Handle dimod SampleView specifically
        if str(type(possible_state)).find('SampleView') != -1:
            try:
                # Convert SampleView to dict and extract values in order
                sample_dict = dict(possible_state)
                return [int(sample_dict[k]) for k in sorted(sample_dict.keys())]
            except Exception as e:
                print(f"DEBUG: Error converting SampleView: {e}")
                pass
        
        # Handle dimod SampleView - convert to dict first
        if hasattr(possible_state, '__iter__') and not isinstance(possible_state, (list, tuple)):
            try:
                # Try to convert to dict if it's a SampleView
                if hasattr(possible_state, 'keys'):
                    result = [int(possible_state[k]) for k in sorted(possible_state.keys())]
                    return result
                # Otherwise try to iterate directly
                result = [int(v) for v in possible_state]
                return result
            except Exception as e:
                pass
        
        # Qiskit OptimizationResult (res.x)
        if hasattr(possible_state, "x"):
            possible_state = possible_state.x
        # numpy arrays -> list
        try:
            import numpy as _np
            if isinstance(possible_state, _np.ndarray):
                return [int(round(v)) for v in possible_state.tolist()]
        except Exception:
            pass
        # generic sequence
        return [int(round(v)) for v in possible_state]

    @staticmethod
    def make_join_order_tree(possible_state, vars_list, relations):
        """
        Convert a selected-variable bitstring into a concrete binary join tree.
        Returns (tree, selected_vars, error_str_or_None).
        """
        bits = Helping_functions._to_bits_array(possible_state)

        # If caller appended the full-set var separately, keep compatibility.
        if len(bits) != len(vars_list):
            last_var = vars_list[-1]
            core_vars = vars_list[:-1]
            state_vars = [core_vars[i] for i, b in enumerate(bits) if b == 1]
            state_vars.append(last_var)
        else:
            state_vars = [vars_list[i] for i, b in enumerate(bits) if b == 1]

        # Forest build: start from leaves
        forest = {frozenset([r]): [r] for r in relations}

        # Attempt to reduce by applying chosen subset merges
        for v in state_vars:
            v_set = set(v)
            # find two disjoint subtrees that union to v
            merged = False
            for left, right in list(combinations(list(forest.keys()), 2)):
                if left.isdisjoint(right) and left.union(right) == v_set:
                    left_val = forest[left][0] if len(left) == 1 else forest[left]
                    right_val = forest[right][0] if len(right) == 1 else forest[right]
                    forest[frozenset(v_set)] = [left_val, right_val]
                    forest.pop(left)
                    forest.pop(right)
                    merged = True
                    break
            # if no matching pair exists, keep going; final validity is checked below

        if len(forest) == 1:
            return next(iter(forest.values())), state_vars, None
        else:
            return [], state_vars, "More than one tree remained at the end"

    # ---------- Variable/cost utilities ----------

    @staticmethod
    def separate_variables(op_vars_list, all_vars, cost_list, length):
        """
        Split chosen variables into:
          - items of exact size==length and all their subsets (temp_1)
          - the remainder (op_vars_list without those subsets)
        Append aggregated costs at the end of each returned list.
        """
        if not op_vars_list:
            return op_vars_list, []

        # collect one representative of cardinality==length
        bucket = [v for v in op_vars_list if len(v) == length]
        if not bucket:
            return op_vars_list, []

        target = bucket[0]
        temp_1 = [v for v in op_vars_list if set(v).issubset(set(target))]
        remainder = [v for v in op_vars_list if v not in temp_1]

        cost_1 = sum(cost_list[all_vars.index(v)] for v in remainder) if remainder else 0
        cost_2 = sum(cost_list[all_vars.index(v)] for v in temp_1) if temp_1 else 0

        remainder_with_cost = remainder + [cost_1]
        temp_1_with_cost = temp_1 + [cost_2]

        return remainder_with_cost, temp_1_with_cost

    @staticmethod
    def single_cost_var(vars_list, weight_assigned, num_relns):
        """
        Extract variables whose cardinality is in num_relns and their costs
        (aligned to vars_list indexing).
        """
        var_list = [var for var in vars_list if len(var) in num_relns]
        cost_list = [weight_assigned[vars_list.index(var)] for var in var_list]
        return var_list, cost_list


class Experiments_class:
    """
    Qiskit experiment runners for QUBO optimization.
    Methods use Qiskit solvers (QAOA, NumPyMinimumEigensolver).
    """

    @staticmethod
    def _check_qiskit_available():
        """Check if Qiskit is available and raise informative error if not."""
        if not QISKIT_AVAILABLE:
            raise ImportError(
                "Qiskit packages not available. Please ensure:\n"
                "- qiskit_optimization is installed\n"
                "- qiskit_algorithms is available\n"
                "- StatevectorSampler is accessible\n\n"
                "Try: pip install qiskit-optimization"
            )

    @staticmethod
    def _select_solver(solver_name, qp_bqm_tuple):
        """Select and run the appropriate Qiskit solver."""
        Experiments_class._check_qiskit_available()
        print(solver_name)
        if solver_name == 'exact_result':
            return Solvers_qiskit.exact_result(qp_bqm_tuple)
        elif solver_name in ('qaoa', 'qiskit_qaoa'):
            return Solvers_qiskit.qaoa(qp_bqm_tuple)
        elif solver_name == 'simulated_annealing':
            if not CLASSICAL_OPT_AVAILABLE:
                raise ValueError("Classical optimization packages not available. Cannot use 'simulated_annealing' solver.")
            return Solvers_qiskit.simulated_annealing(qp_bqm_tuple)
        else:
            raise ValueError(f"Unknown solver '{solver_name}'. Available: 'exact_result', 'qaoa', 'simulated_annealing'")

    @staticmethod
    def get_available_solvers():
        """Return list of available solvers."""
        available = []
        if QISKIT_AVAILABLE:
            available.extend(['exact_result', 'qaoa'])
        if CLASSICAL_OPT_AVAILABLE:
            available.append('simulated_annealing')
        return available

    @staticmethod
    def qiskit_experiment(relations, weights, solver):
        """
        Build QUBO (S1,S2), solve with Qiskit backend, decode.
        """
        s1, s2, n = QUBO_formulation.construct_QUBO(relations)
        qp_bqm = Solvers_qiskit.prepare_model(s1, s2, n, weights)

        result = Experiments_class._select_solver(solver, qp_bqm)

        # Extract solution bits
        if hasattr(result, 'lowest'):
            bits = result.lowest().record[0][0]
        else:
            bits = [int(round(x)) for x in result.x] if hasattr(result, 'x') else []

        relation_combinations = QUBO_formulation.relation_sublists(relations)
        optimized_jo_var = []
        total_cost = 0
        
        for i, bit in enumerate(bits):
            if bit == 1 and i < len(relation_combinations):
                optimized_jo_var.append(relation_combinations[i])
                total_cost += weights[i]

        return result, total_cost, optimized_jo_var

    @staticmethod
    def qiskit_experiment_opt_jo_by_vars_list(lst, cost, solver):
        """
        Custom variable universe (last element is the full-set variable).
        Returns (result: dimod.SampleSet, total_cost: int, optimized_jo_var: list[list[str]], model: QuadraticProgram).
        """
        # Create copies to avoid modifying original data
        vars_list = lst.copy()
        costs = cost.copy()

        last_var = vars_list[-1]
        last_cost = costs[-1]
        vars_core = vars_list[:-1]
        costs_core = costs[:-1]

        s1, s2, n = QUBO_formulation.construct_QUBO_opt_jo_higher_vars(vars_core)
        qp, bqm = Solvers_qiskit.prepare_model(s1, s2, n, costs_core)

        result = Experiments_class._select_solver(solver, (qp, bqm))

        # Extract solution bits
        if hasattr(result, 'lowest'):
            bits = result.lowest().record[0][0]
        else:
            bits = [int(round(x)) for x in result.x] if hasattr(result, 'x') else []

        optimized_jo_var = []
        for i, bit in enumerate(bits):
            if bit == 1 and i < len(vars_core):
                optimized_jo_var.append(vars_core[i])
        optimized_jo_var.append(last_var)

        # Calculate total cost
        full_vars = vars_core + [last_var]
        full_costs = costs_core + [last_cost]
        total_cost = 0
        for v in optimized_jo_var:
            if v in full_vars:
                total_cost += full_costs[full_vars.index(v)]

        return result, total_cost, optimized_jo_var, qp

    @staticmethod
    def qiskit_experiment_inner_split_joo(lst, cost, l1, l2, solver):
        """
        Inner-split variant with S3 constraints.
        Returns (result: dimod.SampleSet, total_cost: int, optimized_jo_var: list[list[str]]).
        """
        s1, s2, s3, n = QUBO_formulation.construct_qubo_for_inner_split(lst, l1, l2)
        qp_bqm = Solvers_qiskit.prepare_model_with_s3(s1, s2, s3, n, cost)

        result = Experiments_class._select_solver(solver, qp_bqm)

        # Extract solution bits
        if hasattr(result, 'lowest'):
            bits = result.lowest().record[0][0]
        else:
            bits = [int(round(x)) for x in result.x] if hasattr(result, 'x') else []

        optimized_jo_var = []
        total_cost = 0
        for i, bit in enumerate(bits):
            if bit == 1 and i < len(lst):
                optimized_jo_var.append(lst[i])
                total_cost += cost[i]

        return result, total_cost, optimized_jo_var

    # Backward compatibility methods (can be removed eventually)
    @staticmethod
    def dwave_experiment(relations, weights, solver):
        """Deprecated - use qiskit_experiment instead."""
        print("Warning: dwave_experiment is deprecated. Use qiskit_experiment instead.")
        return Experiments_class.qiskit_experiment(relations, weights, solver)

    @staticmethod
    def dwave_experiment_opt_jo_by_vars_list(lst, cost, solver):
        """Deprecated - use qiskit_experiment_opt_jo_by_vars_list instead."""
        print("Warning: dwave_experiment_opt_jo_by_vars_list is deprecated. Use qiskit_experiment_opt_jo_by_vars_list instead.")
        return Experiments_class.qiskit_experiment_opt_jo_by_vars_list(lst, cost, solver)

    @staticmethod
    def dwave_experiment_inner_split_joo(lst, cost, l1, l2, solver):
        """Deprecated - use qiskit_experiment_inner_split_joo instead."""
        print("Warning: dwave_experiment_inner_split_joo is deprecated. Use qiskit_experiment_inner_split_joo instead.")
        return Experiments_class.qiskit_experiment_inner_split_joo(lst, cost, l1, l2, solver)


class QUBO_Split_Optimization_func:
    """
    QUBO-based join order optimization using Qiskit quantum and classical solvers.
    Supports various split strategies for different relation counts.
    """

    def __init__(self, filename):
        import os
        os.makedirs("servers/qubo", exist_ok=True)
        self.logfile = open(f"servers/qubo/{filename}.log", "w")

    def __del__(self):
        try:
            self.logfile.close()
        except Exception:
            pass

    def logQiskitResult(self, result, relations, weights, vars_list, dp_optimal_cost, total_cost):
        # result is a dimod.SampleSet (from neal/Qiskit wrapper)
        self.logfile.write(
            "# {} ; {} ; {} ; {}\n".format(
                ",".join(relations),
                ",".join(str(e) for e in weights),
                str(total_cost),
                ",".join(str(e) for e in vars_list),
            )
        )
        self.logfile.flush()

        agg = result.aggregate()
        lowest_energy_state = agg.lowest().record[0][1]  # energy of the best sample

        for r in agg.record:
            # r[0]=sample bits, r[1]=energy, r[2]=num_occurrences
            logData = [",".join(relations)]
            logData.append(r[2])  # occurrences
            logData.append(r[1])  # energy
            order, variables, error = Helping_functions.make_join_order_tree(r[0], vars_list, relations)
            logData.append("Valid" if error is None else "Invalid")
            logData.append("Optimal" if (total_cost == dp_optimal_cost and r[1] == lowest_energy_state) else "Not Optimal")
            logData.append(order)
            logData.append(variables)
            self.logfile.write(";".join([str(e) for e in logData]) + "\n")
            self.logfile.flush()

    # ---- fixed-length helper ----
    def fixed_length_var_optimal_sol(self, relations, weights, length, solver):
        relation_combinations = QUBO_formulation.relation_sublists(relations)
        rel_comb = [list(lst) for lst in combinations(relations, length)]

        total_cost_list = []
        optimal_rel_list = []
        optimal_variables = []

        for rel in rel_comb:
            new_rel_vars_list = QUBO_formulation.relation_sublists(rel)
            new_weights = [weights[relation_combinations.index(var)] for var in new_rel_vars_list]

            result, total_cost, jo_vars = Experiments_class.qiskit_experiment(rel, new_weights, solver)
            optimal_variables.append(jo_vars)
            optimal_rel_list.append(jo_vars[-1])
            total_cost_list.append(total_cost)

        return optimal_variables, optimal_rel_list, total_cost_list

    @staticmethod
    def joo_by_split_SS_outer_Bushy(relations, power_set, weights, length, solver):
        """
        Outer-bushy split search (split relations into 'length' and 'remaining').
        No DP; only Qiskit QAOA/simulated annealing.
        """
        store_jo_details = {}
        temp_1, temp_2 = [], []
        cost_1, cost_2 = [], []
        store_final_vars, store_total_cost, JO_details = [], [], []

        if len(relations) != 2 * length:
            rel_comb = [list(lst) for lst in combinations(relations, length)]
        else:
            rel_comb = [list(lst) for lst in combinations(relations, length)]
            rel_comb = rel_comb[0:int(len(rel_comb) / 2)]

        for rel in rel_comb:
            # variables inside the chosen subset
            temp_lst = QUBO_formulation.relation_sublists(rel)
            new_rel_vars_list = [var for var in temp_lst if var in power_set]

            # variables in the complement
            rel_remaining = [elm for elm in relations if elm not in rel]
            l = len(rel_remaining)
            temp_lst_2 = QUBO_formulation.relation_sublists(rel_remaining)
            new_rel_vars_list_remaining = [var for var in temp_lst_2 if var in power_set]

            # build the working universe (+ full set)
            new_list = new_rel_vars_list + new_rel_vars_list_remaining + [relations]
            new_weights = [weights[power_set.index(var)] for var in new_list]

            # solve
            result, total_cost, JO_vars, _ = Experiments_class.qiskit_experiment_opt_jo_by_vars_list(
                new_list, new_weights, solver
            )
            JO_details.append([total_cost, JO_vars, result, new_weights, new_list])

            # pick the two 'length'-sized vars used (if any)
            JO_vars_copy = JO_vars.copy()
            temp_3 = [var for var in JO_vars_copy if len(var) == length]
            if len(temp_3) == 0:
                continue

            lst1, lst2 = Helping_functions.separate_variables(JO_vars, new_list, new_weights, length)

            if l > 2:
                store_jo_details[frozenset(lst1[-2])] = lst1
                store_jo_details[frozenset(lst2[-2])] = lst2
                temp_1.append(lst1[-2]); cost_1.append(lst1[-1])
                temp_2.append(lst2[-2]); cost_2.append(lst2[-1])
            else:
                store_jo_details[frozenset(lst2[-2])] = lst2
                temp_2.append(lst2[-2]); cost_2.append(lst2[-1])

        opt_JO_details = min(JO_details, key=lambda e: e[0])

        if len(temp_1) != 0:
            store_final_vars.append(temp_1)
            store_final_vars.append(temp_2)
            store_total_cost.append(cost_1)
            store_total_cost.append(cost_2)
        else:
            store_final_vars += temp_2
            store_total_cost += cost_2

        return opt_JO_details, store_jo_details, store_final_vars, store_total_cost, len(JO_details)


    @staticmethod
    def findind_LR_deep_jo(vars_J2R, All_vars, weights, solver):
        """
        Left/right-deep join-tree search over all 2-way seeds (vars_J2R).
        Returns (best_detail, exp_run).
        best_detail = [total_cost, optimized_jo_vars, result, cost_list, vars_list, qb]
        """
        optimal_jo_details = []
        for rel in vars_J2R:
            vars_list, cost_list = [], []
            for i, elem in enumerate(All_vars):
                if set(rel).issubset(set(elem)):
                    vars_list.append(elem)
                    cost_list.append(weights[i])
            result, total_cost, optimized_jo_vars, qb = Experiments_class.qiskit_experiment_opt_jo_by_vars_list(
                vars_list, cost_list, solver
            )
            optimal_jo_details.append([total_cost, optimized_jo_vars, result, cost_list, vars_list, qb])

        exp_run = len(optimal_jo_details)
        optimal_jo_details = min(optimal_jo_details, key=lambda e: e[0])
        print('QUBO : ', optimal_jo_details[-1])
        return optimal_jo_details, exp_run

    @staticmethod
    def special_function_inner_split(vars_J2R, All_vars, cost, solver):
        """
        Find JO for splits like n = m(t) + 1 + 1 + ... where m(t) is a pre-aggregated part.
        Considers disjoint pairs from vars_J2R and solves on their closure.
        Returns (best_detail, exp_run).
        best_detail = [total_cost, optimized_jo_vars, result, cost_list, vars_list]
        """
        possible_jo_details = []
        for var_1 in vars_J2R:
            for var_2 in vars_J2R:
                if set(var_1).isdisjoint(set(var_2)):
                    vars_list, cost_list = [], []
                    for i, elem in enumerate(All_vars):
                        if set(var_1).issubset(set(elem)) or set(var_2).issubset(set(elem)):
                            vars_list.append(elem)
                            cost_list.append(cost[i])
                    result, total_cost, optimized_jo_vars, _ = Experiments_class.qiskit_experiment_opt_jo_by_vars_list(
                        vars_list, cost_list, solver
                    )
                    possible_jo_details.append([total_cost, optimized_jo_vars, result, cost_list, vars_list])

        exp_run = len(possible_jo_details)
        optimal_jo_details = min(possible_jo_details, key=lambda e: e[0])
        return optimal_jo_details, exp_run

    @staticmethod
    def reduce_contraints_for_disjoint_inner_join(lst, cost, l1, l2, solver):
        """
        Inner-split with reduced constraints:
        - First l1 entries are seed totals (group A).
        - Next l2 entries are seed totals (group B).
        - Remaining entries are supersets.
        Build smaller universes per A-seed by taking only B seeds that are disjoint with it,
        plus all supersets that include that A-seed. Solve with inner-split model.
        Returns (best_detail, exp_run).
        best_detail = [total_cost, optimized_jo_vars, result, new_cost_lst, new_list]
        """
        l3 = l1 + l2
        sub_lst = lst[0:l1]
        opt_details = []
        n = len(lst)

        for i in sub_lst:
            dis_vars = [lst[j] for j in range(l1, l3) if set(i).isdisjoint(set(lst[j]))]
            lst2 = [lst[j] for j in range(l3, n) if set(i).issubset(set(lst[j]))]

            new_list = [i] + dis_vars + lst2
            new_cost_lst = [cost[lst.index(j)] for j in new_list]

            l2_local = len(dis_vars)
            result, total_cost, optimized_jo_vars = Experiments_class.qiskit_experiment_inner_split_joo(
                new_list, new_cost_lst, 1, l2_local, solver
            )
            opt_details.append([total_cost, optimized_jo_vars, result, new_cost_lst, new_list])

        exp_run = len(opt_details)
        opt_detail = min(opt_details, key=lambda e: e[0])
        return opt_detail, exp_run



    def finding_opt_jo(self, relations, weights, solver):
        """
        Find optimal join order for n in {4,5,6,7,8} using split strategies + Qiskit solvers.
        Uses Qiskit QAOA, NumPyMinimumEigensolver, and classical simulated annealing.
        """
        power_set = QUBO_formulation.relation_sublists(relations)
        print('number of all variables in power set of a query is:', len(power_set))

        vars_2, cost_2 = Helping_functions.single_cost_var(power_set, weights, [2])
        optimal_jo_details = []
        n = len(relations)

        dp_opt_jo_cost = Helping_functions.dynamic_programming(relations, weights)
        dp_optimal_cost = dp_opt_jo_cost[1]

        # ----- n = 3 -----
        if n == 3:
            count = 1
            optimal_jo, exp_run = QUBO_Split_Optimization_func.findind_LR_deep_jo(vars_2, power_set, weights, solver)
            if optimal_jo[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for the l/r deep join tree & #variables:{len(optimal_jo[3])} and #exp:{exp_run}.')
                optimal_jo_details.append(optimal_jo)
            else:
                print(f'No,optimal JO not found for the l/r deep join tree & #variables:{len(optimal_jo[3])} and #exp:{exp_run}.')
            print(f'# total exps run:{exp_run} & # splits:{count}.')

        # ----- n = 4 -----
        if n == 4:
            # Split -> 1 (deep trees)
            count = 1
            optimal_jo, exp_run = QUBO_Split_Optimization_func.findind_LR_deep_jo(vars_2, power_set, weights, solver)
            if optimal_jo[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for the l/r deep join tree & #variables:{len(optimal_jo[3])} and #exp:{exp_run}.')
                optimal_jo_details.append(optimal_jo)
            else:
                print(f'No,optimal JO not found for the l/r deep join tree & #variables:{len(optimal_jo[3])} and #exp:{exp_run}.')

            # Split -> 2 (2+2)
            count += 1
            vars_4, cost_4 = Helping_functions.single_cost_var(power_set, weights, [4])
            result, total_cost, optimized_jo_vars, _ = Experiments_class.qiskit_experiment_opt_jo_by_vars_list(
                vars_2 + vars_4, cost_2 + cost_4, solver
            )
            if total_cost == dp_optimal_cost:
                print(f'Yes,found optimal JO for split 2+2 & #variables:{len(vars_2 + vars_4)} and #exp:{1}.')
                optimal_jo_details.append([total_cost, optimized_jo_vars, result, cost_2 + cost_4, vars_2 + vars_4])
            else:
                print(f'No,optimal JO not found for split 2+2 & #variables:{len(vars_2 + vars_4)} and #exp:{1}.')
            exp_run += 1
            print(f'# total exps run:{exp_run} & # splits:{count}.')

        # ----- n = 5 -----
        if n == 5:
            count = 1
            optimal_jo, exp_run = QUBO_Split_Optimization_func.findind_LR_deep_jo(vars_2, power_set, weights, solver)
            if (optimal_jo[0] == dp_optimal_cost):
                print(f'Yes,found optimal JO for the l/r deep join tree & #variables:{len(optimal_jo[3])} & #exp:{exp_run}.')
                optimal_jo_details.append(optimal_jo)
            else:
                print(f'No,optimal JO not found for the l/r deep join tree & #variables:{len(optimal_jo[3])} & #exp:{exp_run}.')

        # ----- n = 6 -----
        if n == 6:
            # Split -> 1 (deep)
            count = 1
            optimal_jo, exp_run = QUBO_Split_Optimization_func.findind_LR_deep_jo(vars_2, power_set, weights, solver)
            if optimal_jo[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for the l/r deep join tree & #variables:{len(optimal_jo[3])} and #exp:{exp_run}.')
                optimal_jo_details.append(optimal_jo)
            else:
                print(f'No,optimal JO not found for the l/r deep join tree & #variables:{len(optimal_jo[3])} & #exp:{exp_run}.')

            # Split -> 2 (4+2)
            count += 1
            opt_detail, store_jo_details, store_final_vars_4, store_total_cost_4, exprun_s2 = \
                QUBO_Split_Optimization_func.joo_by_split_SS_outer_Bushy(relations, power_set, weights, 4, solver)
            if opt_detail[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for split 4+2 & #variables:{len(opt_detail[3])} and #exp:{exprun_s2}.')
                optimal_jo_details.append(opt_detail)
            else:
                print(f'No,optimal JO not found for split 4+2 & #variables:{len(opt_detail[3])} & #exp:{exprun_s2}.')
            exp_run += exprun_s2

            # Split -> 3 ((4+1)+1)
            count += 1
            vars_56, cost_56 = Helping_functions.single_cost_var(power_set, weights, [5, 6])
            result, total_cost, optimized_jo_vars = Experiments_class.qiskit_experiment_opt_jo_by_vars_list(
                store_final_vars_4 + vars_56, store_total_cost_4 + cost_56, solver
            )
            if total_cost == dp_optimal_cost:
                print(f'Yes,found optimal JO for split (4+1)+1 & #variables{len(store_final_vars_4 + vars_56)} & #exp:{1}.')
                jo_vars = store_jo_details[frozenset(optimized_jo_vars[0])]
                jo_vars.pop()
                optimized_jo_vars.pop(0)
                optimized_jo_vars = jo_vars + optimized_jo_vars
                optimal_jo_details.append([total_cost, optimized_jo_vars, result, store_total_cost_4 + cost_56, store_final_vars_4 + vars_56])
            else:
                print(f'No,optimal JO not found for split (4+1)+1 & #variables{len(store_final_vars_4 + vars_56)} & #exp:{1}.')
            exp_run += 1

            # Split -> 4 ((3+3), (3+2)+1, (2+2+2))
            count += 1
            vars_3, cost_3 = Helping_functions.single_cost_var(power_set, weights, [3])
            result, total_cost, optimized_jo_vars = Experiments_class.qiskit_experiment_opt_jo_by_vars_list(
                vars_2 + vars_3 + vars_56, cost_2 + cost_3 + cost_56, solver
            )
            if total_cost == dp_optimal_cost:
                print(f'Yes,found optimal JO for split (3+2)+1,(2+2+2)&(3+3) & #variables:{len(vars_2 + vars_3 + vars_56)} & #exp:{1}.')
                optimal_jo_details.append([total_cost, optimized_jo_vars, result, cost_2 + cost_3 + cost_56, vars_2 + vars_3 + vars_56])
            else:
                print(f'No,optimal JO not found for split (3+2)+1,(2+2+2)&(3+3) & #variables:{len(vars_2 + vars_3 + vars_56)} & #exp:{1}.')
            exp_run += 1
            print(f'# total exps run:{exp_run} and #splits{count}.')

        # ----- n = 7 -----
        if n == 7:
            count = 1
            optimal_jo, exp_run = QUBO_Split_Optimization_func.findind_LR_deep_jo(vars_2, power_set, weights, solver)
            if optimal_jo[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for the l/r deep join tree & #variables:{len(optimal_jo[3])} and #exp:{exp_run}.')
                optimal_jo_details.append(optimal_jo)
            else:
                print(f'No,optimal JO not found for the l/r deep join tree & #variables:{len(optimal_jo[3])} & #exp:{exp_run}.')

            # 5+2
            count += 1
            opt_detail, store_jo_details, store_final_vars_5, store_total_cost_5, exprun_s2 = \
                QUBO_Split_Optimization_func.joo_by_split_SS_outer_Bushy(relations, power_set, weights, 5, solver)
            if opt_detail[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for split 5+2 & #variables:{len(opt_detail[3])} and #exp:{exprun_s2}.')
                optimal_jo_details.append(opt_detail)
            else:
                print(f'No,optimal JO not found for split 5+2 & #variables:{len(opt_detail[3])} & #exp:{exprun_s2}.')
            exp_run += exprun_s2

            # (5+1)+1
            count += 1
            vars_67, cost_67 = Helping_functions.single_cost_var(power_set, weights, [6, 7])
            result, total_cost, optimized_jo_vars = Experiments_class.qiskit_experiment_opt_jo_by_vars_list(
                store_final_vars_5 + vars_67, store_total_cost_5 + cost_67, solver
            )
            if total_cost == dp_optimal_cost:
                print(f'Yes,found optimal JO for split (5+1)+1 & #variables{len(store_final_vars_5 + vars_67)} & #exp:{1}.')
                jo_vars = store_jo_details[frozenset(optimized_jo_vars[0])]
                jo_vars.pop()
                optimized_jo_vars.pop(0)
                optimized_jo_vars = jo_vars + optimized_jo_vars
                optimal_jo_details.append([total_cost, optimized_jo_vars, result, store_total_cost_5 + cost_67, store_final_vars_5 + vars_67])
            else:
                print(f'No,optimal JO not found for split (5+1)+1 & #variables{len(store_final_vars_5 + vars_67)} & #exp:{1}.')
            exp_run += 1

            # 4+3
            count += 1
            opt_detail, store_jo_details, store_final_vars_34, store_total_cost_34, exprun_s4 = \
                QUBO_Split_Optimization_func.joo_by_split_SS_outer_Bushy(relations, power_set, weights, 4, solver)
            if opt_detail[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for split 4+3 & #variables:{len(opt_detail[3])} and #exp:{exprun_s4}.')
                optimal_jo_details.append(opt_detail)
            else:
                print(f'No,optimal JO not found for split 4+3 & #variables:{len(opt_detail[3])} & #exp:{exprun_s4}.')
            exp_run += exprun_s4

            # 4(t)+1+1+1
            count += 1
            vars_567, cost_567 = Helping_functions.single_cost_var(power_set, weights, [5, 6, 7])
            result, total_cost, optimized_jo_vars = Experiments_class.qiskit_experiment_opt_jo_by_vars_list(
                store_final_vars_34[1] + vars_567, store_total_cost_34[1] + cost_567, solver
            )
            if total_cost == dp_optimal_cost:
                print(f'Yes,found optimal JO for split (4(t)+1)+1 & #variables{len(store_final_vars_34[1] + vars_567)} & #exp:{1}.')
                jo_vars = store_jo_details[frozenset(optimized_jo_vars[0])]
                jo_vars.pop()
                optimized_jo_vars.pop(0)
                optimized_jo_vars = jo_vars + optimized_jo_vars
                optimal_jo_details.append([total_cost, optimized_jo_vars, result, store_total_cost_34[1] + cost_567, store_final_vars_34[1] + vars_567])
            else:
                print(f'No,optimal JO not found for split (4(t)+1)+1 & #variables{len(store_final_vars_34[1] + vars_567)} & #exp:{1}.')
            exp_run += 1

            # (4+2)+1
            count += 1
            l1 = len(vars_2)
            l2 = len(store_final_vars_34[1])
            vars_67, cost_67 = Helping_functions.single_cost_var(power_set, weights, [6, 7])
            result, total_cost, optimized_jo_vars = Experiments_class.qiskit_experiment_inner_split_joo(
                vars_2 + store_final_vars_34[1] + vars_67, cost_2 + store_total_cost_34[1] + cost_67, l1, l2, solver
            )
            if total_cost == dp_optimal_cost:
                print(f'Yes,found optimal JO for split (4+2)+1 & #variables{len(vars_2 + store_final_vars_34[1] + vars_67)} & #exp:{1}.')
                jo_vars = store_jo_details[frozenset(optimized_jo_vars[1])]
                jo_vars.pop()
                optimized_jo_vars.pop(1)
                optimized_jo_vars = jo_vars + optimized_jo_vars
                optimal_jo_details.append([total_cost, optimized_jo_vars, result, cost_2 + store_total_cost_34[1] + cost_67, vars_2 + store_final_vars_34[1] + vars_67])
            else:
                print(f'No,optimal JO not found for split (4+2)+1 & #variables{len(vars_2 + store_final_vars_34[1] + vars_67)} & #exp:{1}.')
            exp_run += 1

            # (3+3)+1
            count += 1
            result, total_cost, optimized_jo_vars = Experiments_class.qiskit_experiment_opt_jo_by_vars_list(
                store_final_vars_34[0] + vars_67, store_total_cost_34[0] + cost_67, solver
            )
            if total_cost == dp_optimal_cost:
                print(f'Yes,found optimal JO for split (3+3)+1 & #variables{len(store_final_vars_34[0] + vars_67)} & #exp:{1}.')
                jo_vars_1 = store_jo_details[frozenset(optimized_jo_vars[0])]; jo_vars_1.pop()
                jo_vars_2 = store_jo_details[frozenset(optimized_jo_vars[1])]; jo_vars_2.pop()
                optimized_jo_vars.pop(0); optimized_jo_vars.pop(0)
                optimized_jo_vars = jo_vars_1 + jo_vars_2 + optimized_jo_vars
                optimal_jo_details.append([total_cost, optimized_jo_vars, result, store_total_cost_34[0] + cost_67, store_final_vars_34[0] + vars_67])
            else:
                print(f'No,optimal JO not found for split (3+3)+1 & #variables{len(store_final_vars_34[0] + vars_67)} & #exp:{1}.')
            exp_run += 1
            print(f'# total exps run:{exp_run} and #splits{count}.')

        # ----- n = 8 -----
        if n == 8:
            count = 1
            optimal_jo, exp_run = QUBO_Split_Optimization_func.findind_LR_deep_jo(vars_2, power_set, weights, solver)
            if optimal_jo[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for the l/r deep join tree & #variables:{len(optimal_jo[3])} and #exp:{exp_run}.')
                optimal_jo_details.append(optimal_jo)
            else:
                print(f'No,optimal JO not found for the l/r deep join tree & #variables:{len(optimal_jo[3])} & #exp:{exp_run}.')

            # 6+2
            count += 1
            opt_detail, store_jo_details, store_final_vars_6, store_total_cost_6, exprun_s2 = \
                QUBO_Split_Optimization_func.joo_by_split_SS_outer_Bushy(relations, power_set, weights, 6, solver)
            if opt_detail[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for split 6+2 & #variables:{len(opt_detail[3])} and #exp:{opt_detail[4]}.')
                optimal_jo_details.append(opt_detail)
            else:
                print(f'No,optimal JO not found for split 6+2 & #variables:{len(opt_detail[3])} & #exp:{opt_detail[4]}.')
            exp_run += exprun_s2

            # (6(t)+1)+1
            count += 1
            vars_78, cost_78 = Helping_functions.single_cost_var(power_set, weights, [7, 8])
            result, total_cost, optimized_jo_vars = Experiments_class.qiskit_experiment_opt_jo_by_vars_list(
                store_final_vars_6 + vars_78, store_total_cost_6 + cost_78, solver
            )
            if total_cost == dp_optimal_cost:
                print(f'Yes,found optimal JO for split (6(t)+1)+1 & #variables{len(store_final_vars_6 + vars_78)} & #exp:{1}.')
                jo_vars = store_jo_details[frozenset(optimized_jo_vars[0])]
                jo_vars.pop()
                optimized_jo_vars.pop(0)
                optimized_jo_vars = jo_vars + optimized_jo_vars
                optimal_jo_details.append([total_cost, optimized_jo_vars, result, store_total_cost_6 + cost_78, store_final_vars_6 + vars_78])
            else:
                print(f'No,optimal JO not found for split (6(t)+1)+1 & #variables{len(store_final_vars_6 + vars_78)} & #exp:{1}.')
            exp_run += 1

            # 5+3
            count += 1
            opt_detail, store_jo_details, store_final_vars_35, store_total_cost_35, exprun_s4 = \
                QUBO_Split_Optimization_func.joo_by_split_SS_outer_Bushy(relations, power_set, weights, 5, solver)
            if opt_detail[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for split 5+3 & #variables:{len(opt_detail[3])} and #exp:{exprun_s4}.')
                optimal_jo_details.append(opt_detail)
            else:
                print(f'No,optimal JO not found for split 5+3 & #variables:{len(opt_detail[3])} & #exp:{exprun_s4}.')
            exp_run += exprun_s4

            # 5(t)+1+1+1
            count += 1
            vars_6, cost_6 = Helping_functions.single_cost_var(power_set, weights, [6])
            result, total_cost, optimized_jo_vars = Experiments_class.qiskit_experiment_opt_jo_by_vars_list(
                store_final_vars_35[1] + vars_6 + vars_78, store_total_cost_35[1] + cost_6 + cost_78, solver
            )
            if total_cost == dp_optimal_cost:
                print(f'Yes,found optimal JO for split 5(t)+1+1+1 & #variables{len(store_final_vars_35[1] + vars_6 + vars_78)} & #exp:{1}.')
                jo_vars = store_jo_details[frozenset(optimized_jo_vars[0])]
                jo_vars.pop()
                optimized_jo_vars.pop(0)
                optimized_jo_vars = jo_vars + optimized_jo_vars
                optimal_jo_details.append([total_cost, optimized_jo_vars, result, store_total_cost_35[1] + cost_6 + cost_78, store_final_vars_35[1] + vars_6 + vars_78])
            else:
                print(f'No,optimal JO not found for split 5(t)+1+1+1 & #variables{len(store_final_vars_35[1] + vars_6 + vars_78)} & #exp:{1}.')
            exp_run += 1

            # (5+2)+1 and (3+4)+1 via reduced constraints
            count += 1
            l1 = len(vars_2)
            l2 = len(store_final_vars_35[1])
            res, exprun_s6 = QUBO_Split_Optimization_func.reduce_contraints_for_disjoint_inner_join(
                vars_2 + store_final_vars_35[1] + vars_78,
                cost_2 + store_total_cost_35[1] + cost_78,
                l1, l2, solver
            )
            if res[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for split (5+2)+1 and (3+4)+1 & #variables{len(res[4])} & #exp:{exprun_s6}.')
                jo_vars = store_jo_details[frozenset(res[1][1])]
                jo_vars.pop()
                res[1].pop(1)
                res[1] = jo_vars + res[1]
                optimal_jo_details.append(res)
            else:
                print(f'No,optimal JO not found for split (5+2)+1 and (3+4)+1 & #variables{len(res[4])} & #exp:{exprun_s6}.')
            exp_run += exprun_s6

            # 4+4
            count += 1
            opt_detail, store_jo_details, store_final_vars_44, store_total_cost_44, exprun_s7 = \
                QUBO_Split_Optimization_func.joo_by_split_SS_outer_Bushy(relations, power_set, weights, 4, solver)
            if opt_detail[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for split 4+4 & #variables:{len(opt_detail[3])} and #exp:{exprun_s7}.')
                optimal_jo_details.append(opt_detail)
            else:
                print(f'No,optimal JO not found for split 4+4 & #variables:{len(opt_detail[3])} & #exp:{exprun_s7}.')
            exp_run += exprun_s7

            # 4(t)+1+1+1+1
            count += 1
            vars_5, cost_5 = Helping_functions.single_cost_var(power_set, weights, [5])
            vars_4t = store_final_vars_44[0] + store_final_vars_44[1]
            total_cost_4 = store_total_cost_44[0] + store_total_cost_44[1]
            optimal_jo, exprun_s8 = QUBO_Split_Optimization_func.findind_LR_deep_jo(
                vars_4t, vars_4t + vars_5 + vars_6 + vars_78, total_cost_4 + cost_5 + cost_6 + cost_78, solver
            )
            if optimal_jo[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for split 4(t)+1+1+1+1 & #variables:{len(optimal_jo[3])} and #exp:{exprun_s8}.')
                optimal_jo_details.append(optimal_jo)
            else:
                print(f'No,optimal JO not found for split 4(t)+1+1+1+1 & #variables:{len(optimal_jo[3])} & #exp:{exprun_s8}.')
            exp_run += exprun_s8
            print(f'# total exps run:{exp_run} and #splits{count}.')

        # ----- Output selection / return -----
        number_of_solutions = len(optimal_jo_details)

        if number_of_solutions == 1:
            vars_list = optimal_jo_details[0][4]
            qiskit_result = optimal_jo_details[0][2]  # Qiskit optimization result
            
            # Handle dimod SampleSet - extract the actual sample
            if hasattr(qiskit_result, 'lowest'):
                sample = qiskit_result.lowest().record[0][0]  # Extract the bitstring
                return Helping_functions.make_join_order_tree(sample, vars_list, relations)
            else:
                return Helping_functions.make_join_order_tree(qiskit_result, vars_list, relations)

        elif number_of_solutions > 1:
            # Choose by minimal total cost; tie-break by length of vars_list
            best_idx, _ = min(enumerate(optimal_jo_details),
                              key=lambda kv: (kv[1][0], len(kv[1][4])))
            vars_list = optimal_jo_details[best_idx][4]
            qiskit_result = optimal_jo_details[best_idx][2]
            
            # Handle dimod SampleSet - extract the actual sample
            if hasattr(qiskit_result, 'lowest'):
                sample = qiskit_result.lowest().record[0][0]  # Extract the bitstring
                return Helping_functions.make_join_order_tree(sample, vars_list, relations)
            else:
                return Helping_functions.make_join_order_tree(qiskit_result, vars_list, relations)

        else:
            # Fallback: no solution collected; construct from DP result
            print('No solution found!!!')
            return [], [], "No solution found"


if __name__ == "__main__":
    # 1) Define relations
    relations = ['R1', 'R2', 'R3', 'R4']
    print(f"Optimizing join order for {len(relations)} relations: {relations}")

    # 2) Generate random weights for all |S|>=2 subsets
    weights = Helping_functions.init_weights(relations)
    print(f"Generated {len(weights)} random weights for the joins.")

    # 3) Check Qiskit availability for QAOA
    try:
        Experiments_class._check_qiskit_available()
        solver_to_use = 'qaoa'
        print(f"Qiskit available. Using '{solver_to_use}' solver with StatevectorSampler.")
    except ImportError as e:
        print(f"Error: {e}")
        exit(1)

    # 4) Optimizer instance
    optimizer = QUBO_Split_Optimization_func(filename="qaoa_test")

    # 5) Run optimization with QAOA
    print("\nStarting QAOA optimization...")
    start_time = time.time()
    join_tree, selected_joins, error_msg = optimizer.finding_opt_jo(
        relations,
        weights,
        solver=solver_to_use
    )
    end_time = time.time()
    print(f"\nQAOA optimization finished in {end_time - start_time:.2f} seconds.")

    # 6) Report result
    if error_msg:
        print(f"An error occurred: {error_msg}")
    else:
        print("\nOptimal Join Order Found:")
        print(join_tree)