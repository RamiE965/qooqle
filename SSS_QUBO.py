# Imports - General purposes
import random
import numpy as np
from itertools import combinations
import csv
import pandas as pd
import itertools
from math import comb
import time

# Imports - D-Wave
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import ConstrainedQuadraticModel, BinaryQuadraticModel
from dimod.binary_quadratic_model import BinaryQuadraticModel
from itertools import combinations
from neal import SimulatedAnnealingSampler
from dimod.serialization.format import Formatter



class QUBO_formulation:
    """
    The QUBO-formulation class is used to create a quadratic model with the aim of finding the optimal join order.
    """

    def __init__(self, relations):
        self.relations = relations

    def relation_sublists(relations):
        # Creates the power set from the relations
        relations_sublists = sum([list(map(list, combinations(relations, i))) for i in range(len(relations) + 1)], [])

        # Removes the superfluous initial relations and the empty set from the power set
        for i in range(len(relations)+1):
            relations_sublists.pop(0)
        return relations_sublists
    
    def construct_QUBO(relations):
        """
        Method for creating the QUBO model.

        Parameter:
        relations: An n-element list of the relations involved in the join tree.
        """

        # Forming the power set and removing the single relations and the empty set
        relations_sublists = QUBO_formulation.relation_sublists(relations)

        # Number of relation combinations
        n = len(relations_sublists)

        # S_1
        s_1_helper = relations_sublists
        
        # S_2
        s_2_helper = []
        for i in range(n):
            for j in range(i,n):
                if (i != j):
                    if(set(relations_sublists[i]).intersection(set(relations_sublists[j]))):
                        if(len(relations_sublists[i]) <= len(relations_sublists[j])):
                            if(not set(relations_sublists[i]).issubset(set(relations_sublists[j]))):
                                s_2_helper.append([i,j])
                                
        return s_1_helper, s_2_helper, n
    
    def construct_QUBO_opt_jo_higher_vars(lst):
        """
        Method for creating the QUBO model.

        Parameter:
        lst : a list of join order variables.
        
        """

        # Forming the power set and removing the single relations and the empty set
        #relations_sublists = QUBO_formulation.relation_sublists(relations)

        # Number of relation combinations
        n = len(lst)

        # S_1
        s_1_helper = lst 
        
        # S_2
        s_2_helper = []
        for i in range(n):
            for j in range(i,n):
                if (i != j):
                    if(set(lst[i]).intersection(set(lst[j]))):
                        if(len(lst[i]) <= len(lst[j])):
                            if(not set(lst[i]).issubset(set(lst[j]))):
                                s_2_helper.append([i,j])
                                
        return s_1_helper, s_2_helper, n
    
    def construct_qubo_for_inner_split(lst,l1,l2):
        '''
        this function is defined for realizing the contraints for the qubo formulation
        this formulation is specially defined to solve the inner split of a search space.
        e.g. n = (m+p)+1+1...+1    where m & p < n
        Parameter:
        lst: list of variables containing total cost element as well as single cost
        we also need to define the number of variables with total cost.
        l1 : number of total cost varibale that is m.
        l2 : number of total cost varibale that is p.
        '''
        l3 = l1 + l2
        n = len(lst)
        
        # S_1
        s_1_helper = lst

        # S_2
        s_2_helper = []
        for i in range(n):
            for j in range(i,n):
                if(i!=j):
                    if(set(lst[i]).intersection(set(lst[j]))):
                        if(len(lst[i]) <= len(lst[j])):
                            if(not set(lst[i]).issubset(set(lst[j]))):
                                s_2_helper.append([i,j])
        # S_3
        s_3_helper = []
        for i in range(l1):
            for j in range(l1,l3):
                if(not set(lst[i]).isdisjoint(set(lst[j]))):
                    s_3_helper.append([i,j])
                    
        # S_3_adding more constraints
        for i in range(l1):
            for j in range(i,l1):
                if(i!=j):
                    if(set(lst[i]).isdisjoint(set(lst[j]))):
                        s_3_helper.append([i,j])

        return s_1_helper, s_2_helper, s_3_helper, n

class Solvers_dwave:
    """
    The Solvers_dwave class is used to solve the modeled QUBO problem on D-Wave quantum computers.
    """
    def prepare_model(s_1_helper, s_2_helper, n, weights_lst):

        w_max = max(weights_lst) * 2

        constrained_quadratic_model = ConstrainedQuadraticModel() # initialize the quadratic model.
        objective = BinaryQuadraticModel(vartype='BINARY') # initialize the objective.

        for i in range(n):
            objective.add_variable(i)

        # S_1
        for i in range(0, len(s_1_helper)):
            objective.set_linear(i, (weights_lst[i] - w_max))
        
        # S_2
        for i in range(0, len(s_2_helper)):
            objective.set_quadratic(s_2_helper[i][0], s_2_helper[i][1], + w_max)
            
        constrained_quadratic_model.set_objective(objective)
        
        #print(objective)

        return constrained_quadratic_model
    
    def prepare_model_with_s3(s_1_helper, s_2_helper, s_3_helper, n, weights_lst):

        w_max = max(weights_lst) * 2

        constrained_quadratic_model = ConstrainedQuadraticModel() # initialize the quadratic model.
        objective = BinaryQuadraticModel(vartype='BINARY') # initialize the objective.

        for i in range(n):
            objective.add_variable(i)

        # S_1
        for i in range(0, len(s_1_helper)):
            objective.set_linear(i, (weights_lst[i] - w_max))
        
        # S_2
        for i in range(0, len(s_2_helper)):
            objective.set_quadratic(s_2_helper[i][0], s_2_helper[i][1], + w_max)
            
        # S_3
        for i in range(0, len(s_3_helper)):
            objective.set_quadratic(s_3_helper[i][0], s_3_helper[i][1], + w_max)
            
        constrained_quadratic_model.set_objective(objective)
        
        #print(objective)

        return constrained_quadratic_model

    
    def exact_result(constrained_quadratic_model):
        """
        Method for exactly calculating an optimal join order.

        Parameter:
        quadratic_program: Quadratic program on which to run the optimization algorithm.
        """
        binary_quadratic_model, invert = dimod.cqm_to_bqm(constrained_quadratic_model)
        result = dimod.ExactSolver().sample(binary_quadratic_model)
        return result

    def simulated_annealing(constrained_quadratic_model):
        """
        Method for calculating an optimal join order using simulated annealing.

        Parameter:
        quadratic_program: Quadratic program on which to run the optimization algorithm.
        """
        #algorithm_globals.random_seed = 1234
        binary_quadratic_model, invert = dimod.cqm_to_bqm(constrained_quadratic_model)
        result = SimulatedAnnealingSampler().sample(binary_quadratic_model, num_reads=100000)
        return result
    
    def dwave_sampler(constrained_quadratic_model):
        """
        Method for calculating an optimal join order using D-Wave QPU.

        Parameter:
        quadratic_program: Quadratic program on which to run the optimization algorithm.
        """
        endpoint = 'https://cloud.dwavesys.com/sapi'
        token = 'DEV-8ed115f6b1ac9df988c81c724c3f97ae0a9d5856'
        
        #solver = 'DW_2000Q_6'
        solver = 'Advantage_system4.1'

        binary_quadratic_model, invert = dimod.cqm_to_bqm(constrained_quadratic_model)
        dw = DWaveSampler(endpoint=endpoint, token=token, solver=solver)
        sampler = EmbeddingComposite(dw)

        result = sampler.sample(binary_quadratic_model, num_reads=1000)

        return result

    
class Helping_functions:
    """
    This class contains helper functions for the experimental execution of the QUBO algorithm.
    """
    def init_weights(relations):
        """
        Helper method for initializing random join costs.

        Parameter:
        relations: A list of relations.
        """
        powerset = sum([list(map(list, combinations(relations, i))) for i in range(len(relations) + 1)], [])
        return [random.randint(1,100) for _ in range(len(powerset)-(len(relations) + 1))]
    def print_relations_and_weights(relations, weights):
        """
        Auxiliary method for visualizing all possible relation combinations and their weights.

        Parameter:
        relations: A list of relations.
        weights: Cost of each join.
        """
        file = open("Qiskit-Tests.txt", "a", encoding='utf-8')
        relation_combinations = QUBO_formulation.relation_sublists(relations)
        # Overview of all sets and their weights
        for i in range (len(relation_combinations)):
            print('Variable:-', i, ' (weight:-', weights[i], ') -> ', list(relation_combinations[i]),sep="")
            file.write('\n')
            file.write('Variable ')
            file.write(str(i))
            file.write(' (weight: ')
            file.write(str(weights[i]))
            file.write(') -> ')
            file.write(str(list(relation_combinations[i])))
        file.write('\n')
        file.close()
    
    
    def dynamic_programming(relations, weights):
        t = {}
        relations_sublists = QUBO_formulation.relation_sublists(relations)
        sets = [frozenset([e]) for e in relations]
        sets += [frozenset(e) for e in relations_sublists]
        for e in relations:
            t[frozenset([e])] = (e,0)
        n=len(relations)
        for i, S in enumerate(relations_sublists):
            #if(len(S)==s):
            #while(len(S)==s):
                values = []
                for (a,b) in itertools.combinations(sets,2):
                    if a.isdisjoint(b) and a.union(b)==frozenset(S):
                        values.append(([t[a][0],t[b][0]], t[a][1] + t[b][1] + weights[i]))
                t[frozenset(S)] = (min(values, key=lambda e: e[1]))
        #s+=1
        return t[frozenset(relations)]
    
    
    def make_join_order_tree(possible_state, vars_list, relations):
        state_vars = list()
        if len(possible_state) != len(vars_list):
            last_var = vars_list[-1]
            vars_list = vars_list[:-1]
            for i in range(len(possible_state)):
                if possible_state[i]==1:
                    state_vars.append(vars_list[i])
            state_vars.append(last_var)
        else: 
            for i in range(len(possible_state)):
                if possible_state[i]==1:
                    state_vars.append(vars_list[i])
        forest = {}
        for r in relations:
            forest[frozenset([r])] = [r]
        for v in state_vars:
            for [left, right] in itertools.combinations(forest,2):
                # left and right have to be disjoint and together form the set v
                if left.isdisjoint(right) and left.union(right) == set(v):
                    # remove single element list [a]+[b]==[a,b]!=[[a],[b]]
                    leftValue = forest[left][0] if len(left)==1 else forest[left]
                    rightValue = forest[right][0] if len(right)==1 else forest[right]
                    # add new tree to forest
                    forest[frozenset(v)]= [leftValue, rightValue]
                    # remove used trees
                    forest.pop(left)
                    forest.pop(right)
        if len(forest)==1:
            return forest.popitem()[1], state_vars, None
        else: 
            return [] , state_vars, "More than one tree remained at the end"
        
        
    def separate_variables(op_vars_list, all_vars, cost_list, length):
        temp = []
        temp_1 = []
        cost_1 = 0
        cost_2 = 0
        for itm in op_vars_list:
            if(len(itm)==length):
                temp.append(itm)
        for j in op_vars_list:
            if(set(j).issubset(set(temp[0]))):
                temp_1.append(j)
        for i in temp_1:
            op_vars_list.remove(i)
        for itm in op_vars_list:
            cost_1 += cost_list[all_vars.index(itm)]
        for itm in temp_1:
            cost_2 += cost_list[all_vars.index(itm)]
        op_vars_list.append(cost_1)
        temp_1.append(cost_2)
        return op_vars_list, temp_1
    
    
    def single_cost_var(vars_list, weight_assigned, num_relns):
        '''
        This function is to extract the variables from power
        set P for a specified number of relations  with their respective weights.
        '''
        var_list = [var for var in vars_list if len(var) in num_relns]
        cost_list = [weight_assigned[vars_list.index(var)] for var in var_list]
        return var_list, cost_list


class Experiments_class():
    '''
    Dwave experiment class 
    '''
    def dwave_experiment(relations,weights, solver):
        """
        Method of making measurements in the D-Wave framework.

        Parameter:
        relations: A list of input relations.
        weights: Weights of all possible joins
        solver: The solver to use
        """
        a,b,c = QUBO_formulation.construct_QUBO(relations)
        constrained_quadratic_model = Solvers_dwave.prepare_model(a, b, c, weights)

        #Helping_functions.print_relations_and_weights(relations, weights)

        if(solver == 'exact_result'):
            result = Solvers_dwave.exact_result(constrained_quadratic_model)
        if(solver == 'dwave'):
            result = Solvers_dwave.dwave_sampler(constrained_quadratic_model)
#             dwave.inspector.show(result)
#             Helping_functions.show_number_of_qubits_dwave(result)
        if(solver == 'simulated_annealing'):
            result = Solvers_dwave.simulated_annealing(constrained_quadratic_model)

        #file=Helping_functions.print_qubo_results_dwave(constrained_quadratic_model, relations, result)
        #Formatter(width=200).fprint(result)
        possible_state = result.lowest().record[0][0]
        relation_combinations = QUBO_formulation.relation_sublists(relations)
        optimized_jo_var=list()
        total_cost=0
        for i in range(len(possible_state)):
            if possible_state[i]==1:
                optimized_jo_var.append(relation_combinations[i])
                total_cost += weights[i]
        return result, total_cost, optimized_jo_var
    
    def dwave_experiment_opt_jo_by_vars_list(lst, cost, solver):
        """
        Method of making measurements in the D-Wave framework.

        Parameter:
        lst: A list of input variables.
        cost: cost of joins in the lst
        solver: The solver to use
        """
        #algorithm_globals.random_seed = 1234
        last_var = lst[-1]
        lst.pop()
        last_cost = cost[-1]
        cost.pop()
        a,b,c = QUBO_formulation.construct_QUBO_opt_jo_higher_vars(lst)
        constrained_quadratic_model = Solvers_dwave.prepare_model(a, b, c, cost)

        #Helping_functions.print_relations_and_weights(relations, weights)

        if(solver == 'exact_result'):
            result = Solvers_dwave.exact_result(constrained_quadratic_model)
        if(solver == 'dwave'):
            result = Solvers_dwave.dwave_sampler(constrained_quadratic_model)
            #dwave.inspector.show(result)
            #Helping_functions.show_number_of_qubits_dwave(result)
        if(solver == 'simulated_annealing'):
            result = Solvers_dwave.simulated_annealing(constrained_quadratic_model)

        #file=Helping_functions.print_qubo_results_dwave(constrained_quadratic_model, relations, result)
        #Formatter(width=200).fprint(result)
        possible_outcome = result.lowest().record[0][0]
        optimized_jo_var = list()
        for i in range(len(possible_outcome)):
            if possible_outcome[i]==1:
                optimized_jo_var.append(lst[i])
        optimized_jo_var.append(last_var)
        total_cost=0
        cost.append(last_cost)
        lst.append(last_var)
        for var in optimized_jo_var:
            total_cost += cost[lst.index(var)]
        return result, total_cost, optimized_jo_var, constrained_quadratic_model
    
    def dwave_experiment_inner_split_joo(lst, cost, l1, l2, solver):
        a,b,c,d = QUBO_formulation.construct_qubo_for_inner_split(lst,l1,l2)
        constrained_quadratic_model = Solvers_dwave.prepare_model_with_s3(a, b, c, d, cost)
        if(solver == 'simulated_annealing'):
            result = Solvers_dwave.simulated_annealing(constrained_quadratic_model)
        if(solver == 'dwave'):
            result = Solvers_dwave.dwave_sampler(constrained_quadratic_model)
        
        possible_state=result.lowest().record[0][0]
        optimized_jo_var = list()
        total_cost=0
        for i in range(len(possible_state)):
            if possible_state[i]==1:
                optimized_jo_var.append(lst[i])
                total_cost+=cost[i]
        return result, total_cost, optimized_jo_var



class QUBO_Split_Optimization_func():

    def __init__(self, filename):
        self.logfile = open("servers/qubo/{}.log".format(filename),"w")

    def __del__(self):
        self.logfile.close()


    def logDwaveResult(self, result: dimod.SampleSet, relations, weights, vars_list, dp_optimal_cost, total_cost):
        self.logfile.write("# {} ; {} ; {} ; {}\n".format(",".join(relations), ",".join(str(e) for e in weights), str(total_cost), ','.join(str(e) for e in vars_list)))
        lowest_energy_state = result.aggregate().lowest().record[0][1]
        for r in result.aggregate().record:
            logData = [",".join(relations)]
            logData.append(r[2])
            logData.append(r[1])
            order, variables, error = Helping_functions.make_join_order_tree(r[0], vars_list, relations)
            logData.append("Valid" if error == None else "Invalid")
            logData.append("Optimal" if total_cost == dp_optimal_cost and r.energy == lowest_energy_state else "Not Optimal")
            logData.append(order)
            logData.append(variables)
            self.logfile.write(";".join([str(e) for e in logData]))
            self.logfile.write("\n")
            self.logfile.flush()

    '''
    This function is to get the optimal solutions of a particular length
    from given query using the input weights.
    '''
    
    def fixed_length_var_optimal_sol(relations, weights,length, solver):
        relation_combinations = QUBO_formulation.relation_sublists(relations)
        rel_comb=[list(lst) for lst in combinations(relations,length)]
        total_cost_list=[]
        optimal_rel_list=[]
        optimal_variables=[]
        for rel in rel_comb:
            new_rel_vars_list=QUBO_formulation.relation_sublists(rel)
            new_weights=[]
            for var in new_rel_vars_list:
                new_weights.append(weights[relation_combinations.index(var)])
            a,b,c = Experiments_class.dwave_experiment(rel, new_weights, solver)
            optimal_variables.append(c)
            optimal_rel_list.append(c[-1])
            total_cost_list.append(b)
        return optimal_variables, optimal_rel_list, total_cost_list
    
    
    @staticmethod
    def joo_by_split_SS_outer_Bushy(relations, power_set, weights, length, solver):
        
        '''
        This function is defined to find the JO by splitting the splitted search space further.
        we don't use the dynamic programming but the only quantum annealing and simulated annealing.
        This function can be used for the outer bushy trees.
        '''
        store_jo_details = {}
        temp_1 = []
        temp_2 = []
        cost_1 = []
        cost_2 = []
        store_final_vars = []
        store_total_cost = []
        JO_details = []
        if(len(relations) != 2*length):
            rel_comb=[list(lst) for lst in combinations(relations,length)]
        else:
            rel_comb = [list(lst) for lst in combinations(relations,length)]
            rel_comb = rel_comb[0:int(len(rel_comb)/2)]
        for rel in rel_comb:
            temp_lst = QUBO_formulation.relation_sublists(rel)
            new_rel_vars_list = []
            for var in temp_lst:
                if var in power_set:
                   new_rel_vars_list.append(var) 
            #print('new_rel_vars_list:-',new_rel_vars_list)
            rel_remaining = []
            for elm in relations:
                if(elm not in rel):
                    rel_remaining.append(elm)
            l = len(rel_remaining)
            temp_lst_2 = QUBO_formulation.relation_sublists(rel_remaining)
            new_rel_vars_list_remaining = []
            for var in temp_lst_2:
                if var in power_set:
                    new_rel_vars_list_remaining.append(var)
            #print('new_rel_vars_list_rem:-',new_rel_vars_list_remaining)
            new_list = new_rel_vars_list + new_rel_vars_list_remaining + [relations]
            new_weights = []
            for var in new_list:
                new_weights.append(weights[power_set.index(var)])
            result, total_cost, JO_vars = Experiments_class.dwave_experiment_opt_jo_by_vars_list(new_list, new_weights, solver)
            JO_details.append([total_cost, JO_vars, result, new_weights, new_list])
            JO_vars_copy = JO_vars.copy()
            temp_3 = []
            for var in JO_vars_copy:
                if len(var)==length:
                    temp_3.append(var)
            if len(temp_3) == 0:
                continue
            lst1,lst2 = Helping_functions.separate_variables(JO_vars, new_list, new_weights, length)
            if(l>2):
                store_jo_details[frozenset(lst1[-2])] = lst1
                store_jo_details[frozenset(lst2[-2])] = lst2
                temp_1.append(lst1[-2])
                cost_1.append(lst1[-1])
                temp_2.append(lst2[-2])
                cost_2.append(lst2[-1])
            else:
                store_jo_details[frozenset(lst2[-2])] = lst2
                temp_2.append(lst2[-2])
                cost_2.append(lst2[-1])
        opt_JO_details = (min(JO_details, key = lambda e : e[0]))
        if(len(temp_1)!=0):
            store_final_vars.append(temp_1)
            store_final_vars.append(temp_2)
            store_total_cost.append(cost_1)
            store_total_cost.append(cost_2)
        else:
            store_final_vars += temp_2
            store_total_cost += cost_2
        
        return  opt_JO_details, store_jo_details, store_final_vars, store_total_cost, len(JO_details)

    
    @staticmethod
    def findind_LR_deep_jo(vars_J2R, All_vars, weights, solver):
        '''
        This function is defined for the left and right deep join tree.
        '''
        #relation_combinations = QUBO_formulation.relation_sublists(relations)
        #rel_comb=[list(lst) for lst in combinations(relations, 2)]
        optimal_jo_details = []  # to store the optimal join order details
        for rel in vars_J2R:
            vars_list = []
            cost_list = []
            for i, elem in enumerate(All_vars):
                if(set(rel).issubset(set(elem))):
                    vars_list.append(elem)
                    cost_list.append(weights[i]) 
            result, total_cost, optimized_jo_vars, qb = Experiments_class.dwave_experiment_opt_jo_by_vars_list(vars_list, cost_list, solver)
            optimal_jo_details.append([total_cost, optimized_jo_vars, result, cost_list, vars_list, qb])
        exp_run = len(optimal_jo_details) # to store the number of experiments run
        optimal_jo_details = min(optimal_jo_details, key = lambda e:e[0])
        print('QUBO : ',optimal_jo_details[-1])
        return optimal_jo_details, exp_run
    
    @staticmethod
    def special_function_inner_split(vars_J2R, All_vars, cost, solver):
        '''
        this function is defined to find the join order for the split
        like n = m(t)+1+1+1 where m(t) is the total cost upto m and here m<n.
        Parameters:
        vars_J2R = variables of m(t)
        All_vars = all variable upto n
        cost = cost corresponding to the all_vars
        '''
        possible_jo_details = []
        for var_1 in vars_J2R:
            for var_2 in vars_J2R:
                if(set(var_1).isdisjoint(set(var_2))):
                    vars_list = []
                    cost_list = []
                    for i, elem in enumerate(All_vars):
                        if(set(var_1).issubset(set(elem)) or set(var_2).issubset(set(elem))):
                            vars_list.append(elem)
                            cost_list.append(cost[i])
                    result, total_cost, optimized_jo_vars = Experiments_class.dwave_experiment_opt_jo_by_vars_list(vars_list, cost_list, solver)
                    possible_jo_details.append([total_cost, optimized_jo_vars, result, cost_list, vars_list])
        exp_run = len(possible_jo_details)
        optimal_jo_details = (min(possible_jo_details, key = lambda e:e[0]))
        return optimal_jo_details, exp_run
    
    @staticmethod
    def reduce_contraints_for_disjoint_inner_join(lst, cost, l1, l2, solver):
        l3 = l1 + l2
        sub_lst = lst[0:l1]
        opt_details = []
        n = len(lst)
        for i in sub_lst:
            lst2 = []
            new_list = []
            new_cost_lst = []
            dis_vars = []
            dis_vars = [lst[j] for j in range(l1, l3) if set(i).isdisjoint(set(lst[j]))]
            lst2 = [lst[j] for j in range(l3, n) if set(i).issubset(set(lst[j]))]
            new_list.append(i)
            new_list += dis_vars
            new_list += lst2
            new_cost_lst = [cost[lst.index(j)] for j in new_list]
            l2 = len(dis_vars)
            result, total_cost, optimized_jo_vars = Experiments_class.dwave_experiment_inner_split_joo(new_list, new_cost_lst, 1, l2, solver)
            opt_details.append([total_cost, optimized_jo_vars, result, new_cost_lst, new_list])
        exp_run = len(opt_details)
        opt_detail = (min(opt_details, key = lambda e : e[0]))
        return opt_detail, exp_run
    
    
    def finding_opt_jo(self, relations, weights, solver):
        
        '''
        This definition has been enabled to find the optimal join order for relations 4,5,6,7 and 8.
        '''
        power_set = QUBO_formulation.relation_sublists(relations)
        print('number of all variables in power set of a query is:',len(power_set))
        vars_2, cost_2 = Helping_functions.single_cost_var(power_set, weights, [2])
        optimal_jo_details = []
        n = len(relations)
        dp_opt_jo_cost = Helping_functions.dynamic_programming(relations, weights)
        dp_optimal_cost = dp_opt_jo_cost[1]

        '''
        Solving for the 3 relations query.
        '''
        if n == 3:
        #Split -> 1, checking for left and right deep join tree
            count = 1
            optimal_jo, exp_run = QUBO_Split_Optimization_func.findind_LR_deep_jo(vars_2, power_set, weights, solver)
            if optimal_jo[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for the l/r deep join tree & #variables:{len(optimal_jo[3])} and #exp:{exp_run}.')
                optimal_jo_details.append(optimal_jo)
            else:
                print(f'No,optimal JO not found for the l/r deep join tree & #variables:{len(optimal_jo[3])} and #exp:{exp_run}.')
            
            print(f'# total exps run:{exp_run} & # splits:{count}.')
        
        '''
        Solving for the 4 relations query.
        '''
        if n == 4:
        #Split -> 1, checking for left and right deep join tree
            count = 1
            optimal_jo, exp_run = QUBO_Split_Optimization_func.findind_LR_deep_jo(vars_2, power_set, weights, solver)
            if optimal_jo[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for the l/r deep join tree & #variables:{len(optimal_jo[3])} and #exp:{exp_run}.')
                optimal_jo_details.append(optimal_jo)
            else:
                print(f'No,optimal JO not found for the l/r deep join tree & #variables:{len(optimal_jo[3])} and #exp:{exp_run}.')

        #Split -> 2, checking for the split 2+2
            count += 1
            vars_4, cost_4 = Helping_functions.single_cost_var(power_set, weights, [4])
            result, total_cost, optimized_jo_vars = Experiments_class.dwave_experiment_opt_jo_by_vars_list(vars_2 + vars_4, cost_2 + cost_4, solver)
            if total_cost == dp_optimal_cost:
                print(f'Yes,found optimal JO for split 2+2 & #variables:{len(vars_2 + vars_4)} and #exp:{1}.')
                optimal_jo_details.append([total_cost, optimized_jo_vars,result, cost_2 + cost_4,  vars_2 + vars_4])
            else:
                print(f'No,optimal JO not found for split 2+2 & #variables:{len(vars_2 + vars_4)} and #exp:{1}.')
            exp_run += 1

            print(f'# total exps run:{exp_run} & # splits:{count}.')

        '''
        Solving for the 5 relations query.
        '''
        if n == 5:
        #Split -> 1, checking for left and right deep join tree
            count = 1
            optimal_jo, exp_run = QUBO_Split_Optimization_func.findind_LR_deep_jo(vars_2, power_set, weights, solver)
            if(optimal_jo[0] == dp_optimal_cost):
                print(f'Yes,found optimal JO for the l/r deep join tree & #variables:{len(optimal_jo[3])} & #exp:{exp_run}.')
                optimal_jo_details.append(optimal_jo)
            else:
                print(f'No,optimal JO not found for the l/r deep join tree & #variables:{len(optimal_jo[3])} & #exp:{exp_run}.')
            
        # #Split -> 2, checking for the (4+1) as ((2+2)+1)
        #     count += 1
        #     vars_45, cost_45 = Helping_functions.single_cost_var(power_set, weights, [4,5])
        #     total_vars = vars_2 + vars_45
        #     total_num_vars = len(total_vars)
        #     result, total_cost, optimized_jo_vars = Experiments_class.dwave_experiment_opt_jo_by_vars_list(vars_2 + vars_45, cost_2 + cost_45, solver)
        #     if(total_cost == dp_optimal_cost):
        #         print(f'Yes,found optimal JO for split (4+1)->((2+2)+1) & #variables:{len(total_num_vars)} & #exp:{1}.')
        #         optimal_jo_details.append([total_cost, optimized_jo_vars, result])
        #     else:
        #         print(f'No,optimal JO not found for split (4+1)->((2+2)+1) & #variables:{len(total_num_vars)} & #exp:{1}.')
        #     exp_run += 1

        # #Split -> 3, checking for the 3+2
        #     count += 1
        #     opt_detail, store_jo_details, store_final_vars, store_total_cost, exprun_s3 = QUBO_Split_Optimization_func.joo_by_split_SS_outer_Bushy(relations, power_set, weights, 3, solver)
        #     if(opt_detail[0] == dp_optimal_cost):
        #         print(f'Yes,found optimal JO for split 3+2 & #variables:{len(opt_detail[3])} and #exp:{exprun_s3}.')
        #         optimal_jo_details.append(opt_detail)
        #     else:
        #         print(f'No,optimal JO not found for split 3+2 & #variables:{len(opt_detail[3])} & #exp:{exprun_s3}.') 

        #     exp_run += exprun_s3 
        #     print(f'# total exps run:{exp_run} and # splits:{count}.')

                
        '''
        Solving for the 6 relations query.
        '''
        if n == 6:
        # Split -> 1, checking for left/right deep join tree
            count = 1
            optimal_jo, exp_run = QUBO_Split_Optimization_func.findind_LR_deep_jo(vars_2, power_set, weights, solver)
            if optimal_jo[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for the l/r deep join tree & #variables:{len(optimal_jo[3])} and #exp:{exp_run}.')
                optimal_jo_details.append(optimal_jo)
            else:
                print(f'No,optimal JO not found for the l/r deep join tree & #variables:{len(optimal_jo[3])} & #exp:{exp_run}.')
                
        # Split -> 2, checking for 4+2 Bushy tree
            count += 1
            opt_detail, store_jo_details, store_final_vars_4, store_total_cost_4, exprun_s2 = QUBO_Split_Optimization_func.joo_by_split_SS_outer_Bushy(relations, power_set, weights, 4, solver)
            if opt_detail[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for split 4+2 & #variables:{len(opt_detail[3])} and #exp:{exprun_s2}.')
                optimal_jo_details.append(opt_detail)
            else:
                print(f'No,optimal JO not found for split 4+2 & #variables:{len(opt_detail[3])} & #exp:{exprun_s2}.')
            exp_run += exprun_s2

        # Split -> 3, checking for (4+1)+1
            count += 1
            vars_56 , cost_56 = Helping_functions.single_cost_var(power_set, weights, [5,6])
            result, total_cost, optimized_jo_vars = Experiments_class.dwave_experiment_opt_jo_by_vars_list(store_final_vars_4 + vars_56, store_total_cost_4 + cost_56, solver)
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
                
        # Split -> 4, checking for (3+3), (3+2)+1 and (2+2+2)
            count += 1
            vars_3 , cost_3 = Helping_functions.single_cost_var(power_set, weights, [3])
            result, total_cost, optimized_jo_vars = Experiments_class.dwave_experiment_opt_jo_by_vars_list(vars_2 + vars_3 + vars_56, cost_2 + cost_3 + cost_56, solver)
            if total_cost == dp_optimal_cost:
                print(f'Yes,found optimal JO for split (3+2)+1,(2+2+2)&(3+3) & #variables:{len(vars_2 + vars_3 + vars_56)} & #exp:{1}.')
                optimal_jo_details.append([total_cost, optimized_jo_vars, result, cost_2 + cost_3 + cost_56, vars_2 + vars_3 + vars_56])
            else:
                print(f'No,optimal JO not found for split (3+2)+1,(2+2+2)&(3+3) & #variables:{len(vars_2 + vars_3 + vars_56)} & #exp:{1}.')
            exp_run += 1
            print(f'# total exps run:{exp_run} and #splits{count}.')


        '''
        Solving for the 7 relations query.
        '''
        if n == 7:
        # Split -> 1, checking for left/right deep join tree
            count = 1
            optimal_jo, exp_run = QUBO_Split_Optimization_func.findind_LR_deep_jo(vars_2, power_set, weights, solver)
            if optimal_jo[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for the l/r deep join tree & #variables:{len(optimal_jo[3])} and #exp:{exp_run}.')
                optimal_jo_details.append(optimal_jo)
            else:
                print(f'No,optimal JO not found for the l/r deep join tree & #variables:{len(optimal_jo[3])} & #exp:{exp_run}.')
        # Split -> 2, checking for 5+2 Bushy tree
            count += 1
            opt_detail, store_jo_details, store_final_vars_5, store_total_cost_5, exprun_s2 = QUBO_Split_Optimization_func.joo_by_split_SS_outer_Bushy(relations, power_set, weights, 5, solver)
            if opt_detail[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for split 5+2 & #variables:{len(opt_detail[3])} and #exp:{exprun_s2}.')
                optimal_jo_details.append(opt_detail)
            else:
                print(f'No,optimal JO not found for split 5+2 & #variables:{len(opt_detail[3])} & #exp:{exprun_s2}.')
            exp_run += exprun_s2
        # Split -> 3, checking for (5(t)+1)+1
            count += 1
            vars_67 , cost_67 = Helping_functions.single_cost_var(power_set, weights, [6,7])
            result, total_cost, optimized_jo_vars = Experiments_class.dwave_experiment_opt_jo_by_vars_list(store_final_vars_5 + vars_67, store_total_cost_5 + cost_67, solver)
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
        # Split -> 4, checking for 4+3 Bushy tree
            count += 1
            opt_detail, store_jo_details, store_final_vars_34, store_total_cost_34, exprun_s4 = QUBO_Split_Optimization_func.joo_by_split_SS_outer_Bushy(relations, power_set, weights, 4, solver)
            if opt_detail[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for split 4+3 & #variables:{len(opt_detail[3])} and #exp:{exprun_s4}.')
                optimal_jo_details.append(opt_detail)
            else:
                print(f'No,optimal JO not found for split 4+3 & #variables:{len(opt_detail[3])} & #exp:{exprun_s4}.')
            exp_run += exprun_s4
                
        # Split -> 5, checking for 4(t)+1+1+1
            count += 1
            vars_567 , cost_567 = Helping_functions.single_cost_var(power_set, weights, [5,6,7])
            result, total_cost, optimized_jo_vars = Experiments_class.dwave_experiment_opt_jo_by_vars_list(store_final_vars_34[1] + vars_567, store_total_cost_34[1] + cost_567, solver)
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
                
        #Split -> 6, checking for (4+2)+1
            count += 1
            l1 = len(vars_2)
            l2 = len(store_final_vars_34[1])
            vars_67 , cost_67 = Helping_functions.single_cost_var(power_set, weights, [6,7])
            result, total_cost, optimized_jo_vars = Experiments_class.dwave_experiment_inner_split_joo(vars_2 + store_final_vars_34[1] + vars_67, cost_2 + store_total_cost_34[1] + cost_67, l1, l2, solver)
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
                
        #Split -> 7, checking for (3+3)+1
            count += 1
            result, total_cost, optimized_jo_vars = Experiments_class.dwave_experiment_opt_jo_by_vars_list(store_final_vars_34[0] + vars_67, store_total_cost_34[0] + cost_67, solver)
            if total_cost == dp_optimal_cost:
                print(f'Yes,found optimal JO for split (3+3)+1 & #variables{len(store_final_vars_34[0] + vars_67)} & #exp:{1}.')
                jo_vars_1 = store_jo_details[frozenset(optimized_jo_vars[0])]
                jo_vars_1.pop()
                jo_vars_2 = store_jo_details[frozenset(optimized_jo_vars[1])]
                jo_vars_2.pop()
                optimized_jo_vars.pop(0)
                optimized_jo_vars.pop(0)
                optimized_jo_vars = jo_vars_1 + jo_vars_2 + optimized_jo_vars
                optimal_jo_details.append([total_cost, optimized_jo_vars, result, store_total_cost_34[0] + cost_67, store_final_vars_34[0] + vars_67])
            else:
                print(f'No,optimal JO not found for split (3+3)+1 & #variables{len(store_final_vars_34[0] + vars_67)} & #exp:{1}.')
            exp_run += 1
            print(f'# total exps run:{exp_run} and #splits{count}.')

        '''
        Solving for the 8 relations query.
        '''
        if n == 8: 
        #Split -> 1, checking for left/right deep join tree
            count = 1
            optimal_jo, exp_run = QUBO_Split_Optimization_func.findind_LR_deep_jo(vars_2, power_set, weights, solver)
            if optimal_jo[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for the l/r deep join tree & #variables:{len(optimal_jo[3])} and #exp:{exp_run}.')
                optimal_jo_details.append(optimal_jo)
            else:
                print(f'No,optimal JO not found for the l/r deep join tree & #variables:{len(optimal_jo[3])} & #exp:{exp_run}.')  
        #Split -> 2, checking for 6+2 Bushy tree
            count += 1
            opt_detail, store_jo_details, store_final_vars_6, store_total_cost_6, exprun_s2 = QUBO_Split_Optimization_func.joo_by_split_SS_outer_Bushy(relations, power_set, weights, 6, solver)
            if opt_detail[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for split 6+2 & #variables:{len(opt_detail[3])} and #exp:{opt_detail[4]}.')
                optimal_jo_details.append(opt_detail)
            else:
                print(f'No,optimal JO not found for split 6+2 & #variables:{len(opt_detail[3])} & #exp:{opt_detail[4]}.')
            exp_run += exprun_s2 
        #Split -> 3, checking for (6(t)+1)+1
            count += 1
            vars_78 , cost_78 = Helping_functions.single_cost_var(power_set, weights, [7,8])
            result, total_cost, optimized_jo_vars = Experiments_class.dwave_experiment_opt_jo_by_vars_list(store_final_vars_6 + vars_78, store_total_cost_6 + cost_78, solver)
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
        #Split -> 4, checking for 5+3 Bushy tree
            count += 1
            opt_detail, store_jo_details, store_final_vars_35, store_total_cost_35, exprun_s4 = QUBO_Split_Optimization_func.joo_by_split_SS_outer_Bushy(relations, power_set, weights, 5, solver)
            if opt_detail[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for split 5+3 & #variables:{len(opt_detail[3])} and #exp:{exprun_s4}.')
                optimal_jo_details.append(opt_detail)
            else:
                print(f'No,optimal JO not found for split 5+3 & #variables:{len(opt_detail[3])} & #exp:{exprun_s4}.')
            exp_run += exprun_s4
        #Split -> 5, checking for 5(t)+1+1+1
            count += 1
            vars_6 , cost_6 = Helping_functions.single_cost_var(power_set, weights, [6])
            result, total_cost, optimized_jo_vars = Experiments_class.dwave_experiment_opt_jo_by_vars_list(store_final_vars_35[1] + vars_6 + vars_78, store_total_cost_35[1] + cost_6 + cost_78, solver)
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
        #Split -> 6, checking for (5+2)+1 and (3+4)+1
            count += 1
            l1 = len(vars_2)
            l2 = len(store_final_vars_35[1])
            res, exprun_s6 = QUBO_Split_Optimization_func.reduce_contraints_for_disjoint_inner_join(vars_2 + store_final_vars_35[1] + vars_78, cost_2 + store_total_cost_35[1] + cost_78, l1, l2, solver)
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
            
        #Split -> 7, checking for 4+4 Bushy tree
            count += 1
            opt_detail, store_jo_details, store_final_vars_44, store_total_cost_44, exprun_s7 = QUBO_Split_Optimization_func.joo_by_split_SS_outer_Bushy(relations, power_set, weights, 4, solver)
            if opt_detail[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for split 4+4 & #variables:{len(opt_detail[3])} and #exp:{exprun_s7}.')
                optimal_jo_details.append(opt_detail)
            else:
                print(f'No,optimal JO not found for split 4+4 & #variables:{len(opt_detail[3])} & #exp:{exprun_s7}.')
            exp_run += exprun_s7
                
        #Split -> 8, checking for 4(t)+1+1+1+1
            count += 1
            vars_5 , cost_5 = Helping_functions.single_cost_var(power_set, weights, [5])
            vars_4t = store_final_vars_44[0] + store_final_vars_44[1]
            total_cost_4 = store_total_cost_44[0] + store_total_cost_44[1]
            optimal_jo, exprun_s8 = QUBO_Split_Optimization_func.findind_LR_deep_jo(vars_4t, vars_4t + vars_5 + vars_6 + vars_78, total_cost_4 + cost_5 + cost_6 + cost_78, solver)
            if optimal_jo[0] == dp_optimal_cost:
                print(f'Yes,found optimal JO for split 4(t)+1+1+1+1 & #variables:{len(optimal_jo[3])} and #exp:{exprun_s8}.')
                optimal_jo_details.append(optimal_jo)
            else:
                print(f'No,optimal JO not found for split 4(t)+1+1+1+1 & #variables:{len(optimal_jo[3])} & #exp:{exprun_s8}.')
            exp_run += exprun_s8
            print(f'# total exps run:{exp_run} and #splits{count}.')
    
       # Output log generation     
        number_of_solutions = len(optimal_jo_details)
        if number_of_solutions == 1:
            vars_list = optimal_jo_details[0][4]
            lowest_energy_state = optimal_jo_details[0][2].lowest().record[0][0]
            self.logDwaveResult(optimal_jo_details[0][2], relations, optimal_jo_details[0][3], optimal_jo_details[0][4], dp_optimal_cost, optimal_jo_details[0][0])
            return Helping_functions.make_join_order_tree(lowest_energy_state, vars_list, relations)
        elif number_of_solutions > 1:
            print('Multiple solutions found!!!')
            best_solution = []
            for i in range(number_of_solutions):
                best_solution.append([i, optimal_jo_details[i][2].aggregate().lowest().record[0][2]])
            print('all_solution_lists:',best_solution)
            best_solution = max(best_solution , key = lambda e:e[1])
            print('best_solution and index:',best_solution)
            i = best_solution[0]
            vars_list = optimal_jo_details[i][4]
            lowest_energy_state = optimal_jo_details[i][2].lowest().record[0][0]
            self.logDwaveResult(optimal_jo_details[i][2], relations, optimal_jo_details[i][3], optimal_jo_details[i][4], dp_optimal_cost, optimal_jo_details[i][0])
            return Helping_functions.make_join_order_tree(lowest_energy_state, vars_list, relations)
        elif number_of_solutions == 0:
            print('No solution found!!!')
            optimal_jo_details.append(optimal_jo)
            vars_list = optimal_jo_details[0][4]
            lowest_energy_state = optimal_jo_details[0][2].lowest().record[0][0]
            self.logDwaveResult(optimal_jo_details[0][2], relations, optimal_jo_details[0][3], optimal_jo_details[0][4], dp_optimal_cost, optimal_jo_details[0][0])
            return Helping_functions.make_join_order_tree(lowest_energy_state, vars_list, relations)

