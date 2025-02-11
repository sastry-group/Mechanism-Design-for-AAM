import sys
from pathlib import Path
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
import json
import math
from pathlib import Path
from multiprocessing import Pool
import logging
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console
from rich.table import Table
import logging
from rich.logging import RichHandler


# Logging optimization information
file_logger = logging.getLogger("file_logger")
file_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("solver_log.txt")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
file_logger.addHandler(file_handler)


# Add the bluesky package to the path
top_level_path = Path(__file__).resolve().parent.parent
print(str(top_level_path))
sys.path.append(str(top_level_path))

from VertiportStatus import VertiportStatus
from fisher.fisher_int_optimization import agent_allocation_selection, map_goodslist_to_agent_goods, track_delayed_goods
from fisher.FisherGraphBuilder import FisherGraphBuilder
from write_csv import write_output, save_data 
from utils import store_agent_data, process_allocations, rank_allocations, store_market_data, get_next_auction_data, build_edge_information, plot_utility_functions

INTEGRAL_APPROACH = False
UPDATED_APPROACH = True
TOL_ERROR = 1e-3
MAX_NUM_ITERATIONS = 10000


def construct_market(flights, timing_info, sectors, vertiport_usage, default_good_valuation=1, dropout_good_valuation=-1, BETA=1):
    """

    """
    # # building the graph
    # market_auction_time=timing_info["start_time"]
    # start_time_graph_build = time.time()
    # builder = FisherGraphBuilder(vertiport_usage, timing_info)
    # market_graph = builder.build_graph(flights)
    # print(f"Time to build graph: {time.time() - start_time_graph_build}")

    print("Constructing market...")
    start_time_market_construct = time.time()
    # goods_list = list(market_graph.edges) + ['default_good'] + ['dropout_good']
    goods_list = []
    w = []
    u = []
    agent_constraints = []
    agent_goods_lists = []
    
    for flight_id, flight in flights.items():

        print(f"Building graph for flight {flight_id}")
        builder = FisherGraphBuilder(vertiport_usage, timing_info)
        agent_graph = builder.build_graph(flight)
        origin_vertiport = flight["origin_vertiport_id"]
        start_node_time = flight["appearance_time"]
    

        # Add constraints
        nodes = list(agent_graph.nodes)
        edges = list(agent_graph.edges)
        starting_node = origin_vertiport + "_" + str(start_node_time)
        nodes.remove(starting_node)
        nodes = [starting_node] + nodes
        inc_matrix = nx.incidence_matrix(agent_graph, nodelist=nodes, edgelist=edges, oriented=True).toarray()
        # print(f"Agent nodes: {nodes}")
        # print(f"Agent edges: {edges}")
        # if flight_id == "AC005":
        # for row in inc_matrix:

        #     positive_indices = [edges[index] for index in np.where(row == 1)[0]]
        #     negative_indices = [edges[index] for index in np.where(row == -1)[0]]
        #     print(f"{positive_indices} - {negative_indices}")
        # print(row)
        # print(f"Incidence matrix: {inc_matrix}")
        rows_to_delete = []
        for i, row in enumerate(inc_matrix):
            if -1 not in row:
                rows_to_delete.append(i)
        A = np.delete(inc_matrix, rows_to_delete, axis=0)
        A[0] = -1 * A[0]
        valuations = []
        for edge in edges:
            valuations.append(agent_graph.edges[edge]["valuation"]) 

        for i, constraint in enumerate(builder.additional_constraints):
            arrival_index = edges.index(constraint[-1])
            for edge in constraint[:-1]:
                edge_index = edges.index(edge)
                A = np.vstack((A, np.zeros(len(A[0]))))
                A[-1, edge_index] = 1
                A[-1, arrival_index] = -1
                # valuations.append(0)

        if flight_id == "AC005":
            for row in A:
        # row = inc_matrix[-15,:]
                positive_indices = [edges[index] for index in np.where(row == 1)[0]]
                negative_indices = [edges[index] for index in np.where(row == -1)[0]]
                # print(f"{positive_indices} - {negative_indices}")

        for i, constraint in enumerate(builder.additional_constraints):
            arrival_index = edges.index(constraint[-1])
            for edge in constraint[:-1]:
                edge_index = edges.index(edge)
                A = np.vstack((A, np.zeros(len(A[0]))))
                A[-1, edge_index] = 1
                A[-1, arrival_index] = -1
                # valuations.append(0)

        if flight_id == "AC005":
            for row in A:
        # row = inc_matrix[-15,:]
                positive_indices = [edges[index] for index in np.where(row == 1)[0]]
                negative_indices = [edges[index] for index in np.where(row == -1)[0]]
                # print(f"{positive_indices} - {negative_indices}")

        b = np.zeros((len(A),1))
        b[0] = 1
        
        A_with_default_good = np.hstack((A, np.zeros((A.shape[0], 1)), np.zeros((A.shape[0], 1)))) # outside/default and dropout good
        A_with_default_good[0, -1] = 1 # droupout good
        # A_with_default_good = np.hstack((A, np.zeros((A.shape[0], 1)))) # outside/default good
        goods = edges + ['default_good'] + ['dropout_good']
        # print(f"Agent {flight_id}'s goods: {goods}")
        # Appending values for default and dropout goods
        valuations.append(default_good_valuation) # Small positive valuation for default good
        valuations.append(dropout_good_valuation) # Small positive valuation for dropout good

        w.append(flight["budget_constraint"])
        u.append(valuations)
        agent_constraints.append((A_with_default_good, b))
        agent_goods_lists.append(goods)
        goods_list += edges

    goods_list = goods_list + ['default_good'] + ['dropout_good']
    # Remove duplicate goods from goods_list
    goods_list = list(dict.fromkeys(goods_list))
    supply = find_capacity(goods_list, sectors, vertiport_usage)
    # for sup, good in zip(supply, goods_list):
    #     print(f"Supply for {good}: {sup}")
    # print(f"Supply: {supply}")
    
  

    print(f"Time to construct market: {round(time.time() - start_time_market_construct)}")
    return (u, agent_constraints, agent_goods_lists), (w, supply, BETA), (goods_list)


def find_capacity(goods_list, route_data, vertiport_data):
    # Create a dictionary for route capacities, for now just connectin between diff vertiports
    # sector_dict = {sid: sector["hold_capacity"] for sid, sector in sectors_data.items()}
    # route_dict = {(route["origin_vertiport_id"], route["destination_vertiport_id"]): route["capacity"] for route in route_data}

    capacities = np.zeros(len(goods_list)) 
    for i, (origin, destination) in enumerate(goods_list[:-2]): # excluding default/outside good - consider changing this to remove "dropout_good" and "default_good"
        # print(f"Origin: {origin} - Destination: {destination}")
        origin_base = origin.split("_")[0]
        destination_base = destination.split("_")[0]
        if origin_base[0] == 'S' and destination_base[0] == 'S':
            if origin_base[3:] == destination_base[3:]:
                # Staying within a sector
                edge = vertiport_data.get_edge_data(origin, destination)
                capacity = edge['hold_capacity'] - edge['hold_usage']
            else:
                # Traveling between sectors
                capacity = 10
            # capacity = sector_dict.get(origin_base, None)
        # if origin_base != destination_base:
        #     # Traveling between vertiport
        #     capacity = route_dict.get((origin_base, destination_base), None)
        # else:
        elif origin_base[0] == 'V' and destination_base[0] == 'V':
            # Staying within a vertiport
            if origin.endswith('_arr'):
                edge = vertiport_data.get_edge_data(origin, destination)
                capacity = edge['landing_capacity'] - edge['landing_usage']
                # origin_time = origin.replace('_arr', '')
                # node = vertiport_data._node.get(origin_time)
                # capacity = node.get('landing_capacity') - node.get('landing_usage') 
            elif destination.endswith('_dep'):
                edge = vertiport_data.get_edge_data(origin, destination)
                capacity = edge['takeoff_capacity'] - edge['takeoff_usage']
                # destination_time = destination.replace('_dep', '')
                # node = vertiport_data._node.get(destination_time)
                # capacity = node.get('takeoff_capacity') - node.get('takeoff_usage') 
            else:
                edge = vertiport_data.get_edge_data(origin, destination)
                # node = vertiport_data._node.get(origin)
                # if node is None:
                    # print(f"Origin: {origin} - Destination: {destination}")
                    # print(f"Nodes: {vertiport_data.nodes}")
                capacity = edge['hold_capacity'] - edge['hold_usage']
                # print(f"Node hold capacity: {node.get('hold_capacity')}")
                # print(f"Node usage capacity: {node.get('hold_usage')}")
                # print(f"Capacity on edge {origin} to {destination}: {capacity}")
        else:
            if origin_base[0] == 'V':
                # Traveling from vertiport to sector
                origin_time = origin.replace('_dep', '')
                edge = vertiport_data.get_edge_data(origin_time, origin)
                capacity = edge['takeoff_capacity'] - edge['takeoff_usage']
                # node = vertiport_data._node.get(origin_time)
                # capacity = node.get('takeoff_capacity') - node.get('takeoff_usage')
            elif destination_base[0] == 'V':
                # Traveling from sector to vertiport
                destination_time = destination.replace('_arr', '')
                edge = vertiport_data.get_edge_data(destination, destination_time)
                capacity = edge['landing_capacity'] - edge['landing_usage']
                # node = vertiport_data._node.get(destination_time)
                # # if node is None:
                # #     print(f"Origin: {origin} - Destination: {destination}")
                # #     print(f"Nodes: {vertiport_data.nodes}")
                # capacity = node.get('landing_capacity') - node.get('landing_usage')
        # print(f"Capacity on edge {origin} to {destination}: {capacity}")
        capacities[i] = capacity
    
    capacities[-2] = 100 # default/outside good
    capacities[-1] = 100 # dropout good

    return capacities



def update_market(x, values_k, market_settings, constraints, agent_goods_lists, goods_list, 
                  price_default_good, problem, sparse_representation, update_rebates=True, integral=False, price_upper_bound=1000):
    '''
    Update market consumption, prices, and rebates
    '''
    start_time = time.time()
    num_goods = len(goods_list)
    num_agents = len(agent_goods_lists)
    k, p_k, r_k = values_k
    supply, beta = market_settings
    x_sparse_array, y_sparse_array, sparse_agent_x_inds, sparse_agent_y_inds, y_sum_matrix = sparse_representation
    
    # Update consumption
    # y = cp.Variable((num_agents, num_goods - 2)) # dropout and default removed
    # y = cp.Variable((num_agents, num_goods - 1)) # dropout removed (4)
    warm_start = False
    # if problem is None:
    # y = cp.Variable((num_agents, num_goods - 2), integer=integral) 
    y = cp.Variable((len(y_sparse_array), 1), integer=integral)
    y_bar = cp.Variable((num_goods - 2, 1), integer=integral)
    # Do we remove drop out here or not? - remove the default and dropout good
    # objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(x[:,:-1] - y, 'fro')) - (beta / 2) * cp.square(cp.norm(cp.sum(y, axis=0) - supply[:-1], 2))) # (4) (5)
    # objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(x[:,:-2] - y, 'fro')) - (beta / 2) * cp.square(cp.norm(cp.sum(y, axis=0) - supply[:-2], 2)))
    # objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(x - y, 'fro')) - (beta / 2) * cp.square(cp.norm(cp.sum(y, axis=0) - supply, 2)))
    # y_sum = cp.sum(y, axis=0)
    short_p_k = np.array(p_k[:-2]).reshape(-1,1)
    x = np.array(x).reshape(-1,1)
    short_supply = np.array(supply[:-2]).reshape(-1,1)
    y_sum = y_sum_matrix @ y
    # print(f"y summing matrix: {y_summing_matrix}")
    # y_sum = cp.vstack([cp.sum(y[y_sparse_array == i]) for i in range(num_goods - 2)])
    # print(f"Num goods: {num_goods}")
    # print(f"Y sum shape: {y_sum.shape}")
    short_x = np.array(np.concatenate([x[inds[:-2]] for inds in sparse_agent_x_inds])).reshape(-1,1)
    # print(f"Short x shape: {short_x.shape}")
    # print(f"y bar shape: {y_bar.shape}")
    # print(f"Short supply shape: {short_supply.shape}")
    # print(f"Short p_k shape: {short_p_k.shape}")
    objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(short_x - y, 2)) - (beta / 2) * cp.square(cp.norm(y_sum + y_bar - short_supply, 2))  - short_p_k.T @ y_bar)
    cp_constraints = [y_bar >= 0, y<=1, y_bar<=short_supply] # remove default and dropout good
    problem = cp.Problem(objective, cp_constraints)
    warm_start = False
    # problem = cp.Problem(objective)
    # else:
    #     y = problem.variables()[0]
    #     y_bar = problem.variables()[1]
    #     p_k = problem.param_dict['p_k']
    #     x = problem.param_dict['x']
    # p_k.value = p_k_val
    # x.value = x_val
    build_time = time.time() - start_time

    start_time = time.time()
    if integral:
        solvers = [cp.MOSEK]
    else:
        solvers = [cp.CLARABEL, cp.SCS, cp.OSQP, cp.ECOS, cp.CVXOPT]
    for solver in solvers:
        try:
            result = problem.solve(solver=solver, warm_start=warm_start, ignore_dpp=True)
            logging.info(f"Problem solved with solver {solver}")
            break
        except cp.error.SolverError as e:
            logging.error(f"Solver {solver} failed: {e}")
            continue
        except Exception as e:
            logging.error(f"An unexpected error occurred with solver {solver}: {e}")
            continue
    solve_time = time.time() - start_time
    # print(f"Market: Build time: {round(build_time,6)} - Solve time: {round(solve_time,6)} with solver {solver}")

    # Check if the problem was solved successfully
    if problem.status != cp.OPTIMAL:
        logging.error("Failed to solve the problem with all solvers.")
    else:
        logging.info("Optimization result: %s", result)

    y_k_plus_1 = y.value
    y_bar_k_plus_1 = y_bar.value

    # Update prices
    # p_k_plus_1 = np.zeros(p_k.shape)
    # try the options (default good): 
    # (3) do not update price, dont add it in optimization, 
    # (4) update price and add it in optimization, 
    # (5) dont update price but add it in optimization
    # print(f"Current prices: {p_k[:-2].shape}")
    # print(f"Current y bar: {y_bar_k_plus_1.shape}")
    # print(f"Supply: {supply[:-2].shape}")
    # print(f"y_k shape: {np.array([np.sum(y_k_plus_1[y_sparse_array == i]) for i in range(num_goods - 2)]).shape}")
    p_k_plus_1 = np.array(p_k[:-2]).reshape(-1,1) + beta * (np.array([np.sum([y_k_plus_1[y_sparse_array == i]]) for i in range(num_goods - 2)]).reshape(-1,1) + y_bar_k_plus_1.reshape(-1,1) - np.array(supply[:-2]).reshape(-1,1)) #(3) default 
    
    print(f"Updated prices: {p_k_plus_1}")
    # p_k_plus_1 = p_k[:-1] + beta * (np.sum(y_k_plus_1, axis=0) - supply[:-1]) #(4)
    # p_k_plus_1 = p_k[:-2] + beta * (np.sum(y_k_plus_1[:,:-2], axis=0) - supply[:-2]) #(5)
    # p_k_plus_1 = p_k + beta * (np.sum(y_k_plus_1, axis=0) - supply)
    for i in range(len(p_k_plus_1)):
        if p_k_plus_1[i] < 0:
            p_k_plus_1[i] = 0  
        if p_k_plus_1[i] > price_upper_bound:
            p_k_plus_1[i] = price_upper_bound 
    # p_k_plus_1[-1] = 0  # dropout good
    # p_k_plus_1[-2] = price_default_good  # default good
    p_k_plus_1 = np.append(p_k_plus_1.T, [price_default_good,0]) # default and dropout good
    # print(f"Updated prices: {p_k_plus_1}")
    # p_k_plus_1 = np.append(p_k_plus_1, 0)  #  (4) update default good


    # Update each agent's rebates
    if update_rebates:
        r_k_plus_1 = []
        for i in range(num_agents):
            agent_constraints = constraints[i]
            # agent_x = np.array([x.value[i, goods_list.index(good)] for good in agent_goods_lists[i]])
            agent_x = np.array([x[sparse_agent_x_inds[i]]]).reshape(-1,1)
            if UPDATED_APPROACH:
                # agent_x = np.array([x[i, goods_list[:-1].index(good)] for good in agent_goods_lists[i][:-1]])
                agent_x = np.array([x[sparse_agent_x_inds[i]]]).reshape(-1,1)
                print(f"Constraints: {agent_constraints[1]}")
                constraint_violations = np.array([agent_constraints[0][j] @ agent_x - agent_constraints[1][j] for j in range(len(agent_constraints[1]))])
            else:
                constraint_violations = np.array([max(agent_constraints[0][j] @ agent_x - agent_constraints[1][j], 0) for j in range(len(agent_constraints[1]))])
            # print(f"Constraint violations: {constraint_violations}")
            r_k_plus_1.append(r_k[i] + beta * constraint_violations[0][0])
    else:
        r_k_plus_1 = r_k
    return k + 1, y_k_plus_1, p_k_plus_1, r_k_plus_1, problem, solve_time




def update_agents(w, u, p, r, constraints, goods_list, agent_goods_lists, y, beta, x_iter, update_frequency, sparse_representation, rational=False, parallel=False, integral=False):
    num_agents, num_goods = len(w), len(p)
    x_sparse_array, y_sparse_array, sparse_agent_x_inds, sparse_agent_y_inds, _ = sparse_representation

    agent_indices = range(num_agents)
    agent_prices = [np.array([p[goods_list.index(good)] for good in agent_goods_lists[i]]) for i in agent_indices]
    agent_utilities = [np.array(u[i]) for i in agent_indices]
    agent_ys = [np.array(y[inds]).reshape(-1,1) for inds in sparse_agent_y_inds]
    # agent_ys = [np.array([y[i, goods_list[:-2].index(good)] for good in agent_goods_lists[i][:-2]]) for i in agent_indices] # removing dropout and detault good (3)
    # agent_ys = [np.array([y[i, goods_list[:-1].index(good)] for good in agent_goods_lists[i][:-1]]) for i in agent_indices] # removing dropout (4)
    args = [(w[i], agent_utilities[i], agent_prices[i], r[i], constraints[i], agent_ys[i], beta, x_iter, update_frequency, rational, integral) for i in agent_indices]

    # Update agents in parallel or not depending on parallel flag
    # parallel = True
    if not parallel:
        results = []
        adjusted_budgets = []
        build_times = []
        solve_times = []
        for arg in args:
            updates =  update_agent(*arg)
            results.append(updates[0])
            adjusted_budgets.append(updates[1])
            build_times.append(updates[2][0])
            solve_times.append(updates[2][1])
        # results = [update_agent(*arg) for arg in args]
        # print(f"Average build time: {np.mean(build_times)} - Average solve time: {np.mean(solve_times)}")
    else:
        num_processes = 4 # increase based on available resources
        with Pool(num_processes) as pool:
            pooled_results = pool.starmap(update_agent, args)
            results = [result[0] for result in pooled_results]
            adjusted_budgets = [result[1] for result in pooled_results]
            build_times = [result[2][0] for result in pooled_results]
            solve_times = [result[2][1] for result in pooled_results]
        print(f"Average build time: {np.mean(build_times)} - Average solve time: {np.mean(solve_times)}")

    x = np.concatenate(results, axis=0)
    # x = np.zeros((len(x_sparse_array), 1))
    # for inds, agent_x in zip(sparse_agent_x_inds, results):
    #     for good in goods_list:
    #         if good in agent_goods_lists[i]:
    #             x[i, goods_list.index(good)] = agent_x[agent_goods_lists[i].index(good)]
    return x, adjusted_budgets, sum(solve_times)

def update_agent(w_i, u_i, p, r_i, constraints, y_i, beta, x_iter, update_frequency, rational=False, integral=True, solver=cp.SCS):
    """
    Update individual agent's consumption given market settings and constraints
    """
    start_time = time.time()
    # Individual agent optimization
    A_i, b_i = constraints
    # A_bar = A_i[0]

    num_constraints = len(b_i)
    num_goods = len(p)

    if x_iter % update_frequency == 0:
        print(f"{x_iter % update_frequency}")
        # lambda_i = r_i.T @ b_i # update lambda
        lambda_i = r_i * b_i[0][0]
        w_adj = w_i + lambda_i
        # print(w_adj)
        w_adj = max(w_adj, 0)
        # print(f"Adjusted budget: {w_adj}")
    else:
        w_adj = w_i
    # w_adj = abs(w_adj) 

        # print(f"Non-adjusted budget: {w_adj}")
    # optimizer check
    x_i = cp.Variable((num_goods,1), integer=integral)
    if rational:
        regularizers = - (beta / 2) * cp.square(cp.norm(x_i - y_i, 2)) - (beta / 2) * cp.sum([cp.square(cp.maximum(A_i[t] @ x_i - b_i[t], 0)) for t in range(num_constraints)])
        lagrangians = - p.T @ x_i - cp.sum([r_i[t] * cp.maximum(A_i[t] @ x_i - b_i[t], 0) for t in range(num_constraints)])
        objective = cp.Maximize(u_i.T @ x_i + regularizers + lagrangians)
        cp_constraints = [x_i >= 0, x_i<= 1]
        # cp_constraints = [x_i >= 0, p.T @ x_i <= w_adj]
        # objective = cp.Maximize(u_i.T @ x_i)
        # cp_constraints = [x_i >= 0, p.T @ x_i <= w_adj, A_i @ x_i <= b_i]
    elif UPDATED_APPROACH:
        # objective_terms = v_i.T @ x_i[:-2] + v_i_o * x_i[-2] + v_i_d * x_i[-1]
        objective_terms = u_i.T @ x_i
        # regularizers = - (beta / 2) * cp.square(cp.norm(x_i[:-1] - y_i, 2)) - (beta / 2) * cp.square(cp.norm(A_i @ x_i - b_i, 2))  #(4)
        regularizers = - (beta / 2) * cp.square(cp.norm(x_i[:-2] - y_i, 2)) - (beta / 2) * cp.square(cp.norm(A_i[0][:] @ x_i - b_i[0], 2)) # - (beta / 2) * cp.square(cp.norm(A_i @ x_i - b_i, 2))        
        # regularizers = - (beta / 2) * cp.square(cp.norm(x_i - y_i, 2)) - (beta / 2) * cp.square(cp.norm(A_i @ x_i - b_i, 2)) 
        # lagrangians = - p.T @ x_i - r_i.T @ (A_i @ x_i - b_i) # the price of dropout good is 0
        lagrangians = - p.T @ x_i - r_i * (A_i[0][:] @ x_i - b_i[0])
        nominal_objective = w_adj * cp.log(objective_terms)
        objective = cp.Maximize(nominal_objective + lagrangians + regularizers)
        cp_constraints = [x_i >= 0, x_i<= 1, A_i[1:] @ x_i == b_i[1:]]
        # cp_constraints = [x_i >= 0, A_bar @ x_i[:-2] + x_i[-2] >= 0]
    else:
        regularizers = - (beta / 2) * cp.square(cp.norm(x_i - y_i, 2)) - (beta / 2) * cp.sum([cp.square(cp.maximum(A_i[t] @ x_i - b_i[t], 0)) for t in range(num_constraints)])
        lagrangians = - p.T @ x_i - cp.sum([r_i[t] * cp.maximum(A_i[t] @ x_i - b_i[t], 0) for t in range(num_constraints)])
        nominal_objective = w_adj * cp.log(u_i.T @ x_i)
        objective = cp.Maximize(nominal_objective + lagrangians + regularizers)
        cp_constraints = [x_i >= 0, x_i<= 1]
    # check_time = time.time()
    problem = cp.Problem(objective, cp_constraints)
    # problem.solve(solver=solver, verbose=False)

    build_time = time.time() - start_time
    start_time = time.time()
    if integral:
        solvers = [cp.MOSEK]
    else:
        solvers = [cp.SCS, cp.CLARABEL, cp.MOSEK, cp.OSQP, cp.ECOS, cp.CVXOPT]
    for solver in solvers:
        try:
            if solver == cp.MOSEK:
                result = problem.solve(solver=solver, mosek_params={"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-7})
            else:
                result = problem.solve(solver=solver)
            logging.info(f"Agent Opt - Problem solved with solver {solver}")
            break
        except cp.error.SolverError as e:
            # print(f"Solver error {e}")
            logging.error(f"Agent Opt - Solver {solver} failed: {e}")
            continue
        except Exception as e:
            # print(f"Solver error {e}")
            logging.error(f"Agent Opt - An unexpected error occurred with solver {solver}: {e}")
            continue
    solve_time = time.time() - start_time
    # print(f"Solver used: {problem.solver_stats.solver_name}")
    # print(f"Solver stats: {problem.solver_stats}")
    # print(f"Problem status: {problem.status}")
    
    # Check if the problem was solved successfully
    if problem.status != cp.OPTIMAL:
        logging.error("Agent Opt - Failed to solve the problem with all solvers.")
    else:
        logging.info("Agent opt - Optimization result: %s", result)


    return x_i.value, w_adj, (build_time, solve_time)



def run_market(initial_values, agent_settings, market_settings, bookkeeping, sparse_representation, rational=False, price_default_good=10, lambda_frequency=1, price_upper_bound=1000):
    """
    
    """
    print(f"Rebate frequency: {lambda_frequency}, Price upper bound: {price_upper_bound}")
    u, agent_constraints, agent_goods_lists, agent_indices = agent_settings
    y, p, r = initial_values
    w, supply, beta = market_settings
    goods_list = bookkeeping
    x_sparse_array, y_sparse_array, sparse_agent_x_inds, sparse_agent_y_inds, _ = sparse_representation

    x_iter = 0
    prices = []
    rebates = []
    overdemand = []
    x_allocations = []
    market_clearing = []
    y_allocations= []
    error = [] * len(agent_constraints)
    abs_error = [] * len(agent_constraints)
    social_welfare_vector = []
    
    # Algorithm 1
    num_agents = len(agent_goods_lists)
    tolerance = num_agents * np.sqrt(len(supply)-2) * TOL_ERROR  # -1 to ignore default goods
    market_clearing_error = float('inf')
    x_iter = 0
    start_time_algorithm = time.time()  
    
    problem = None
    console = Console(force_terminal=True)
    console.print("[bold green]Starting Market Simulation...[/bold green]")

    iter_start = time.time()
    market_run_times = []

    while x_iter <= MAX_NUM_ITERATIONS:  # max(abs(np.sum(opt_xi, axis=0) - C)) > epsilon:
        # if x_iter == 0: 
        #     beta_init = beta
        # else:
        #     beta = beta_init/np.sqrt(x_iter)
            # beta = beta_init / (x_iter + 1)
        
        # if x_iter == 0:
        #     x = np.zeros((len(x_sparse_array), 1))
        #     # x[:-2] = y # Maybe change back later
        #     adjusted_budgets = w
        # else:
        x, adjusted_budgets, agent_solve_t = update_agents(w, u, p, r, agent_constraints, goods_list, agent_goods_lists, y, beta, x_iter, lambda_frequency, sparse_representation, rational=rational, integral=INTEGRAL_APPROACH)
        x_allocations.append(x) # 
        # x_sum = np.hstack([np.sum(x[sparse_agent_x_inds[i][:-2]]) for i in range(len(agent_goods_lists))])
        x_sum = np.array([np.sum(x[x_sparse_array == i]) for i in range(len(goods_list))])[:-2]
        overdemand.append(x_sum - supply[:-2].flatten())
        # print(f"Overdemand: {x_sum - supply[:-2].flatten()}")
        # overdemand.append(np.sum(x[:,:-2], axis=0) - supply[:-2].flatten())

        iter_constraint_error = 0
        iter_constraint_x_y = 0
        for agent_index in range(len(agent_constraints)):
            # agent_x = np.array([x[agent_index, goods_list.index(good)] for good in agent_goods_lists[agent_index]])
            # agent_y = np.array([y[agent_index, goods_list.index(good)] for good in agent_goods_lists[agent_index][:-2]])
            agent_x = np.concatenate([x[sparse_agent_x_inds[agent_index]]])
            agent_y = np.concatenate([y[sparse_agent_y_inds[agent_index]]])
            constraint_error = agent_constraints[agent_index][0] @ agent_x - agent_constraints[agent_index][1]
            abs_constraint_error = np.sqrt(np.sum(np.square(constraint_error)))
            iter_constraint_error += abs_constraint_error 
            # print(f"Agent x: {agent_x}")
            agent_error = np.sqrt(np.sum(np.square(np.array(agent_x[:-2]).reshape(agent_y.shape) - agent_y)))
            iter_constraint_x_y += agent_error
            if x_iter == 0:
                error.append([abs_constraint_error])
                abs_error.append([agent_error])
            else:
                error[agent_index].append(abs_constraint_error)
                abs_error[agent_index].append(agent_error)
    
        
        # if x_iter % rebate_frequency == 0:
        if True:
            update_rebates = True

        # Update market
        k, y, p, r, problem, market_solve_t = update_market(x, (1, p, r), (supply, beta), agent_constraints, agent_goods_lists, goods_list, 
                                            price_default_good, problem, sparse_representation,
                                            update_rebates=update_rebates, integral=INTEGRAL_APPROACH, price_upper_bound=price_upper_bound)
        y_allocations.append(y)
        rebates.append([[rebate] for rebate in r])
        prices.append(p)
        # current_social_welfare = social_welfare(x, p, u, supply, agent_indices)
        current_social_welfare = 0
        social_welfare_vector.append(current_social_welfare)

        # x_ij = np.sum(x[:,:-2], axis=0) # removing default and dropout good
        excess_demand = x_sum - supply[:-2]
        p = p.T
        market_clearing_error = abs(p[:-2].T @ excess_demand)
        market_clearing.append(market_clearing_error)

        x_iter += 1

        iter_end = time.time()
        market_run_times.append(market_solve_t)

        # Create a table with current metrics
        table = Table.grid(expand=True)

        table = Table(title=f"Iteration {x_iter}")
        table.add_column("Metric", justify="left")
        table.add_column("Value", justify="right")
        table.add_row("Case Tolerance: N*sqrt(Supply)*TOL_ERROR", f"{tolerance:.7f}")
        table.add_row("Tolerance Error: (TOL_ERROR)", f"{TOL_ERROR:.7f}")
        table.add_row("Market Clearing Error (MCE)", f"{market_clearing_error:.7f}")
        table.add_row("Ax - b Error", f"{iter_constraint_error:.7f}")
        table.add_row("x - y Error", f"{iter_constraint_x_y:.7f}")
        table.add_row("Time to solve market", f"{market_solve_t:.7f}")
        table.add_row("Time to run algorithm", f"{agent_solve_t:.7f}")
        table.add_row("Iter time: ", f"{iter_end - iter_start:.7f}")
        

        console.clear()
        console.print(table)
        print(f"Excess demand: {excess_demand}")
        print(f"Prices: {p}")
        # if (market_clearing_error <= tolerance) and (iter_constraint_error <= 0.0001) and (x_iter>=10) and (iter_constraint_x_y <= 0.01):
        if (market_clearing_error <= tolerance) and (iter_constraint_error <= 0.01) and (x_iter>=10) and (iter_constraint_x_y <= 0.1):
            break
        if x_iter == 5:
            break

        iter_start = time.time()

        # print("Iteration: ", x_iter, "- MCE: ", round(market_clearing_error, 5), "-Ax-b. Err: ", iter_constraint_error, " - Tol: ", round(tolerance,3), "x-y error:", iter_constraint_x_y)
        logging.info(f"Iteration: {x_iter}, Market Clearing Error: {market_clearing_error}, Tolerance: {tolerance}")

    console.print("[bold green]Simulation Complete! Optimization results in file: solver_log.txt[/bold green]")
    print(f"Average market time: {np.mean(market_run_times)}")

        # if market_clearing_error <= tolerance:
        #     break

    print(f"Time to run algorithm: {round(time.time() - start_time_algorithm,5)}")

    # Todo: Convert sparse x and y to dense representation
    yplot = []
    for y in y_allocations:
        dense_y = np.zeros((num_agents, len(goods_list) - 2))
        for i in range(num_agents):
            dense_y[i, np.array(y_sparse_array[sparse_agent_y_inds[i]]).reshape(-1,1)] = np.array(y[sparse_agent_y_inds[i]]).reshape(-1,1)
        yplot.append(dense_y)

    agent_allocations = []
    for x in x_allocations:
        dense_x = np.zeros((num_agents, len(goods_list)))
        for i in range(num_agents):
            dense_x[i, np.array(x_sparse_array[sparse_agent_x_inds[i]]).reshape(-1,1)] = np.array(x[sparse_agent_x_inds[i]]).reshape(-1,1)
        agent_allocations.append(dense_x)


    data_to_plot ={
        "x_iter": x_iter,
        "prices": prices,
        "p": p,
        "overdemand": overdemand,
        "error": error,
        "abs_error": abs_error,
        "rebates": rebates,
        "x_allocations": x_allocations,
        "agent_allocations": agent_allocations,
        "market_clearing": market_clearing,
        "agent_constraints": agent_constraints,
        "yplot": yplot,
        "social_welfare_vector": social_welfare_vector,
    }


    # data_to_plot = [x_iter, prices, p, overdemand, error, abs_error, rebates, agent_allocations, market_clearing, agent_constraints, yplot, social_welfare_vector]

    last_prices = np.array(prices[-1])
    # final_prices = last_prices[last_prices > 0]

    # print(f"Error: {[error[i][-1] for i in range(len(error))]}")
    # print(f"Overdemand: {overdemand[-1][:]}")
    return dense_x, last_prices, r, overdemand, agent_constraints, adjusted_budgets, data_to_plot


def social_welfare(x, p, u, supply, agent_indices):

    welfare = 0
    #assuming the utilties are stacked vertically with the same edge order per agent (as x)
    # and removing dropuout and default from eveyr agent utility list
    utility_lists = np.zeros((len(x), len(p)))
    for i, utility_list in enumerate(u):
        agent_utility_mapped = np.zeros(len(p))
        agent_utility_mapped[agent_indices[i]] = utility_list
        utility_lists[i] = agent_utility_mapped
    # utility = [item for sublist in u for item in sublist[:-2]]
    # current_capacity = supply[:-2] - np.sum(x[:,:-2], axis=0) # should this be ceil or leave it as fraciton for now?
    agent_welfare = np.dot(utility_lists[:,:-2], x[:,:-2].T) / len(x) #- np.dot(p[:-2], x[:,:-2].T)
    welfare = np.sum(agent_welfare)

    return welfare



def plotting_market(data_to_plot, desired_goods, output_folder, market_auction_time=None, lambda_frequency=1):

    x_iter = data_to_plot["x_iter"]
    prices = data_to_plot["prices"]
    p = data_to_plot["p"]
    overdemand = data_to_plot["overdemand"]
    error = data_to_plot["error"]
    abs_error = data_to_plot["abs_error"]
    rebates = data_to_plot["rebates"]
    agent_allocations = data_to_plot["agent_allocations"]
    market_clearing = data_to_plot["market_clearing"]
    agent_constraints = data_to_plot["agent_constraints"]
    yplot = data_to_plot["yplot"]
    social_welfare = data_to_plot["social_welfare_vector"]



    # x_iter, prices, p, overdemand, error, abs_error, rebates, agent_allocations, market_clearing, agent_constraints, yplot, social_welfare, desired_goods = data_to_plot
    def get_filename(base_name):
        case_name = output_folder.split("/")[-1]
        if market_auction_time:
            return f"{output_folder}/{base_name}_a{market_auction_time}_{case_name}.png"
        else:
            return f"{output_folder}/{base_name}_{case_name}.png"
    
    # Price evolution
    plt.figure(figsize=(10, 5))
    for good_index in range(len(p) - 2):
        plt.plot(range(1, x_iter + 1), [prices[i][good_index] for i in range(len(prices))])
    plt.plot(range(1, x_iter + 1), [prices[i][-2] for i in range(len(prices))], 'b--', label="Default Good")
    plt.plot(range(1, x_iter + 1), [prices[i][-1] for i in range(len(prices))], 'r-.', label="Dropout Good")
    plt.xlabel('x_iter')
    plt.ylabel('Prices')
    plt.title("Price evolution")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(get_filename("price_evolution"),  bbox_inches='tight')
    plt.close()

    # Overdemand evolution
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, x_iter + 1), overdemand)
    plt.xlabel('x_iter')
    plt.ylabel('Demand - Supply')
    plt.title("Overdemand evolution")
    plt.savefig(get_filename("overdemand_evolution"))
    plt.close()

    # Constraint error evolution
    plt.figure(figsize=(10, 5))
    for agent_index in range(len(agent_constraints)):
        plt.plot(range(1, x_iter + 1), error[agent_index])
    plt.ylabel('Constraint error')
    plt.title("Constraint error evolution $\sum ||Ax - b||^2$")
    plt.savefig(get_filename("linear_constraint_error_evolution"))
    plt.close()

    # Absolute error evolution
    plt.figure(figsize=(10, 5))
    for agent_index in range(len(agent_constraints)):
        plt.plot(range(1, x_iter + 1), abs_error[agent_index])
    plt.ylabel('Constraint error')
    plt.title("Absolute error evolution $\sum ||x_i - y_i||^2$")
    plt.savefig(get_filename("x-y_error_evolution"))
    plt.close()

    # Rebate evolution
    plt.figure(figsize=(10, 5))
    for constraint_index in range(len(rebates[0])):
        plt.plot(range(1, x_iter + 1), [rebates[i][constraint_index] for i in range(len(rebates))])
    plt.xlabel('x_iter')
    plt.ylabel('rebate')
    plt.title("Rebate evolution")
    plt.savefig(get_filename("rebate_evolution"))
    plt.close()

    # Agent allocation evolution
    # plt.figure(figsize=(10, 5))
    # for agent_index in range(len(agent_allocations[0])):
    #     plt.plot(range(1, x_iter + 1), [agent_allocations[i][agent_index][:-2] for i in range(len(agent_allocations))])
    #     plt.plot(range(1, x_iter + 1), [agent_allocations[i][agent_index][-2] for i in range(len(agent_allocations))], 'b--', label=f"{agent_index} - Default Good")
    #     plt.plot(range(1, x_iter + 1), [agent_allocations[i][agent_index][-1] for i in range(len(agent_allocations))], 'r-.', label=f"{agent_index} - Dropout Good")
    # plt.legend()
    # plt.xlabel('x_iter')
    # plt.title("Agent allocation evolution")
    # plt.savefig(get_filename("agent_allocation_evolution"))
    # plt.close()

    # Payment 
    
    plt.figure(figsize=(10, 5))
    # agent allocations 
    for agent_index in range(len(agent_allocations[0])):
        # payment = prices[agent_index] @ agent_allocations[agent_index][0]
        label = f"Flight:{agent_index}" 
        plt.plot(range(1, x_iter + 1), [prices[i] @ agent_allocations[i][agent_index] for i in range(len(prices))], label=label)
    plt.xlabel('x_iter')
    plt.title("Payment evolution")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(get_filename("payment"), bbox_inches='tight')
    plt.close()


    # Desired goods evolution
    plt.figure(figsize=(10, 5))
    # print(f"Allocations: {agent_allocations}")
    agent_desired_goods_list = []
    for agent in enumerate(desired_goods):
        agent_id = agent[0]
        agent_name = agent[1]       
        # dep_index = desired_goods[agent_name]["desired_good_dep"]
        # arr_index = desired_goods[agent_name]["desired_good_arr"]
        label = f"Flight:{agent_name}, {desired_goods[agent_name]['desired_dep_edge']}" 
        # plt.plot(range(1, x_iter + 1), [agent_allocations[i][agent_id][dep_index] for i in range(len(agent_allocations))], '-', label=f"{agent_name}_dep good")
        dep_index = desired_goods[agent_name]["desired_dep_edge_idx"]
        agent_desired_goods = [agent_allocations[i][agent_id][dep_index] for i in range(len(agent_allocations))]
        agent_desired_goods_list.append(agent_desired_goods)
        plt.plot(range(1, x_iter + 1), agent_desired_goods, '--', label=label)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('x_iter')
    plt.title("Desired Goods Agent allocation evolution")
    plt.savefig(get_filename("desired_goods_allocation_evolution"), bbox_inches='tight')
    plt.close()
    print(f"Final Desired Goods Allocation: {[desired_goods[-1] for desired_goods in agent_desired_goods_list]}")

    # Market Clearing Error
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, x_iter + 1), market_clearing)
    plt.xlabel('x_iter')
    plt.title("Market Clearing Error")
    plt.savefig(get_filename("market_clearing_error"))
    plt.close()

    # y
    plt.figure(figsize=(10, 5))
    for agent_index in range(len(yplot[0])):
        plt.plot(range(1, x_iter + 1), [yplot[i][agent_index][:-2] for i in range(len(yplot))])
        # plt.plot(range(1, x_iter + 1), [yplot[i][agent_index][-2] for i in range(len(yplot))], 'b--', label="Default Good")
        # plt.plot(range(1, x_iter + 1), [yplot[i][agent_index][-1] for i in range(len(yplot))], 'r-.', label="Dropout Good")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('x_iter')
    plt.title("Y-values")
    plt.savefig(get_filename("y-values"), bbox_inches='tight')
    plt.close()


    # Social Welfare
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, x_iter + 1), social_welfare)
    plt.xlabel('x_iter')
    plt.ylabel('Social Welfare')
    plt.title("Social Welfare")
    plt.savefig(get_filename("social_welfare"))
    plt.close()

    # Rebate error
    plt.figure(figsize=(10, 5))
    # print(rebates)
    # print(f"Rebate frequency: {lambda_frequency}")
    rebate_error = [[rebates[i][j][0] - rebates[i - i % int(lambda_frequency)][j][0] for j in range(len(rebates[0]))] for i in range(len(rebates))]
    plt.plot(range(1, x_iter + 1), rebate_error)
    plt.xlabel('x_iter')
    plt.ylabel('Rebate error')
    plt.title("Rebate error")
    plt.savefig(get_filename("rebate_error"))
    plt.close()

def load_json(file=None):
    """
    Load a case file for a fisher market test case from a JSON file.
    """
    if file is None:
        return None
    assert Path(file).is_file(), f"File {file} does not exist."

    # Load the JSON file
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        print(f"Opened file {file}")
    return data



def track_desired_goods(flights, goods_list):
    "return the index of the desired goods for each flight"
    desired_goods = {}
    for i, flight_id in enumerate(flights.keys()):
        flight = flights[flight_id]
        appearance_time = flight["appearance_time"]
        desired_request = flight["requests"]["001"]
        origin_vertiport = flight["origin_vertiport_id"]
        desired_dep_time = desired_request["request_departure_time"]
        desired_vertiport = desired_request["destination_vertiport_id"]
        desired_arrival_time = desired_request["request_arrival_time"]
        desired_transit_edges = []
        desired_dep_edge = (f"{origin_vertiport}_{desired_dep_time}", f"{origin_vertiport}_{desired_dep_time}_dep")
        flights_desired_goods = [desired_dep_edge]
        for i in range(appearance_time, desired_dep_time):
            flights_desired_goods.append((f"{origin_vertiport}_{i}", f"{origin_vertiport}_{i+1}"))
        flights_desired_goods.append((f"{origin_vertiport}_{desired_dep_time}_dep", f"{desired_request['sector_path'][0]}_{desired_request['sector_times'][0]}"))
        desired_transit_edges.append((f"{origin_vertiport}_{desired_dep_time}_dep", f"{desired_request['sector_path'][0]}_{desired_request['sector_times'][0]}"))
        for i in range(len(desired_request["sector_path"])):
            sector = desired_request["sector_path"][i]
            start_time = desired_request["sector_times"][i]
            end_time = desired_request["sector_times"][i+1]
            for sector_time in range(start_time, end_time):
                flights_desired_goods.append((f"{sector}_{sector_time}", f"{sector}_{sector_time+1}"))

            if i < len(desired_request["sector_path"]) - 1:
                next_sector = desired_request["sector_path"][i+1]
                flights_desired_goods.append((f"{sector}_{end_time}", f"{next_sector}_{end_time}"))
        if desired_vertiport is not None:
            desired_arr_edge = (f"{sector}_{end_time}", f"{desired_vertiport}_{desired_arrival_time}_arr")
            flights_desired_goods.append((f"{sector}_{end_time}", f"{desired_vertiport}_{desired_arrival_time}_arr"))
            flights_desired_goods.append((f"{desired_vertiport}_{desired_arrival_time}_arr", f"{desired_vertiport}_{desired_arrival_time}"))
            # desired_good_dep_to_arr = (f"{origin_vertiport}_{desired_dep_time}_dep", f"{desired_vertiport}_{desired_arrival_time}_arr")
            # good_id_arr = goods_list.index(desired_good_arr)
            # good_id_dep_to_arr = goods_list.index(desired_good_dep_to_arr)
            # desired_goods[flight_id]["desired_good_dep_to_arr"] = good_id_dep_to_arr
        # good_id_dep = goods_list.index(desired_good_dep)
        # desired_goods[flight_id] = {"desired_good_arr": good_id_arr, "desired_good_dep": good_id_dep, "desired_good_dep_to_arr": good_id_dep_to_arr}
        # desired_goods[flight_id]["desired_good_dep"] = good_id_dep
        # print(f"Desired goods for flight {flight_id}: {flights_desired_goods}")
        index_list = []
        for good in flights_desired_goods:
            index_list.append(goods_list.index(good))
        desired_goods[flight_id] = {"good_indices": index_list, "desired_dep_edge_idx": goods_list.index(desired_dep_edge), 
                                    "desired_dep_edge": desired_dep_edge, "desired_paths": flights_desired_goods, 
                                    "desired_arr_edge": desired_arr_edge, "desired_transit_edges": desired_transit_edges, 
                                    "desired_arr_edge_idx": goods_list.index(desired_arr_edge), 
                                    "desired_transit_edges_idx": [goods_list.index(desired_transit_edges[i]) for i in range(len(desired_transit_edges))]}

    return desired_goods



def fisher_allocation_and_payment(vertiport_usage, flights, timing_info, sectors_data, vertiports, 
                                  output_folder=None, save_file=None, initial_allocation=True, design_parameters=None):

    # # building the graph
    market_auction_time=timing_info["auction_start"]
    if False:
    # if market_auction_time>5:
        price_default_good = 10
        default_good_valuation = 1
        dropout_good_valuation = 40
        BETA = 50
        lambda_frequency = 50
        price_upper_bound = 3000
    else:
        price_default_good = design_parameters["price_default_good"]
        default_good_valuation = design_parameters["default_good_valuation"]
        dropout_good_valuation = design_parameters["dropout_good_valuation"]
        BETA = design_parameters["beta"]
        lambda_frequency = design_parameters["lambda_frequency"]
        price_upper_bound = design_parameters["price_upper_bound"]       
    # start_time_graph_build = time.time()
    # builder = FisherGraphBuilder(vertiport_usage, timing_info)
    # market_graph = builder.build_graph(flights)
    # print(f"Time to build graph: {time.time() - start_time_graph_build}")

    #Extracting design parameters
    # we should create a config file for this
    # if design_parameters:
    #     price_default_good = design_parameters["price_default_good"]
    #     default_good_valuation = design_parameters["default_good_valuation"]
    #     dropout_good_valuation = design_parameters["dropout_good_valuation"]
    #     BETA = design_parameters["beta"]
    #     lambda_frequency = design_parameters["lambda_frequency"]
    #     price_upper_bound = design_parameters["price_upper_bound"]

    # else:
    #     BETA = 1 # chante to 1/T
    #     dropout_good_valuation = -1
    #     default_good_valuation = 1
    #     price_default_good = 10
    #     lambda_frequency = 1
    #     price_upper_bound = 1000

    # Construct market
    agent_information, market_information, bookkeeping = construct_market(flights, timing_info, sectors_data, vertiport_usage, 
                                                                          default_good_valuation=default_good_valuation, 
                                                                          dropout_good_valuation=dropout_good_valuation, BETA=BETA)
    
    # Run market
    goods_list = bookkeeping
    num_goods, num_agents = len(goods_list), len(flights)
    u, agent_constraints, agent_goods_lists = agent_information
    _ , capacity, _ = market_information
    agent_indices = map_goodslist_to_agent_goods(goods_list, agent_goods_lists)
    agent_information = (*agent_information, agent_indices)

    # Sparse Representation
    sparse_agent_x_inds = []
    sparse_agent_y_inds = []
    x_start = 0
    y_start = 0
    y_agent_indices = []
    for inds in agent_indices:
        x_end = len(inds) + x_start
        y_end = len(inds) - 2 + y_start
        sparse_agent_x_inds.append(list(np.arange(x_start, x_end)))
        sparse_agent_y_inds.append(list(np.arange(y_start, y_end)))
        y_agent_indices.append(inds[:-2])
        x_start = x_end
        y_start = y_end
    y_sparse_array = np.concatenate(y_agent_indices)
    x_sparse_array = np.concatenate(agent_indices)
    y_sum_matrix = np.array([[1 if elem == i else 0 for elem in y_sparse_array] for i in range(num_goods - 2)])

    # y = np.random.rand(num_agents, num_goods-2)*10
    dense_y = np.zeros((num_agents, num_goods - 2))
    # y = np.zeros((len(sparse_goods_representation), 1))
    desired_goods = track_desired_goods(flights, goods_list)
    for i, agent_id in enumerate(desired_goods):
        # dept_id = desired_goods[agent_ids]["desired_good_arr"]
        # arr_id = desired_goods[agent_ids]["desired_good_dep"] 
        # dept_to_arr_id = desired_goods[agent_ids]["desired_good_dep_to_arr"]
        # y[i][dept_id]= 1
        # y[i][arr_id] = 1
        # y[i][desired_goods[agent_id]["good_indices"][0]] = 1
        for good_idx in desired_goods[agent_id]["good_indices"]:
            dense_y[i][good_idx] = 1
        # print(f"Initial allocation for agent {i}: {y[i]}")
        # print(f"Goods: {agent_goods_lists[i]}")
        # ybar = np.array([y[i, goods_list.index(good)] for good in agent_goods_lists[i][:-2]] + [0,0])
        # print(f"y bar: {ybar}")
        # agent_goods = agent_goods_lists[i]
        # print(f"Agent goods: {agent_goods}")
        # print(f"ybar is 1: {[agent_goods[ind] for ind in np.where(ybar == 1)[0]]}")
        # print(f"Ax - b: {agent_constraints[i][0] @ ybar - agent_constraints[i][1]}")
        # invalid_constraints = np.where(agent_constraints[i][0] @ ybar - agent_constraints[i][1] == 1)
        # print(f"A: {agent_constraints[i][0]}")
        # assert all(agent_constraints[i][0] @ ybar - agent_constraints[i][1] == 0), f"Initial allocation for agent {i} does not satisfy constraints for agent {i}"
    # y = np.random.rand(num_agents, num_goods)
    y = np.concatenate([[agent_y[ind] for ind in y_sparse_array[inds]] for agent_y, inds in zip(dense_y, sparse_agent_y_inds)])
    p = np.zeros(num_goods)
    p[-2] = price_default_good 
    p[-1] = 0 # dropout good
    # r = [np.zeros(len(agent_constraints[i][1])) for i in range(num_agents)]
    r = np.zeros(num_agents)

    x, prices, r, overdemand, agent_constraints, adjusted_budgets, data_to_plot = run_market((y,p,r), agent_information, market_information, 
                                                             bookkeeping, (x_sparse_array, y_sparse_array, sparse_agent_x_inds, sparse_agent_y_inds, y_sum_matrix),
                                                             rational=False, price_default_good=price_default_good, 
                                                             lambda_frequency=lambda_frequency, price_upper_bound=price_upper_bound)
    

    print("---FINAL ALLOCATION---")
    for agent_x, desired_good in zip(x, desired_goods):
        print(f"Agent allocation of goods: {[goods_list[i] for i in np.where(x[0] > 0.1)[0]]}")
        print(f"Partially allocated good values: {[x[0][i] for i in np.where(x[0] > 0.1)[0]]}")
    # print(f"Partial allocation for 0th agent: {[goods_list[i] for i in np.where(x[0] > 0.1)[0]]}")
    # print(f"Partial allocation for 1st agent: {[goods_list[i] for i in np.where(x[1] > 0.1)[0]]}")
    # print(f"Partially allocated good values: {[x[0][i] for i in np.where(x[0] > 0.1)[0]]}")
    # print(f"Partially allocated good values: {[x[1][i] for i in np.where(x[1] > 0.1)[0]]}")
    # print(f"Full agent allocations: {x[2][-2]}")
    # print(f"Goods list: {goods_list}")
    # print(f"{np.where(np.array(goods_list) == ('S001_37', 'S002_37'))}")
    # print(f"Weird allocation: {[x[2][ind] for ind in np.where(goods_list == ('S001_37', 'S002_37'))[0]]}")
    # print(f"Weird allocation: {[x[2][ind] for ind in np.where(goods_list == ('S002_37', 'S002_38'))[0]]}")
        # allocated_goods = [agent_goods_list[i] for i in np.where(row > 0.9)[0]]
        # print(f"{row}")
    # print(f"Final allocation: {x}")
    extra_data = {
    'x_prob': x,
    'prices': prices,
    'rebates': r,
    'agent_constraints': agent_constraints,
    'adjusted_budgets': adjusted_budgets,
    'desired_goods': desired_goods,
    'goods_list': goods_list,
    'capacity': capacity,
    'data_to_plot': data_to_plot,
    'agent_goods_lists': agent_goods_lists}
    save_data(output_folder, "fisher_data", market_auction_time, **extra_data)
    plotting_market(data_to_plot, desired_goods, output_folder, market_auction_time)
    
    # Building edge information for mapping - move this to separate function
    # move this part to a different function
    edge_information = build_edge_information(goods_list)


    agents_data_dict = store_agent_data(flights, x, agent_information, adjusted_budgets, desired_goods, agent_goods_lists, edge_information)
    market_data_dict = store_market_data(extra_data, design_parameters, market_auction_time)
    agents_data_dict = track_delayed_goods(agents_data_dict, market_data_dict)
    # Rank agents based on their allocation and settling any contested goods
    sorted_agent_dict, ranked_list = rank_allocations(agents_data_dict, market_data_dict)
    agents_data_dict, market_data_dict= agent_allocation_selection(ranked_list, agents_data_dict, market_data_dict)
    valuations = {key: agents_data_dict[key]["valuation"] for key in agents_data_dict.keys()}

    # Getting data for next auction
    allocation, rebased, dropped, = get_next_auction_data(agents_data_dict, market_data_dict)
    market_data_dict = plot_utility_functions(agents_data_dict, market_data_dict, output_folder)

    # print(f"Allocation: {allocation}")

    output_data = {"market_data":market_data_dict, "agents_data":agents_data_dict, "ranked_list":ranked_list, "valuations":valuations}
    save_data(output_folder, "fisher_data_after", market_auction_time, **output_data)


    write_output(flights, edge_information, market_data_dict, 
                agents_data_dict, market_auction_time, output_folder)

    return allocation, rebased, dropped, valuations
    



if __name__ == "__main__":
    pass

