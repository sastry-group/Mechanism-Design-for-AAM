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


logging.basicConfig(filename='solver_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Add the bluesky package to the path
top_level_path = Path(__file__).resolve().parent.parent
print(str(top_level_path))
sys.path.append(str(top_level_path))

from VertiportStatus import VertiportStatus
from fisher.sampling_graph import build_edge_information, agent_probability_graph_extended, sample_path, plot_sample_path_extended, process_allocations, mapping_agent_to_full_data, mapping_goods_from_allocation
from fisher.fisher_int_optimization import int_optimization
from write_csv import write_output

UPDATED_APPROACH = True
TOL_ERROR = 1e-3
MAX_NUM_ITERATIONS = 300
# BETA = 1
# dropout_good_valuation = -1
# default_good_valuation = 1
# price_default_good = 10



def build_graph(vertiport_status, timing_info):
    """

    """
    print("Building graph...")
    start_time_graph_build = time.time()
    max_time, time_step = timing_info["end_time"], timing_info["time_step"]

    auxiliary_graph = nx.DiGraph()
    ## Construct nodes
    #  Create dep, arr, and standard nodes for each initial node (vertiport + time step)
    for node in vertiport_status.nodes:
        auxiliary_graph.add_node(node + "_dep")
        auxiliary_graph.add_node(node + "_arr")
        auxiliary_graph.add_node(node)

    ## Construct edges
        # Create arr -> standard edges
        # auxiliary_graph.add_edge(node + "_arr", node, time=0, weight=0)
        auxiliary_graph.add_edge(node + "_arr", node)

        # Create standard -> dep edges
        # auxiliary_graph.add_edge(node, node + "_dep", time=max_time, weight=0)
        auxiliary_graph.add_edge(node, node + "_dep")

        # Connect standard nodes to node at next time step
        if vertiport_status.nodes[node]["time"] != max_time:
            vertiport_id = vertiport_status.nodes[node]["vertiport_id"]
            next_time = vertiport_status.nodes[node]["time"] + time_step
            auxiliary_graph.add_edge(node, vertiport_id + "_" + str(next_time))

    for edge in vertiport_status.edges:
        origin_vertiport_id_with_depart_time, destination_vertiport_id_with_arrival_time = edge
        # auxiliary_graph.add_edge(origin_vertiport_id_with_depart_time + "_dep", destination_vertiport_id_with_arrival_time + "_arr", time=0, weight=0)
        auxiliary_graph.add_edge(origin_vertiport_id_with_depart_time + "_dep", destination_vertiport_id_with_arrival_time + "_arr")

    print(f"Time to build graph: {time.time() - start_time_graph_build}")
    return auxiliary_graph


def construct_market(market_graph, flights, timing_info, routes, vertiport_usage, default_good_valuation=1, dropout_good_valuation=-1, BETA=1):
    """

    """
    max_time, time_step = timing_info["end_time"], timing_info["time_step"]
    times_list = list(range(timing_info["start_time"], max_time + time_step, time_step))

    print("Constructing market...")
    start_time_market_construct = time.time()
    goods_list = list(market_graph.edges) + ['default_good'] + ['dropout_good']
    w = []
    u = []
    agent_constraints = []
    agent_goods_lists = []
    for flight_id, flight in flights.items():
        origin_vertiport = flight["origin_vertiport_id"]
        # Create agent graph
        agent_graph = nx.DiGraph()
        # for node_time in times_list:
        #     agent_graph.add_edge(origin_vertiport + "_" + str(node_time), origin_vertiport + "_" + str(node_time) + "_dep")
        for request_id, request in flight["requests"].items():
            if request["request_departure_time"] == 0:
                for start_time, end_time in zip(times_list[:-1],times_list[1:]):
                    start_node, end_node = origin_vertiport + "_" + str(start_time), origin_vertiport + "_" + str(end_time)
                    if end_time == times_list[-1]:
                        attributes = {"valuation": request["valuation"]}
                    else:
                        attributes = {"valuation": 0}
                        agent_graph.add_edge(start_node, end_node, **attributes)
            else:
                dep_time = request["request_departure_time"]
                arr_time = request["request_arrival_time"]
                destination_vertiport = request["destination_vertiport_id"]
                decay = flights[flight_id]["decay_factor"]
                for i in range(5):
                    start_node, end_node = origin_vertiport + "_" + str(dep_time + i) + "_dep", destination_vertiport + "_" + str(arr_time+i) + "_arr"
                    valuation = request["valuation"] * decay**i
                    attributes = {"valuation": valuation}
                    agent_graph.add_edge(start_node, end_node, **attributes)
                    dep_start_node, dep_end_node = origin_vertiport + "_" + str(dep_time + i), origin_vertiport + "_" + str(dep_time + i) + "_dep"
                    arr_start_node, arr_end_node = destination_vertiport + "_" + str(arr_time + i) + "_arr", destination_vertiport + "_" + str(arr_time + i)
                    agent_graph.add_edge(dep_start_node, dep_end_node, **{"valuation": 0})
                    agent_graph.add_edge(arr_start_node, arr_end_node, **{"valuation": 0})
                # stationary_times = [time for time in times_list if time >= (arr_time)]
                stationary_times = times_list[times_list.index(arr_time):]
                for start_time, end_time in zip(stationary_times[:-1], stationary_times[1:]):
                    start_node, end_node = destination_vertiport + "_" + str(start_time), destination_vertiport + "_" + str(end_time)
                    attributes = {"valuation": 0}
                    agent_graph.add_edge(start_node, end_node, **attributes)

        # Add constraints
        nodes = list(agent_graph.nodes)
        edges = list(agent_graph.edges)
        starting_node = origin_vertiport + "_" + str(timing_info["start_time"])
        nodes.remove(starting_node)
        nodes = [starting_node] + nodes
        inc_matrix = nx.incidence_matrix(agent_graph, nodelist=nodes, edgelist=edges, oriented=True).toarray()
        rows_to_delete = []
        for i, row in enumerate(inc_matrix):
            if -1 not in row:
                rows_to_delete.append(i)
        A = np.delete(inc_matrix, rows_to_delete, axis=0)
        A[0] = -1 * A[0]
        valuations = []
        for edge in edges:
            valuations.append(agent_graph.edges[edge]["valuation"])

        b = np.zeros(len(A))
        b[0] = 1
        
        A_with_default_good = np.hstack((A, np.zeros((A.shape[0], 1)), np.zeros((A.shape[0], 1)))) # outside/default and dropout good
        A_with_default_good[0, -1] = 1
        # A_with_default_good = np.hstack((A, np.zeros((A.shape[0], 1)))) # outside/default good
        goods = edges + ['default_good'] + ['dropout_good']
        # Appending values for default and dropout goods
        valuations.append(default_good_valuation) # Small positive valuation for default good
        valuations.append(dropout_good_valuation) # Small positive valuation for dropout good

        w.append(flight["budget_constraint"])
        u.append(valuations)
        agent_constraints.append((A_with_default_good, b))
        agent_goods_lists.append(goods)

    supply = find_capacity(goods_list, routes, vertiport_usage)
  

    print(f"Time to construct market: {time.time() - start_time_market_construct}")
    return (u, agent_constraints, agent_goods_lists), (w, supply, BETA), (goods_list, times_list)


def find_capacity(goods_list, route_data, vertiport_data):
    # Create a dictionary for route capacities, for now just connectin between diff vertiports
    route_dict = {(route["origin_vertiport_id"], route["destination_vertiport_id"]): route["capacity"] for route in route_data}

    capacities = np.zeros(len(goods_list)) 
    for i, (origin, destination) in enumerate(goods_list[:-2]): # excluding default/outside good - consider changing this to remove "dropout_good" and "default_good"
        origin_base = origin.split("_")[0]
        destination_base = destination.split("_")[0]
        if origin_base != destination_base:
            # Traveling between vertiport
            capacity = route_dict.get((origin_base, destination_base), None)
        else:
            # Staying within a vertiport
            if origin.endswith('_arr'):
                origin_time = origin.replace('_arr', '')
                node = vertiport_data._node.get(origin_time)
                capacity = node.get('landing_capacity') - node.get('landing_usage') 
            elif destination.endswith('_dep'):
                destination_time = destination.replace('_dep', '')
                node = vertiport_data._node.get(destination_time)
                capacity = node.get('takeoff_capacity') - node.get('takeoff_usage') 
            else:
                node = vertiport_data._node.get(origin)
                capacity = node.get('hold_capacity') - node.get('hold_usage') 

        capacities[i] = capacity
    
    capacities[-2] = 100 # default/outside good
    capacities[-1] = 100 # dropout good

    return capacities

def update_basic_market(x, values_k, market_settings, constraints):
    '''Update market consumption, prices, and rebates'''
    shape = np.shape(x)
    num_agents = shape[0]
    num_goods = shape[1]
    k, p_k, r_k = values_k
    supply, beta = market_settings
    
    # Update consumption
    y = cp.Variable((num_agents, num_goods))
    objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(x - y, 'fro')) - (beta / 2) * cp.square(cp.norm(cp.sum(y, axis=0) - supply, 2)))
    # cp_constraints = [y >= 0]
    # problem = cp.Problem(objective, cp_constraints)
    problem = cp.Problem(objective)
    problem.solve(solver=cp.CLARABEL)
    y_k_plus_1 = y.value

    # Update prices
    p_k_plus_1 = p_k + beta * (np.sum(y_k_plus_1, axis=0) - supply)
    for i in range(len(p_k_plus_1)):
        if p_k_plus_1[i] < 0:
            p_k_plus_1[i] = 0

    # Update each agent's rebates
    r_k_plus_1 = []
    for i in range(num_agents):
        agent_constraints = constraints[i]
        if UPDATED_APPROACH:
            constraint_violations = np.array([agent_constraints[0][j] @ x[i] - agent_constraints[1][j] for j in range(len(agent_constraints[1]))])

        else:
            constraint_violations = np.array([max(agent_constraints[0][j] @ x[i] - agent_constraints[1][j], 0) for j in range(len(agent_constraints[1]))])
        r_k_plus_1.append(r_k[i] + beta * constraint_violations)
    return k + 1, y_k_plus_1, p_k_plus_1, r_k_plus_1


def update_market(x, values_k, market_settings, constraints, agent_goods_lists, goods_list, price_default_good):
    '''
    Update market consumption, prices, and rebates
    '''
    shape = np.shape(x)
    num_agents = shape[0]
    num_goods = shape[1]
    k, p_k, r_k = values_k
    supply, beta = market_settings
    
    # Update consumption
    y = cp.Variable((num_agents, num_goods - 2)) # dropout and default removed
    # y = cp.Variable((num_agents, num_goods- 1)) # dropout removed
    # y = cp.Variable((num_agents, num_goods)) 
    # Do we remove drop out here or not? - remove the default and dropout good
    # objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(x[:,:-1] - y, 'fro')) - (beta / 2) * cp.square(cp.norm(cp.sum(y, axis=0) - supply[:-1], 2)))
    objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(x[:,:-2] - y, 'fro')) - (beta / 2) * cp.square(cp.norm(cp.sum(y, axis=0) - supply[:-2], 2)))
    # objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(x - y, 'fro')) - (beta / 2) * cp.square(cp.norm(cp.sum(y, axis=0) - supply, 2)))
    cp_constraints = [y >= 0] 
    problem = cp.Problem(objective, cp_constraints)
    # problem = cp.Problem(objective)

    solvers = [cp.CLARABEL, cp.SCS, cp.OSQP, cp.ECOS, cp.CVXOPT]
    for solver in solvers:
        try:
            result = problem.solve(solver=solver)
            logging.info(f"Problem solved with solver {solver}")
            break
        except cp.error.SolverError as e:
            logging.error(f"Solver {solver} failed: {e}")
            continue
        except Exception as e:
            logging.error(f"An unexpected error occurred with solver {solver}: {e}")
            continue

    # Check if the problem was solved successfully
    if problem.status != cp.OPTIMAL:
        logging.error("Failed to solve the problem with all solvers.")
    else:
        logging.info("Optimization result: %s", result)

    y_k_plus_1 = y.value

    # Update prices
    # p_k_plus_1 = np.zeros(p_k.shape)
    # try the options: 
    # (4) do not update price but dont add it in optimization, 
    # (5) update price and add it in optimization, 
    # (6) dont update price but add it in optimization
    p_k_plus_1 = p_k[:-2] + beta * (np.sum(y_k_plus_1, axis=0) - supply[:-2])
    # p_k_plus_1 = p_k + beta * (np.sum(y_k_plus_1, axis=0) - supply)
    for i in range(len(p_k_plus_1)):
        if p_k_plus_1[i] < 0:
            p_k_plus_1[i] = 0
    # p_k_plus_1[-1] = 0  # dropout good
    # p_k_plus_1[-2] = price_default_good  # default good
    p_k_plus_1 = np.append(p_k_plus_1, [price_default_good,0]) # default and dropout good

    # Update each agent's rebates
    r_k_plus_1 = []
    for i in range(num_agents):
        agent_constraints = constraints[i]
        # Why are we doing this twice?
        agent_x = np.array([x[i, goods_list.index(good)] for good in agent_goods_lists[i]])
        if UPDATED_APPROACH:
            # agent_x = np.array([x[i, goods_list[:-1].index(good)] for good in agent_goods_lists[i][:-1]])
            constraint_violations = np.array([agent_constraints[0][j] @ agent_x - agent_constraints[1][j] for j in range(len(agent_constraints[1]))])
        else:
            constraint_violations = np.array([max(agent_constraints[0][j] @ agent_x - agent_constraints[1][j], 0) for j in range(len(agent_constraints[1]))])
        r_k_plus_1.append(r_k[i] + beta * constraint_violations)
    return k + 1, y_k_plus_1, p_k_plus_1, r_k_plus_1


def update_basic_agents(w, u, p, r, constraints, y, beta, rational=False):
    num_agents = len(w)
    num_goods = len(p)
    x = np.zeros((num_agents, num_goods))
    for i in range(num_agents):
        x[i,:] = update_agent(w[i], u[i,:], p, r[i], constraints[i], y[i,:], beta, rational=rational)
    # print(x)
    return x


def update_agents(w, u, p, r, constraints, goods_list, agent_goods_lists, y, beta, rational=False, parallel=False):
    num_agents, num_goods = len(w), len(p)

    agent_indices = range(num_agents)
    agent_prices = [np.array([p[goods_list.index(good)] for good in agent_goods_lists[i]]) for i in agent_indices]
    agent_utilities = [np.array(u[i]) for i in agent_indices]
    # agent_ys = [np.array([y[i, goods_list.index(good)] for good in agent_goods_lists[i]]) for i in agent_indices]
    agent_ys = [np.array([y[i, goods_list[:-2].index(good)] for good in agent_goods_lists[i][:-2]]) for i in agent_indices] # removing dropout and detault good

    args = [(w[i], agent_utilities[i], agent_prices[i], r[i], constraints[i], agent_ys[i], beta, rational) for i in agent_indices]

    # Update agents in parallel or not depending on parallel flag
    if not parallel:
        results = []
        adjusted_budgets = []
        for arg in args:
            updates =  update_agent(*arg)
            results.append(updates[0])
            adjusted_budgets.append(updates[1]) 
        # results = [update_agent(*arg) for arg in args]
    else:
        num_processes = 4 # increase based on available resources
        with Pool(num_processes) as pool:
            results = pool.starmap(update_agent, args)

    x = np.zeros((num_agents, num_goods))
    for i, agent_x in zip(agent_indices, results):
        for good in goods_list:
            if good in agent_goods_lists[i]:
                x[i, goods_list.index(good)] = agent_x[agent_goods_lists[i].index(good)]
    return x, adjusted_budgets


def update_agent(w_i, u_i, p, r_i, constraints, y_i, beta, rational=False, solver=cp.SCS):
    """
    Update individual agent's consumption given market settings and constraints
    """
    # Individual agent optimization
    A_i, b_i = constraints
    # A_bar = np.hstack((A_i, np.zeros((len(A_i), 1))))
    # A_bar[0, -1] = 1
    num_constraints = len(b_i)
    num_goods = len(p)

    budget_adjustment = r_i.T @ b_i
    w_adj = w_i + budget_adjustment
    w_adj = max(w_adj, 0)

    # print(f"Adjusted budget: {w_adj}")
    # optimizer check
    x_i = cp.Variable(num_goods)
    if rational:
        regularizers = - (beta / 2) * cp.square(cp.norm(x_i - y_i, 2)) - (beta / 2) * cp.sum([cp.square(cp.maximum(A_i[t] @ x_i - b_i[t], 0)) for t in range(num_constraints)])
        lagrangians = - p.T @ x_i - cp.sum([r_i[t] * cp.maximum(A_i[t] @ x_i - b_i[t], 0) for t in range(num_constraints)])
        objective = cp.Maximize(u_i.T @ x_i + regularizers + lagrangians)
        cp_constraints = [x_i >= 0]
        # cp_constraints = [x_i >= 0, p.T @ x_i <= w_adj]
        # objective = cp.Maximize(u_i.T @ x_i)
        # cp_constraints = [x_i >= 0, p.T @ x_i <= w_adj, A_i @ x_i <= b_i]
    elif UPDATED_APPROACH:
        # objective_terms = v_i.T @ x_i[:-2] + v_i_o * x_i[-2] + v_i_d * x_i[-1]
        objective_terms = u_i.T @ x_i
        regularizers = - (beta / 2) * cp.square(cp.norm(x_i[:-2] - y_i, 2)) - (beta / 2) * cp.square(cp.norm(A_i @ x_i - b_i, 2)) 
        # regularizers = - (beta / 2) * cp.square(cp.norm(x_i - y_i, 2)) - (beta / 2) * cp.square(cp.norm(A_i @ x_i - b_i, 2)) 
        lagrangians = - p.T @ x_i - r_i.T @ (A_i @ x_i - b_i)
        nominal_objective = w_adj * cp.log(objective_terms)
        objective = cp.Maximize(nominal_objective + lagrangians + regularizers)
        cp_constraints = [x_i >= 0]
        # cp_constraints = [x_i >= 0, A_bar @ x_i[:-2] + x_i[-2] >= 0]
    else:
        regularizers = - (beta / 2) * cp.square(cp.norm(x_i - y_i, 2)) - (beta / 2) * cp.sum([cp.square(cp.maximum(A_i[t] @ x_i - b_i[t], 0)) for t in range(num_constraints)])
        lagrangians = - p.T @ x_i - cp.sum([r_i[t] * cp.maximum(A_i[t] @ x_i - b_i[t], 0) for t in range(num_constraints)])
        nominal_objective = w_adj * cp.log(u_i.T @ x_i)
        objective = cp.Maximize(nominal_objective + lagrangians + regularizers)
        cp_constraints = [x_i >= 0]
    # check_time = time.time()
    problem = cp.Problem(objective, cp_constraints)
    # problem.solve(solver=solver, verbose=False)

    solvers = [cp.SCS,cp.CLARABEL, cp.OSQP, cp.ECOS, cp.CVXOPT]
    for solver in solvers:
        try:
            result = problem.solve(solver=solver)
            logging.info(f"Agent Opt - Problem solved with solver {solver}")
            break
        except cp.error.SolverError as e:
            logging.error(f"Agent Opt - Solver {solver} failed: {e}")
            continue
        except Exception as e:
            logging.error(f"Agent Opt - An unexpected error occurred with solver {solver}: {e}")
            continue

    # Check if the problem was solved successfully
    if problem.status != cp.OPTIMAL:
        logging.error("Agent Opt - Failed to solve the problem with all solvers.")
    else:
        logging.info("Agent opt - Optimization result: %s", result)


    # print(f"Solvers time: {time.time() - check_time}")
    return x_i.value, w_adj


def run_basic_market(initial_values, agent_settings, market_settings, plotting=False, rational=False):
    u, agent_constraints = agent_settings
    y, p, r = initial_values
    w, supply, beta = market_settings

    x_iter = 0
    prices = []
    rebates = []
    overdemand = []
    agent_allocations = []
    error = [] * len(agent_constraints)
    while x_iter <= 100:  # max(abs(np.sum(opt_xi, axis=0) - C)) > epsilon:
        # Update agents
        x = update_basic_agents(w, u, p, r, agent_constraints, y, beta, rational=rational)
        agent_allocations.append(x)
        overdemand.append(np.sum(x, axis=0) - supply.flatten())
        for agent_index in range(len(agent_constraints)):
            constraint_error = agent_constraints[agent_index][0] @ x[agent_index] - agent_constraints[agent_index][1]
            if x_iter == 0:
                error.append([constraint_error])
            else:
                error[agent_index].append(constraint_error)

        # Update market
        k, y, p, r = update_basic_market(x, (1, p, r), (supply, beta), agent_constraints)
        rebates.append([rebate_list for rebate_list in r])
        prices.append(p)
        x_iter += 1
    if plotting:
        for good_index in range(len(p)):
            plt.plot(range(1, x_iter+1), [prices[i][good_index] for i in range(len(prices))], label=f"Good {good_index}")
        plt.xlabel('x_iter')
        plt.ylabel('Prices')
        plt.title("Price evolution")
        plt.legend()
        plt.show()
        plt.plot(range(1, x_iter+1), overdemand)
        plt.xlabel('x_iter')
        plt.ylabel('Demand - Supply')
        plt.title("Overdemand evolution")
        plt.show()
        for agent_index in range(len(agent_constraints)):
            plt.plot(range(1, x_iter+1), error[agent_index])
        plt.title("Constraint error evolution")
        plt.show()
        for constraint_index in range(len(rebates[0])):
            plt.plot(range(1, x_iter+1), [rebates[i][constraint_index] for i in range(len(rebates))])
        plt.title("Rebate evolution")
        plt.show()
        for agent_index in range(len(agent_allocations[0])):
            plt.plot(range(1, x_iter+1), [agent_allocations[i][agent_index] for i in range(len(agent_allocations))])
        plt.title("Agent allocation evolution")
        plt.show()
    print(f"Error: {[error[i][-1] for i in range(len(error))]}")
    print(f"Overdemand: {overdemand[-1][:]}")
    return x, p, r, overdemand



def run_market(initial_values, agent_settings, market_settings, bookkeeping, rational=False, price_default_good=10):
    """
    
    """
    u, agent_constraints, agent_goods_lists = agent_settings
    y, p, r = initial_values
    w, supply, beta = market_settings
    goods_list, times_list = bookkeeping

    x_iter = 0
    prices = []
    rebates = []
    overdemand = []
    agent_allocations = []
    market_clearing = []
    error = [] * len(agent_constraints)
    
    # Algorithm 1
    num_agents = len(agent_goods_lists)
    tolerance = num_agents * np.sqrt(len(supply)-2) * TOL_ERROR  # -1 to ignore default goods
    market_clearing_error = float('inf')
    x_iter = 0
    start_time_algorithm = time.time()  

    while market_clearing_error > tolerance and x_iter <= MAX_NUM_ITERATIONS:
    # while x_iter <= 100:
        # Update agents
        x, adjusted_budgets = update_agents(w, u, p, r, agent_constraints, goods_list, agent_goods_lists, y, beta, rational=rational)
        agent_allocations.append(x[:,:-1]) # removing dropout good
        overdemand.append(np.sum(x[:,:-2], axis=0) - supply[:-2].flatten())
        x_ij = np.sum(x[:,:-2], axis=0) # removing default and dropout good
        excess_demand = x_ij - supply[:-2]
        clipped_excess_demand = np.where(p[:-2] > 0,excess_demand, np.maximum(0, excess_demand)) # price removing default and dropout good
        market_clearing_error = np.linalg.norm(clipped_excess_demand, ord=2)
        market_clearing.append(market_clearing_error)

        for agent_index in range(len(agent_constraints)):
            agent_x = np.array([x[agent_index, goods_list.index(good)] for good in agent_goods_lists[agent_index]])
            constraint_error = agent_constraints[agent_index][0] @ agent_x - agent_constraints[agent_index][1]
            if x_iter == 0:
                error.append([constraint_error])
            else:
                error[agent_index].append(constraint_error)

        # Update market
        k, y, p, r = update_market(x, (1, p, r), (supply, beta), agent_constraints, agent_goods_lists, goods_list, price_default_good)
        rebates.append([rebate_list for rebate_list in r])
        prices.append(p)
        x_iter += 1
        print("Iteration: ", x_iter, "- Market Clearing Error: ", market_clearing_error, " - Tolerance: ", tolerance)
        logging.info(f"Iteration: {x_iter}, Market Clearing Error: {market_clearing_error}, Tolerance: {tolerance}")
    
        # if market_clearing_error <= tolerance:
        #     break

    print(f"Time to run algorithm: {time.time() - start_time_algorithm}")

    data_to_plot = [x_iter, prices, p, overdemand, error, rebates, agent_allocations, market_clearing, agent_constraints]

    last_prices = np.array(prices[-1])
    # final_prices = last_prices[last_prices > 0]

    # print(f"Error: {[error[i][-1] for i in range(len(error))]}")
    # print(f"Overdemand: {overdemand[-1][:]}")
    return x, last_prices, r, overdemand, agent_constraints, adjusted_budgets, data_to_plot

def plotting_market(data_to_plot, output_folder, market_auction_time=None):
    x_iter, prices, p, overdemand, error, rebates, agent_allocations, market_clearing, agent_constraints = data_to_plot
    plt.figure(figsize=(20, 20))
    plt.subplot(2, 3, 1)
    for good_index in range(len(p)-2):
        plt.plot(range(1, x_iter+1), [prices[i][good_index] for i in range(len(prices))])
    plt.plot(range(1, x_iter+1), [prices[i][-2] for i in range(len(prices))], 'b--' ,label="Default Good")
    plt.plot(range(1, x_iter+1), [prices[i][-1] for i in range(len(prices))], 'r-.' ,label="Dropout Good")
    plt.xlabel('x_iter')
    plt.ylabel('Prices')
    plt.title("Price evolution")
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(range(1, x_iter+1), overdemand)
    plt.xlabel('x_iter')
    plt.ylabel('Demand - Supply')
    plt.title("Overdemand evolution")

    plt.subplot(2, 3, 3)
    for agent_index in range(len(agent_constraints)):
        plt.plot(range(1, x_iter+1), error[agent_index])
    plt.ylabel('Constraint error')
    plt.title("Constraint error evolution")

    plt.subplot(2, 3, 4)
    for constraint_index in range(len(rebates[0])):
        plt.plot(range(1, x_iter+1), [rebates[i][constraint_index] for i in range(len(rebates))])
    plt.xlabel('x_iter')
    plt.ylabel('rebate')
    plt.title("Rebate evolution")
    

    plt.subplot(2, 3, 5)
    for agent_index in range(len(agent_allocations[0])):
        plt.plot(range(1, x_iter+1), [agent_allocations[i][agent_index] for i in range(len(agent_allocations))])
    plt.title("Agent allocation evolution")
    plt.xlabel('x_iter')

    plt.subplot(2, 3, 6)
    plt.plot(range(1, x_iter+1), market_clearing)
    plt.xlabel('x_iter')
    plt.title("Market Clearing Error")
    # plt.show()
    if market_auction_time:
        save_file = f"{output_folder}/market_plot_a{market_auction_time}.png" 
    else:
        save_file = f"{output_folder}/market_plot.png"
    plt.savefig(save_file)
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


def find_dep_and_arrival_nodes(edges):
    dep_node_found = False
    arrival_node_found = False
    
    for edge in edges:
        if "dep" in edge[0]:
            dep_node_found = edge[0]
            arrival_node_found = edge[1]
            assert "arr" in arrival_node_found, f"Arrival node not found: {arrival_node_found}"
            return dep_node_found, arrival_node_found
    
    return dep_node_found, arrival_node_found


def fisher_allocation_and_payment(vertiport_usage, flights, timing_info, routes_data, vertiports, 
                                  output_folder=None, save_file=None, initial_allocation=True, design_parameters=None):

    market_auction_time=timing_info["start_time"]
    # Build Fisher Graph
    market_graph = build_graph(vertiport_usage, timing_info)

    #Extracting design parameters
    if design_parameters:
        price_default_good = design_parameters["price_default_good"]
        default_good_valuation = design_parameters["default_good_valuation"]
        dropout_good_valuation = design_parameters["dropout_good_valuation"]
        BETA = design_parameters["beta"]
    else:
        BETA = 1 # chante to 1/T
        dropout_good_valuation = -1
        default_good_valuation = 1
        price_default_good = 10

    # Construct market
    agent_information, market_information, bookkeeping = construct_market(market_graph, flights, timing_info, routes_data, vertiport_usage, 
                                                                          default_good_valuation=default_good_valuation, 
                                                                          dropout_good_valuation=dropout_good_valuation, BETA=BETA)

    # Run market
    goods_list, times_list = bookkeeping
    num_goods, num_agents = len(goods_list), len(flights)
    u, agent_constraints, agent_goods_lists = agent_information
    # y = np.random.rand(num_agents, num_goods-2)*10
    y = np.random.rand(num_agents, num_goods)*10
    p = np.random.rand(num_goods)*10
    p[-2] = price_default_good 
    p[-1] = 0 # dropout good
    r = [np.zeros(len(agent_constraints[i][1])) for i in range(num_agents)]
    # x, p, r, overdemand = run_market((y,p,r), agent_information, market_information, bookkeeping, plotting=True, rational=False)
    x, prices, r, overdemand, agent_constraints, adjusted_budgets, data_to_plot = run_market((y,p,r), agent_information, market_information, 
                                                             bookkeeping, rational=False, price_default_good=price_default_good)
    
    
    plotting_market(data_to_plot, output_folder, market_auction_time)
    
    # Building edge information for mapping - move this to separate function
    edge_information = build_edge_information(goods_list)
    agent_allocations, agent_dropout_x, agent_indices, agent_edge_information = process_allocations(x, edge_information, agent_goods_lists)
    
    int_allocations = []
    utilities = []
    dropouts = []
    budgets = []
    constraints = []
    agents_indices = []
    int_allocations_full = []
    start_time_sample = time.time()
    print("Sampling edges ...")
    for i in range(num_agents):
        agent_number = i + 1
        frac_allocations = agent_allocations[i]
        start_node= list(agent_edge_information[i].values())[0][0]
        extended_graph, agent_allocation = agent_probability_graph_extended(agent_edge_information[i], frac_allocations, agent_number, output_folder)
        sampled_path_extended, sampled_edges, int_allocation, dropout_flag = sample_path(extended_graph, start_node, agent_allocation, agent_dropout_x[i])
        if dropout_flag:
            dropouts.append(flights[i]["flight_id"])
        else:
            # print("Sampled Path:", sampled_path_extended)
            # print("Sampled Edges:", sampled_edges)
            # plot_sample_path_extended(extended_graph, sampled_path_extended, agent_number, output_folder)
            int_allocations.append(int_allocation)
            int_allocation_full = mapping_agent_to_full_data(edge_information, sampled_edges)
            int_allocations_full.append(int_allocation_full)
            utilities.append(u[i]) # removing default and dropout good
            budgets.append(adjusted_budgets[i])
            constraints.append(agent_constraints[i])
            agents_indices.append(agent_indices)

    int_allocations_full = np.array(int_allocations_full)
    print(f"Time to sample: {time.time() - start_time_sample:.5f}")

    # IOP for contested goods
    _ , capacity, _ = market_information
    capacity = capacity[:-2] # removing default and dropout good
    prices = prices[:-2] # removing default and dropout good
    new_allocations, new_prices = int_optimization(int_allocations_full, capacity, budgets, prices, utilities, constraints, agents_indices, int_allocations, output_folder)
    payment = np.sum(new_prices * new_allocations, axis=1)
    end_capacity = capacity - np.sum(new_allocations, axis=0)

    new_allocations_goods = mapping_goods_from_allocation(new_allocations, goods_list)

    # Allocation and Rebased
    allocation = []
    rebased = []
    for i, (flight_id, flight) in enumerate(flights.items()):
        dep_node, arrival_node = find_dep_and_arrival_nodes(new_allocations_goods[i])
        if dep_node:
            good = (dep_node, arrival_node)
            allocation.append((flight_id, good))
        else:
            rebased.append((flight_id, '000'))
    end_agent_status_data = (allocation, rebased, dropouts)



    write_output(flights, agent_constraints, edge_information, prices, new_prices, capacity, end_capacity,
                agent_allocations, agent_indices, agent_edge_information, agent_goods_lists, 
                int_allocations, new_allocations_goods, u, adjusted_budgets, payment, end_agent_status_data, market_auction_time, output_folder)
    
    return allocation, rebased, None



if __name__ == "__main__":
    pass

