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
from ascending_auc.asc_auc_int_optimization import int_optimization
from write_csv import write_output

UPDATED_APPROACH = True
TOL_ERROR = 1e-3
MAX_NUM_ITERATIONS = 1000
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
    # y = cp.Variable((num_agents, num_goods - 2)) # dropout and default removed
    # y = cp.Variable((num_agents, num_goods - 1)) # dropout removed (4)
    y = cp.Variable((num_agents, num_goods)) 
    y_bar = cp.Variable(num_goods)
    # Do we remove drop out here or not? - remove the default and dropout good
    # objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(x[:,:-1] - y, 'fro')) - (beta / 2) * cp.square(cp.norm(cp.sum(y, axis=0) - supply[:-1], 2))) # (4) (5)
    # objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(x[:,:-2] - y, 'fro')) - (beta / 2) * cp.square(cp.norm(cp.sum(y, axis=0) - supply[:-2], 2)))
    # objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(x - y, 'fro')) - (beta / 2) * cp.square(cp.norm(cp.sum(y, axis=0) - supply, 2)))
    y_sum = cp.sum(y, axis=0)
    objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(x - y, 'fro')) - (beta / 2) * cp.square(cp.norm(y_sum + y_bar - supply, 2))  - p_k.T @ y_bar)
    cp_constraints = [y_bar >= 0] 
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
    # try the options (default good): 
    # (3) do not update price, dont add it in optimization, 
    # (4) update price and add it in optimization, 
    # (5) dont update price but add it in optimization
    p_k_plus_1 = p_k[:-2] + beta * (np.sum(y_k_plus_1[:,:-2], axis=0) - supply[:-2]) #(3)
    # p_k_plus_1 = p_k[:-1] + beta * (np.sum(y_k_plus_1, axis=0) - supply[:-1]) #(4)
    # p_k_plus_1 = p_k[:-2] + beta * (np.sum(y_k_plus_1[:,:-2], axis=0) - supply[:-2]) #(5)
    # p_k_plus_1 = p_k + beta * (np.sum(y_k_plus_1, axis=0) - supply)
    for i in range(len(p_k_plus_1)):
        if p_k_plus_1[i] < 0:
            p_k_plus_1[i] = 0
    # p_k_plus_1[-1] = 0  # dropout good
    # p_k_plus_1[-2] = price_default_good  # default good
    p_k_plus_1 = np.append(p_k_plus_1, [price_default_good,0]) # default and dropout good
    # p_k_plus_1 = np.append(p_k_plus_1, 0)  #  (4) update default good


    # Update each agent's rebates
    r_k_plus_1 = []
    for i in range(num_agents):
        agent_constraints = constraints[i]
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
    agent_ys = [np.array([y[i, goods_list.index(good)] for good in agent_goods_lists[i]]) for i in agent_indices]
    # agent_ys = [np.array([y[i, goods_list[:-2].index(good)] for good in agent_goods_lists[i][:-2]]) for i in agent_indices] # removing dropout and detault good (3)
    # agent_ys = [np.array([y[i, goods_list[:-1].index(good)] for good in agent_goods_lists[i][:-1]]) for i in agent_indices] # removing dropout (4)
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
    # A_bar = A_i[0]

    num_constraints = len(b_i)
    num_goods = len(p)

    budget_adjustment = r_i.T @ b_i
    w_adj = w_i + budget_adjustment
    w_adj = max(w_adj, 0)
    # w_adj = abs(w_adj) 

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
        # regularizers = - (beta / 2) * cp.square(cp.norm(x_i[:-1] - y_i, 2)) - (beta / 2) * cp.square(cp.norm(A_i @ x_i - b_i, 2))  #(4)
        regularizers = - (beta / 2) * cp.square(cp.norm(x_i - y_i, 2)) - (beta / 2) * cp.square(cp.norm(A_i @ x_i - b_i, 2))        
        # regularizers = - (beta / 2) * cp.square(cp.norm(x_i - y_i, 2)) - (beta / 2) * cp.square(cp.norm(A_i @ x_i - b_i, 2)) 
        lagrangians = - p.T @ x_i - r_i.T @ (A_i @ x_i - b_i) # the price of dropout good is 0
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

    solvers = [cp.SCS, cp.CLARABEL, cp.MOSEK, cp.OSQP, cp.ECOS, cp.CVXOPT]
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
    abs_error = [] * len(agent_constraints)
    
    # Algorithm 1
    num_agents = len(agent_goods_lists)
    tolerance = num_agents * np.sqrt(len(supply)-2) * TOL_ERROR  # -1 to ignore default goods
    market_clearing_error = float('inf')
    x_iter = 0
    start_time_algorithm = time.time()  

    while market_clearing_error > tolerance and x_iter <= MAX_NUM_ITERATIONS:
    # while x_iter <= 100:
        x, adjusted_budgets = update_agents(w, u, p, r, agent_constraints, goods_list, agent_goods_lists, y, beta, rational=rational)
        agent_allocations.append(x) # 
        overdemand.append(np.sum(x[:,:-2], axis=0) - supply[:-2].flatten())
        x_ij = np.sum(x[:,:-2], axis=0) # removing default and dropout good
        excess_demand = x_ij - supply[:-2]
        clipped_excess_demand = np.where(p[:-2] > 0,excess_demand, np.maximum(0, excess_demand)) # price removing default and dropout good
        market_clearing_error = np.linalg.norm(clipped_excess_demand, ord=2)
        market_clearing.append(market_clearing_error)

        for agent_index in range(len(agent_constraints)):
            agent_x = np.array([x[agent_index, goods_list.index(good)] for good in agent_goods_lists[agent_index]])
            agent_y = np.array([y[agent_index, goods_list.index(good)] for good in agent_goods_lists[agent_index]])
            constraint_error = agent_constraints[agent_index][0] @ agent_x - agent_constraints[agent_index][1]
            agent_error = np.sum(np.square(agent_x - agent_y)) +  np.sum(np.square(constraint_error))  
            if x_iter == 0:
                error.append([constraint_error])
                abs_error.append([agent_error])
            else:
                error[agent_index].append(constraint_error)
                abs_error[agent_index].append(agent_error)
        

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

    data_to_plot = [x_iter, prices, p, overdemand, error, abs_error, rebates, agent_allocations, market_clearing, agent_constraints]

    last_prices = np.array(prices[-1])
    # final_prices = last_prices[last_prices > 0]

    # print(f"Error: {[error[i][-1] for i in range(len(error))]}")
    # print(f"Overdemand: {overdemand[-1][:]}")
    return x, last_prices, r, overdemand, agent_constraints, adjusted_budgets, data_to_plot

def plotting_market(data_to_plot, output_folder, market_auction_time=None):
    x_iter, prices, p, overdemand, error, abs_error, rebates, agent_allocations, market_clearing, agent_constraints = data_to_plot
    plt.figure(figsize=(20, 20))
    plt.subplot(2, 4, 1)
    for good_index in range(len(p)-2):
        plt.plot(range(1, x_iter+1), [prices[i][good_index] for i in range(len(prices))])
    plt.plot(range(1, x_iter+1), [prices[i][-2] for i in range(len(prices))], 'b--' ,label="Default Good")
    plt.plot(range(1, x_iter+1), [prices[i][-1] for i in range(len(prices))], 'r-.' ,label="Dropout Good")
    plt.xlabel('x_iter')
    plt.ylabel('Prices')
    plt.title("Price evolution")
    plt.legend()

    plt.subplot(2, 4, 2)
    plt.plot(range(1, x_iter+1), overdemand)
    plt.xlabel('x_iter')
    plt.ylabel('Demand - Supply')
    plt.title("Overdemand evolution")

    plt.subplot(2, 4, 3)
    for agent_index in range(len(agent_constraints)):
        plt.plot(range(1, x_iter+1), error[agent_index])
    plt.ylabel('Constraint error')
    plt.title("Constraint error evolution")

    plt.subplot(2, 4, 4)
    for agent_index in range(len(agent_constraints)):
        plt.plot(range(1, x_iter+1), abs_error[agent_index])
    plt.ylabel('Constraint error')
    plt.title("Absolute error evolution")

    plt.subplot(2, 4, 5)
    for constraint_index in range(len(rebates[0])):
        plt.plot(range(1, x_iter+1), [rebates[i][constraint_index] for i in range(len(rebates))])
    plt.xlabel('x_iter')
    plt.ylabel('rebate')
    plt.title("Rebate evolution")
    

    plt.subplot(2, 4, 6)
    for agent_index in range(len(agent_allocations[0])):
        plt.plot(range(1, x_iter+1), [agent_allocations[i][agent_index] for i in range(len(agent_allocations))])
        # plt.plot(range(1, x_iter+1), agent_allocations[-2][agent_index], 'b--' ,label="Default Good")
        # plt.plot(range(1, x_iter+1), agent_allocations[-1][agent_index], 'r-.' ,label="Dropout Good")
    plt.title("Agent allocation evolution")
    plt.xlabel('x_iter')

    plt.subplot(2, 4, 7)
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

class time_step:
    flight_id = ""
    time_no = -1
    spot = ""
    price = 0
    def __init__(self, id_, time_, spot_):
        self.time_no = time_
        self.flight_id = id_
        self.spot = spot_
    def raise_val(self):
        self.price += 1

class bundle:
    flight_id = ""
    time = -1
    flight = ()
    times = []
    value = 0
    def __init__(self, id_, start, end, price, spot):
        flight_id = id_
        value = price
        for i in range(start,end):
            self.times += [time_step(flight_id, i, spot)]
    def update_flight_path(self, depart_time, depart_port, arrive_time, arrive_port):
        dep_id = depart_port + '_' + str(depart_time) + '_dep'
        arr_id = arrive_port + '_' + str(arrive_time) + '_arr'
        self.flight = (self.flight_id, (dep_id, arr_id))
        return self.flight
    
    def __init__(self, f,t,v):
        self.flight_id = f
        self.times = t
        self.value =v
    
    
    def findCost(self, current_time):
        tot = 0
        for i in range(current_time,len(self.times)):
            tot += self.times[i].price
        return tot


#[('AC004', ('V001_13_dep', 'V002_18_arr')), ('AC005', ('V007_17_dep', 'V002_55_arr')), ('AC008', ('V003_20_dep', 'V006_42_arr'))]

def process_request(id_, depart_port, arrive_port, depart_time, arrive_time, maxBid, start_time, end_time, step, decay):
    reqs = []
    if(depart_port == arrive_port):
        reqs += [bundle(id_, 1, 100, maxBid, depart_port)]
        return reqs
    curtimesarray = [time_step(id_, i, 'NA') for i in range(start_time, end_time + 1, step)]

    for i in range(start_time, depart_time, step):
        curtimesarray[i].spot =  depart_port

    curtimesarray[depart_time].spot = depart_port+'_dep'

    for i in range(depart_time + 1, arrive_time, step):
        curtimesarray[i].spot = depart_port+arrive_port

    curtimesarray[arrive_time].spot += arrive_port+'_arr'

    for i in range(arrive_time + 1, end_time, step):
        curtimesarray[i].spot = arrive_port
    

    delayed_dep_t = depart_time 
    delayed_arr_t = arrive_time
    
    nb = bundle(id_, curtimesarray, maxBid)
    nb.update_flight_path(depart_time, depart_port, arrive_time, arrive_port)
    reqs += [nb]
    while(delayed_arr_t + 1 < end_time): # arrive_port + _arr on last timestep
        c2 = [tm for tm in reqs[-1].times]
        c2[delayed_dep_t].spot = depart_port
        c2[delayed_dep_t + 1].spot = depart_port + '_dep'
        c2[delayed_arr_t].spot = depart_port + arrive_port
        c2[delayed_arr_t + 1].spot = arrive_port + '_arr'
        delayedBundle = bundle(id_, c2, maxBid * decay)
        delayedBundle.update_flight_path(delayed_dep_t, depart_port, delayed_arr_t, arrive_port)
        reqs += [delayedBundle]
        decay *= 0.95
        delayed_dep_t += 1
        delayed_arr_t += 1
    return reqs 

def multiplicitiesDict(vals): # can optimize
    print(vals)
    k = set(vals)
    s = {}
    for i in k:
        s[i] = vals.count(i)
    return s

def run_auction(reqs, start_time, end_time, capacities):
    numreq = len(reqs)
    price_per_req = [0] * numreq
    final = False
    pplcnt_log = []
    maxprice_log = []
    while(not final):
        final = True
        print(numreq)
        print(price_per_req)
        for t in range(start_time, end_time):
            print(t)
            #look through all requests at a time step
            spots_reqs = []
            for r in reqs:
                #print(type(r.times[t]))
                spots_reqs += [[r.times[t].spot, r.flight_id]]
            

            print()

            spots_reqs = [e[0] for e in list({tuple(i) for i in spots_reqs})] # ensuring same flights arent competing by creating set of used spots including flight ids
            print('A  -----------------')
            multiplicities = multiplicitiesDict(spots_reqs)
            print('B ----------')
            pricedOut = []
            for r_ix in range(len(reqs)):
                r = reqs[r_ix]
                #compare each request size to capacity
                #print(r.times[t].spot,multiplicities[r.times[t].spot], capacities[r.times[t].spot])
                if(multiplicities[r.times[t].spot] > capacities[r.times[t].spot]): # contested
                    final = False
                     #if larger increase price of time step at each request & total flight price
                    r.times[t].raise_val()
                    price_per_req[r_ix] += 1
                    if(price_per_req[r_ix] >= r.value):
                        pricedOut += [r_ix]
            for n in pricedOut[::-1]:
                reqs.pop(n)
                price_per_req.pop(n)
            pplcnt_log += [[t, len(price_per_req)]]
            maxprice_log += [[t, max(price_per_req)]]
            numreq = len(reqs)
    
    return reqs, price_per_req, pplcnt_log, maxprice_log


def define_capacities(vertiport_data, route_data):
    capacities = {}
    for i in route_data:
        dep_port = i["origin_vertiport_id"]
        arr_port = i["destination_vertiport_id"]
        capacities[dep_port+arr_port] = i["capacity"]
    for i in vertiport_data:
        capacities[i] = vertiport_data[i]["hold_capacity"]
        capacities[i+i] = vertiport_data[i]["hold_capacity"] #route to self
        capacities[i+'_dep'] = vertiport_data[i]["takeoff_capacity"]
        capacities[i+'_arr'] = vertiport_data[i]["landing_capacity"]


    return capacities

def pickHighest(requests, start_time):
    mxMap = {}
    mxReq = {}
    for r in requests:
        mxMap[r.flight_id] = -1
        mxReq[r.flight_id] = None
    for r in requests:
        tot = r.findCost(start_time)
        if(tot>mxMap[r.flight_id]):
            mxMap[r.flight_id] = tot 
            mxReq[r.flight_id] = r
    return mxReq.values()


def ascending_auc_allocation_and_payment(vertiport_usage, flights, timing_info, routes_data, vertiports, 
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



    capacities = define_capacities(vertiport_usage.vertiports, routes_data)

    requests = []
    for f in flights.keys():
        flight_data = flights[f]
        origin_vp = flight_data["origin_vertiport_id"]
        flight_req = flight_data["requests"]
        decay = flight_data["decay_factor"]
        for req_index in flight_req.keys():
            fr = flight_req[req_index]
            dest_vp = fr["destination_vertiport_id"]
            if(origin_vp == dest_vp):
                continue
            dep_time = fr["request_departure_time"]
            arr_time = fr["request_arrival_time"]
            val = fr["valuation"]
            requests += process_request(f, origin_vp, dest_vp, dep_time, arr_time, val, timing_info["start_time"], timing_info["end_time"], timing_info["time_step"], decay)
    
    #for i in requests:
    #    print(i.flight)


    allocated_requests, final_prices_per_req, agents_left_time, price_change = run_auction(requests, timing_info["start_time"], timing_info["auction_end_time"], capacities)

    allocated_requests = pickHighest(allocated_requests, market_auction_time)

    print('FINAL')

    #print(allocated_requests)
    
    allocation = []

    #print(allocated_requests)                                       

    allocation = [ar.flight for ar in allocated_requests]
    rebased = None

    #write_output(flights, agent_constraints, edge_information, prices, new_prices, capacity, end_capacity,
    #            agent_allocations, agent_indices, agent_edge_information, agent_goods_lists, 
    #            int_allocations, new_allocations_goods, u, adjusted_budgets, payment, end_agent_status_data, market_auction_time, output_folder)
    
    return allocation, None



if __name__ == "__main__":
    pass

