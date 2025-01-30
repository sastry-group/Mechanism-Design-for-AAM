import time
import numpy as np
import logging
import cvxpy as cp
import sys
from pathlib import Path
import networkx as nx
from rich.console import Console
from rich.table import Table
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import csv
import json


top_level_path = Path(__file__).resolve().parent.parent
sys.path.append(str(top_level_path))

from fisher.FisherGraphBuilder import FisherGraphBuilder

INTEGRAL_APPROACH = False
UPDATED_APPROACH = True
TOL_ERROR = 1e-3
MAX_NUM_ITERATIONS = 10000

logger = logging.getLogger("global_logger")



def construct_market(flights, timing_info, sectors, vertiport_usage, output_folder, default_good_valuation=1, dropout_good_valuation=-1, BETA=1):
    """
    Constructs a market for the given flights, timing information, sectors, and vertiport usage.
    Parameters:
    flights (dict): A dictionary where keys are flight IDs and values are flight information.
    timing_info (dict): A dictionary containing timing information, including "start_time".
    sectors (list): A list of sectors involved in the market.
    vertiport_usage (dict): A dictionary containing vertiport usage information.
    default_good_valuation (int, optional): Valuation for the default good. Default is 1.
    dropout_good_valuation (int, optional): Valuation for the dropout good. Default is -1.
    BETA (int, optional): A parameter for the market. Default is 1.
    Returns:
    tuple: A tuple containing:
        - A tuple of agent utilities, agent constraints, and agent goods lists.
        - A tuple of agent budgets, supply, and BETA.
        - A list of goods in the market.
    """

    if not flights or not timing_info or not sectors or not vertiport_usage:
        raise ValueError("Invalid input: flights, timing_info, sectors, or vertiport_usage is empty.")
    # # building the graph
    # market_auction_time=timing_info["start_time"]
    # start_time_graph_build = time.time()
    # builder = FisherGraphBuilder(vertiport_usage, timing_info)
    # market_graph = builder.build_graph(flights)
    # print(f"Time to build graph: {time.time() - start_time_graph_build}")

    start_time_market_construct = time.time()
    # goods_list = list(market_graph.edges) + ['default_good'] + ['dropout_good']
    goods_list = []
    w = []
    u = []
    agent_constraints = []
    agent_goods_lists = []
    
    for flight_id, flight in flights.items():

        # print(f"Building graph for flight {flight_id}")
        logger.info(f"Building graph for flight {flight_id}")
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

        # if flight_id == "AC005":
        #     for row in A:
        # # row = inc_matrix[-15,:]
        #         positive_indices = [edges[index] for index in np.where(row == 1)[0]]
        #         negative_indices = [edges[index] for index in np.where(row == -1)[0]]
        #         # print(f"{positive_indices} - {negative_indices}")
        #         logger.debug(f"Construct market AC005: {positive_indices} - {negative_indices}")


        b = np.zeros(len(A))
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
    
  

    # print(f"Time to construct market: {round(time.time() - start_time_market_construct)}")
    logger.info(f"Time to construct market: {round(time.time() - start_time_market_construct)}")
    return (u, agent_constraints, agent_goods_lists), (w, supply, BETA), (goods_list)



def find_capacity(goods_list, route_data, vertiport_data):
    # Create a dictionary for route capacities, for now just connectin between diff vertiports
    # sector_dict = {sid: sector["hold_capacity"] for sid, sector in sectors_data.items()}
    # route_dict = {(route["origin_vertiport_id"], route["destination_vertiport_id"]): route["capacity"] for route in route_data}

    capacities = np.zeros(len(goods_list)) 
    for i, (origin, destination) in enumerate(goods_list[:-2]): # excluding default/outside good - consider changing this to remove "dropout_good" and "default_good"
        # print(f"Origin: {origin} - Destination: {destination}")
        logger.debug(f"Origin: {origin} - Destination: {destination}")
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
        logger.debug(f"Capacity on edge {origin} to {destination}: {capacity}")
        capacities[i] = capacity
    
    capacities[-2] = 100 # default/outside good
    capacities[-1] = 100 # dropout good

    return capacities





def update_market(x_val, values_k, market_settings, constraints, agent_goods_lists, goods_list, 
                  price_default_good, problem, update_rebates=True, integral=False, price_upper_bound=1000):
    '''
    Update market consumption, prices, and rebates
    '''
    start_time = time.time()
    shape = np.shape(x_val)
    num_agents = shape[0]
    num_goods = shape[1]
    k, p_k_val, r_k = values_k
    supply, beta = market_settings
    
    # Update consumption
    # y = cp.Variable((num_agents, num_goods - 2)) # dropout and default removed
    # y = cp.Variable((num_agents, num_goods - 1)) # dropout removed (4)
    warm_start = False
    if problem is None:
        y = cp.Variable((num_agents, num_goods - 2), integer=integral) 
        y_bar = cp.Variable(num_goods - 2, integer=integral)
        p_k = cp.Parameter(num_goods, name='p_k')
        x = cp.Parameter((num_agents, num_goods), name='x')
        # Do we remove drop out here or not? - remove the default and dropout good
        # objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(x[:,:-1] - y, 'fro')) - (beta / 2) * cp.square(cp.norm(cp.sum(y, axis=0) - supply[:-1], 2))) # (4) (5)
        # objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(x[:,:-2] - y, 'fro')) - (beta / 2) * cp.square(cp.norm(cp.sum(y, axis=0) - supply[:-2], 2)))
        # objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(x - y, 'fro')) - (beta / 2) * cp.square(cp.norm(cp.sum(y, axis=0) - supply, 2)))
        y_sum = cp.sum(y, axis=0)
        objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(x[:,:-2] - y, 'fro')) - (beta / 2) * cp.square(cp.norm(y_sum + y_bar - supply[:-2], 2))  - p_k[:-2].T @ y_bar)
        cp_constraints = [y_bar >= 0, y<=1, y_bar<=supply[:-2]] # remove default and dropout good
        problem = cp.Problem(objective, cp_constraints)
        warm_start = False
        # problem = cp.Problem(objective)
    else:
        y = problem.variables()[0]
        y_bar = problem.variables()[1]
        p_k = problem.param_dict['p_k']
        x = problem.param_dict['x']
    p_k.value = p_k_val
    x.value = x_val
    build_time = time.time() - start_time

    start_time = time.time()
    if integral:
        solvers = [cp.MOSEK]
    else:
        solvers = [cp.CLARABEL, cp.SCS, cp.OSQP, cp.ECOS, cp.CVXOPT]
    for solver in solvers:
        try:
            result = problem.solve(solver=solver, warm_start=warm_start, ignore_dpp=True)
            logger.info(f"Problem solved with solver {solver}")
            break
        except cp.error.SolverError as e:
            logger.error(f"Solver {solver} failed: {e}")
            continue
        except Exception as e:
            logger.error(f"An unexpected error occurred with solver {solver}: {e}")
            continue
    solve_time = time.time() - start_time
    # print(f"Market: Build time: {round(build_time,6)} - Solve time: {round(solve_time,6)} with solver {solver}")

    # Check if the problem was solved successfully
    if problem.status != cp.OPTIMAL:
        logger.error("Failed to solve the problem with all solvers.")
        raise RuntimeError("Unable to solve the optimization problem.")
    else:
        logger.info("Optimization result: %s", result)

    y_k_plus_1 = y.value

    # Update prices
    # p_k_plus_1 = np.zeros(p_k.shape)
    # try the options (default good): 
    # (3) do not update price, dont add it in optimization, 
    # (4) update price and add it in optimization, 
    # (5) dont update price but add it in optimization
    p_k_plus_1 = p_k_val[:-2] + beta * (np.sum(y_k_plus_1, axis=0) - supply[:-2]) #(3) default 
    
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
    p_k_plus_1 = np.append(p_k_plus_1, [price_default_good,0]) # default and dropout good
    # p_k_plus_1 = np.append(p_k_plus_1, 0)  #  (4) update default good


    # Update each agent's rebates
    if update_rebates:
        r_k_plus_1 = []
        for i in range(num_agents):
            agent_constraints = constraints[i]
            agent_x = np.array([x.value[i, goods_list.index(good)] for good in agent_goods_lists[i]])
            if UPDATED_APPROACH:
                # agent_x = np.array([x[i, goods_list[:-1].index(good)] for good in agent_goods_lists[i][:-1]])
                constraint_violations = np.array([agent_constraints[0][j] @ agent_x - agent_constraints[1][j] for j in range(len(agent_constraints[1]))])
            else:
                constraint_violations = np.array([max(agent_constraints[0][j] @ agent_x - agent_constraints[1][j], 0) for j in range(len(agent_constraints[1]))])

            r_k_plus_1.append(r_k[i] + beta * constraint_violations[0])
    else:
        r_k_plus_1 = r_k
    return k + 1, y_k_plus_1, p_k_plus_1, r_k_plus_1, problem


def update_agents(w, u, p, r, constraints, goods_list, agent_goods_lists, y, beta, x_iter, update_frequency, rational=False, parallel=False, integral=False):
    
    num_agents, num_goods = len(w), len(p)
    if num_agents < 10:
        parallel = False
    agent_indices = range(num_agents)
    agent_prices = [np.array([p[goods_list.index(good)] for good in agent_goods_lists[i]]) for i in agent_indices]
    agent_utilities = [np.array(u[i]) for i in agent_indices]
    agent_ys = [np.array([y[i, goods_list.index(good)] for good in agent_goods_lists[i][:-2]]) for i in agent_indices]

    args = [(w[i], agent_utilities[i], agent_prices[i], r[i], constraints[i], agent_ys[i], beta, x_iter, update_frequency, rational, integral) for i in agent_indices]

    results = []
    adjusted_budgets = []

    if parallel:
        # num_cores = cpu_count()
        num_threads = min(12, len(agent_indices)) 
        logging.info(f"Running update_agents in parallel with {num_threads} threads/ CPUs.") 
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(lambda a: update_agent(*a), args))

    else:
        for arg in args:
            results.append(update_agent(*arg))

    x = np.zeros((num_agents, num_goods))
    for i, (agent_x, adj_budget, _) in enumerate(results):
        for good in goods_list:
            if good in agent_goods_lists[i]:
                x[i, goods_list.index(good)] = agent_x[agent_goods_lists[i].index(good)]
        adjusted_budgets.append(adj_budget)

    return x, adjusted_budgets

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
        logger.debug(f"Agent {x_iter} - Updating budget: {w_i}")
        # print(f"{x_iter % update_frequency}")
        # lambda_i = r_i.T @ b_i # update lambda
        lambda_i = r_i * b_i[0]
        w_adj = w_i + lambda_i
        # print(w_adj)
        w_adj = max(w_adj, 0)
    else:
        w_adj = w_i
    # w_adj = abs(w_adj) 

    # print(f"Adjusted budget: {w_adj}")
    # optimizer check
    x_i = cp.Variable(num_goods, integer=integral)
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
        regularizers = - (beta / 2) * cp.square(cp.norm(x_i[:-2] - y_i, 2)) - (beta / 2) *cp.square(cp.norm(A_i[0][:] @ x_i - b_i[0], 2)) # - (beta / 2) * cp.square(cp.norm(A_i @ x_i - b_i, 2))        
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

    # timing again
    start_time = time.time()
    solution_found = False
    result = np.zeros(num_goods)  # Default fallback

    if integral:
        solvers = [cp.MOSEK]
    else:
        solvers = [cp.SCS, cp.CLARABEL, cp.MOSEK, cp.OSQP, cp.ECOS, cp.CVXOPT]
    
    # for solver in solvers:
    #     try:
    #         if solver == cp.MOSEK:
    #             result = problem.solve(solver=solver, mosek_params={"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-7})
    #         else:
    #             result = problem.solve(solver=solver)
    #         logger.info(f"Agent Opt - Problem solved with solver {solver}")
    #         break
    #     except cp.error.SolverError as e:
    #         # print(f"Solver error {e}")
    #         logger.error(f"Agent Opt - Solver {solver} failed: {e}")
    #         continue
    #     except Exception as e:
    #         # print(f"Solver error {e}")
    #         logger.error(f"Agent Opt - An unexpected error occurred with solver {solver}: {e}")
    #         continue
    # solve_time = time.time() - start_time
    # # print(f"Solver used: {problem.solver_stats.solver_name}")
    # # print(f"Solver stats: {problem.solver_stats}")
    # # print(f"Problem status: {problem.status}")
    
    # # Check if the problem was solved successfully
    # if problem.status != cp.OPTIMAL:
    #     logger.error("Agent Opt - Failed to solve the problem with all solvers.")
    # else:
    #     logger.info("Agent opt - Optimization result: %s", result)

    for solver in solvers:
        try:
            problem.solve(solver=solver, warm_start=True)  # Warm-start enabled
            if problem.status == cp.OPTIMAL:
                result = x_i.value  # Store result
                solution_found = True
                logging.info(f"Agent Opt - Problem solved with {solver} in {time.time() - start_time:.4f} sec")
                break  # Stop trying solvers once a valid solution is found
            else:
                logging.warning(f"Agent Opt - Solver {solver} did not return an optimal solution. Status: {problem.status}")
        except cp.error.SolverError as e:
            logging.warning(f"Agent Opt - Solver {solver} failed: {e}")
        except Exception as e:
            logging.warning(f"Agent Opt - Unexpected error with solver {solver}: {e}")

    solve_time = time.time() - start_time

    if not solution_found:
        logging.error(f"Agent Opt - All solvers failed. Using zero allocation fallback.")


    return x_i.value, w_adj, (build_time, solve_time)



def run_market(initial_values, agent_settings, market_settings, bookkeeping, rational=False, price_default_good=10, lambda_frequency=1, price_upper_bound=1000):
    """
    
    """
    logger.debug(f"Rebate frequency: {lambda_frequency}, Price upper bound: {price_upper_bound}")
    # print(f"Rebate frequency: {lambda_frequency}, Price upper bound: {price_upper_bound}")
    u, agent_constraints, agent_goods_lists, agent_indices = agent_settings
    y, p, r = initial_values
    w, supply, beta = market_settings
    goods_list = bookkeeping

    x_iter = 0
    prices = []
    rebates = []
    overdemand = []
    agent_allocations = []
    market_clearing = []
    yplot= []
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

    iteration_data = []  

    while x_iter <= MAX_NUM_ITERATIONS:  # max(abs(np.sum(opt_xi, axis=0) - C)) > epsilon:
        # if x_iter == 0: 
        #     beta_init = beta
        # else:
        #     beta = beta_init/np.sqrt(x_iter)
            # beta = beta_init / (x_iter + 1)
        
        if x_iter == 0:
            x = np.zeros((num_agents, len(p)))
            x[:,:-2] = y
            adjusted_budgets = w
        else:
            x, adjusted_budgets = update_agents(w, u, p, r, agent_constraints, goods_list, agent_goods_lists, y, beta, x_iter, lambda_frequency, rational=rational, integral=INTEGRAL_APPROACH)
        agent_allocations.append(x) # 
        overdemand.append(np.sum(x[:,:-2], axis=0) - supply[:-2].flatten())
        x_ij = np.sum(x[:,:-2], axis=0) # removing default and dropout good
        excess_demand = x_ij - supply[:-2]
        clipped_excess_demand = np.where(p[:-2] > 0, excess_demand, np.maximum(0, excess_demand)) # price removing default and dropout good
        market_clearing_error = np.linalg.norm(clipped_excess_demand, ord=2)
        market_clearing.append(market_clearing_error)

        iter_constraint_error = 0
        iter_constraint_x_y = 0
        for agent_index in range(len(agent_constraints)):
            agent_x = np.array([x[agent_index, goods_list.index(good)] for good in agent_goods_lists[agent_index]])
            agent_y = np.array([y[agent_index, goods_list.index(good)] for good in agent_goods_lists[agent_index][:-2]])
            constraint_error = agent_constraints[agent_index][0] @ agent_x - agent_constraints[agent_index][1]
            abs_constraint_error = np.sqrt(np.sum(np.square(constraint_error)))
            iter_constraint_error += abs_constraint_error 
            agent_error = np.sqrt(np.sum(np.square(agent_x[:-2] - agent_y)))
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
        k, y, p, r, problem = update_market(x, (1, p, r), (supply, beta), agent_constraints, agent_goods_lists, goods_list, 
                                            price_default_good, problem, 
                                            update_rebates=update_rebates, integral=INTEGRAL_APPROACH, price_upper_bound=price_upper_bound)
        yplot.append(y)
        rebates.append([[rebate] for rebate in r])
        prices.append(p)
        current_social_welfare = social_welfare(x, p, u, supply, agent_indices)
        social_welfare_vector.append(current_social_welfare)

        iteration_snapshot = {
            "iteration": x_iter,
            "prices": p.tolist(),  # Convert NumPy arrays to lists for JSON/CSV compatibility
            "market_clearing_error": market_clearing_error,
            "agent_allocations": x.tolist(),  # Agent allocations
            "social_welfare": current_social_welfare,
        }
        iteration_data.append(iteration_snapshot)
        x_iter += 1



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

        console.clear()
        console.print(table)
        
        # if (market_clearing_error <= tolerance) and (iter_constraint_error <= 0.0001) and (x_iter>=10) and (iter_constraint_x_y <= 0.01):
        if (market_clearing_error <= tolerance) and (iter_constraint_error <= 0.01) and (x_iter>=10) and (iter_constraint_x_y <= 0.1):
            break
        if x_iter == 1000:
            break


        # print("Iteration: ", x_iter, "- MCE: ", round(market_clearing_error, 5), "-Ax-b. Err: ", iter_constraint_error, " - Tol: ", round(tolerance,3), "x-y error:", iter_constraint_x_y)
        logger.info(f"Iteration: {x_iter}, Market Clearing Error: {market_clearing_error}, Tolerance: {tolerance}")

    console.print("[bold green]Simulation Complete! Optimization results in file: /results/log[/bold green]")


    save_iteration_data(iteration_data, "iteration_data", output_dir="results")

        # if market_clearing_error <= tolerance:
        #     break

    # print(f"Time to run algorithm: {round(time.time() - start_time_algorithm,5)}")
    logger.info(f"Time to run algorithm: {round(time.time() - start_time_algorithm,5)}")    

    data_to_plot ={
        "x_iter": x_iter,
        "prices": prices,
        "p": p,
        "overdemand": overdemand,
        "error": error,
        "abs_error": abs_error,
        "rebates": rebates,
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
    return x, last_prices, r, overdemand, agent_constraints, adjusted_budgets, data_to_plot


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

def save_iteration_data(data, filename="iteration_data", output_dir="results"):
    """
    Save iteration data to a file. Supports CSV and JSON formats based on file extension.

    Args:
        data (list): The iteration data to save. Should be a list of dictionaries for CSV.
        filename (str): The name of the file, including extension (.csv or .json).
        output_dir (str): The directory to save the file in.
    """
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists


    with output_path.open('w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

    with output_path.open('w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)


    print(f"Data saved to {output_path}")