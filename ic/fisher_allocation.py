import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
import json
from VertiportStatus import VertiportStatus
from sampling_graph import build_edge_information, agent_probability_graph_extended, sample_path, plot_sample_path_extended, build_agent_edge_utilities
from fisher_int_optimization import int_optimization
from pathlib import Path
import math


UPDATED_APPROACH = True


def build_graph(vertiport_status, timing_info):
    """
    (2)
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


def construct_market(market_graph, flights, timing_info):
    """
    (3)
    """
    max_time, time_step = timing_info["end_time"], timing_info["time_step"]
    times_list = list(range(timing_info["start_time"], max_time + time_step, time_step))

    print("Constructing market...")
    start_time_market_construct = time.time()
    goods_list = list(market_graph.edges) + ['default_good']
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
                start_node, end_node = origin_vertiport + "_" + str(dep_time) + "_dep", destination_vertiport + "_" + str(arr_time) + "_arr"
                attributes = {"valuation": request["valuation"]}
                agent_graph.add_edge(start_node, end_node, **attributes)
                dep_start_node, dep_end_node = origin_vertiport + "_" + str(dep_time), origin_vertiport + "_" + str(dep_time) + "_dep"
                arr_start_node, arr_end_node = destination_vertiport + "_" + str(arr_time) + "_arr", destination_vertiport + "_" + str(arr_time)
                agent_graph.add_edge(dep_start_node, dep_end_node, **{"valuation": 0})
                agent_graph.add_edge(arr_start_node, arr_end_node, **{"valuation": 0})
                stationary_times = [time for time in times_list if time >= arr_time]
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
        valuations.append(1) # Small positive valuation for default good

        b = np.zeros(len(A))
        b[0] = 1
        
        A_with_default_good = np.hstack((A, np.zeros((A.shape[0], 1))))
        goods = edges + ['default_good']

        w.append(flight["budget_constraint"])
        u.append(valuations)
        agent_constraints.append((A_with_default_good, b))
        agent_goods_lists.append(goods)

        supply = np.ones(len(goods_list))
        supply[-1] = 100
        beta = 1

    print(f"Time to construct market: {time.time() - start_time_market_construct}")
    return (u, agent_constraints, agent_goods_lists), (w, supply, beta), (goods_list, times_list)


def update_basic_market(x, values_k, market_settings, constraints):
    '''Update market consumption, prices, and rebates'''
    shape = np.shape(x)
    num_agents = shape[0]
    num_goods = shape[1]
    k, p_k, r_k = values_k
    supply, beta = market_settings
    
    # Update consumption
    y = cp.Variable((num_agents, num_goods))
    objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(x - y, 2)) - (beta / 2) * cp.square(cp.norm(cp.sum(y, axis=0) - supply, 2)))
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


def update_market(x, values_k, market_settings, constraints, agent_goods_lists, goods_list):
    '''
    Update market consumption, prices, and rebates
    (7)
    '''
    shape = np.shape(x)
    num_agents = shape[0]
    num_goods = shape[1]
    k, p_k, r_k = values_k
    supply, beta = market_settings
    
    # Update consumption
    y = cp.Variable((num_agents, num_goods))
    objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(x - y, 2)) - (beta / 2) * cp.square(cp.norm(cp.sum(y, axis=0) - supply, 2)))
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
        agent_x = np.array([x[i, goods_list.index(good)] for good in agent_goods_lists[i]])
        if UPDATED_APPROACH:
            agent_x = np.array([x[i, goods_list.index(good)] for good in agent_goods_lists[i]])
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


def update_agents(w, u, p, r, constraints, goods_list, agent_goods_lists, y, beta, rational=False):
    """
    (5)
    """
    num_agents = len(w)
    num_goods = len(p)
    x = np.zeros((num_agents, num_goods))
    for i in range(num_agents):
        agent_p = np.array([p[goods_list.index(good)] for good in agent_goods_lists[i]])
        agent_u = np.array(u[i])
        agent_y = np.array([y[i, goods_list.index(good)] for good in agent_goods_lists[i]])
        agent_x = update_agent(w[i], agent_u, agent_p, r[i], constraints[i], agent_y, beta, rational=rational)
        for good in goods_list:
            if good in agent_goods_lists[i]:
                x[i, goods_list.index(good)] = agent_x[agent_goods_lists[i].index(good)]
    # print(x)
    return x


def update_agent(w_i, u_i, p, r_i, constraints, y_i, beta, rational=False):
    """
    (4) Update individual agent's consumption given market settings and constraints
    (6)
    """
    # Individual agent optimization
    A_i, b_i = constraints
    num_constraints = len(b_i)
    num_goods = len(p)

    budget_adjustment = r_i.T @ b_i
    w_adj = w_i + budget_adjustment
    w_adj = max(w_adj, 0)

    # print(f"Adjusted budget: {w_adj}")
    # optimizer check
    x_i = cp.Variable(num_goods)
    if rational:
        start_time_update = time.time()
        regularizers = - (beta / 2) * cp.square(cp.norm(x_i - y_i, 2)) - (beta / 2) * cp.sum([cp.square(cp.maximum(A_i[t] @ x_i - b_i[t], 0)) for t in range(num_constraints)])
        lagrangians = - p.T @ x_i - cp.sum([r_i[t] * cp.maximum(A_i[t] @ x_i - b_i[t], 0) for t in range(num_constraints)])
        objective = cp.Maximize(u_i.T @ x_i + regularizers + lagrangians)
        cp_constraints = [x_i >= 0]
        print(f"1 Time to compute regularizers and lagrangians: {time.time() - start_time_update}")
        # cp_constraints = [x_i >= 0, p.T @ x_i <= w_adj]
        # objective = cp.Maximize(u_i.T @ x_i)
        # cp_constraints = [x_i >= 0, p.T @ x_i <= w_adj, A_i @ x_i <= b_i]
    elif UPDATED_APPROACH:
        start_time_update = time.time()
        regularizers = - (beta / 2) * cp.square(cp.norm(x_i - y_i, 2)) - (beta / 2) * cp.sum([cp.square(A_i[t] @ x_i - b_i[t]) for t in range(num_constraints)])
        lagrangians = - p.T @ x_i - cp.sum([r_i[t] * (A_i[t] @ x_i - b_i[t]) for t in range(num_constraints)])
        nominal_objective = w_adj * cp.log(u_i.T @ x_i)
        objective = cp.Maximize(nominal_objective + lagrangians + regularizers)
        cp_constraints = [x_i >= 0]
        print(f"2 Time to compute regularizers and lagrangians: {time.time() - start_time_update}")
    else:
        start_time_update = time.time()
        regularizers = - (beta / 2) * cp.square(cp.norm(x_i - y_i, 2)) - (beta / 2) * cp.sum([cp.square(cp.maximum(A_i[t] @ x_i - b_i[t], 0)) for t in range(num_constraints)])
        lagrangians = - p.T @ x_i - cp.sum([r_i[t] * cp.maximum(A_i[t] @ x_i - b_i[t], 0) for t in range(num_constraints)])
        nominal_objective = w_adj * cp.log(u_i.T @ x_i)
        objective = cp.Maximize(nominal_objective + lagrangians + regularizers)
        cp_constraints = [x_i >= 0]
        print(f"3 Time to compute regularizers and lagrangians: {time.time() - start_time_update}")
    check_time = time.time()
    problem = cp.Problem(objective, cp_constraints)
    problem.solve(solver=cp.CLARABEL)
    print(f"Solvers time: {time.time() - check_time}")
    return x_i.value


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



def run_market(initial_values, agent_settings, market_settings, bookkeeping, plotting=True, rational=False, output_folder=False):
    """
    (4)
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
    while x_iter <= 300:  # max(abs(np.sum(opt_xi, axis=0) - C)) > epsilon:
        # Update agents
        x = update_agents(w, u, p, r, agent_constraints, goods_list, agent_goods_lists, y, beta, rational=rational)
        agent_allocations.append(x)
        overdemand.append(np.sum(x, axis=0) - supply.flatten())
        clipped_excess_demand = np.where(p > 0, overdemand, np.maximum(0, overdemand))
        market_clearning_error = np.linalg.norm(clipped_excess_demand, ord=2)
        market_clearing.append(market_clearning_error)
        for agent_index in range(len(agent_constraints)):
            agent_x = np.array([x[agent_index, goods_list.index(good)] for good in agent_goods_lists[agent_index]])
            constraint_error = agent_constraints[agent_index][0] @ agent_x - agent_constraints[agent_index][1]
            if x_iter == 0:
                error.append([constraint_error])
            else:
                error[agent_index].append(constraint_error)

        # Update market
        k, y, p, r = update_market(x, (1, p, r), (supply, beta), agent_constraints, agent_goods_lists, goods_list)
        rebates.append([rebate_list for rebate_list in r])
        prices.append(p)
        x_iter += 1

    if plotting:
        plt.subplot(2, 3, 1)
        for good_index in range(len(p)):
            plt.plot(range(1, x_iter+1), [prices[i][good_index] for i in range(len(prices))])
        plt.xlabel('x_iter')
        plt.ylabel('Prices')
        plt.title("Price evolution")
        # plt.legend(p[-1], title="Fake Good")


        plt.subplot(2, 3, 2)
        plt.plot(range(1, x_iter+1), overdemand)
        plt.xlabel('x_iter')
        plt.ylabel('Demand - Supply')
        plt.title("Overdemand evolution")

        plt.subplot(2, 3, 3)
        for agent_index in range(len(agent_constraints)):
            plt.plot(range(1, x_iter+1), error[agent_index])
        plt.title("Constraint error evolution")

        plt.subplot(2, 3, 4)
        for constraint_index in range(len(rebates[0])):
            plt.plot(range(1, x_iter+1), [rebates[i][constraint_index] for i in range(len(rebates))])
        plt.title("Rebate evolution")

        plt.subplot(2, 3, 5)
        for agent_index in range(len(agent_allocations[0])):
            plt.plot(range(1, x_iter+1), [agent_allocations[i][agent_index] for i in range(len(agent_allocations))])
        plt.title("Agent allocation evolution")

        plt.subplot(2, 3, 6)
        plt.plot(range(1, x_iter+1), market_clearing)
        plt.xlabel('x_iter')
        plt.title("Market Clearing Error")
        # plt.show()
        if output_folder:
            plt.savefig(f"{output_folder}/market_plot.png")
    

    last_prices = np.array(prices[-1])
    # final_prices = last_prices[last_prices > 0]

    # print(f"Error: {[error[i][-1] for i in range(len(error))]}")
    # print(f"Overdemand: {overdemand[-1][:]}")
    return x, last_prices, r, overdemand, agent_constraints

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


if __name__ == "__main__":
    file_path = "test_cases/case2_fisher.json"
    file_name = file_path.split("/")[-1].split(".")[0]
    data = load_json(file_path)
    output_folder = f"ic/results/{file_name}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)


    flights = data["flights"]
    vertiports = data["vertiports"]
    timing_info = data["timing_info"]

    # Create vertiport graph and add starting aircraft positions
    vertiport_usage = VertiportStatus(vertiports, data["routes"], timing_info)

    # Build Fisher Graph
    market_graph = build_graph(vertiport_usage, timing_info)

    # Construct market
    agent_information, market_information, bookkeeping = construct_market(market_graph, flights, timing_info)

    # Run market
    goods_list, times_list = bookkeeping
    num_goods = len(goods_list)
    num_agents = len(flights)
    u, agent_constraints, agent_goods_lists = agent_information
    y = np.random.rand(num_agents, num_goods)*10
    p = np.random.rand(num_goods)*10
    r = [np.zeros(len(agent_constraints[i][1])) for i in range(num_agents)]
    x, prices, r, overdemand, agent_constraints = run_market((y,p,r), agent_information, market_information, bookkeeping, plotting=True, rational=False)

    # u_1 = np.array([2, 6, 2, 4, 2, 0, 0, 0] * math.ceil(num_agents/2)).reshape((math.ceil(num_agents/2), num_goods))
    # u_2 = np.array([0, 0, 1, 0, 1, 1, 6, 4] * math.floor(num_agents/2)).reshape((math.floor(num_agents/2), num_goods))
    # u = np.concatenate((u_1, u_2), axis=0).reshape(num_agents, num_goods) + np.random.rand(num_agents, num_goods)*0.2
    # print(x, prices, r, overdemand, agent_constraints)
    # Sampling fractional edges
    edge_information = build_edge_information(goods_list)
    all_agents_utilities = build_agent_edge_utilities(edge_information, agent_goods_lists, u)

    int_allocations = np.zeros((num_agents, num_goods-1))
    for i in range(x.shape[0]):
        frac_allocations = x[i][:-1]
        start_edge_index = np.nonzero(x[i])[0][0]
        start_node = edge_information[f"e{start_edge_index+1}"][0]
        extended_graph, agent_allocation = agent_probability_graph_extended(edge_information, frac_allocations, output_folder)
        sampled_path_extended, sampled_edges, int_allocation = sample_path(extended_graph, start_node, agent_allocation)
        print("Sampled Path:", sampled_path_extended)
        print("Sampled Edges:", sampled_edges)
        plot_sample_path_extended(extended_graph, sampled_path_extended, output_folder)
        int_allocations[i] = int_allocation
    # test_run_market(plotting=True, rational=False, homogeneous=True)

    print(int_allocations)
    # IOP for contested goods
    budget, capacity, _ = market_information
    capacity = capacity[:-1]
    print("Budget:", budget)
    print("Capacity:", capacity)
    # print(agent_constraints)
    new_allocations = int_optimization(int_allocations, capacity, budget, prices, all_agents_utilities, agent_constraints)
    print(new_allocations)

    # testing temp, remove later
    output_file = f"{output_folder}/output.txt"
    with open(output_file, "w") as f:
        f.write("From Fisher Markets:\n")
        f.write("Allocations:\n")
        f.write(str(x))        
        f.write("Prices:\n")
        f.write(str(prices))
        f.write("\n")
        f.write("Constraints:\n")
        f.write(str(agent_constraints))
        f.write("\n------------------------------------------------")
        f.write("Agent Allocation:\n")
        f.write(str(agent_allocation))
        f.write("\n")
        f.write("Utilties:\n")
        f.write(str(all_agents_utilities))
        f.write("Int Allocation:\n")
        f.write(str(int_allocations))
        f.write("\n")
        f.write("New Int Allocation:\n")
        f.write(str(new_allocations))
    # For testing purposes
    print("Output written to", output_file)


