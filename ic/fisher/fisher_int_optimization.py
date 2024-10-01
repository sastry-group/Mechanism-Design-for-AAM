import cvxpy as cp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import time


def agent_allocation_selection(ranked_list, agent_data, market_data):
    capacity = market_data['capacity']
    capacity_temp = capacity
    prices = market_data['prices'] 
    demand = market_data['demand']
    flag = False
    contested = []
    allocated = []
    for agent in ranked_list:
        Aarray = agent_data[agent]["constraints"][0]
        Aarray = Aarray[:,:-2] #removing default and dropout goods
        barray = agent_data[agent]["constraints"][1]
        fisher_allocation = agent_data[agent]["allocation_short"]
        utility = agent_data[agent]["utility"][:-2]
        budget = agent_data[agent]["adjusted_budget"]
        agent_indices = agent_data[agent]["agent_edge_indices"]
        agent_prices = prices[agent_indices]
        agent_values = find_optimal_xi(len(fisher_allocation), utility, Aarray, barray, agent_prices, budget)
        if agent_values is None:
            print("Warning: Could not find optimal xi value for agent", agent)
        else:
            # we need to do this in vertiport status as well"
            agent_values_to_full_size = np.zeros(len(prices))
            agent_values_to_full_size[agent_indices] = agent_values
            check_capacity = capacity_temp - agent_values_to_full_size
            idx_contested_edges = np.where(demand > check_capacity)[0]
            if np.all(check_capacity >= 0):
                agent_data[agent]["int_allocation"] = agent_values
                agent_data[agent]["status"] = "int_allocated"
                allocated.append(agent)
                capacity_temp = check_capacity
                market_data['capacity'] = capacity_temp

                # equilibrium_reached = check_equilibrium(demand, capacity)
            else:
                agent_data[agent]["status"] = "contested"
                contested.append(agent)
                market_data['prices'][idx_contested_edges] = market_data['prices'][idx_contested_edges] + 10000
                flag = True
        agent_data[agent]["final_allocation"] = agent_values
    market_data['int_allocated_agents'] = allocated
    market_data['contested_agents'] = contested


    return agent_data, market_data, flag

def settling_contested_allocations(agent_data, market_data):
    k = 0
    ALPHA = 0.1
    equilibrium_reached = False
    while not equilibrium_reached:
        
        allocated_agents, droppped_agents, no_solution_agents = [], [], []
        agent_xs_full_size = []
        agent_xs = []
        for agent in market_data['contested_agents']:
            Aarray = agent_data[agent]["constraints"][0]
            Aarray = Aarray[:,:-2]
            barray = agent_data[agent]["constraints"][1]

            fisher_allocation = agent_data[agent]["allocation_short"]
            utility = agent_data[agent]["utility"][:-2]
            budget = agent_data[agent]["adjusted_budget"]
            agent_indices = agent_data[agent]["agent_edge_indices"]
            prices = market_data['prices']
            agent_prices = prices[agent_indices]
            agent_values = find_optimal_xi(len(fisher_allocation), utility, Aarray, barray, agent_prices, budget)
            if agent_values is None:
                print("Warning: Could not find optimal xi value for agent", agent)
                no_solution_agents.append(agent)
            else:
                agent_xs_full_size = np.zeros(len(prices))
                agent_xs_full_size[agent_indices] = agent_values
                agent_xs.append(agent_xs_full_size)
                allocated_agents.append(agent)
                
        demand = np.sum(agent_xs, axis=0)
        idx_contested_edges = np.where(demand > market_data["capacity"])
        if idx_contested_edges == []:
            market_data["prices"] = [idx_contested_edges] + ALPHA
        
        equilibrium_reached = check_equilibrium(demand, market_data["capacity"])
        k += 1
    agent_data = update_agent_status(agent_data, agent_values, no_solution_agents, droppped_agents, allocated_agents)    

    return agent_data, market_data

def update_agent_status(agent_data, agent_values, no_solution_agents, droppped_agents, allocated_agents):
    if len(droppped_agents) > 0:
        for agent in droppped_agents:
            agent_data[agent]["status"] = "dropped"
            agent_data[agent]["final_allocation"] = np.zeros(len(agent_values))
    if len(allocated_agents) > 0:
        for agent in allocated_agents:
            agent_data[agent]["final_allocation"] = agent_values[allocated_agents.index(agent)]
            agent_data[agent]["status"] = "allocated"
    return agent_data

def int_optimization(full_allocations, capacity, budget, prices, utility, agents_constraints, agent_indices, agents_allocations, output_folder):
    #int_optimization(int_allocations_full, capacity, budget, prices, u, agent_constraints, int_allocations, output_folder)
    """
    Function to solve an integer optimization problem
    Args:
    x_agents (np.array): stacked allocation matrix for all agents, integer (n_agents,n_goods)
    capacity (np.array): capacity vector (n_goods,)
    budget (np.array): budget vector (n_agents,)
    prices (np.array): prices vector (n_goods,)
    utility (np.array): utility vector (n_agents,n_goods)
    A (list,n_agents): constraint matrix [np.array(n_constranst,n_goods), ...]
    b (np.array): constraint vector

    """
    
    # Checking contested allocations
    start_time = time.time()
    print("Checking contested allocations")
    contested_edges, agents_with_contested_allocations, contested_agent_allocations = contested_allocations(full_allocations, capacity)
    print(f"Time taken to check contested allocations: {time.time() - start_time}")

    if len(contested_edges) > 0:
        print("Contested allocations found, running integer optimization algorithm")
        start_time_int = time.time()
        ALPHA = 0.1

        # increasing the prices to contested edges
        for contested_edge in contested_edges:
            prices[contested_edge] = prices[contested_edge] + ALPHA

        # creating a new market - consider changing this to np.arrays for efficiency
        new_market_capacity = capacity - np.sum(full_allocations, axis=0)
        new_market_capacity[contested_edges] = capacity[contested_edges]  # +1  to account for the reduced capacity if substracted before
        new_market_A = []
        new_market_b = []
        for agent in agents_with_contested_allocations:
            Aarray = agents_constraints[agent][0]
            new_market_A.append(Aarray[:,:-2]) #removing default and dropout goods
            barray = agents_constraints[agent][1]
            new_market_b.append(barray)

        # Setting up for optimization
        k = 0
        equilibrium_reached = False
        contested_agent_allocations = contested_agent_allocations.tolist()
        # new_market_utility = new_market_utility.tolist()
        n_agents = len(agents_with_contested_allocations)
        xi_values = np.zeros((n_agents, len(prices)))
        while not equilibrium_reached:

            for i, agent in enumerate(agents_with_contested_allocations):
                # print(len(contested_agent_allocations[i]), new_market_utility,new_market_budget[i], Aprime[i], bprime[i])
                agent_values = np.array([0]*len(agents_allocations[agent]))
                agent_prices = prices[agent_indices[agent]]
                agent_values = find_optimal_xi(len(agents_allocations[agent]), utility[agent][:-2], new_market_A[i], new_market_b[i], agent_prices, budget[agent])
                if agent_values is None:
                    print("Warning: Could not find optimal xi value for agent", agent)
                    agent_values = np.array([0]*len(agents_allocations[i]))
                xi_values[i,:] = map_agent_values(len(prices), agent_indices[agent], agent_values)
            
            demand = np.sum(xi_values, axis=0)
            for j in range(len(new_market_capacity)):
                if demand[j] > new_market_capacity[j]:
                    prices[j] = prices[j] + ALPHA
            equilibrium_reached = check_equilibrium(demand, new_market_capacity)
            
            k += 1

        new_allocation = update_allocation(full_allocations, xi_values, agents_with_contested_allocations)
        print_equilibrium_results(k, equilibrium_reached, prices, xi_values, demand)

        print(f"Time taken to run integer optimization algorithm: {time.time() - start_time_int}")
        return new_allocation, prices
        
    else:
        print("No contested allocations")
        return full_allocations, prices
    

def update_allocation(x_agents, xi_values, agents_with_contested_allocations):
    """
    Function to update the allocation matrix with the new xi values
    Args:
    x_agents (list, nxm): allocation matrix
    xi_values (list, nxm): xi values
    agents_with_contested_allocations (list, nx1): agents with contested allocations
    """
    new_x_agents = x_agents
    for i in range(len(agents_with_contested_allocations)):
        new_x_agents[agents_with_contested_allocations[i]] = xi_values[i]
    # print("New allocation: ", new_x_agents)
    return new_x_agents



def print_equilibrium_results(iteration, equilibrium_reached, prices, xi_values, demand):
    data = {
        'Iteration': [iteration],
        'Equilibrium reached': [equilibrium_reached],
        'Prices': [prices],
        'Xi values': [xi_values],
        'Demand': [demand]
    }
    df = pd.DataFrame(data)
    df = df.transpose()
    # print(df)



def check_equilibrium(demand, capacity):
    return np.all(demand <= capacity)

def find_optimal_xi(n, utility, A, b, prices, budget):
    """
    Finds the optimal value of xi that maximizes the utility function for agent i.

    Parameters:
    - n (int): The number of agents for the new market
    - utility (list): The utility values for the agent's goods .
    - A (numpy.ndarray): The coefficient matrix for the linear equality constraints for the agent.
    - b (numpy.ndarray): The constant vector for the linear equality constraints.
    - prices (numpy.ndarray): The prices for each good.
    - budget (float): The maximum budget constraint for each agent.
    Returns:
    - numpy.ndarray: The optimal values of xi.
    """
    x = cp.Variable(n, integer=True)
    objective = cp.Maximize(cp.sum(cp.multiply(utility, x)))
    constraints = [A @ x == b, cp.sum(cp.matmul(prices, x)) <= budget, x >=0] 
    problem = cp.Problem(objective, constraints)
    result = problem.solve()
    
    print("Problem status:", problem.status)
    # print("Optimal value:", result)
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        message = f"Warning: The problem status is: {problem.status}"
        print(message)
        return None
    
    return x.value




def contested_allocations(integer_allocations, capacity):
    """
    Function to check contested allocations
    Args:
    integer_allocations (np.array): stacked allocation matrix for all agents, integer (n_agents,n_goods)
    capacity (np.array): capacity vector (n_goods,)
    Returns:
    contested_edges (list): list of contested edges
    agents_with_contested_allocations (list): list of agents with contested allocations
    contested_edges_col (np.array): column of contested edges
    contested_agent_allocations (np.array): contested agent allocations

    """

    contested_edges = []
    # contested_edges_col = np.array([])
    contested_goods_per_agents = []
    agents_with_contested_allocations = []
    integer_allocations = np.array(integer_allocations)
    capacity = np.array(capacity)
    demand = np.sum(integer_allocations, axis=0)
    contested_edges = np.where(demand > capacity)[0]
    contested_agent_allocations = []

    for id, agent in enumerate(integer_allocations):
        if np.any(agent[contested_edges] > 0):
            # contested_edges_col = np.append(contested_edges_col, contested_edges)
            # contested_edges_col.append(contested_edges)
        # contested_edges_col.append(integer_allocations[id])
            agents_with_contested_allocations.append(id)

    if agents_with_contested_allocations == []:
        return [], [], []
    else:
        contested_agent_allocations = integer_allocations[agents_with_contested_allocations]
        # contested_agent_allocations = np.vstack(allocations)

        return contested_edges, agents_with_contested_allocations, contested_agent_allocations

def map_agent_values(full_x_array, agent_indices, agent_values):
    """
    Function to map agent values to the index in agent indices
    Args:
    agent_indices (list): list of indices corresponding to the agents
    agent_values (list): list of values for each agent
    Returns:
    mapped_values (np.array): array of mapped values
    """
    mapped_values = np.zeros(full_x_array)
    for agent_index, agent_value in zip(agent_indices, agent_values):
        mapped_values[agent_index] = agent_value
    return mapped_values

# Test
# num_agents, num_goods, constraints_per_agent = 5, 8, [6] * 5

# u_1 = np.array([2, 6, 2, 4, 2, 0, 0, 0] * math.ceil(num_agents/2)).reshape((math.ceil(num_agents/2), num_goods))
# u_2 = np.array([0, 0, 1, 0, 1, 1, 6, 4] * math.floor(num_agents/2)).reshape((math.floor(num_agents/2), num_goods))
# utility = np.concatenate((u_1, u_2), axis=0).reshape(num_agents, num_goods) + np.random.rand(num_agents, num_goods)*0.2
# # utiliy = np.array([[2.10628590e+00, 6.13463177e+00, 2.12402939e+00, 4.06331267e+00,
# #   2.19681180e+00 ,2.04596833e-03, 6.15723078e-02, 3.08863481e-02],
# #  [2.16271364e+00, 6.03811631e+00, 2.00095090e+00, 4.19687028e+00,
# #   2.17477883e+00, 2.14251137e-02, 1.90627363e-01, 7.58981569e-02],
# #  [2.04961097e+00, 6.12345572e+00, 2.04791694e+00, 4.05964121e+00,
# #   2.11196920e+00, 6.03082793e-02, 1.24299362e-02, 8.59124297e-02],
# #  [1.18106818e-01 ,1.22718131e-01, 1.08330508e+00, 1.12469963e-01,
# #   1.09494796e+00 ,1.08835093e+00, 6.13612516e+00, 4.04153363e+00],
# #  [9.38598097e-02 ,1.36914154e-01, 1.01101805e+00, 3.57687317e-02,
# #   1.12010625e+00 ,1.07778423e+00, 6.06547205e+00, 4.03022307e+00]])
# # print(utility)
# x_agents = np.array([[1.00397543e+00,4.80360433e-07,1.92159788e-06,1.79412229e-06,
#     1.00599048e+00,2.49121963e-07,6.26215796e-09, 2.42991058e-08],
#  [1.00411052e+00 ,1.21665341e-07 ,1.80407082e-03, 1.00637552e+00,
#     4.84328060e-07, 4.82682667e-07 ,7.56841742e-09 ,2.18305684e-08],
#  [7.85322580e-08, 9.88166643e-01, 9.84427659e-01 ,1.86267598e-06,
#     1.30186140e-07,6.49328992e-07, 1.08405199e-08, 1.74791140e-08],
#  [1.38309823e-06, 5.76908587e-09, 1.35556694e-08 ,1.13796523e-08,
#     4.79966863e-05 ,1.00276772e+00, 3.71215693e-06 ,1.00482195e+00],
#  [3.58705716e-06, 6.56599609e-09 ,1.11557250e-08, 9.46150329e-09,
#     1.00542076e+00, 6.86702827e-08, 1.00451420e+00, 1.85567648e-03]])

# x_agents_rounded = np.round(x_agents, 1)

# x_contested  = np.array([[1., 0., 0., 0., 1., 0., 0., 0.],
#  [1. ,0. ,0., 1., 0., 0., 0., 0.],
#  [1., 0., 0., 1., 0., 0., 0., 0.],
#  [0., 0., 0., 0., 0., 1., 0., 1.],
#  [0., 0., 0., 0., 1., 0., 1., 0.]])
# # print(x_agents_rounded)
# capacity = np.array([10 ,  1.,  10.  , 1. , 10. , 10. ,  1.  , 1.])
# budget = np.ones(5)*100 + np.random.rand(5)*10 - 5
# # A = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, -1, -1, 0, 0, 0, 0], [0, 1, -1, 0, 0, 0, 0, 0, 0], \
# #                 [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], \
# #                 [0, 0, 0, 0, 0, 1, 1, 0, 0], [0, 0, -1, 0, 0, 1, 0, -1, 0], [0, 0, 0, 0, -1, 0, 1, 0, 0], \
# #                 [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0]])
# A =[
#     np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0],
#                [1, 0, 0, -1, -1, 0, 0, 0, 0],
#                [0, 1, -1, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 1, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 1, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 1, 0]]), 
#     np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0],
#                [1, 0, 0, -1, -1, 0, 0, 0, 0],
#                [0, 1, -1, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 1, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 1, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 1, 0]]), 
#     np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0],
#                [1, 0, 0, -1, -1, 0, 0, 0, 0],
#                [0, 1, -1, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 1, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 1, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 1, 0]]), 
#     np.array([[0, 0, 0, 0, 0, 1, 1, 0, 0],
#                [0, 0, -1, 0, 0, 1, 0, -1, 0],
#                [0, 0, 0, 0, -1, 0, 1, 0, 0],
#                [1, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 1, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 1, 0, 0, 0, 0, 0]]),
#     np.array([[0, 0, 0, 0, 0, 1, 1, 0, 0],
#                [0, 0, -1, 0, 0, 1, 0, -1, 0],
#                [0, 0, 0, 0, -1, 0, 1, 0, 0],
#                [1, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 1, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 1, 0, 0, 0, 0, 0]])]
# prices = np.array([0., 16.60415211, 0., 7.83724561, 0., 0., 16.97227583, 7.55527186])
# b = np.array([[1., 0., 0., 0., 0., 0.],[1., 0., 0., 0., 0., 0.],[1., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0.]])

# # int_optimization(x_contested, capacity, budget, prices, utility, A, b)