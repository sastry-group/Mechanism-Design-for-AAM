import cvxpy as cp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import time


def agent_allocation_selection(ranked_list, agent_data, market_data):
    temp_prices = market_data['prices'] 
    contested = []
    allocated = []
    contested_goods_id = []
    for agent in ranked_list:
        agent_data[agent]["status"] = "contested"
        while agent_data[agent]["status"] == "contested":
            Aarray = agent_data[agent]["constraints"][0]
            Aarray = np.hstack((Aarray[:, :-2], Aarray[:, -1].reshape(-1, 1)))

            # Aarray = np.append(Aarray, Aarray[:, -1]) #keeping dropout good
            # Aarray = Aarray[:,:-2] #removing default and dropout goods
            barray = agent_data[agent]["constraints"][1]
            n_vals = len(agent_data[agent]["utility"]) - 1 # to remove default good
            utility = agent_data[agent]["utility"][:-2] + [agent_data[agent]["utility"][-1]] # do not remove dropout
            budget = agent_data[agent]["original_budget"]
            agent_indices = agent_data[agent]["agent_edge_indices"]
            agent_prices = temp_prices[agent_indices] 
            agent_prices = np.append(agent_prices, temp_prices[-1]) # adding dropout good
            agent_values, valuation = find_optimal_xi(n_vals, utility, Aarray, barray, agent_prices, budget)
            if agent_values is None:
                print("Warning: Could not find optimal xi value for agent", agent)
            else:
                # we need to do this in vertiport status as well"
                agent_values_to_full_size = np.zeros(len(temp_prices))
                agent_values_to_full_size[agent_indices] = agent_values[:-1]
                agent_values_to_full_size[-1] =  agent_values[-1]
                check_capacity = market_data["capacity"] - agent_values_to_full_size
                if np.all(check_capacity >= 0):
                    agent_data[agent]["final_allocation"] = agent_values
                    agent_data[agent]["status"] = "allocated"
                    allocated.append(agent)
                    market_data['capacity'] = check_capacity
                elif agent_values[-1] == 1:
                    agent_data[agent]["final_allocation"] = agent_values
                    agent_data[agent]["status"] = "dropped"
                else:
                    contested.append(agent)
                    idx_contested_edges = np.where(check_capacity < 0)[0]
                    temp_prices[idx_contested_edges] += 10000
                    contested_goods_id.append(idx_contested_edges)
        print(f"Agent values: {agent_values} with valuation {valuation}")
        agent_data[agent]["valuation"] = valuation
                


    return agent_data, market_data



def track_delayed_goods(agents_data_dict, market_data_dict):

    goods_list = market_data_dict['goods_list']
    for agent_id, agent_data in agents_data_dict.items():
        delayed_goods = []
        for good in agent_data["agent_goods_list"]:
            if "dep" in good[0] and "arr" in good[1]:
                good_tuple = (good[0], good[1])
                delayed_goods.append(good_tuple)
        delayed_goods.pop(0)
        agent_data['delayed_goods'] = delayed_goods
        delayed_goods_indices = [goods_list.index(good) for good in delayed_goods]
        agent_data["idx_delayed_goods"] = delayed_goods_indices
    return agents_data_dict


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
    
    # print("Problem status:", problem.status)
    # print("Optimal value:", result)
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        message = f"Warning: The problem status is: {problem.status}"
        print(message)
        return None
    
    return x.value, result




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
    agent_indices (list): list of indices corresponding to the agent goods in the full array
    agent_values (list): list of values for each agent ( this is only the values for each agent)
    Returns:
    mapped_values (np.array): the size here is n_goods
    """
    mapped_values = np.zeros(full_x_array)
    for agent_index, agent_value in zip(agent_indices, agent_values):
        mapped_values[agent_index] = agent_value
    return mapped_values

def map_goodslist_to_agent_goods(goods_list, agent_goods):
    """
    Function to map goods list to agent goods
    Args:
    goods_list (list): list of goods (this the entire list of goods)
    agent_goods (list): list of agent goods
    Returns:
    mapped_goods (np.array): 
    """
    ind2master_goodsidx = []
    for agent, agent_goods_list in enumerate(agent_goods):
        mapped_goods = np.zeros(len(agent_goods_list))
        mapped_goods = [goods_list.index(good) for good in agent_goods_list]
        ind2master_goodsidx.append(mapped_goods)

    

    return ind2master_goodsidx

