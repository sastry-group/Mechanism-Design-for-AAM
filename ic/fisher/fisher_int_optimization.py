import cvxpy as cp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
# from int_allocation import update_agents, update_market


def int_optimization(x_agents, capacity, budget, prices, utility, A, b):
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
    contested_edges, agents_with_contested_allocations, contested_edges_col, contested_agent_allocations = contested_allocations(x_agents, capacity)
    if contested_edges:
        ALPHA = 0.1

        # increasing the prices to contested edges
        for contested_edge in contested_edges:
            prices[contested_edge] = prices[contested_edge] + ALPHA

        # creating a new market
        new_market_capacity = capacity - np.sum(x_agents, axis=0)
        new_market_capacity[contested_edges] = capacity[contested_edges] 
        new_market_budget = budget[agents_with_contested_allocations]
        new_market_budget = new_market_budget.tolist()
        new_market_utility = utility[agents_with_contested_allocations]

        new_market_A = []
        new_market_b = []
        # constraints for the new market
        for index in agents_with_contested_allocations:
            Aarray = A[index]
            last_index = Aarray.shape[1] - 1
            barray = b[index]
            new_market_A.append(np.delete(Aarray, last_index, axis=1))
            new_market_b.append(barray)
   

        Aprime = np.array(new_market_A)
        bprime = np.array(new_market_b)

        print("Starting information for new market:")
        print(f"New capacity: {new_market_capacity}") 
        print("Prices: ", prices)
        print("Utility: ", new_market_utility)
        # print("A: ", Aprime)
        # print("b: ", bprime)
        print("Budget: ", new_market_budget)
        print("Capacity: ", new_market_capacity)
        print("Utility: ", new_market_utility)
        print("Agents with contested allocations: ", agents_with_contested_allocations)

        # Setting up for optimization
        k = 0
        equilibrium_reached = False
        contested_agent_allocations = contested_agent_allocations.tolist()
        new_market_utility = new_market_utility.tolist()
        n_agents = len(agents_with_contested_allocations)
        xi_values = np.zeros((n_agents, len(prices)))
        while not equilibrium_reached:

            for i in range(n_agents):
                # print(len(contested_agent_allocations[i]), new_market_utility,new_market_budget[i], Aprime[i], bprime[i])
                xi_values[i,:] = find_optimal_xi(len(contested_agent_allocations[i]), new_market_utility[i], Aprime[i], bprime[i], prices, new_market_budget[i])
               
            
            demand = np.sum(xi_values, axis=0)
            for j in range(len(capacity)):
                if demand[j] > capacity[j]:
                    prices[j] = prices[j] + ALPHA
            equilibrium_reached = check_equilibrium(demand, capacity)
            
            k += 1

        new_allocation = update_allocation(x_agents, contested_edges_col, xi_values, agents_with_contested_allocations)
        print_equilibrium_results(k, equilibrium_reached, prices, xi_values, demand)
        return new_allocation
        
    else:
        print("No contested allocations")
        return x_agents
    

def update_allocation(x_agents, contested_edges_col, xi_values, agents_with_contested_allocations):
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
    print("New allocation: ", new_x_agents)
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
    print(df)



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
    problem.solve()
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
    contested_edges_col = np.array([])
    agents_with_contested_allocations = []
    for i in range(len(capacity)):
        if np.sum(integer_allocations[:, i]) > capacity[i]:
            agents_index = np.where(integer_allocations[:, i] > 0)[0]
            contested_edges_col = np.append(contested_edges_col, integer_allocations[:,i], axis=0)
            contested_edges.append(i)
            agents_with_contested_allocations.append([idx for idx in agents_index])

    allocations = []
    for agent in agents_with_contested_allocations:
        allocation = integer_allocations[agent]
        allocations.append(allocation)
    
    contested_agent_allocations = np.vstack(allocations)

    return contested_edges, agents_with_contested_allocations[0], contested_edges_col, contested_agent_allocations



# Test
num_agents, num_goods, constraints_per_agent = 5, 8, [6] * 5

u_1 = np.array([2, 6, 2, 4, 2, 0, 0, 0] * math.ceil(num_agents/2)).reshape((math.ceil(num_agents/2), num_goods))
u_2 = np.array([0, 0, 1, 0, 1, 1, 6, 4] * math.floor(num_agents/2)).reshape((math.floor(num_agents/2), num_goods))
utility = np.concatenate((u_1, u_2), axis=0).reshape(num_agents, num_goods) + np.random.rand(num_agents, num_goods)*0.2
# utiliy = np.array([[2.10628590e+00, 6.13463177e+00, 2.12402939e+00, 4.06331267e+00,
#   2.19681180e+00 ,2.04596833e-03, 6.15723078e-02, 3.08863481e-02],
#  [2.16271364e+00, 6.03811631e+00, 2.00095090e+00, 4.19687028e+00,
#   2.17477883e+00, 2.14251137e-02, 1.90627363e-01, 7.58981569e-02],
#  [2.04961097e+00, 6.12345572e+00, 2.04791694e+00, 4.05964121e+00,
#   2.11196920e+00, 6.03082793e-02, 1.24299362e-02, 8.59124297e-02],
#  [1.18106818e-01 ,1.22718131e-01, 1.08330508e+00, 1.12469963e-01,
#   1.09494796e+00 ,1.08835093e+00, 6.13612516e+00, 4.04153363e+00],
#  [9.38598097e-02 ,1.36914154e-01, 1.01101805e+00, 3.57687317e-02,
#   1.12010625e+00 ,1.07778423e+00, 6.06547205e+00, 4.03022307e+00]])
# print(utility)
x_agents = np.array([[1.00397543e+00,4.80360433e-07,1.92159788e-06,1.79412229e-06,
    1.00599048e+00,2.49121963e-07,6.26215796e-09, 2.42991058e-08],
 [1.00411052e+00 ,1.21665341e-07 ,1.80407082e-03, 1.00637552e+00,
    4.84328060e-07, 4.82682667e-07 ,7.56841742e-09 ,2.18305684e-08],
 [7.85322580e-08, 9.88166643e-01, 9.84427659e-01 ,1.86267598e-06,
    1.30186140e-07,6.49328992e-07, 1.08405199e-08, 1.74791140e-08],
 [1.38309823e-06, 5.76908587e-09, 1.35556694e-08 ,1.13796523e-08,
    4.79966863e-05 ,1.00276772e+00, 3.71215693e-06 ,1.00482195e+00],
 [3.58705716e-06, 6.56599609e-09 ,1.11557250e-08, 9.46150329e-09,
    1.00542076e+00, 6.86702827e-08, 1.00451420e+00, 1.85567648e-03]])

x_agents_rounded = np.round(x_agents, 1)

x_contested  = np.array([[1., 0., 0., 0., 1., 0., 0., 0.],
 [1. ,0. ,0., 1., 0., 0., 0., 0.],
 [1., 0., 0., 1., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0., 1., 0., 1.],
 [0., 0., 0., 0., 1., 0., 1., 0.]])
# print(x_agents_rounded)
capacity = np.array([10 ,  1.,  10.  , 1. , 10. , 10. ,  1.  , 1.])
budget = np.ones(5)*100 + np.random.rand(5)*10 - 5
# A = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, -1, -1, 0, 0, 0, 0], [0, 1, -1, 0, 0, 0, 0, 0, 0], \
#                 [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], \
#                 [0, 0, 0, 0, 0, 1, 1, 0, 0], [0, 0, -1, 0, 0, 1, 0, -1, 0], [0, 0, 0, 0, -1, 0, 1, 0, 0], \
#                 [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0]])
A =[
    np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0],
               [1, 0, 0, -1, -1, 0, 0, 0, 0],
               [0, 1, -1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1, 0]]), 
    np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0],
               [1, 0, 0, -1, -1, 0, 0, 0, 0],
               [0, 1, -1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1, 0]]), 
    np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0],
               [1, 0, 0, -1, -1, 0, 0, 0, 0],
               [0, 1, -1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1, 0]]), 
    np.array([[0, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, -1, 0, 0, 1, 0, -1, 0],
               [0, 0, 0, 0, -1, 0, 1, 0, 0],
               [1, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0, 0]]),
    np.array([[0, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, -1, 0, 0, 1, 0, -1, 0],
               [0, 0, 0, 0, -1, 0, 1, 0, 0],
               [1, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0, 0]])]
prices = np.array([0., 16.60415211, 0., 7.83724561, 0., 0., 16.97227583, 7.55527186])
b = np.array([[1., 0., 0., 0., 0., 0.],[1., 0., 0., 0., 0., 0.],[1., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0.]])

int_optimization(x_contested, capacity, budget, prices, utility, A, b)