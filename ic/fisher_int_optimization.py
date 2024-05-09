import cvxpy as cp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math



def int_optimization(x_agents, capacity, budget, prices, utility, A, b):
    """
    Function to solve an integer optimization problem
    Args:
    utility (list, nx1): utility vector
    A (list, nxm): constraint matrix
    b (list, nx1): constraint vector
    x_agents: stacked allocation matrix for all agents, integer [n_agents x n_goods]
    """
    # Check contested allocations
    contested_edges, agents_with_contested_allocations, contested_edges_col, contested_agent_allocations = contested_allocations(x_agents, capacity)
    if contested_edges:
        # print(f"Contested edge index: {contested_edges}")
        # print(f"Agents with contested allocations: {agents_with_contested_allocations}")
        # print(f"Contested edges cols: {contested_edges_col}")
        for contested_edge in contested_edges:
            prices[contested_edge] = prices[contested_edge] 
        # print(f"Updated prices: {prices}")
        new_market_capacity = capacity - np.sum(x_agents, axis=0)
        new_market_capacity[contested_edges] = capacity[contested_edges]
        # print(f"New capacity: {new_market_capacity}")  

        new_market_budget = budget[agents_with_contested_allocations][0]
        # print(f"Congested budget: ", new_market_budget)
        # x = contested_agent_allocations
        # print("New market probabilities: ", contested_agent_allocations)
        new_market_utility = utility[agents_with_contested_allocations][0]
        # print("New utility matrix: ", new_market_utility)

        new_market_A = []
        new_market_b = []
        # constraints for the new market
        for index in agents_with_contested_allocations:
            Aarray = A[index]  
            Acleaned_array = Aarray[:-1]  # Remove the last element
            barray = b[index]
            new_market_A.append(Acleaned_array)
            new_market_b.append(barray)
   
    

        Aprime = np.array(new_market_A)
        bprime = np.array(new_market_b)
        # print(Aprime)
        # print(bprime)
        print(contested_agent_allocations)

        k = 0
        equilibrium_reached = False
        ALPHA = 0.1

        while not equilibrium_reached:
            xi_values = find_optimal_xi(contested_agent_allocations, utility, Aprime, bprime)
            
            demand = np.sum(xi_values, axis=0)
            for j in range(len(capacity)):
                if demand[j] > capacity[j]:
                    prices[k+1] = prices[k] + ALPHA
            equilibrium_reached = check_equilibrium(demand, capacity)
            
            k += 1
        print(xi_values)

        
    else:
        return x_agents

def check_equilibrium(demand, capacity):
    return np.all(demand <= capacity)

def find_optimal_xi(n, utility, A, b):
    x = cp.Variable(n, integer=True)
    objective = cp.Maximize(cp.sum(cp.multiply(utility, x)))
    constraints = [A @ x == b]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return x.value




def contested_allocations(integer_allocations, capacity):
    """
    Function to check contested allocations
    Args:
    integer_allocations (list, nxm): integer allocation matrix
    capacity (list, nx1): capacity vector
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



num_agents, num_goods, constraints_per_agent = 5, 8, [6] * 5

u_1 = np.array([2, 6, 2, 4, 2, 0, 0, 0] * math.ceil(num_agents/2)).reshape((math.ceil(num_agents/2), num_goods))
u_2 = np.array([0, 0, 1, 0, 1, 1, 6, 4] * math.floor(num_agents/2)).reshape((math.floor(num_agents/2), num_goods))
utility = np.concatenate((u_1, u_2), axis=0).reshape(num_agents, num_goods) + np.random.rand(num_agents, num_goods)*0.2
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
capacity = np.array([1. ,  1.,  1.  , 1. , 10. , 10. ,  1.  , 1.])
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
prices = np.array([0., 16.60415211, 0., 7.83724561, 0., 0., 16.97227583, 7.55527186, 4.53592165])
b = np.array([[1., 0., 0., 0., 0., 0.],[1., 0., 0., 0., 0., 0.],[1., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0.]])
int_optimization(x_agents_rounded, capacity, budget, prices, utility, A, b)