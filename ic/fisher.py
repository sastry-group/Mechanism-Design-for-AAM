import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

num_agents  = 10 
num_goods = 10
budget = np.random.rand(num_agents) # (n x 1)
capacity = (num_agents / num_goods) * np.ones(num_goods) # (1 x n) containing the maximum number of agents that can use a given good
valuation = np.random.rand(num_agents, num_goods) #  (n x m) 
initial_prices = np.random.rand(num_goods) # initialize the Prices in the market


opt_xi = np.zeros((num_agents, num_goods)) # optimal valuation, xi

# Convergence criteria
epsilon = 0.001

x_iter = 1

p = initial_prices
supply_demand2 = []

beta = 1

y_in = (1/num_goods) * np.ones((num_agents, num_goods))

while x_iter <= 100:  # max(abs(np.sum(opt_xi, axis=0) - C)) > epsilon:
    # Find optimal x values
    for i in range(num_agents):
        x = cp.Variable(num_goods)
        objective = cp.Maximize(budget[i] * cp.log(valuation[i, :] @ x) - p @ x - (beta / 2) * cp.square(cp.norm(x - y_in[i, :], 2)))
        constraints = [x >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        opt_xi[i, :] = x.value
    
    # Find optimal y values
    y = cp.Variable((num_agents, num_goods))
    objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(opt_xi - y, 2)) - (beta / 2) * cp.square(cp.norm(cp.sum(y, axis=0) - capacity, 2)))
    problem = cp.Problem(objective)
    problem.solve()
    y_in = y.value
    
    p = p + beta * (np.sum(y_in, axis=0) - capacity.flatten())
    
    sup_dem = np.sum(opt_xi, axis=0) - capacity.flatten()
    supply_demand2.append(np.sum(np.square(sup_dem)))
    
    x_iter += 1
print("Price: ", p)

plt.semilogy(range(1, x_iter), supply_demand2)
plt.xlabel('Iterations')
plt.ylabel('Difference in Supply and Demand')
plt.show()
