import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# MODEL PARAMETERS
epsilon = 0.001
beta = 1


def update_market(x, values_k, market_settings, constraints):
    '''Update market consumption, prices, and rebates'''
    shape = np.shape(x)
    num_agents = shape[0]
    num_goods = shape[1]
    k, p_k, r_k = values_k
    supply, beta = market_settings
    
    # Update consumption
    y = cp.Variable((num_agents, num_goods))
    objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(x - y, 2)) - (beta / 2) * cp.square(cp.norm(cp.sum(y, axis=0) - supply, 2)))
    cp_constraints = [y >= 0]
    problem = cp.Problem(objective, cp_constraints)
    problem.solve()
    y_k_plus_1 = y.value

    # Update prices
    p_k_plus_1 = p_k + beta * (np.sum(y_k_plus_1, axis=0) - supply)

    # Update each agent's rebates
    r_k_plus_1 = []
    for i in range(num_agents):
        agent_constraints = constraints[i]
        r_k_plus_1.append(r_k[i] + beta * np.array([max(agent_constraints[0][j] @ x[i] - agent_constraints[1][j], 0) for j in range(len(agent_constraints[1]))]))
    return k + 1, y_k_plus_1, p_k_plus_1, r_k_plus_1


def update_agents(w, u, p, r, constraints, y, beta):
    num_agents = len(w)
    num_goods = len(p)
    x = np.zeros((num_agents, num_goods))
    for i in range(num_agents):
        x[i,:] = update_agent(w[i], u[i,:], p, r[i], constraints[i], y[i,:], beta)
    print(x)
    return x


def update_agent(w_i, u_i, p, r_i, constraints, y_i, beta):
    # Individual agent optimization
    A_i, b_i = constraints
    num_constraints = len(b_i)

    budget_adjustment = r_i.T @ b_i
    w_adj = w_i + budget_adjustment

    x_i = cp.Variable(num_goods)
    regularizers = - (beta / 2) * cp.square(cp.norm(x_i - y_i, 2)) - (beta / 2) * cp.sum([cp.square(cp.maximum(A_i[t] @ x_i - b_i[t], 0)) for t in range(num_constraints)])
    lagrangians = - p.T @ x_i - cp.sum([r_i[t] * cp.maximum(A_i[t] @ x_i - b_i[t], 0) for t in range(num_constraints)])
    nominal_objective = w_adj * cp.log(u_i.T @ x_i)
    objective = cp.Maximize(nominal_objective + lagrangians + regularizers)
    cp_constraints = [x_i >= 0]
    problem = cp.Problem(objective, cp_constraints)
    problem.solve(verbose=True)
    return x_i.value


def run_market(initial_values, agent_settings, market_settings):
    u, agent_constraints = agent_settings
    y, p, r = initial_values
    w, supply, beta = market_settings

    x_iter = 0
    while x_iter <= 100:  # max(abs(np.sum(opt_xi, axis=0) - C)) > epsilon:
        # Update agents
        x = update_agents(w, u, p, r, agent_constraints, y, beta)

        # Update market
        k, y, p, r = update_market(x, (1, p, r), (supply, beta), agent_constraints)
    return x, p, r


if __name__ == "__main__":
    # x = test_update_agents(5, 10, [2, 3, 4, 5, 6])
    # y, p, r = test_update_market(5, 10, [2, 3, 4, 5, 6])
    x, p, r = test_run_market(5, 10, [2, 3, 4, 5, 6])
    # p, x_iter, supply_demand = fisher_allocation(num_agents, num_goods)
    # print("Price: ", p)

    # plt.semilogy(range(1, x_iter), supply_demand)
    # plt.xlabel('Iterations')
    # plt.ylabel('Difference in Supply and Demand')
    # plt.show()