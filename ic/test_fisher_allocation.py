from fisher_allocation import update_agents, update_market, run_market
import numpy as np


def test_update_agents():
    print("Testing agent update")
    num_agents, num_goods, constraints_per_agent = 5, 10, [2, 3, 4, 5, 6]
    w = np.random.rand(num_agents)*10
    u = np.random.rand(num_agents, num_goods)*5
    p = np.random.rand(num_goods)*10
    r = [np.random.rand(constraints_per_agent[i])*10 for i in range(num_agents)]
    constraints = [(np.random.rand(constraints_per_agent[i], num_goods)*10, np.random.rand(constraints_per_agent[i])*10) for i in range(num_agents)]
    y = np.random.rand(num_agents, num_goods)*10
    beta = 1
    x = update_agents(w, u, p, r, constraints, y, beta)


def test_update_market():
    print("Testing market update")
    num_agents, num_goods, constraints_per_agent = 5, 10, [2, 3, 4, 5, 6]
    x = np.random.rand(num_agents, num_goods)*10
    s = np.random.rand(num_goods)*10
    p = np.random.rand(num_goods)*10
    r = [np.random.rand(constraints_per_agent[i])*10 for i in range(num_agents)]
    constraints = [(np.random.rand(constraints_per_agent[i], num_goods)*10, np.random.rand(constraints_per_agent[i])*10) for i in range(num_agents)]
    beta = 1
    _, y_new, p_new, r_new = update_market(x, (1, p, r), (s, beta), constraints)
    print(y_new, p_new, r_new)


def test_run_market(plotting=False, rational=False):
    """
    Function to update agents parameters
    num_agents (int): number of agents 
    num_goods (int): number of goods
    constraints_per_agent (list, nx1): number of constraints per agent
    w (list, num_agetns x 1): budget of each agent 
    p (list, num_goods x 1): price of each good
    r (list, num_agents x 1): rebates of each agent
    constraints (matrix, num_constraints x num_goods): constraints of each agent
    """
    print("(1) Testing full market run")
    num_agents, num_goods, constraints_per_agent = 5, 10, [2, 3, 4, 5, 6]
    w = np.random.rand(num_agents)*10
    u = np.random.rand(num_agents, num_goods)*5
    p = np.random.rand(num_goods)*10
    r = [np.random.rand(constraints_per_agent[i])*10 for i in range(num_agents)]
    constraints = []
    for i in range(num_agents):
        A = np.random.rand(constraints_per_agent[i], num_goods)*10
        # print('A: ', A)
        # for j in range(num_goods):
        #     A[0, j] = 1
        # A[:, -1] = 0
        # b = np.zeros(constraints_per_agent[i])
        # b[0] = np.random.rand()*10
        b = np.random.rand(constraints_per_agent[i])*10
        # print('b: ', b)
        constraints.append((A, b))
        # constraints.append((np.concatenate((A, -A), axis=0), np.concatenate((b,-b), axis=0)+0.01))
    y = np.random.rand(num_agents, num_goods)*10
    # supply = np.random.rand(num_goods)*10
    supply = np.ones(num_goods)*1
    supply[-1] = 100
    beta = 1
    x, p, r, overdemand = run_market((y, p, r), (u, constraints), (w, supply, beta), plotting=plotting, rational=rational)
    print("x: ", x, "p: ", p, "r: ", r)


if __name__ == "__main__":
    test_run_market(plotting=True, rational=False)
