import argparse
import json
from pathlib import Path
import numpy as np
import math
import os
import pickle
import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fisher_allocation import update_basic_agents, update_market, run_basic_market, run_market, build_graph, construct_market
from VertiportStatus import VertiportStatus


# parser = argparse.ArgumentParser(description="Inputs to test Fisher market implementation.")
# parser.add_argument(
#     "--file", type=str, required=True, help="The path to the test case json file."
# )
# args = parser.parse_args()


def test_update_basic_agents():
    print("Testing agent update")
    num_agents, num_goods, constraints_per_agent = 5, 10, [2, 3, 4, 5, 6]
    w = np.random.rand(num_agents)*10
    u = np.random.rand(num_agents, num_goods)*5
    p = np.random.rand(num_goods)*10
    r = [np.random.rand(constraints_per_agent[i])*10 for i in range(num_agents)]
    constraints = [(np.random.rand(constraints_per_agent[i], num_goods)*10, np.random.rand(constraints_per_agent[i])*10) for i in range(num_agents)]
    y = np.random.rand(num_agents, num_goods)*10
    beta = 1
    x = update_basic_agents(w, u, p, r, constraints, y, beta)
    print(x)


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


def test_run_market(plotting=False, rational=False, homogeneous=False):
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
    st0 = np.random.get_state()
    dbfile = open('ic/Picklerandomstate', 'ab')
    pickle.dump(st0, dbfile)                    
    dbfile.close()
    print("Testing full market run")
    num_agents, num_goods, constraints_per_agent = 5, 9, [6] * 5
    # w = np.random.rand(num_agents)*100
    w = np.ones(num_agents)*100 + np.random.rand(num_agents)*10 - 5
    # u = np.random.rand(num_agents, num_goods)*5
    # u = np.array([2, 6, 2, 4, 2, 1] * num_agents).reshape((num_agents, num_goods)) + np.random.rand(num_agents, num_goods)*0.2 - 0.1
    u_1 = np.array([2, 6, 2, 4, 2, 0, 0, 0, 1] * math.ceil(num_agents/2)).reshape((math.ceil(num_agents/2), num_goods))
    # u_2 = np.array([0, 0, 1, 0, 1, 1, 6, 4, 1] * math.floor(num_agents/2)).reshape((math.floor(num_agents/2), num_goods))
    u_2 = np.array([0, 0, 2, 0, 2, 2, 6, 4, 1] * math.floor(num_agents/2)).reshape((math.floor(num_agents/2), num_goods))
    u = np.concatenate((u_1, u_2), axis=0).reshape(num_agents, num_goods) + np.random.rand(num_agents, num_goods)*0.2 - 0.1
    p = np.random.rand(num_goods)*10
    # r = [np.random.rand(constraints_per_agent[i])*10 for i in range(num_agents)]
    constraints = []
    if not homogeneous:
        r = [np.zeros(constraints_per_agent[i]) for i in range(num_agents)]
        for i in range(num_agents):
            A = np.random.rand(constraints_per_agent[i], num_goods)*10
            # for j in range(num_goods):
            #     A[0, j] = 1
            A[:, -1] = 0
            # b = np.zeros(constraints_per_agent[i])
            # b[0] = np.random.rand()*10
            b = np.random.rand(constraints_per_agent[i])*10
            constraints.append((A, b))
            # constraints.append((np.concatenate((A, -A), axis=0), np.concatenate((b,-b), axis=0)+0.01))
    else:
        # A = np.array([[1, 1, 0, 0, 0, 0], [1, 0, 0, -1, -1, 0], [0, 1, -1, 0, 0, 0], [0, 0, 1, 1, 1, 0]]) # with unnecessary constraint
        A_1 = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, -1, -1, 0, 0, 0, 0], [0, 1, -1, 0, 0, 0, 0, 0, 0], \
                        [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0]])
        A_2 = np.array([[0, 0, 0, 0, 0, 1, 1, 0, 0], [0, 0, -1, 0, 0, 1, 0, -1, 0], [0, 0, 0, 0, -1, 0, 1, 0, 0], \
                        [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0]])
        # A = np.random.rand(constraints_per_agent[0], num_goods)*10
        # A[:, -1] = 0
        b = np.zeros(constraints_per_agent[0])
        b[0] = 1
        r = [np.zeros(constraints_per_agent[0]) for i in range(num_agents)]
        constraints = [(A_1, b) for i in range(math.ceil(num_agents/2))] + [(A_2, b) for i in range(math.floor(num_agents/2))]
    y = np.random.rand(num_agents, num_goods)*10
    # supply = np.random.rand(num_goods)*10
    supply = np.ones(num_goods)*1
    supply[0] = 10
    supply[4] = 10
    supply[5] = 10
    supply[2] = 10
    supply[-1] = 100
    beta = 1
    x, p, r, overdemand = run_basic_market((y, p, r), (u, constraints), (w, supply, beta), plotting=plotting, rational=rational)
    print(f"Agent allocations: {x}")
    # print(x, p, r)


def test_construct_and_run_market(data):
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
    x, p, r, overdemand = run_market((y,p,r), agent_information, market_information, bookkeeping, plotting=True, rational=False)

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
    # file_path = args.file
    # assert Path(file_path).is_file(), f"File at {file_path} does not exist."
    file_path = "test_cases/case2_fisher.json"
    test_case_data = load_json(file_path)
    test_construct_and_run_market(test_case_data)
    # test_run_market(plotting=True, rational=False, homogeneous=True)
