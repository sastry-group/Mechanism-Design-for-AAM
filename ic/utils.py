import numpy as np
import matplotlib.pyplot as plt


def set_file_name(file_name, output_folder, method, force_overwrite):
    """
    Set the file name for the output file.
    """
    if not file_name:
        file_name = "output"
    if not output_folder:
        output_folder = "output"
    if not method:
        method = "fisher"
    if force_overwrite:
        return f"{output_folder}/{file_name}_{method}.json"
    return f"{output_folder}/{file_name}_{method}_{np.random.randint(1000000)}.json"


def process_allocations(x, edge_information, agent_goods_lists, flights):
    """
    Process the allocation matrix to output agent-specific goods allocations
    and corresponding indices in goods_list (master list). We also remove the default good
    for every agent to process sampling and integer allocaton
    
    Parameters:
    - x: np.ndarray of shape (len(goods_list), num_agents)
    - edge_information: dictionary of edge information ('edge_label': ('origin_node', 'destination_node')
    - agent_goods_lists: list of lists, each containing tuples of goods for each agent
    
    Returns:
    - agent_allocations: list of np.ndarrays, each containing the fractional allocations for an agent from the fisher market
    - agent_indices: list of np.ndarrays, each containing the indices of the agents goods mapping to the master goods_list
    - agent_edge_information: list of dictionaries, each containing the edge information for an agent goods  (edge_label: ('origin_node', 'destination_node'))
    """
    # Create a dictionary for quick lookup of goods in goods_list
    goods_index_map = {good: idx for idx, good in enumerate(edge_information.values())}

    # creating an agent's dictionary of goods allocations
    agents_data = {}
    
    agent_allocations = []
    agent_indices = []
    agent_edge_information = []
    edge_labels_list = list(edge_information.keys())
    agents_dropout_allocations = []
    
    for agent, agent_data in enumerate(agent_goods_lists):
        # Remove 'default_good' if it exists
        agent_goods = [good for good in agent_data if good not in ['default_good', 'dropout_good']]
        agents_dropout_allocations.append(x[agent][goods_index_map[('dropout_good')]])
        
        # Find indices of agent_goods in goods_list
        indices = [goods_index_map[good] for good in agent_goods if good in goods_index_map]
        
        # Get allocations for the agent
        allocations = x[agent][indices]
        agent_edges = {}
        for id in indices:
            agent_edge = edge_labels_list[id]
            agent_edges[agent_edge] = edge_information[agent_edge]
            
        agents_data
        agent_allocations.append(allocations)
        agent_indices.append(np.array(indices))
        agent_edge_information.append(agent_edges)
    
    return agent_allocations, agents_dropout_allocations, agent_indices, agent_edge_information

def store_agent_data(flights, fisher_allocations, agent_information, 
                    adjusted_budgets, desired_goods, agent_goods_lists, edge_information):

    
    
    flights_list = list(flights.keys())
    utility, agent_constraints, agent_goods_lists, _ = agent_information
    agent_allocations, agent_dropout_x, agent_indices, agent_edge_information = process_allocations(fisher_allocations, edge_information, agent_goods_lists, flights)

    agents_data = {}   
    for i, flight_id in enumerate(flights_list):
        agents_data[flight_id] = {'status': 'fisher_allocated'}
        agents_data[flight_id]['agent_id'] = i
        agents_data[flight_id]['original_budget'] = flights[flight_id]['budget_constraint']
        agents_data[flight_id]['utility'] = utility[i]
        agents_data[flight_id]['constraints'] = agent_constraints[i]
        agents_data[flight_id]['adjusted_budget'] = adjusted_budgets[i]
        agents_data[flight_id]['fisher_allocation'] = fisher_allocations[i]
        agents_data[flight_id]['agent_goods_list'] = agent_goods_lists[i]
        agents_data[flight_id]['desired_good_info'] = desired_goods[flight_id]
        agents_data[flight_id]['deconficted_goods'] = None
        agents_data[flight_id]['allocation_short'] = agent_allocations[i]
        agents_data[flight_id]['agent_edge_indices'] = agent_indices[i]
        agents_data[flight_id]['delayed_goods'] = []
        agents_data[flight_id]["payment"] = 0
        agents_data[flight_id]["flight_info"] = flights[flight_id] 

    return agents_data

def store_market_data(extra_data, design_parameters, market_auction_time):

    market_data = {
        'prices': extra_data["prices"],
        'rebates': extra_data["rebates"],
        'goods_list': extra_data['goods_list'],
        'capacity': extra_data["capacity"],
        'original_capacity': extra_data["capacity"],
        'demand': np.sum(extra_data["x_prob"], axis=0),
        'market_auction_time': market_auction_time,
        'num_iterations': extra_data["data_to_plot"]["x_iter"],
        'market_parameters': {key: value for key, value in design_parameters.items()},
        'fisher_run_time': extra_data["data_to_plot"]["fisher_run_time"],
        'fisher_iterations': extra_data["data_to_plot"]["x_iter"],
        'num_agents': extra_data["num_agents"],
        'num_goods': extra_data["num_goods"],
        'num_contested_goods': extra_data["num_contested_goods"],
        'num_contested_paths': extra_data["num_contested_paths"],
    }
    
    return market_data


def rank_allocations(agents_data, market_data):
    """
    Rank the allocations based on fisher allocation of the agents
    
    Parameters:
    - agents_data: dictionary of agents data
    
    Returns:
    - ranked_agents: list of tuples, each containing the agent id and the fisher allocation
    """

    ranked_agents = {}
    for flight_id, data in agents_data.items():
        desired_good_idx = data['desired_good_info']["desired_dep_edge_idx"]
        desired_good_allocation = data['fisher_allocation'][desired_good_idx]
        ranked_agents[flight_id] = {"fisher_desired_good_allocation": desired_good_allocation}
        ranked_agents[flight_id]["desired_good_id"] = desired_good_idx
        ranked_agents[flight_id]["desired_good"] = data['desired_good_info']["desired_dep_edge"]
    # Sort agents by their fisher allocation for the desired good in descending order
    sorted_agent_dict = sorted(ranked_agents.items(), key=lambda item: item[1]["fisher_desired_good_allocation"], reverse=True)
    
    # Convert sorted_agents to a list of agent ids
    ranked_agents_list = [agent[0] for agent in sorted_agent_dict]
    market_data["ranked_agents"] = ranked_agents_list
    
    return sorted_agent_dict, ranked_agents_list, market_data

def find_dep_and_arrival_nodes(edges):
    if edges is None or len(edges) == 0:
        return None
    dep_edge = False
    arr_edge = False
    
    for edge in edges:
        if "dep" in edge[1]:
            assert not dep_edge, f"Multiple departure edges found: {dep_edge} and {edge}"
            dep_edge = edge
        if "arr" in edge[0]:
            assert not arr_edge, f"Multiple arrival nodes found: {arr_edge} and {edge}"
            arr_edge = edge
    # assert "arr" in arrival_node_found, f"Arrival node not found: {arrival_node_found}"
    return dep_edge[1], arr_edge[0]


def get_next_auction_data(agent_data, market_data):
    allocation, rebased, dropped = [], [], []
    for  flight_id, data in agent_data.items():
        # print(f"Processing flight {flight_id}")
        # print(f"Goods list: {data['agent_goods_list']}")
        # print(f"Final allocation: {data['final_allocation']}")
        allocation_indices = np.where(data["final_allocation"] == 1)[0]
        allocation_indices = [int(alloc) for alloc in allocation_indices]
        # print(f"Allocation indices: {allocation_indices}")
        # print(f"Allocated goods: {[data['agent_goods_list'][index] for index in allocation_indices]}")
        if data['status'] == 'allocated':
            desired_good_idx = data['desired_good_info']["desired_dep_edge_idx"]
            int_allocation_long = np.zeros(len(data["fisher_allocation"]))
            int_allocation_long[data["agent_edge_indices"]] = data["final_allocation"][:-1] 
            int_allocation_long[-1] = data["final_allocation"][-1]
            if round(int_allocation_long[desired_good_idx]) == 1:
                good_tuple = (data['desired_good_info']["desired_dep_edge"])
                # allocation.append((flight_id, good_tuple))
                agent_data[flight_id]["good_allocated"] = good_tuple
                allocation.append((flight_id, good_tuple))
            elif round(int_allocation_long[-1]) == 1:
                data['status'] = 'rebased'
                rebased.append(flight_id)
                good_tuple = ('VOOO', 'V000')
                agent_data[flight_id]["good_allocated"] = good_tuple
            else:
                edges_id = np.where(data["final_allocation"] == 1)[0]
                edges = [data["agent_goods_list"][edge_id] for edge_id in edges_id]
                good_tuple = find_dep_and_arrival_nodes(edges)
                if good_tuple:
                    data['status'] = 'delayed'
                    agent_data[flight_id]["good_allocated"] = good_tuple
                    allocation.append((flight_id, good_tuple))
                else:
                    data['status'] = 'parked'

                # print("Check allocation")
                # else:
                #     data['status'] = 'rebased'
                #     good_tuple = ('VOOO', 'V000')
                #     agent_data[flight_id]["good_allocated"] = good_tuple
            
        # elif data['status'] == 'dropped':
        #     dropped.append(flight_id)
        #     good_tuple = ('VOOO', 'V000')
        #     agent_data[flight_id]["good_allocated"] = good_tuple
        # else:
        #     data['status'] = 'rebased'
        #     edges_id = np.where(data["final_allocation"] == 1)[0]
        #     edges = [data["agent_goods_list"][edge_id] for edge_id in edges_id]
        #     rebased.append(flight_id)
        #     good_tuple = find_dep_and_arrival_nodes(edges)
        #     agent_data[flight_id]["good_allocated"] = good_tuple
    
    return allocation, rebased, dropped

def build_edge_information(goods_list):
    """
    Build edge information from goods list.
    Outputs: a dictionary with all the labeled edges. This would a master list.
    """
    edge_information = {}
    for i, goods in enumerate(goods_list[:-2]): # without default and dropout good
        edge_information[f"e{i+1}"] = (goods[0], goods[1])

    edge_information['default_good'] = ('default_good')
    edge_information['dropout_good'] = ('dropout_good')


    return edge_information

def compute_utilites(agent_data):
    """
    Compute the utility for an agent
    """

    agent_utility_vec = np.array(agent_data["utility"]) 
    dropout_utility = agent_utility_vec[-1] 
    default_utility = agent_utility_vec[-2]
    allocated_good_idx = np.where(agent_data["final_allocation"] == 1)[0]
    allocated_good_utility = agent_utility_vec[allocated_good_idx]
    x_allocated = agent_data["final_allocation"][allocated_good_idx]
    desired_good_utility = agent_data["flight_info"]["requests"]["001"]["valuation"]
    agent_fu = np.sum(allocated_good_utility *  x_allocated) + default_utility * agent_data["final_allocation"][-2] + dropout_utility * agent_data["final_allocation"][-1]
    agent_fu_max = desired_good_utility
    return agent_fu, agent_fu_max

