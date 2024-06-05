import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
import re
import time

def agent_probability_graph(edge_information, x):


    G = nx.DiGraph()

    # Creating the nodes
    for i, edge in enumerate(edge_information.values()):
        origin_node = edge[0]
        destination_node = edge[1]
        origin_timestep = int(origin_node.split("_")[1])
        destination_timestep = int(destination_node.split("_")[1])

        G.add_node(f"{origin_node}", subset=origin_timestep)
        G.add_node(f"{destination_node}", subset=destination_timestep)
        G.add_edge(f"{origin_node}", f"{destination_node}", weight=x[i])


    pos = nx.multipartite_layout(G, subset_key="subset")
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
    plt.title("Time Extended Decision Tree Graph with Branching and Constraints")
    plt.show()

    return G


def parse_node_label(node):
    vertiport_match = re.search(r'V(\d+)', node)
    time_step_match = re.search(r'(\d+)(?!.*\d)', node)
    suffix_match = re.search(r'(arr|dep)', node)
    
    vertiport_number = int(vertiport_match.group(1)) if vertiport_match else None
    time_step = int(time_step_match.group(0)) if time_step_match else None
    suffix = suffix_match.group(0) if suffix_match else 'mid'  # 'mid' if no suffix
    
    return vertiport_number, time_step, suffix

def custom_layout(G):
    pos = {}
    layer_map = {'arr': 0, 'mid': 1, 'dep': 2}
    nodes_per_time_step = {}
    
    for node in G.nodes():
        vertiport_number, time_step, suffix = parse_node_label(node)
        if time_step not in nodes_per_time_step:
            nodes_per_time_step[time_step] = {}
        if vertiport_number not in nodes_per_time_step[time_step]:
            nodes_per_time_step[time_step][vertiport_number] = {}
        nodes_per_time_step[time_step][vertiport_number][suffix] = node
    
    for time_step in sorted(nodes_per_time_step):
        for vertiport_number in sorted(nodes_per_time_step[time_step]):
            offset = 0
            for suffix in ['arr', 'mid', 'dep']:
                if suffix in nodes_per_time_step[time_step][vertiport_number]:
                    node = nodes_per_time_step[time_step][vertiport_number][suffix]
                    pos[node] = (time_step, -vertiport_number * 10 - layer_map[suffix])
    return pos

def agent_probability_graph_extended(edge_information, x, agent_number=1, output_folder=False):
    G = nx.DiGraph()

    # Add edges with weights and labels
    agent_label = f"Agent_{agent_number}"
    agent_allocations = {}
    for (key, edge), weight in zip(edge_information.items(), x):
        if weight == 0:
            agent_allocations[key] = 0
            continue
        origin_node, destination_node = edge
        # G.add_edge(origin_node, destination_node, weight=x[list(edge_information.keys()).index(key)], label=key)
        G.add_edge(origin_node, destination_node, weight=weight, label=key)
        agent_allocations[key] = weight

    # Create custom layout and plot the graph
    pos = custom_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', edge_color='#909090', font_size=9)
    edge_labels = {(u, v): f"{d['label']} ({d['weight']:.4f})" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, font_color='red')
    plt.title("Time Extended Decision Tree Graph with Branching and Constraints")

    if output_folder:
        plt.savefig(f'{output_folder}/extended_graph_{agent_label}.png')
    # plt.show()
    plt.close()

    return G, agent_allocations



def sample_path(G, start_node, agent_allocations):
    """Sample a path in the graph G starting from the given node."""

    # start_time_sample = time.time()
    # print("Sampling edges ...")

    current_node = start_node
    path = [current_node]
    edges = []
    allocation_updates = {}

    while True:
        possible_edges = list(G.out_edges(current_node, data=True))
        if not possible_edges:
            break
        
        # Normalize weights
        total_weight = sum(edge[2]['weight'] for edge in possible_edges)
        weights = [edge[2]['weight'] / total_weight for edge in possible_edges]
        next_nodes = [edge[1] for edge in possible_edges]

        # Select next node based on normalized weights
        chosen_edge = random.choices(possible_edges, weights=weights, k=1)[0]
        edge_label = chosen_edge[2]['label']
        allocation_updates[edge_label] = 1  
        next_node = chosen_edge[1]

        # Update path and edges
        path.append(next_node)
        edges.append(edge_label)  # Save the edge name from the data attributes
        current_node = next_node

    agent_allocations = {key: 0 for key in agent_allocations}
    agent_allocations.update(allocation_updates)
    agent_int_allocations = list(agent_allocations.values())

    # print(f"Time to sample: {time.time() - start_time_sample:.5f}")

    return path, edges, agent_int_allocations


def plot_sample_path(G, sampled_path):
    # Add new path to the G graph with a thicker edge to highlight the chosen path
    for i in range(len(sampled_path) - 1):
        origin_node = sampled_path[i]
        destination_node = sampled_path[i + 1]
        G.add_edge(origin_node, destination_node, style='highlighted')

    # Draw the updated graph
    pos = nx.multipartite_layout(G, subset_key="subset")
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
    highlighted_edges = [(u, v) for u, v, d in G.edges(data=True) if 'style' in d and d['style'] == 'highlighted']
    nx.draw_networkx_edges(G, pos, edgelist=highlighted_edges, width=3, edge_color='red')
    plt.title("Time Extended Decision Tree Graph with Branching and Constraints")
    # plt.show()
    return 

def plot_sample_path_extended(G, sampled_path, agent_number, output_folder=False):
    # Create a copy of the original graph to avoid modifying it
    H = G.copy()
    agent_label = f"Agent_{agent_number}"
    # Add new path to the graph with a thicker edge to highlight the chosen path
    for i in range(len(sampled_path) - 1):
        origin_node = sampled_path[i]
        destination_node = sampled_path[i + 1]
        if H.has_edge(origin_node, destination_node):
            H[origin_node][destination_node]['style'] = 'highlighted'
        else:
            print(f"Edge {origin_node} to {destination_node} does not exist in the graph.")

    # Draw the updated graph
    pos = custom_layout(H)  # Use the same layout as the original graph
    plt.figure(figsize=(12, 8))
    nx.draw(H, pos, with_labels=True, node_size=700, node_color='lightblue')
    edge_labels = {(u, v): f"{d['label']} ({d['weight']:.2f})" for u, v, d in H.edges(data=True)}
    nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels)
    highlighted_edges = [(u, v) for u, v, d in H.edges(data=True) if 'style' in d and d['style'] == 'highlighted']
    nx.draw_networkx_edges(H, pos, edgelist=highlighted_edges, width=3, edge_color='red')
    plt.title("Time Extended Decision Tree Graph with Highlighted Chosen Path")
    # plt.show()
    if output_folder:
        plt.savefig(f'{output_folder}/sampled_graph_{agent_label}.png')
    plt.close()
    return H


def build_edge_information(goods_list):
    """
    Build edge information from goods list.
    The last good is the default good so it is removed
    Outputs: a dictionary with all the labeled edges. This would a master list.
    """
    edge_information = {}
    for i, goods in enumerate(goods_list[:-1], start=1):
        edge_information[f"e{i}"] = (goods[0], goods[1])
    return edge_information



def build_agent_edge_utilities(edge_information, agents_goods_list, utility_values):
    """
    Build a dictionary of edges with node tuples and utility values for each agent.
    This creates utility values for agents goods mapped to the master list
    """
    # Initialize list to store the utilities for each agent
    all_agents_utilities = []

    # Iterate over each agent's goods list and utility values
    for agent_goods, utilities in zip(agents_goods_list, utility_values):
        # Create a set of agent's goods for quick lookup
        agent_goods_set = set(agent_goods[:-1])  # Exclude the default good

        # Initialize a list to store utility values in the order of edge_information
        agent_utilities = []
        utility_index = 0

        # Populate the agent_utilities list according to the edge_information order
        for nodes in edge_information.values():
            if nodes in agent_goods_set:
                agent_utilities.append(utilities[utility_index])
                utility_index += 1
            else:
                agent_utilities.append(0)

        # Append the agent's utility list to the list of all agents' utilities
        all_agents_utilities.append(agent_utilities)

    return all_agents_utilities

def process_allocations(x, edge_information, agent_goods_lists):
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
    - agent_indices: list of np.ndarrays, each containing the indices in goods_list for an agent to map the indices to the master list
    - agent_edge_information: list of dictionaries, each containing the edge information for an agent goods  (edge_label: ('origin_node', 'destination_node'))
    """
    # Create a dictionary for quick lookup of goods in goods_list
    goods_index_map = {good: idx for idx, good in enumerate(edge_information.values())}
    
    agent_allocations = []
    agent_indices = []
    agent_edge_information = []
    edge_labels_list = list(edge_information.keys())
    
    for agent, agent_data in enumerate(agent_goods_lists):
        # Remove 'default_good' if it exists
        agent_goods = [good for good in agent_data if good != 'default_good']
        
        # Find indices of agent_goods in goods_list
        indices = [goods_index_map[good] for good in agent_goods if good in goods_index_map]
        
        # Get allocations for the agent
        allocations = x[agent][indices]
        agent_edges = {}
        for id in indices:
            agent_edge = edge_labels_list[id]
            agent_edges[agent_edge] = edge_information[agent_edge]
        
        agent_allocations.append(allocations)
        agent_indices.append(np.array(indices))
        agent_edge_information.append(agent_edges)
    
    return agent_allocations, agent_indices, agent_edge_information


def mapping_agent_to_full_data(full_edge_information, sampled_edges):
    allocation_array = [0] * len(full_edge_information)
    edge_to_index = {edge: index for index, edge in enumerate(full_edge_information)}
    for edge in sampled_edges:
        if edge in edge_to_index:
            allocation_array[edge_to_index[edge]] = 1
    
    return allocation_array


def mapping_goods_from_allocation(new_allocations, goods_list):
    agent_goods = []

    # Iterate over each array in new_allocations
    for i in range(new_allocations.shape[0]):
        indices = np.where(new_allocations[i] == 1)[0]
        allocation_goods = [goods_list[idx] for idx in indices]
        agent_goods.append(allocation_goods)
    return agent_goods


# def create_int_to_frac_datasize(agent_edge_information, sampled_edges):
#     frac_to_int_index_mapping = [0] * len(agent_edge_information)
#     for i, edge in enumerate(agent_edge_information.values()):
#         if edge in sampled_edges:
#             frac_to_int_index_mapping[i] = 1

#     return frac_to_int_index_mapping
    

# edge_information = {
#     'e1': ('V001_1', 'V001_2'),
#     'e2': ('V001_1', 'V002_2'),
#     'e3': ('V001_2', 'V001_3'),
#     'e4': ('V001_2', 'V002_3'),
#     'e5': ('V002_2', 'V002_3')
# }

# edge_information_extended = {
#     'e1': ('V001_1', 'V001_2'),
#     'e2': ('V001_1_dep', 'V002_2_arr'),
#     'e3': ('V001_2', 'V001_2_dep'),
#     'e4': ('V002_2_arr', 'V002_2'),
#     'e5': ('V002_2', 'V002_2_dep'),
#     'e6': ('V001_2', 'V001_3'),
#     'e7': ('V001_2_dep', 'V002_3_arr'),
#     'e8': ('V002_2_dep', 'V002_3_arr')
# }

# # constrain, flow matrix
# A = np.array([[1, 1, 0, 0, 0],
#               [1, 0, -1, -1,0],
#               [0, 1, 0, 0, -1],
#               [0, 0, 1, 1, 1]])
# # x = np.array([0.3, 0.7, 0.1, 0.2, 0.7])  # probabilities or weights of edges
# x2 = np.array([0.3, 0.7, 0.2, 0.7, 0.7, 0.1,0.2,0.7])  # probabilities or weights of edges


## Simple graph
# start_node = edge_information['e1'][0]
# graph = agent_probability_graph(edge_information, x)
# sampled_path = sample_path(graph, start_node)
# print("Sampled Path:", sampled_path)
# plot_sample_path(graph, sampled_path)

# # Extended graph
# extended_graph = agent_probability_graph_extended(edge_information_extended, x2)
# sampled_path_extended, sampled_edges = sample_path(extended_graph, edge_information_extended['e1'][0])
# print("Sampled Path:", sampled_path_extended)
# print("Sampled Edges:", sampled_edges)
# plot_sample_path_extended(extended_graph, sampled_path_extended)

