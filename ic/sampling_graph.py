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

def agent_probability_graph_extended(edge_information, x, output_folder=False):
    G = nx.DiGraph()

    # Add edges with weights and labels
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
        plt.savefig(f'{output_folder}/extended_graph.png')
    # plt.show()

    return G, agent_allocations



def sample_path(G, start_node, agent_allocations):
    """Sample a path in the graph G starting from the given node."""

    start_time_sample = time.time()
    print("Sampling edges ...")

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
    agent_allocations = list(agent_allocations.values())

    print(f"Time to sample: {time.time() - start_time_sample:.2f}")

    return path, edges, agent_allocations


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

def plot_sample_path_extended(G, sampled_path, output_folder=False):
    # Create a copy of the original graph to avoid modifying it
    H = G.copy()

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
        plt.savefig(f'{output_folder}/sampled_graph.png')

    return H


def build_edge_information(goods_list):
    """
    Build edge information from goods list.
    The last good is the default good. 
    """
    edge_information = {}
    for i, goods in enumerate(goods_list[:-1], start=1):
        edge_information[f"e{i}"] = (goods[0], goods[1])
    return edge_information

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

