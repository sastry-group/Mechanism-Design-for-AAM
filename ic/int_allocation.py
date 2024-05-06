import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx


## Create a graph sample to turn into an A matrix
# setp a simple guribo optrmization that takes a set of prices A, b pair and calculates an optimal allocatoin for 1 agent.
# integer program

ALPHA = 0.5

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

def sample_path(G, start_node):
    path = [start_node]
    current_node = start_node

    while True:
        edges = list(G.out_edges(current_node, data=True))
        if not edges:
            break
        
        # Normalize weights
        total_weight = sum(edge[2]['weight'] for edge in edges)
        weights = [edge[2]['weight'] / total_weight for edge in edges]
        next_nodes = [edge[1] for edge in edges]

        # Select next node based on normalized weights
        current_node = random.choices(next_nodes, weights=weights, k=1)[0]

        path.append(current_node)
        

    return path


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
    plt.show()
    return 



edge_information = {
    'e1': ('V001_1', 'V001_2'),
    'e2': ('V001_1', 'V002_2'),
    'e3': ('V001_2', 'V001_3'),
    'e4': ('V001_2', 'V002_3'),
    'e5': ('V002_2', 'V002_3')
}



# constrain, flow matrix
A = np.array([[1, 1, 0, 0, 0],
              [1, 0, -1, -1,0],
              [0, 1, 0, 0, -1],
              [0, 0, 1, 1, 1]])
x = np.array([0.3, 0.7, 0.1, 0.2, 0.7])  # probabilities or weights of edges

start_node = edge_information['e1'][0]
graph = agent_probability_graph(edge_information, x)
sampled_path = sample_path(graph, start_node)
print("Sampled Path:", sampled_path)
plot_sample_path(graph, sampled_path)

