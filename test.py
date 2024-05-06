import networkx as nx

def build_flow_network(A, x, num_cols, time_steps):
    G = nx.DiGraph()  # Directed graph for flow network

    # Iterate through each time step, excluding the first and last row initially
    for t in range(1, time_steps - 1):
        for i in range(num_cols):
            if A[t-1, i] == 1 and A[t, i] == -1:
                # Flow from time t to time t+1 for vertiport i
                source_node = f"V00{i+1}_{t}"
                destination_node = f"V00{i+1}_{t+1}"
                G.add_edge(source_node, destination_node, capacity=x[i])
                print(f"Adding edge from {source_node} to {destination_node} with capacity {x[i]}")

    # Handle the first and last row specifically for global source and sink
    for i in range(num_cols):
        if A[0, i] == 1:
            # Global source to first timestep
            global_source = "Source"
            first_step_node = f"V00{i+1}_1"
            G.add_edge(global_source, first_step_node, capacity=x[i])
            print(f"Adding edge from {global_source} to {first_step_node} with capacity {x[i]}")

        if A[-1, i] == -1:
            # Last timestep to global sink
            last_step_node = f"V00{i+1}_{time_steps}"
            global_sink = "Sink"
            G.add_edge(last_step_node, global_sink, capacity=x[i])
            print(f"Adding edge from {last_step_node} to {global_sink} with capacity {x[i]}")

    return G

# Example usage:
import numpy as np
A = np.array([
    [1, 0, 0],
    [0, 0, 0],
    [-1, 1, 0],
    [0, -1, 1],
    [0, 0, -1]
])
x = np.array([5, 3, 2, 4, 1])
num_cols = 3
time_steps = 5

G = build_flow_network(A, x, num_cols, time_steps)

# Additional: Visualize the graph if desired
import matplotlib.pyplot as plt
pos = nx.spring_layout(G)  # positions for all nodes
nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='k', width=1, linewidths=1, node_size=500, alpha=0.9)
labels = nx.get_edge_attributes(G, 'capacity')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()