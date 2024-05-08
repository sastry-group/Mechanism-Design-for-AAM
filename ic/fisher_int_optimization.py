import cvxpy as cp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


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
    contested_edges = contested_allocations(x_agents, capacity)
    if contested_edges:
        print(f"Contested allocations: {contested_edges}")
        # Define the variables
        contested_edges = x_agents.shape[1]
        x = cp.Variable(contested_edges, integer=True)
        # Define the constraints
        constraints = [A @ x == b]
        objective = cp.Maximize(cp.sum(cp.multiply(utility, x)))
        # Define the problem
        problem = cp.Problem(objective, constraints)
        # Solve the problem
        problem.solve()
        # Get the optimal solution
        optimal_x = x.value
        return optimal_x
        
    else:
        return x_agents

    


def contested_allocations(integer_allocations, capacity):
    """
    Function to check contested allocations
    Args:
    integer_allocations (list, nxm): integer allocation matrix
    capacity (list, nx1): capacity vector
    """
    contested_allocations = []
    for j in range(len(integer_allocations[0])):
        for i in range(len(integer_allocations)):
            if integer_allocations[i][j] > capacity[j]:
                contested_allocations.append((i, j))
    return contested_allocations


def create_example():
    
    G = nx.DiGraph()
    n_vertiports = 2
    time_steps = 3
    edge_counter = 1  # Start a counter for the edge labels

    # Creating the nodes with consistent formatting
    for i in range(1, n_vertiports + 1):
        for j in range(1, time_steps + 1):
            node_label = f"V00{i}_{j}"
            G.add_node(node_label, subset=j)
    
    n_subsets = len(set(nx.get_node_attributes(G, 'subset').values()))
    edge_counter =1
    for subset in range(1, n_subsets):
        subset_nodes = [node for node, data in G.nodes(data=True) if data['subset'] == subset]

        next_nodes = [node for node, data in G.nodes(data=True) if data['subset'] == subset + 1]
        print(subset_nodes, next_nodes)
        for node in subset_nodes:
            for next_node in next_nodes:
                target_node = next_node
                edge_label = f"e{edge_counter}"
                G.add_edge(node, next_node, label=edge_label, weight=1.0 / len(next_nodes))
                edge_counter += 1



            # source_nodes = [f"V00{k}_{t}" for i in range(1, n_vertiports + 1)]
            # target_nodes = [f"V00{k}_{t+1}" for k in range(1, n_vertiports + 1)]
            # for source in source_nodes:
            #     total_edges_from_source = len(target_nodes)
            #     for target in target_nodes:
            #         edge_label = f"e{edge_counter}"
            #         print(edge_counter)
            #         weight = 1.0 / total_edges_from_source
            #         G.add_edge(source, target, label=edge_label, weight=weight)
            #         edge_counter += 1

    # Position nodes for visualization
    pos = nx.multipartite_layout(G, subset_key="subset")
    plt.figure(figsize=(12, 8))
    
    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue')

    # Draw edge labels with offsets
    edge_labels = {(u, v): f"{d['label']} ({d['weight']:.2f})" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=.4)

    plt.title("Time Extended Decision Tree Graph with Flow Constraints")
    plt.show()


    return G



# create_example()
x_agents = 
int_optimization(x_agents, capacity, budget, prices, A, b)