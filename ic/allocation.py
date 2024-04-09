import networkx as nx
from VertiportStatus import draw_graph
import numpy as np
import gurobipy as gp
import time


rho = 1

def build_auxiliary(vertiport_status, flights, timing_info):
    """
    Build auxiliary graph for a given time and set of requests.

    Args:
        graph (VertiportStatus): Graph from which to build the auxiliary graph.
        flights (list): List of flights making requests at this time step.
        time (int): Time step for which to build the auxiliary graph.
        max_time (int): The max time step value of the vertiport graph.
        time_steps (list): List of time steps for the graph.
    """
    print("Building auxiliary graph...")
    start_time_graph_build = time.time()
    max_time, time_step = timing_info["end_time"], timing_info["time_step"]
    auxiliary_graph = nx.MultiDiGraph()
    ## Construct nodes
    #  V1. Create dep, arr, and standard nodes for each initial node (vertiport + time step)
    for node in vertiport_status.nodes:
        auxiliary_graph.add_node(node + "_dep")
        auxiliary_graph.add_node(node + "_arr")
        auxiliary_graph.add_node(node)

    #  V2. Create a node for each unique departure time for each agent
    unique_departure_times = {}
    for flight_id, flight in flights.items():
        flight_unique_departure_times = []
        for request_id, request in flight["requests"].items():
            if request["request_departure_time"] not in flight_unique_departure_times:
                # assert request["request_departure_time"] != 0, "Request departure times cannot be 0."
                flight_unique_departure_times.append(request["request_departure_time"])
        assert 0 in flight_unique_departure_times, "Request departure times must include 0."
        for depart_time in flight_unique_departure_times:
            auxiliary_graph.add_node(flight_id + "_" + str(depart_time))
        unique_departure_times[flight_id] = flight_unique_departure_times

    #  V3. Add source and sink nodes
    auxiliary_graph.add_node("source")
    auxiliary_graph.add_node("sink")


    ## Construct edges
    for node in vertiport_status.nodes:
        # E1. Connect arrival to main nodes
        attributes = {"upper_capacity": vertiport_status.nodes[node]["landing_capacity"],
                      "lower_capacity": 0,
                      "weight": 0,
                      "edge_group": "E1"}
        auxiliary_graph.add_edge(node + "_arr", node, **attributes)

        # E2. Connect main nodes to departure
        attributes = {"upper_capacity": vertiport_status.nodes[node]["takeoff_capacity"],
                      "lower_capacity": 0,
                      "weight": 0,
                      "edge_group": "E2"}
        auxiliary_graph.add_edge(node, node + "_dep", **attributes)

        # E3. Connect time steps together for each vertiport
        if vertiport_status.nodes[node]["time"] == max_time:
            continue
        for val in range(vertiport_status.nodes[node]["hold_capacity"]):
            lambda_val = 0
            factor = 1
            weight = lambda_val * factor
            attributes = {"upper_capacity": 1,
                        "lower_capacity": 0,
                        "weight": weight,
                        "edge_group": "E3_" + str(val + 1)} # Assume lambda is 0 for now
            # Todo: Add case when lambda is not 0
            vertiport_id = vertiport_status.nodes[node]["vertiport_id"]
            next_time = vertiport_status.nodes[node]["time"] + time_step
            auxiliary_graph.add_edge(node, vertiport_id + "_" + str(next_time), **attributes)

    for flight_id, flight in flights.items():
        origin = flight["origin_vertiport_id"]
        # E4. Connect departure node (V1) to flight departure time node (V2)
        for depart_time in unique_departure_times[flight_id]:
            if depart_time == 0:
                continue
            attributes = {"upper_capacity": f"d_{flight_id}_{depart_time}",
                          "lower_capacity": f"d_{flight_id}_{depart_time}",
                          "weight": 0,
                          "edge_group": "E4"}
            auxiliary_graph.add_edge(origin + "_" + str(depart_time) + "_dep", flight_id + "_" + str(depart_time), **attributes)
            
        for request_id, request in flight["requests"].items():
            # E7. Connect source node to flight 0 node
            if request["request_departure_time"] == 0:
                attributes = {"upper_capacity": f"d_{flight_id}_0",
                            "lower_capacity": f"d_{flight_id}_0",
                            "weight": rho * request["bid"], # rho * b, rho=1 for now
                            "edge_group": "E7"}
                auxiliary_graph.add_edge("source", flight_id + "_0", **attributes)
            else:
                # E5. Connect flight departure time node (V2) to arrival node (V1)
                destination = request["destination_vertiport_id"]
                depart_time = request["request_departure_time"]
                arrival_time = request["request_arrival_time"]
                attributes = {"upper_capacity": 1,
                            "lower_capacity": 0,
                            "weight": rho * request["bid"],  # rho * b, rho=1 for now
                            "edge_group": "E5",
                            "flight_id": flight_id,
                            "request_id": request_id}
                auxiliary_graph.add_edge(flight_id + "_" + str(depart_time), \
                                        destination + "_" + str(arrival_time) + "_arr", **attributes)
            
        # E9. Connect flight 0 node to origin
        attributes = {"upper_capacity": f"d_{flight_id}_0",
                    "lower_capacity": f"d_{flight_id}_0",
                    "weight": 0,
                    "edge_group": "E9"}
        auxiliary_graph.add_edge(flight_id + "_0", origin + "_" + str(time_step), **attributes)

    for vertiport in vertiport_status.vertiports.items():
        # E6. Connect source to each node at the first time step
        attributes = {"upper_capacity": f"E6_{vertiport[0]}_cap",
                      "lower_capacity": f"E6_{vertiport[0]}_cap",
                      "weight": 0,
                      "edge_group": "E6"}
        auxiliary_graph.add_edge("source", vertiport[0] + "_" + str(time_step), **attributes)

        # E8. Connect each node at the last time step to sink per park allowance
        for val in range(vertiport[1]["hold_capacity"]):
            attributes = {"upper_capacity": 1,
                        "lower_capacity": 0,
                        "weight": 0, #lambda = 0
                        "edge_group": "E8_" + str(val + 1)}
            auxiliary_graph.add_edge(vertiport[0] + "_" + str(max_time), "sink", **attributes)  
    
    print(f"Time to build graph: {time.time() - start_time_graph_build}")
    # Print edges for debugging
    # for edge in auxiliary_graph.edges(data=True):
    #     print(edge)
    # draw_graph(auxiliary_graph)
    return auxiliary_graph, unique_departure_times


def determine_allocation(vertiport_usage, flights, auxiliary_graph, unique_departure_times):
    vertiports = vertiport_usage.vertiports
    aircraft_ids = list(unique_departure_times.keys())

    # Create the incidence matrix (I) with the pulled node order so that we know what each column and row corresponds to
    node_order = list(auxiliary_graph.nodes())
    edges = auxiliary_graph.edges(data=True)
    edge_order = list(edges)
    I = nx.incidence_matrix(auxiliary_graph, oriented=True, nodelist=node_order, edgelist=edge_order).toarray()

    # Remove source and sink rows to create I_star
    source_index = node_order.index("source")
    sink_index = node_order.index("sink")
    if source_index > sink_index:
        node_order.pop(source_index)
        node_order.pop(sink_index)
    else:
        node_order.pop(sink_index)
        node_order.pop(source_index)
    I_star = np.delete(I, [source_index, sink_index], axis=0)
    # print(f"Node_order:")
    # for node in node_order:
    #     print(node)
    # print(f"\nI_star:")
    # for i in range(len(I_star)):
    #     for j in range(len(I_star[i])):
    #         if I_star[i][j] != 0:
    #             print(f"({i}, {j}, {I_star[i][j]})")
    # print(f"\nEdge_order:")
    # for edge in edge_order:
    #     print(edge)

    # Start building allocation optimization problem
    start_time_build_model = time.time()
    m = gp.Model("allocation")

    # Define the decision variables
    delta = [
        [m.addVar(vtype=gp.GRB.BINARY, name=f"delta_{flight}_time{t}") for t in unique_departure_times[flight]]
        for flight in aircraft_ids
    ]
    A = m.addVars(len(edge_order), lb=0, name="A")

    # Pull weight values (W)
    W = np.zeros(len(edge_order))
    for i, edge in enumerate(edge_order):
        assert len(edge) == 3, "Missing attributes in edge."
        _, _, attr = edge
        W[i] = attr["weight"]

    # Define the objective function
    m.setObjective(gp.quicksum(W[i] * A[i] for i in range(len(edge_order))), sense=gp.GRB.MAXIMIZE)

    # Define the constraints
    for i in range(len(delta)):
        m.addConstr(gp.quicksum(delta[i][j] for j in range(len(delta[i]))), gp.GRB.EQUAL, 1, f"unique_departure_time_flight{i}")
    for row in I_star:
        m.addConstr(gp.quicksum(row[i] * A[i] for i in range(len(edge_order))), gp.GRB.EQUAL, 0, f"flow_conservation")
    for k, edge in enumerate(edge_order):
        assert len(edge) == 3, "Missing attributes in edge."
        _, _, attr = edge
        c_upper = attr["upper_capacity"]
        if isinstance(c_upper, str):
            upper_parts = c_upper.split("_")
            if upper_parts[0] == "d":
                flight = upper_parts[1]
                idx = aircraft_ids.index(flight)
                upper_time = int(upper_parts[2])
                time_idx = unique_departure_times[flight].index(upper_time)
                m.addConstr(A[k] <= delta[idx][time_idx], f"upper_capacity_edge{k}")
            elif upper_parts[0] == "E6":
                vertiport = upper_parts[1]
                S_r = 0
                delta_indices = []
                for flight_id, flight in flights.items():
                    if flight["origin_vertiport_id"] == vertiport:
                        S_r += 1
                        idx = aircraft_ids.index(flight_id)
                        delta_indices.append(idx)
                delta_sum_r = gp.quicksum(delta[idx][0] for idx in delta_indices)
                m.addConstr(S_r - delta_sum_r <= A[k], f"upper_capacity_edge{k}")
                m.addConstr(A[k] <= S_r - delta_sum_r, f"lower_capacity_edge{k}")
        else:
            m.addConstr(A[k] <= c_upper, f"upper_capacity_edge{k}")

        c_lower = attr["lower_capacity"]
        if isinstance(c_lower, str):
            lower_parts = c_lower.split("_")
            if lower_parts[0] == "d":
                flight = lower_parts[1]
                idx = aircraft_ids.index(flight)
                lower_time = int(lower_parts[2])
                time_idx = unique_departure_times[flight].index(lower_time)
                m.addConstr(delta[idx][time_idx] <= A[k], f"lower_capacity_edge{k}")
        else:
            m.addConstr(c_lower <= A[k], f"lower_capacity_edge{k}")

    # Set initial guess for delta variables
    for i in range(len(delta)):
        delta[i][-1].start = 1
    
    print(f"Time to build model: {time.time() - start_time_build_model}")

    # Optimize the model
    print("Optimizing...")
    start_time_optimization = time.time()
    m.optimize()
    print(f"Time to optimize: {time.time() - start_time_optimization}")
    
    # Retrieve the allocation values
    if m.status == gp.GRB.OPTIMAL:
        nonzero_indices = np.where([A[i].x != 0 for i in range(len(A))])[0].tolist()
        allocated_edges = [edge_order[i] for i in nonzero_indices]
        # If the edge is in group E5, pull the flight and request information.
        allocation = []
        for _, _, attr in allocated_edges:
            if attr["edge_group"] == "E5":
                allocation.append((attr["flight_id"], attr["request_id"]))
    else:
        print("Optimization was not successful.")
        allocation = None

    return allocation


def allocation_and_payment(vertiport_usage, flights, timing_info):
    """
    Allocate flights for a given time and set of requests.

    Args:
        graph (VertiportStatus): Graph from which to allocate flights.
        time (int): Time step for which to allocate flights.
        flights (list): List of flights making requests at this time step.
    """
    # Create auxiliary graph and determine allocation
    # Todo: Also determine payment here
    auxiliary_graph, unique_departure_times = build_auxiliary(vertiport_usage, flights, timing_info)
    allocation = determine_allocation(vertiport_usage, flights, auxiliary_graph, unique_departure_times)

    # Print outputs
    print(f"Allocation\n{allocation}")
    payment = None
    print(f"\nPayment\n{payment}")

    return allocation, payment
