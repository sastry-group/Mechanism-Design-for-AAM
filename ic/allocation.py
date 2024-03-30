import networkx as nx
from VertiportStatus import draw_graph

rho = 1

def build_auxiliary(vertiport_status, flights, time, max_time, time_steps=None):
    """
    Build auxiliary graph for a given time and set of requests.

    Args:
        graph (VertiportStatus): Graph from which to build the auxiliary graph.
        flights (list): List of flights making requests at this time step.
        time (int): Time step for which to build the auxiliary graph.
        max_time (int): The max time step value of the vertiport graph.
        time_steps (list): List of time steps for the graph.
    """
    assert time == 1, "Only time step 1 is currently supported."
    # assert time_steps[0] == time, "Time steps must start at time."
    # NOTE: WE ASSUME TIME STEPS ARE IN UNITS OF ONE

    auxiliary_graph = nx.MultiDiGraph()
    ## Construct nodes
    #  V1. Create dep, arr, and standard nodes for each initial node (vertiport + time step)
    for node in vertiport_status.nodes:
        auxiliary_graph.add_node(node + "_dep")
        auxiliary_graph.add_node(node + "_arr")
        auxiliary_graph.add_node(node)

    #  V2. Create a node for each unique departure time for each agent
    unique_departure_times = {}
    for flight in flights:
        flight_unique_departure_times = []
        for request in flight["requests"]:
            if request["request_departure_time"] not in flight_unique_departure_times:
                # assert request["request_departure_time"] != 0, "Request departure times cannot be 0."
                flight_unique_departure_times.append(request["request_departure_time"])
        assert 0 in flight_unique_departure_times, "Request departure times must include 0."
        for time in flight_unique_departure_times:
            auxiliary_graph.add_node(flight["aircraft_id"] + "_" + str(time))
        unique_departure_times[flight["aircraft_id"]] = flight_unique_departure_times

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
            next_time = vertiport_status.nodes[node]["time"] + 1
            auxiliary_graph.add_edge(node, vertiport_id + "_" + str(next_time), **attributes)

    for flight in flights:
        origin = flight["origin_vertiport_id"]
        # E4. Connect departure node (V1) to flight departure time node (V2)
        for depart_time in unique_departure_times[flight["aircraft_id"]]:
            if depart_time == 0:
                continue
            attributes = {"upper_capacity": f"d_{flight['aircraft_id']}_{depart_time}",
                          "lower_capacity": f"d_{flight['aircraft_id']}_{depart_time}",
                          "weight": 0,
                          "edge_group": "E4"}
            auxiliary_graph.add_edge(origin + "_" + str(depart_time) + "_dep", flight["aircraft_id"] + "_" + str(depart_time), **attributes)
            
        for request in flight["requests"]:
            # E7. Connect source node to flight 0 node
            if request["request_departure_time"] == 0:
                attributes = {"upper_capacity": f"d_{flight['aircraft_id']}_0",
                            "lower_capacity": f"d_{flight['aircraft_id']}_0",
                            "weight": rho * request["bid"], # rho * b, rho=1 for now
                            "edge_group": "E7"}
                auxiliary_graph.add_edge("source", flight["aircraft_id"] + "_0", **attributes)
            else:
                # E5. Connect flight departure time node (V2) to arrival node (V1)
                destination = request["destination_vertiport_id"]
                depart_time = request["request_departure_time"]
                arrival_time = request["request_arrival_time"]
                attributes = {"upper_capacity": 1,
                            "lower_capacity": 0,
                            "weight": rho * request["bid"],  # rho * b, rho=1 for now
                            "edge_group": "E5"}
                auxiliary_graph.add_edge(flight["aircraft_id"] + "_" + str(depart_time), \
                                        destination + "_" + str(arrival_time) + "_arr", **attributes)
            
        # E9. Connect flight 0 node to origin
        attributes = {"upper_capacity": f"d_{flight['aircraft_id']}_0",
                    "lower_capacity": f"d_{flight['aircraft_id']}_0",
                    "weight": 0,
                    "edge_group": "E9"}
        auxiliary_graph.add_edge(flight["aircraft_id"] + "_0", origin + "_1", **attributes)

    for vertiport in vertiport_status.vertiports.items():
        # E6. Connect source to each node at the first time step
        attributes = {"upper_capacity": f"E6_{vertiport[0]}_cap",
                      "lower_capacity": f"E6_{vertiport[0]}_cap",
                      "weight": 0,
                      "edge_group": "E6"}
        auxiliary_graph.add_edge("source", vertiport[0] + "_1", **attributes)

        # E8. Connect each node at the last time step to sink per park allowance
        for val in range(vertiport[1]["hold_capacity"]):
            attributes = {"upper_capacity": 1,
                        "lower_capacity": 0,
                        "weight": 0, #lambda = 0
                        "edge_group": "E8_" + str(val + 1)}
            auxiliary_graph.add_edge(vertiport[0] + "_" + str(max_time), "sink", **attributes)  
            
    # Print edges using pretty print
    # for edge in auxiliary_graph.edges(data=True):
    #     print(edge)
    # draw_graph(auxiliary_graph)
    return auxiliary_graph


def determine_allocation(vertiport_usage, flights, time, max_time):
    """
    Allocate flights for a given time and set of requests.

    Args:
        graph (VertiportStatus): Graph from which to allocate flights.
        time (int): Time step for which to allocate flights.
        flights (list): List of flights making requests at this time step.
    """
    auxiliary_graph = build_auxiliary(vertiport_usage, flights, time, max_time)
    allocated_flights = flights
    return allocated_flights
