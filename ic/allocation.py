import networkx as nx


def build_auxiliary(vertiport_status, time, flights):
    """
    Build auxiliary graph for a given time and set of requests.

    Args:
        graph (VertiportStatus): Graph from which to build the auxiliary graph.
        time (int): Time step for which to build the auxiliary graph.
        flights (list): List of flights making requests at this time step.
    """
    assert time == 1, "Only time step 1 is currently supported."

    auxiliary_graph = nx.DiGraph()
    ## Construct nodes
    #  Create dep, arr, and standard nodes for each initial node (vertiport + time step)
    for node in vertiport_status.nodes:
        auxiliary_graph.add_node(node + "_dep")
        auxiliary_graph.add_node(node + "_arr")
        auxiliary_graph.add_node(node)

    #  Create a node for each unique departure time for each agent
    for flight in flights:
        unique_departure_times = []
        for request in flight["requests"]:
            if request["request_departure_time"] not in unique_departure_times:
                unique_departure_times.append(request["request_departure_time"])
        for time in unique_departure_times:
            auxiliary_graph.add_node(flight["aircraft_id"] + "_" + str(time))

    # Add source and sink nodes
    auxiliary_graph.add_node("source")
    auxiliary_graph.add_node("sink")

    ## Construct edges
    #  


def determine_allocation(vertiport_usage, time, flights):
    """
    Allocate flights for a given time and set of requests.

    Args:
        graph (VertiportStatus): Graph from which to allocate flights.
        time (int): Time step for which to allocate flights.
        flights (list): List of flights making requests at this time step.
    """
    auxiliary_graph = build_auxiliary(vertiport_usage, time, flights)
    allocated_flights = flights
    return allocated_flights
