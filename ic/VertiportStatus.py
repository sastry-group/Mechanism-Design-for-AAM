import networkx as nx
import matplotlib.pyplot as plt


class VertiportStatus(nx.DiGraph):
    """
    Class for storing the status of vertiports.
    """
    
    def __init__(self, vertiports, edges, timing, data=None, **attr):
        """
        Create time-extended vertiport graph.

        Args:
            vertiports (dictionary): Dictionary of vertiports with ids as keys containing 
                latitude, longitude, landing_capacity, takeoff_capacity, and hold_capacity.
            edges (list): List of edges where an edge has an origin_vertiport_id and destination_vertport_id.
            timing (dictionary): Dictionary of timing information for the simulation including
                start_time, end_time, and time_step.
        """
        super().__init__(data, **attr)
        self.time_steps = list(range(timing["start_time"], timing["end_time"] + timing["time_step"], timing["time_step"]))
        self.vertiports = vertiports

        # Create time extended graph of vertiports
        for step in self.time_steps:
            for vertiport in vertiports.items():
                time_extended_vertiport_id = vertiport[0] + "_" + str(step)
                self.add_node(time_extended_vertiport_id, **vertiport[1])
                self.nodes[time_extended_vertiport_id]["landing_usage"] = 0
                self.nodes[time_extended_vertiport_id]["takeoff_usage"] = 0
                self.nodes[time_extended_vertiport_id]["hold_usage"] = 0
                self.nodes[time_extended_vertiport_id]["time"] = step
                self.nodes[time_extended_vertiport_id]["vertiport_id"] = vertiport[0]

        # Add edges to time extended graph
        for step in self.time_steps:
            for edge in edges:
                arrival_time = step + edge["travel_time"]
                if arrival_time > timing["end_time"]:
                    continue
                assert arrival_time in self.time_steps, f"Timing setup incorrect. Arrival time {arrival_time} not in time steps."
                time_extended_start = edge["origin_vertiport_id"] + "_" + str(step)
                time_extended_end = edge["destination_vertiport_id"] + "_" + str(arrival_time)
                self.add_edge(time_extended_start, time_extended_end)


    def add_aircraft(self, flights):
        """
        Add starting vertiport usage information.

        Args:
            flights (list): List of flights with at least origin_vertiport_id information.
        """
        for flight in flights:
            start_vertiport = flight["origin_vertiport_id"]
            for time in self.time_steps:
                time_extended_start = start_vertiport + "_" + str(time)
                self.nodes[time_extended_start]["hold_usage"] += 1
                assert self.nodes[time_extended_start]["hold_usage"] <= self.nodes[time_extended_start]["hold_capacity"], \
                    f"Vertiport {start_vertiport} at time {time} over capacity."  


    def move_aircraft(self, origin_vertiport, request):
        """
        Move aircraft from one vertiport to another.

        Args:
            origin_vertiport (str): Vertiport id where the aircraft is coming from.
            request (dict): Dictionary containing information about the aircraft movement request
                including destination_vertiport_id, request_departure_time, and request_arrival_time.
        """
        destination_vertiport = request["destination_vertiport_id"]
        departure_time = request["request_departure_time"]
        arrival_time = request["request_arrival_time"]

        # Update the hold usage of the origin and destination vertiports
        for time in self.time_steps:
            # Move the aircraft from the origin vertiport
            if time < departure_time:
                continue
            time_extended_origin = origin_vertiport + "_" + str(time)
            self.nodes[time_extended_origin]["hold_usage"] -= 1
            assert self.nodes[time_extended_origin]["hold_usage"] >= 0, \
                f"Vertiport {origin_vertiport} at time {time} has negative hold usage."
            
            # Add the aircraft to the destination vertiport
            if time >= arrival_time:
                time_extended_destination = destination_vertiport + "_" + str(time)
                self.nodes[time_extended_destination]["hold_usage"] += 1
                assert self.nodes[time_extended_destination]["hold_usage"] >= 0, \
                    f"Vertiport {destination_vertiport} at time {time} has negative hold usage."
        
        # Add the aircrafts takeoff and landing usage
        time_extended_origin = origin_vertiport + "_" + str(departure_time)
        time_extended_destination = destination_vertiport + "_" + str(arrival_time)
        self.nodes[time_extended_origin]["takeoff_usage"] += 1
        self.nodes[time_extended_destination]["landing_usage"] += 1


def draw_graph(graph):
    # Todo: This could be significantly improved
    # Draw the graph
    pos = nx.shell_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='k')

    # Show the plot
    plt.show()