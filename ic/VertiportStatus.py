import networkx as nx

class VertiportStatus(nx.DiGraph):
    """
    Class for storing the status of vertiports.
    """
    
    def __init__(self, vertiports, edges, timing, data=None, **attr):
        """
        Constructor description goes here.

        Args:
            vertiports (dictionary): Dictionary of vertiports with ids as keys containing 
                latitude, longitude, landing_capacity, takeoff_capacity, and hold_capacity.
            edges (list): List of edges where an edge has an origin_vertiport_id and destination_vertport_id.
            timing (dictionary): Dictionary of timing information for the simulation including
                start_time, end_time, and time_step.
        """
        super().__init__(data, **attr)
        time_steps = list(range(timing["start_time"], timing["end_time"] + timing["time_step"], timing["time_step"]))

        # Create time extended graph of vertiports
        for step in time_steps:
            for vertiport in vertiports.items():
                time_extended_vertiport_id = vertiport[0] + "_" + str(step)
                self.add_node(time_extended_vertiport_id, **vertiport[1])
                self.nodes[time_extended_vertiport_id]["landing_usage"] = 0
                self.nodes[time_extended_vertiport_id]["takeoff_usage"] = 0
                self.nodes[time_extended_vertiport_id]["hold_usage"] = 0

        # Add edges to time extended graph
        for step in time_steps:
            for edge in edges:
                arrival_time = step + edge["travel_time"]
                if arrival_time > timing["end_time"]:
                    continue
                assert arrival_time in time_steps, f"Timing setup incorrect. Arrival time {arrival_time} not in time steps."
                time_extended_start = edge["origin_vertiport_id"] + "_" + str(step)
                time_extended_end = edge["destination_vertiport_id"] + "_" + str(arrival_time)
                self.add_edge(time_extended_start, time_extended_end)

    def method_name(self, param1, param2):
        """
        Method description goes here.

        Args:
            param1 (type): Description of param1.
            param2 (type): Description of param2.

        Returns:
            type: Description of the return value.
        """
        # Method implementation goes here
        pass