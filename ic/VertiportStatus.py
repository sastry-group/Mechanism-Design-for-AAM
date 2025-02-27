import networkx as nx
import matplotlib.pyplot as plt
import math

class VertiportStatus(nx.DiGraph):
    """
    Class for storing the status of vertiports.
    """
    
    def __init__(self, vertiports, sectors, timing, data=None, **attr):
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
        for start_step, end_step in zip(self.time_steps[:-1], self.time_steps[1:]):
            for vertiport in vertiports.items():
                time_extended_vertiport_id_start = vertiport[0] + "_" + str(start_step)
                time_extended_vertiport_id_end = vertiport[0] + "_" + str(end_step)
                self.add_node(time_extended_vertiport_id_start, **vertiport[1])
                attributes = {"hold_usage": 0, "time": start_step,
                              "hold_capacity": vertiport[1]["hold_capacity"], "vertiport_id": vertiport[0]}
                self.add_edge(time_extended_vertiport_id_start, time_extended_vertiport_id_end, **attributes)

        
        for step in self.time_steps:
            for vertiport in vertiports.items():
                time_extended_vertiport_id = vertiport[0] + "_" + str(step)
                time_extended_vertiport_depart = vertiport[0] + "_" + str(step) + "_dep"
                time_extended_vertiport_arrive = vertiport[0] + "_" + str(step) + "_arr"
                dep_attributes = {"takeoff_usage": 0, "takeoff_capacity": vertiport[1]["takeoff_capacity"], "time": step, "vertiport_id": vertiport[0]}
                arr_attributes = {"landing_usage": 0, "landing_capacity": vertiport[1]["landing_capacity"], "time": step, "vertiport_id": vertiport[0]}
                self.add_edge(time_extended_vertiport_id, time_extended_vertiport_depart, **dep_attributes)
                self.add_edge(time_extended_vertiport_arrive, time_extended_vertiport_id, **arr_attributes)

        # # Add edges to time extended graph
        # for step in self.time_steps:
        #     for edge in edges:
        #         arrival_time = step + edge["travel_time"]
        #         if arrival_time > timing["end_time"]:
        #             continue
        #         assert arrival_time in self.time_steps, f"Timing setup incorrect. Arrival time {arrival_time} not in time steps."
        #         time_extended_start = edge["origin_vertiport_id"] + "_" + str(step)
        #         time_extended_end = edge["destination_vertiport_id"] + "_" + str(arrival_time)
        #         self.add_edge(time_extended_start, time_extended_end)


        # print(f"Added nodes: {self.nodes} and edges {self.edges}")
        # Adding sector capacities to time extended graph
        for i in range(len(self.time_steps) - 1):
            for id, sector in sectors.items():
                start_time, end_time = self.time_steps[i], self.time_steps[i+1]
                time_extended_start = id + "_" + str(start_time)
                time_extended_end = id + "_" + str(end_time)
                attributes = {"hold_capacity": sector["hold_capacity"], "hold_usage": 0}
                self.add_edge(time_extended_start, time_extended_end, **attributes)

        # print(f"Added nodes: {self.nodes} and edges {self.edges}")


    def add_aircraft(self, flights):
        """
        Add starting vertiport usage information.

        Args:
            flights (list): List of flights with at least origin_vertiport_id information.
        """
        for flight_id, flight in flights.items():
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
        assert self.nodes[time_extended_origin]["takeoff_usage"] <= self.nodes[time_extended_origin]["takeoff_capacity"], \
            f"Vertiport {origin_vertiport} at time {departure_time} over takeoff capacity."
        self.nodes[time_extended_destination]["landing_usage"] += 1
        assert self.nodes[time_extended_destination]["landing_usage"] <= self.nodes[time_extended_destination]["landing_capacity"], \
            f"Vertiport {destination_vertiport} at time {arrival_time} over landing capacity."

    def allocate_aircraft(self, origin_vertiport, flight, request, auction_period):
        """
        Allocate the aircraft to its requested vertiports and paths. For use when the aircraft are not pre-allocated to positions.
        
        Args:
            origin_vertiport (str): Vertiport id where the aircraft is coming from.
            request (dict): Dictionary containing information about the aircraft movement request
                including destination_vertiport_id, request_departure_time, and request_arrival_time.
        """
        destination_vertiport = request["destination_vertiport_id"]
        appearance_time = flight["appearance_time"]
        departure_time = request["request_departure_time"]
        arrival_time = request["request_arrival_time"]

        # Update the hold usage of the origin and destination vertiports
        for start_time, next_time in zip(self.time_steps[:-1], self.time_steps[1:]):
            end_of_auction = math.ceil(arrival_time / auction_period) * auction_period
            # Add the aircraft to the origin vertiport
            if start_time < appearance_time or start_time >= end_of_auction:
                continue
            if next_time < departure_time:
                time_extended_origin_start = origin_vertiport + "_" + str(start_time)
                time_extended_origin_end = origin_vertiport + "_" + str(next_time)
                self[time_extended_origin_start][time_extended_origin_end]["hold_usage"] += 1
                assert self[time_extended_origin_start][time_extended_origin_end]["hold_usage"] >= 0 and \
                    self[time_extended_origin_start][time_extended_origin_end]["hold_usage"] <= self[time_extended_origin_start][time_extended_origin_end]["hold_capacity"], \
                    f"Vertiport {origin_vertiport} at time {start_time} to {next_time} has incorrect hold usage. One reason if this is the starting node for an agent is that the holding capacity is lower than the number of agents set to start at that location."
                # print(f"Adding {time_extended_origin_start} to {time_extended_origin_end} with current hold_usage {self[time_extended_origin_start][time_extended_origin_end]['hold_usage']} and capacity {self[time_extended_origin_start][time_extended_origin_end]['hold_capacity']}")
                
            
            # Add the aircraft to the destination vertiport
            elif start_time >= arrival_time:
                time_extended_destination_start = destination_vertiport + "_" + str(start_time)
                time_extended_destination_end = destination_vertiport + "_" + str(next_time)
                self[time_extended_destination_start][time_extended_destination_end]["hold_usage"] += 1
                assert self[time_extended_destination_start][time_extended_destination_end]["hold_usage"] >= 0 and \
                    self[time_extended_destination_start][time_extended_destination_end]["hold_usage"] <= self[time_extended_destination_start][time_extended_destination_end]["hold_capacity"], \
                    f"Vertiport {destination_vertiport} at time {start_time} to {next_time} has incorrect hold usage."
                # print(f"Adding {time_extended_destination_start} to {time_extended_destination_end} with current hold_usage {self[time_extended_destination_start][time_extended_destination_end]['hold_usage']} and capacity {self[time_extended_destination_start][time_extended_destination_end]['hold_capacity']}")
                
        # Add aircraft sector traversal
        sector_path = request["sector_path"]
        sector_times = request["sector_times"]
        for i in range(len(request["sector_path"])):
            current_sector, current_time = sector_path[i], sector_times[i]
            next_time = sector_times[i + 1]
            for ts in range(current_time, next_time):
                start_node = f"{current_sector}_{ts}"
                end_node = f"{current_sector}_{ts+1}"
                self[start_node][end_node]["hold_usage"] += 1
                assert self[start_node][end_node]["hold_usage"] >= 0 and self[start_node][end_node]["hold_usage"] <= self[start_node][end_node]["hold_capacity"], \
                    f"Sector {current_sector} at time {ts} to time {ts+1} has incorrect hold usage."
                # print(f"Adding {start_node} to {end_node} with current hold_usage {self[start_node][end_node]['hold_usage']} and capacity {self[start_node][end_node]['hold_capacity']}")
                # if start_node == "S002_35" and end_node == "S002_36":
                #     print(f"Adding ({start_node}, {end_node})") 
                
        
        # Add the aircrafts takeoff and landing usage
        time_extended_origin = origin_vertiport + "_" + str(departure_time)
        time_extended_destination = destination_vertiport + "_" + str(arrival_time)
        self[time_extended_origin][time_extended_origin + "_dep"]["takeoff_usage"] += 1
        assert self[time_extended_origin][time_extended_origin + "_dep"]["takeoff_usage"] <= self[time_extended_origin][time_extended_origin + "_dep"]["takeoff_capacity"], \
            f"Vertiport {origin_vertiport} at time {departure_time} over takeoff capacity."
        # print(f"Adding {time_extended_origin} to {time_extended_origin + '_dep'} with current takeoff_usage {self[time_extended_origin][time_extended_origin + '_dep']['takeoff_usage']} and capacity {self[time_extended_origin][time_extended_origin + '_dep']['takeoff_capacity']}")
        self[time_extended_destination + "_arr"][time_extended_destination]["landing_usage"] += 1
        assert self[time_extended_destination + "_arr"][time_extended_destination]["landing_usage"] <= self[time_extended_destination + "_arr"][time_extended_destination]["landing_capacity"], \
            f"Vertiport {destination_vertiport} at time {arrival_time} over landing capacity."
        # print(f"Adding {time_extended_destination + '_arr'} to {time_extended_destination} with current landing_usage {self[time_extended_destination + '_arr'][time_extended_destination]['landing_usage']} and capacity {self[time_extended_destination + '_arr'][time_extended_destination]['landing_capacity']}")

def draw_graph(graph):
    """
        Draw networkx graph.

        Args:
            graph (nx.Graph): Graph to draw.
        """
    # Todo: This could be significantly improved
    # Draw the graph
    pos = nx.shell_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='k')

    # Show the plot
    plt.show()