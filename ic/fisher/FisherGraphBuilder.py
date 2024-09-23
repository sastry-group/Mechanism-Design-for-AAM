import networkx as nx
import time

class FisherGraphBuilder:
    def __init__(self, vertiport_status, timing_info):
        self.vertiport_status = vertiport_status
        self.timing_info = timing_info
        self.graph = nx.DiGraph()

    def build_graph(self, flights):
        for flight_id, flight_data in flights.items():
            origin_vertiport = flight_data["origin_vertiport_id"]
            destination_vertiport = flight_data["requests"]["001"]["destination_vertiport_id"]
            appearance_time = flight_data["appearance_time"]
            departure_time = flight_data["requests"]["001"]["request_departure_time"]
            arrival_time = flight_data["requests"]["001"]["request_arrival_time"]
            auction_frequency = self.timing_info["auction_frequency"]

            # Create parking nodes from appearance to end of auction
            end_auction_time = self._get_end_auction_time(arrival_time, auction_frequency)

            # Debugging: Print the times being used
            print(f"Processing flight {flight_id}:")
            print(f"  Origin: {origin_vertiport}, Destination: {destination_vertiport}")
            print(f"  Appearance Time: {appearance_time}, Departure Time: {departure_time}, Arrival Time: {arrival_time}")
            print(f"  End Auction Time: {end_auction_time}")

            # if staying at origin vertiport
            self._create_parking_nodes(origin_vertiport, appearance_time, end_auction_time)
            # Create nodes for the time window (appearance to departure, and arrival window)
            self._create_time_window_nodes(origin_vertiport, appearance_time, departure_time)
            # Create nodes for the time window (arrival to end of auction)
            self._create_time_window_nodes(destination_vertiport, arrival_time, end_auction_time)

            # Create edges for the origin vertiport from appearance to end of auction
            self._create_edges(origin_vertiport, appearance_time, departure_time)
 
            # Create dep and arr nodes for the origin and destination vertiports and their edges
            self._create_dep_arr_elements(origin_vertiport, destination_vertiport, departure_time, arrival_time)
            
            # Create edges for the destination vertiport from arrival to end of auction
            self._create_edges(destination_vertiport, arrival_time, end_auction_time)


        return self.graph
    
    def _create_dep_arr_elements(self, origin_vertiport, destination_vertiport, departure_time, arrival_time):
        """Create dep and arr nodes for the origin and destination vertiports at the exact times."""
        dep_node = f"{origin_vertiport}_{departure_time}_dep"
        arr_node = f"{destination_vertiport}_{arrival_time}_arr"
        self._add_node_if_not_exists(dep_node)
        self._add_node_if_not_exists(arr_node)

        # connect park nodes with dep node
        self._create_dep_arr_edges(f"{origin_vertiport}_{departure_time}", dep_node)
        self._create_dep_arr_edges(dep_node, arr_node)
        # connect arr nodes with park nodes
        self._create_dep_arr_edges(arr_node, f"{destination_vertiport}_{arrival_time}")

    def _create_parking_nodes(self, vertiport, start_time, end_time):
        """Create parking nodes from appearance to end of auction time at the origin vertiport."""
        for ts in range(start_time, end_time):
            node = f"{vertiport}_{ts}"
            self._add_node_if_not_exists(node)

    def _create_time_window_nodes(self, vertiport, start_time, end_time):
        """Create nodes from start_time to end_time"""
        for ts in range(start_time, end_time):
            node = f"{vertiport}_{ts}"
            self._add_node_if_not_exists(node)
        
    def _create_edges(self, vertiport, start_time, end_time):
        """Create edges between time window nodes for the given vertiport."""
        for ts in range(start_time, end_time - 1):  # Avoid out of range for next time step
            current_node = f"{vertiport}_{ts}"
            next_node = f"{vertiport}_{ts + 1}"
            self.graph.add_edge(current_node, next_node)
            # print(f"  Added Edge: {current_node} -> {next_node}")

    def _create_dep_arr_edges(self, current_node, next_node):
        """Create edges between time window nodes for the given vertiport."""
        self.graph.add_edge(current_node, next_node)


    def _get_end_auction_time(self, arrival_time, auction_frequency):
        """Calculate the end of the auction window based on arrival time and auction frequency."""
        return ((arrival_time // auction_frequency) + 1) * auction_frequency

    def _add_node_if_not_exists(self, node):
        """Check if node exists in VertiportStatus and add to the graph if it doesn't already exist."""
        if node not in self.graph:
            if node in self.vertiport_status.nodes:
                self.graph.add_node(node, **self.vertiport_status.nodes[node])
            else:
                self.graph.add_node(node)
    
                
