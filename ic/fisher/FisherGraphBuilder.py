import networkx as nx
import time

class FisherGraphBuilder:
    def __init__(self, vertiport_status, timing_info):
        self.vertiport_status = vertiport_status
        self.timing_info = timing_info
        self.graph = nx.DiGraph()

    def build_graph(self, flight_data):

        auction_frequency = self.timing_info["auction_frequency"]

        for request_id, request in flight_data["requests"].items():
            origin_vertiport = flight_data["origin_vertiport_id"]
            appearance_time = flight_data["appearance_time"]
            arrival_time = flight_data["requests"]["001"]["request_arrival_time"]
            auction_frequency = self.timing_info["auction_frequency"]
            end_auction_time = self._get_end_auction_time(arrival_time, auction_frequency)
            departure_time = flight_data["requests"]["001"]["request_departure_time"]

            if request_id == "000":
                attributes = {"valuation": request["valuation"]}
                # if staying at origin vertiport
                # Create parking nodes and edges from appearance to end of auction
                self._create_parking_nodes(origin_vertiport, appearance_time, end_auction_time)
                self._create_edges(origin_vertiport, appearance_time, end_auction_time, attributes = {"valuation": 0})
            else:
                destination_vertiport = flight_data["requests"]["001"]["destination_vertiport_id"]
                # Create nodes for the time window (arrival to end of auction)
                self._create_time_window_nodes(destination_vertiport, arrival_time, end_auction_time)
                # Create edges for the origin vertiport from appearance to end of auction
                valuation = request["valuation"]
                attributes = {"valuation": valuation}
                # Create edges for the destination vertiport from arrival to end of auction
                self._create_edges(destination_vertiport, arrival_time, end_auction_time, attributes = {"valuation": 0})
                self._create_dep_arr_elements(origin_vertiport, destination_vertiport, departure_time, arrival_time, attributes)
                
                decay = flight_data["decay_factor"]
                for ts_delay in range(1, 5): #change this to a variable
                    new_arrival_time = arrival_time + ts_delay
                    new_departure_time = departure_time + ts_delay
                    decay_valuation = request["valuation"] * decay**ts_delay
                    new_end_auction_time = self._get_end_auction_time(new_arrival_time, auction_frequency)
                    self._create_dep_arr_elements(origin_vertiport, destination_vertiport, new_departure_time, new_arrival_time, attributes = {"valuation": decay_valuation})
                    # Create edges for the destination vertiport from arrival to end of auction
                    self._create_edges(destination_vertiport, new_arrival_time, new_end_auction_time, attributes = {"valuation": 0})

        return self.graph
    
    def _create_dep_arr_elements(self, origin_vertiport, destination_vertiport,
                                departure_time, arrival_time, attributes=None):
        """Create dep and arr nodes for the origin and destination vertiports at the exact times."""
        dep_node = f"{origin_vertiport}_{departure_time}_dep"
        arr_node = f"{destination_vertiport}_{arrival_time}_arr"
        self._add_node_if_not_exists(dep_node)
        self._add_node_if_not_exists(arr_node)

        # connect park nodes with dep node
        attributes_park = {"valuation": 0}
        self._create_dep_arr_edges(f"{origin_vertiport}_{departure_time}", dep_node, attributes_park)

        self._create_dep_arr_edges(dep_node, arr_node, attributes)
        # connect arr nodes with park nodes
        self._create_dep_arr_edges(arr_node, f"{destination_vertiport}_{arrival_time}", attributes_park)


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
        
    def _create_edges(self, vertiport, start_time, end_time, attributes=None):
        """Create edges between time window nodes for the given vertiport."""
        for ts in range(start_time, end_time - 1):  # Avoid out of range for next time step
            current_node = f"{vertiport}_{ts}"
            next_node = f"{vertiport}_{ts + 1}"
            self.graph.add_edge(current_node, next_node, **attributes)
            # print(f"  Added Edge: {current_node} -> {next_node}")

    def _create_dep_arr_edges(self, current_node, next_node, attributes=None):
        """Create edges between time window nodes for the given vertiport."""
        self.graph.add_edge(current_node, next_node, **attributes)


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
    
                
