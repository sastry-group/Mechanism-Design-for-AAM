import networkx as nx
import time
import sys
from pathlib import Path

# Add the bluesky package to the path
top_level_path = Path(__file__).resolve().parent.parent
print(str(top_level_path))
sys.path.append(str(top_level_path))

from VertiportStatus import VertiportStatus

class FisherGraphBuilder:
    def __init__(self, vertiport_status, timing_info, previous_capacity=[]):
        self.vertiport_status = vertiport_status
        self.timing_info = timing_info
        self.graph = nx.DiGraph()
        self.additional_constraints = []
        self.previous_capacity = previous_capacity

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
                # if destination_vertiport is not None:
                    # self._create_time_window_nodes(destination_vertiport, arrival_time, end_auction_time)
                # Create edges for the origin vertiport from appearance to end of auction
                valuation = request["valuation"]
                sector_path = request["sector_path"]
                sector_times = request["sector_times"]
                
                decay = flight_data["decay_factor"]
                for ts_delay in range(5):
                    new_arrival_time = arrival_time + ts_delay
                    new_departure_time = departure_time + ts_delay
                    decay_valuation = request["valuation"] * decay**ts_delay
                    new_end_auction_time = self._get_end_auction_time(new_arrival_time, auction_frequency)
                    # self._create_dep_arr_elements(origin_vertiport, destination_vertiport, new_departure_time, new_arrival_time, attributes = {"valuation": decay_valuation})
                    # Create edges for the destination vertiport from arrival to end of auction
                    if destination_vertiport is not None:
                        self._create_edges(destination_vertiport, new_arrival_time, new_end_auction_time, attributes = {"valuation": 0})
                    
                    # Add edges for the path
                    attributes = {"valuation": decay_valuation}
                    new_sector_times = [ts + ts_delay for ts in sector_times]
                    self._create_path_elements(origin_vertiport, destination_vertiport, sector_path, new_sector_times,
                              new_departure_time, new_arrival_time, attributes=attributes)

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

    def _create_path_elements(self, origin_vertiport, destination_vertiport, sector_path, sector_times,
                              departure_time, arrival_time, attributes=None):
        """Create path elements for the given path."""
        path_additional_constraints = []

        # Departing vertiport
        self._add_node_if_not_exists(f"{origin_vertiport}_{departure_time}")
        dep_node = f"{origin_vertiport}_{departure_time}_dep"
        self._add_node_if_not_exists(dep_node)
        self._add_node_if_not_exists(f"{sector_path[0]}_{sector_times[0]}")
        dep_attributes = {"valuation": 0}
        self._create_dep_arr_edges(f"{origin_vertiport}_{departure_time}", dep_node, dep_attributes)
        if (f"{origin_vertiport}_{departure_time}", dep_node) not in self.previous_capacity:
            path_additional_constraints.append((f"{origin_vertiport}_{departure_time}", dep_node))

        self._add_edge_if_not_exists(f"{origin_vertiport}_{departure_time}_dep", f"{sector_path[0]}_{sector_times[0]}", attributes)

        # Traversing sectors
        for i in range(len(sector_path)):
            current_sector, current_time = sector_path[i], sector_times[i]
            next_time = sector_times[i + 1]
            sector_attributes = {"valuation": 0}
            for ts in range(current_time, next_time):
                start_node = f"{current_sector}_{ts}"
                end_node = f"{current_sector}_{ts+1}" 
                # if start_node == "S002_35" and end_node == "S002_36":
                #     print(f"Adding ({start_node}, {end_node})") 
                self._add_node_if_not_exists(end_node)
                self._add_edge_if_not_exists(start_node, end_node, sector_attributes)

            if i < len(sector_path) - 1:
                next_sector = sector_path[i + 1]
                self._add_edge_if_not_exists(f"{current_sector}_{next_time}", f"{next_sector}_{next_time}", sector_attributes)
                if (f"{current_sector}_{next_time}", f"{next_sector}_{next_time}") not in self.previous_capacity:
                    path_additional_constraints.append((f"{current_sector}_{next_time}", f"{next_sector}_{next_time}"))

        # Arriving at vertiport
        if destination_vertiport is not None:
            arr_node = f"{destination_vertiport}_{arrival_time}_arr"
            # print(f"Adding {arr_node} and {destination_vertiport}_{arrival_time}")
            self._add_node_if_not_exists(arr_node)
            self._add_node_if_not_exists(f"{destination_vertiport}_{arrival_time}")
            self._add_edge_if_not_exists(f"{sector_path[-1]}_{sector_times[-1]}", arr_node, {"valuation": 0})
            self._add_edge_if_not_exists(arr_node, f"{destination_vertiport}_{arrival_time}", {"valuation": 0})
            if (arr_node, f"{destination_vertiport}_{arrival_time}") not in self.previous_capacity:
                path_additional_constraints.append((arr_node, f"{destination_vertiport}_{arrival_time}"))
        
        if path_additional_constraints:
            self.additional_constraints.append(path_additional_constraints)

    def _create_sector_nodes(self, sector, start_time, end_time):
        """Create nodes for the given sector from start_time to end_time."""
        for ts in range(start_time, end_time):
            node = f"{sector}_{ts}"
            self._add_node_if_not_exists(node)

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
            # edge = (current_node, next_node)
            # self._add_edge_if_not_exists(edge, **attributes)
            # self.graph.add_edge(current_node, next_node, **attributes)
            self._add_edge_if_not_exists(current_node, next_node, attributes)
            # print(f"  Added Edge: {current_node} -> {next_node}")

    def _create_dep_arr_edges(self, current_node, next_node, attributes=None):
        """Create edges between time window nodes for the given vertiport."""
        # edge = (current_node, next_node)
        # if edge not in self.graph.edges:
        #     if edge in self.vertiport_status.edges:
        #         self.graph.add_edge(edge, **self.vertiport_status.edges[edge])
        #     else:
        #         self.graph.add_edge(edge)
        # edge = (current_node, next_node)
        # self._add_edge_if_not_exists(edge, **attributes)
        self._add_edge_if_not_exists(current_node, next_node, attributes) 


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
    
    # def _add_edge_if_not_exists(self, edge, **attributes):
        # """Check if edge exists in VertiportStatus and add to the graph if it doesn't already exist."""
        # if edge not in self.graph.edges:
        #     current_node, next_node = edge
        #     if edge in self.vertiport_status.edges:
        #         self.graph.add_edge(current_node, next_node, **self.vertiport_status.edges[edge])
        #     else:
        #         self.graph.add_edge(current_node, next_node, **attributes)
    def _add_edge_if_not_exists(self, node1, node2, attributes=None):
        """Check if edge exists in VertiportStatus and add to the graph if it doesn't already exist."""
        if not self.graph.has_edge(node1, node2):
            if not self.previous_capacity or (node1, node2) not in self.previous_capacity:
                self.graph.add_edge(node1, node2, **(attributes or {}))
    
if __name__ == "__main__":
    vertiports = {"A": {
            "latitude": 1.0,
            "longitude": 1.0,
            "landing_capacity": 1,
            "takeoff_capacity": 1,
            "hold_capacity": 2
        }, "B": {
            "latitude": 2.0,
            "longitude": 2.0,
            "landing_capacity": 1,
            "takeoff_capacity": 1,
            "hold_capacity": 2
        }
    }
    routes = [
        {
            "origin_vertiport_id": "A",
            "destination_vertiport_id": "B",
            "travel_time": 1,
            "capacity": 4
        }
    ]
    timing_info = {
        "start_time" : 1,
        "end_time": 5,
        "time_step": 1,
        "auction_frequency": 5,
        "auction_end": 5
    }
    vertiport_status = VertiportStatus(vertiports, routes, timing_info)
    graph_builder = FisherGraphBuilder(vertiport_status, timing_info)

    flight_data = {
        "origin_vertiport_id": "A",
        "appearance_time": 1,
        "requests": {
            "000": {
                "valuation": 10
            },
            "001": {
                "destination_vertiport_id": "B",
                "request_arrival_time": 3,
                "request_departure_time": 2,
                "valuation": 5
            }
        },
        "decay_factor": 0.9
    }
    graph = graph_builder.build_graph(flight_data)
                
