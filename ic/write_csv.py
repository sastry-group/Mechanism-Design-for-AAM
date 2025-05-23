import numpy as np
import pandas as pd
import os
import pickle
from utils import process_allocations

def full_list_string(lst):
    return ', '.join([str(item) for item in lst])


def write_to_csv(dataframe, filepath, mode='w', header=True):
    """Helper function to write DataFrame to a CSV file with error handling."""
    try:
        # filepath = f"{filepath}"
        dataframe.to_csv(filepath, index=False, mode=mode, header=header)
        # print(f"Data written to {filepath}")
    except Exception as e:
        print(f"Error writing data to {filepath}: {e}")




def write_market_interval(auction_start, auction_end, interval_flights, output_folder):
    """
    Writes the market interval data (start, end, and flights in interval) to a CSV file.
    """
    print("Writing market interval to file...")
    market_interval_data = {
        "Auction Start": [auction_start],
        "Auction End": [auction_end],
        "Flights in Interval": [full_list_string(interval_flights)]
    }
    market_interval_df = pd.DataFrame(market_interval_data)
    market_interval_file = os.path.join(f"{output_folder}/results/", "market_interval.csv")
    mode = 'w' if not os.path.isfile(market_interval_file) else 'a'
    write_to_csv(market_interval_df, market_interval_file, mode, header=(mode == 'w'))



def write_market_data(edge_information, prices, new_prices, capacity, end_capacity, market_auction_time, output_folder):
    """
    Writes the market data to a CSV file with details of edges, prices, capacities, etc.
    """
    market_data = []
    for i, (key, value) in enumerate(edge_information.items()):
        market_data.append([
            market_auction_time, key, ', '.join(value), prices[i], new_prices[i], capacity[i], end_capacity[i]
        ])
    
    columns = ["Auction Time", "Edge Label", "Good", "Fisher Prices", "New Prices", "Capacity", "End Capacity"]
    market_df = pd.DataFrame(market_data, columns=columns)
    market_file = os.path.join(f"{output_folder}/results/", f"market_{market_auction_time}.csv")
    mode = 'w' if not os.path.isfile(market_file) else 'a'
    write_to_csv(market_df, market_file, mode, header=(mode == 'w'))

    
def write_agent_data(agent_id, agents_data, edge_information, output_folder):
    """
    Writes the individual agent data to a CSV file, including allocations, utility, and budget.
    """
    data_to_write = {
        "Fishers Allocations": agents_data[agent_id]["fisher_allocation"],
        # "Edge Information": [f"{k}: {v}" for k, v in edge_information.items()],
        "Goods Lists": agents_data[agent_id]["agent_goods_list"],
        "Final Allocations": agents_data[agent_id]["final_allocation"],
        "Payment": agents_data[agent_id]["payment"],
        "Utility": agents_data[agent_id]["utility"],
        "Budget": agents_data[agent_id]["adjusted_budget"],
    }
    
    agent_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data_to_write.items()]))
    agent_file = os.path.join(f"{output_folder}/results/", f"{agent_id}.csv")
    mode = 'w' if not os.path.isfile(agent_file) else 'a'
    write_to_csv(agent_df, agent_file, mode, header=(mode == 'w'))


def write_results_table(flights, agents_data, output_folder):
    """
    Writes the summary of market results, including status, budgets, valuations, and allocations.
    """
    market_results_data = []
    for key, value in agents_data.items():
        status = value["status"]
        request_dep_time = flights[key]['requests']["001"]["request_departure_time"]
        original_budget = value["original_budget"]
        valuation = flights[key]['requests']["001"]["valuation"]
        modified_budget = value["adjusted_budget"]
        if status == "dropped":
            allocated_flight = ("Dropped", "Dropped")
        elif status == "parked":
            allocated_flight = ("Parked", "Parked")
        else:
            allocated_flight = value["good_allocated"]
        agent_payment = value["payment"]  
        origin_destination_tuple = (flights[key]["origin_vertiport_id"], flights[key]["requests"]["001"]["destination_vertiport_id"])
        
          
        market_results_data.append([
            key, status, modified_budget, original_budget, valuation, origin_destination_tuple, 
            request_dep_time, allocated_flight, agent_payment
        ])
    
    market_results_df = pd.DataFrame(market_results_data, columns=[
        "Agent", "Status", "Mod. Budget", "Ori. Budget", "Valuation", "(O,D)", 
        "Desired Departure (ts)", "Allocation (ts)", "Payment"
    ])

    market_results_file = os.path.join(f"{output_folder}/results/", "market_results_table.csv")
    mode = 'w' if not os.path.isfile(market_results_file) else 'a'
    write_to_csv(market_results_df, market_results_file, mode, header=(mode == 'w'))


def write_contested_goods(market_performance_data, output_folder):
    """
    Writes the market auction time, high-priced goods, and agents requesting high-priced goods to a CSV file.
    """
    market_results_data = [
        ["Market Auction Time", market_performance_data["market_auction_time"]],
        ["High-Priced Good", market_performance_data["high_priced_goods"]],
        ["Agents Requesting High Priced Goods", market_performance_data["agents_requesting_high_priced_goods"]]
    ]

    market_results_df = pd.DataFrame(market_results_data, columns=["Description", "Information"])
    market_results_file = os.path.join(f"{output_folder}/results/", "contested_goods.csv")
    mode = 'w' if not os.path.isfile(market_results_file) else 'a'
    write_to_csv(market_results_df, market_results_file, mode, header=(mode == 'w'))


def write_market_performance(market_performance_data, output_folder):

    market_results_data = []

    market_results_data.append([market_performance_data["market_auction_time"], market_performance_data["num_iterations"], 
                                market_performance_data["num_agents"], market_performance_data["num_goods"], len(market_performance_data["zero_cap_goods"][0]),
                                market_performance_data["num_contested_paths"], market_performance_data['num_contested_goods'], market_performance_data["fisher_run_time"]])
    

    market_performance_df = pd.DataFrame(market_results_data, columns=[
        "Market_Auction_Start_Time", "Number_Interations", "Number_Agents", "Num_Goods","Num_ZeroCap_Goods","Number_Contested_Path", "Number_Contested_DesGood","Run_Time"
    ])


    market_results_file = os.path.join(f"{output_folder}/results/", "market_performance_table.csv")
    mode = 'w' if not os.path.isfile(market_results_file) else 'a'
    write_to_csv(market_performance_df, market_results_file, mode, header=(mode == 'w'))


def write_output(flights, edge_information, market_data_dict, 
                agents_data, market_auction_time, output_folder):

    data = []
    prices = market_data_dict.get("prices", [])
    new_prices = market_data_dict.get("prices", []) # this is not accurate
    capacity = market_data_dict.get("original_capacity", [])
    end_capacity = market_data_dict.get("capacity", [])

    for agent_id, agent_data in agents_data.items():
        flight_info = flights.get(agent_id, {}).get("requests", {}).get("001", {})
        origin = flights.get(agent_id, {}).get("origin_vertiport_id", None)
        destination = flight_info.get("destination_vertiport_id", None)
        requested_dep_time = flight_info.get("request_departure_time", None)
        requested_arr_time = flight_info.get("request_arrival_time", None)
        dep_capacity = market_data_dict["original_capacity"][agent_data["desired_good_info"]["desired_dep_edge_idx"]]
        arr_capacity = market_data_dict["original_capacity"][agent_data["desired_good_info"]["desired_arr_edge_idx"]]
        next_sector_capacity = [market_data_dict["original_capacity"][idx] for idx in agent_data["desired_good_info"]["desired_transit_edges_idx"]]


        allocated_flight = agent_data.get("good_allocated", [])
        allocated_dep_time, allocated_arr_time = None, None
        if agent_data.get("status") == "allocated" or agent_data.get("status") == "delayed":
            allocated_dep_time = allocated_flight[0].split("_")[1]
            allocated_arr_time = allocated_flight[1].split("_")[1]
        elif agent_data.get("status")== "rebased":
            allocated_dep_time = 0
            allocated_arr_time = 0
        elif agent_data.get("status") == "dropped":
            allocated_dep_time = 0
            allocated_arr_time = 0
        else:
            allocated_dep_time = "unknown"
            allocated_arr_time = "unknown"

        max_capacity = (dep_capacity, next_sector_capacity, arr_capacity)
        status = agent_data.get("status", "unknown")
        payment = agent_data.get("payment", 0)
        initial_air_credits = agent_data.get("original_budget", 0)
        final_air_credits = agent_data.get("adjusted_budget", 0)
        utility = agent_data.get("valuation", 0)
        rank_id = np.where(np.array(market_data_dict["ranked_agents"]) == agent_id)[0] + 1

        data.append([
            agent_id, (origin, destination), (requested_arr_time, requested_dep_time),
            max_capacity, (allocated_arr_time, allocated_dep_time), status, payment,
            initial_air_credits, final_air_credits, utility, rank_id[0]
        ])

    columns = [
        "Aircraft", "Requested Route (Origin, Destination)", "Requested Time (Arrival, Departure)",
        "Max Capacity (Dep, Rout, Arr)", "Allocated Time (Arrival, Departure)", "Status", "Payment",
        "Initial Air Credits", "Final Air Credits", "Utility", "Rank"
    ]

    df = pd.DataFrame(data, columns=columns)
    output_file = os.path.join(f"{output_folder}/results/", f"results_table_{market_auction_time}.csv")
    write_results_table(flights, agents_data, output_folder)
    write_market_performance(market_data_dict, output_folder)
    write_market_data(edge_information, prices, new_prices, capacity, end_capacity, market_auction_time, output_folder)
    write_contested_goods(market_data_dict, output_folder)
    write_to_csv(df, output_file)

    print(f"Results table written to {output_file}")

# def write_output(flights, edge_information, market_data_dict, 
#                 agents_data_dict, market_auction_time, output_folder):
#     """
#     Orchestrates the writing of output data including market data, agent data, and result tables.
#     """
#     print("Writing output to file...")

#     prices = market_data_dict["prices"]
#     new_prices = prices # change this
#     capacity = market_data_dict["original_capacity"]
#     end_capacity = market_data_dict["capacity"] # change this
#     write_results_table(flights, agents_data_dict, output_folder)
#     write_market_data(edge_information, prices, new_prices, capacity, end_capacity, market_auction_time, output_folder)
#     write_customer_board(flights, agents_data_dict, output_folder)
#     write_SP_board(edge_information, agents_data_dict, output_folder)
#     write_network_board(market_data_dict, agents_data_dict, output_folder)
#     write_to_csv(pd.DataFrame(market_data_dict["ranked_agents"]), os.path.join(output_folder, "ranked_list.csv"))
    
#     for key in agents_data_dict:
#         if agents_data_dict[key]['status'] != 'dropped':
#             write_agent_data(key, agents_data_dict, edge_information, output_folder)

#     print("Output files written to", output_folder)


def save_data(output_folder, file_name, market_auction_time, **kwargs):
    """
    Saves additional data to a pickle file for later use.
    """
    try:
        file_path = f'{output_folder}/results/{file_name}_{market_auction_time}.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(kwargs, f)

        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")



# def write_customer_board(flights, agents_data, output_folder):
#     # here we need to create a fleet operator facing board including the following:
#     # 1. The list of all aircraft similar to airport naming convention
#     # 2. their respective requests as a tuple (origin, destination), (desired departure time, arrival time)
#     # 3. the allocated flight for each request
#     # 4. the payment for each request
#     # status: on-time, delayed, dropped
#     """
#     Writes a customer-facing board that includes details of aircraft, their requests, and flight allocation.
#     """
#     customer_board_data = []
#     for key, value in agents_data.items():
#         allocated_flight = value["good_allocated"]
#         request = flights[key]["requests"]["001"]
#         origin_vertiport_id, dep_time, destination_vertiport_id, arr_time = None, None, None, None
#         payment = value["payment"]
        
#         if value.get("status") == "allocated":
#             dep_info, arr_info = allocated_flight[0], allocated_flight[1]
#             origin_vertiport_id, dep_time = dep_info.split('_')[0], dep_info.split('_')[1]
#             destination_vertiport_id, arr_time = arr_info.split('_')[0], arr_info.split('_')[1]
#             # payment = value["payment"]
#             status = value["status"]
#         elif value.get("status") == "rebased":
#             status = "rebased"
#             origin_vertiport_id = flights[key]["origin_vertiport_id"]
#             # dep_time = request["request_departure_time"]
#             destination_vertiport_id = request["destination_vertiport_id"]
#             # arr_time = request["request_arrival_time"]
#         else:
#             status = "cancelled"
#             origin_vertiport_id = flights[key]["origin_vertiport_id"]
#             # dep_time = request["request_departure_time"]
#             destination_vertiport_id = request["destination_vertiport_id"]
#             # arr_time = request["request_arrival_time"]
        
#         customer_board_data.append([
#             key, dep_time, origin_vertiport_id, arr_time, destination_vertiport_id, status, payment
#         ])
#         # status = value["status"]
#         # customer_board_data.append([
#         #     key, request["request_departure_time"], flights[key]["origin_vertiport_id"],  
#         #     request["destination_vertiport_id"], request["request_arrival_time"], status, payment
#         # ])

#     customer_board_df = pd.DataFrame(customer_board_data, columns=[
#         "Aircraft", "Departure Time", "Departure Vertiport",  "Arrival Time", "Arrival Vertiport", "Status", "Payment"
#     ])
    
#     customer_board_file = os.path.join(output_folder, "customer_board.csv")
#     write_to_csv(customer_board_df, customer_board_file)

# def write_SP_board(edge_information, agents_data, output_folder):
#     # here we need to create a service provider  facing board including the following:
#     # 1. Number of requests
#     # 2. Congestion level of each request
#     # 3. The Fisher allocation for each request for their desired goods or delays if they are being delayed
#     """
#     Writes a service provider-facing board with request details, congestion, and Fisher allocations.
#     """
#     sp_board_data = []

#     for key, value in agents_data.items():
#         allocated_flight = value["good_allocated"]
#         request = value["flight_info"]["requests"]["001"]
#         origin_vertiport_id = value["flight_info"]["origin_vertiport_id"]
#         dep_time = request["request_departure_time"]
#         arr_time = request["request_arrival_time"]
#         destination_vertiport_id = request["destination_vertiport_id"]
#         allocated_origin_vertiport, allocated_dep_time, allocated_destination_vertiport,allocated_arr_time = None, None, None, None        
#         if value.get("status") == "allocated":
#             dep_info, arr_info = allocated_flight[0], allocated_flight[1]
#             allocated_origin_vertiport, allocated_dep_time = dep_info.split('_')[0], dep_info.split('_')[1]
#             allocated_destination_vertiport, allocated_arr_time = arr_info.split('_')[0], arr_info.split('_')[1]
#             payment = value["payment"]
#         elif value.get("status") == "rebased":
#             origin_vertiport_id = value["flight_info"]["origin_vertiport_id"]
#             # dep_time = request["request_departure_time"]
#             destination_vertiport_id = request["destination_vertiport_id"]
#             # arr_time = request["request_arrival_time"]
#         else:
#             origin_vertiport_id = value["flight_info"]["origin_vertiport_id"]
#             # dep_time = request["request_departure_time"]
#             destination_vertiport_id = request["destination_vertiport_id"]

#         sp_board_data.append([
#             key, len(value["flight_info"]["requests"]), (origin_vertiport_id, destination_vertiport_id), (dep_time, arr_time), 
#             value["status"], (allocated_origin_vertiport, allocated_destination_vertiport), (allocated_dep_time, allocated_arr_time)
#         ])
        
    
#     sp_board_df = pd.DataFrame(sp_board_data, columns=["Agent", "Number of Requests", "Requested Route", "Requested Time (Departure, Arrival)", "Status", "Allocated Route", "Allocated Time (Departure, Arrival)" ])
    
#     sp_board_file = os.path.join(output_folder, "sp_board.csv")
#     write_to_csv(sp_board_df, sp_board_file)

# def write_network_board(market_data_dict, agent_data_dict, output_folder):
#     # here we need to create a network facing board including the following:
#     # This is information private to the SP
#     # 1. Number of iterations to converge
#     # 2. Parameters used in the market (beta, etc)
#     # 3. The final prices for each desired good
#     # 4. Valuations of the agents 
#     # original budget, modified budget
#     """
#     Writes a network-facing board including market parameters, prices, and valuations.
#     """
#     num_iterations = market_data_dict["num_iterations"]
#     design_parameters = market_data_dict["market_parameters"]
#     network_board_data = []

#     # for agent_id, agent_data in agent_data_dict.items():
#     #     origin_vertiport = agent_data["flight_info"]["origin_vertiport_id"]
#     #     destination_vertiport = agent_data["flight_info"]["requests"]["001"]["destination_vertiport_id"]
#     #     original_budget = agent_data["original_budget"]
#     #     valuation = agent_data["valuation"]
#     #     network_board_data.append([
#     #         num_iterations, num_agents, contested_routes, design_parameters, final_prices, valuations,
#     #         agent_id, origin_vertiport, destination_vertiport, original_budget, valuation
#     #     ])

#     initial_capacity = market_data_dict["original_capacity"]
#     end_capacity = market_data_dict["capacity"]
#     num_agents = len(agent_data_dict)
#     final_prices = market_data_dict["prices"]


#     network_board_data.append([
#         num_iterations, num_agents, initial_capacity, end_capacity, design_parameters, final_prices
#     ])    

#     network_board_df = pd.DataFrame(network_board_data, columns=[
#         "No. Iterations", "No. Agents", "Initial Capacity", "End Capacity", 
#         "Design Parameters (price of default good, defautlt good valuation, dropout good val, beta, price upper bound)", "Final Prices"
#     ])
    
#     network_board_file = os.path.join(output_folder, "network_board.csv")
#     write_to_csv(network_board_df, network_board_file)