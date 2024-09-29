import numpy as np
import pandas as pd
import os
import pickle
from utils import process_allocations

def full_list_string(lst):
    return ', '.join([str(item) for item in lst])

def write_market_interval(auction_start, auction_end, interval_flights, output_folder):
    """
    """
    print("Writing market interval to file...")
    market_interval_data = {
        "Auction Start": [auction_start],
        "Auction End": [auction_end],
        "Flights in Interval": [full_list_string(interval_flights)]
    }
    market_interval_df = pd.DataFrame(market_interval_data)
    market_interval_file = os.path.join(output_folder, "market_interval.csv")
    if not os.path.isfile(market_interval_file):
        market_interval_df.to_csv(market_interval_file, index=False, mode='w')
    else:
        market_interval_df.to_csv(market_interval_file, index=False, mode='a', header=False)
    
    print("Market interval written to", market_interval_file)


def write_market_data(edge_information, prices, new_prices, capacity, end_capacity, market_auction_time, output_folder):

    # Market data
    market_data = []
    for i, (key, value) in enumerate(edge_information.items()):
        market_data.append([market_auction_time, key, ', '.join(value), prices[i], new_prices[i], capacity[i], end_capacity[i]])
    
    # Create the DataFrame with the appropriate columns
    columns = ["Auction Time", "Edge Label", "Good", "Fisher Prices", "New Prices", "Capacity", "End Capacity"]
    market_df = pd.DataFrame(market_data, columns=columns)
    market_file = os.path.join(output_folder, f"market_{market_auction_time}.csv")

    # Write to the file, ensuring the header is included only when creating the file
    if not os.path.isfile(market_file):
        market_df.to_csv(market_file, index=False, mode='w')
    else:
        market_df.to_csv(market_file, index=False, mode='a', header=False)

    


def write_output(flights, edge_information, prices, new_prices, capacity, end_capacity, 
                agents_data, market_auction_time, output_folder):
    """

    """
    print("Writing output to file...")
    # we need to separate this data writing later


    write_results_table(flights, agents_data, output_folder)
    write_market_data(edge_information, prices, new_prices, capacity, end_capacity, market_auction_time, output_folder)

    # extra_data = {
    #     "capacity": capacity,
    #     "end Capacity": end_capacity,
    #     "prices": new_prices,
    #     "agents_data": agents_data
    # }
    # save_data(output_folder, "animation", market_auction_time, **extra_data)



    # Agent data
    for key, value in agents_data.items():
        if agents_data[key]['status'] == 'dropped':
            continue
        data_to_write = {
            # "Fisher Allocations": [f"{x:.4f}" for x in agent_allocations[i]],
            "Fishers Allocations": agents_data[key]["fisher_allocation"],
            "Indices": agents_data[key]['agent_edge_indices'],
            # "Edge Information": agent_edge_information[i],
            "Goods Lists": agents_data[key]["agent_goods_list"],
            "Sample and Int Allocations": agents_data[key]["int_allocation"],
            "Deconflicted Allocations": agents_data[key]["deconflicted_goods"],
            "Utility": agents_data[key]["utility"],
            "Budget": agents_data[key]["adjusted_budget"],
            "Payment": agents_data[key]["payment"]
        }
        agent_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data_to_write.items()]))
        agent_file = os.path.join(output_folder, f"{key}.csv")

        if not os.path.isfile(agent_file):
            agent_df.to_csv(agent_file, index=False, mode='w')
        else:
            agent_df.to_csv(agent_file, index=False, mode='a', header=False)

    print("Output files written to", output_folder)

def write_results_table(flights, agents_data, output_folder):
    """
    """


    market_results_data = []
    for key, value in flights.items():
        agent_flight_data = key
        request_dep_time = value["requests"]["001"]["request_departure_time"]
        original_budget = value["budget_constraint"]
        valuation = value['requests']["001"]["valuation"]
        modified_budget = agents_data[key]["adjusted_budget"]
        allocated_flight = agents_data[key]["good_allocated"]
        agent_payment = agents_data[key]["payment"]
        origin_destination_tuple = (value["origin_vertiport_id"], value['requests']["001"]["destination_vertiport_id"])
        status = agents_data[key]["status"]
          
        # allocated_flight = next((allocation[1] for allocation in allocations if flight_id == allocation[0]), None)
        market_results_data.append([agent_flight_data, status, modified_budget, original_budget, valuation, origin_destination_tuple, 
                        request_dep_time, allocated_flight, agent_payment])
    market_results_df = pd.DataFrame(market_results_data, columns=["Agent", "Status", "Mod. Budget", "Ori. Budget", "Valuation", "(O,D)", "Desired Departure (ts)",
                                                            "Allocation (ts)", "Payment"])

    market_results_file = os.path.join(output_folder, "market_results_table.csv")
    if not os.path.isfile(market_results_file):
        market_results_df.to_csv(market_results_file, index=False, mode='w')
    else:
        market_results_df.to_csv(market_results_file, index=False, mode='a', header=False)
    
    print("Market interval written to", market_results_file)



def save_data(output_folder, file_name, market_auction_time, **kwargs):
    with open(f'{output_folder}/{file_name}_{market_auction_time}.pkl', 'wb') as f:
        pickle.dump(kwargs, f)


