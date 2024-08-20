import numpy as np
import pandas as pd
import os

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

    


def write_output(flights, agent_constraints, edge_information, prices, new_prices, capacity, end_capacity, 
                 agent_allocations, agent_indices, agent_edge_information, agent_goods_lists, 
                 int_allocations, new_allocations_goods, utilities, budget, payment, end_agent_status_data, market_auction_time, output_folder):
    """

    """
    print("Writing output to file...")
    # we need to separate this data writing later

    write_results_table(flights, end_agent_status_data, budget, payment, output_folder)
    write_market_data(edge_information, prices, new_prices, capacity, end_capacity, market_auction_time, output_folder)



    # Agent data
    dropouts = end_agent_status_data[2]
    for i, flight_id in enumerate(list(flights.keys())):
        if flight_id in dropouts:
            continue
        agent_data = {
            "Allocations": [f"{x:.4f}" for x in agent_allocations[i]],
            "Indices": agent_indices[i].astype(int),
            "Edge Information": agent_edge_information[i],
            "Goods Lists": agent_goods_lists[i][:-1],
            "Sample and Int Allocations": int_allocations[i],
            "Deconflicted Allocations": new_allocations_goods[i],
            "Utility": utilities[i],
            "Budget": str(budget[i]),
            "Payment": str(payment[i])
        }
        agent_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in agent_data.items()]))
        agent_file = os.path.join(output_folder, f"{flight_id}.csv")

        if not os.path.isfile(agent_file):
            agent_df.to_csv(agent_file, index=False, mode='w')
        else:
            agent_df.to_csv(agent_file, index=False, mode='a', header=False)

    print("Output files written to", output_folder)

def write_results_table(flights, end_agent_status_data, budget, payment, output_folder):
    """
    """

    allocations, rebased_allocations, dropout_agents = end_agent_status_data


    market_results_data = []
    for i, flight_id in enumerate(list(flights.keys())):
        flight = flights[flight_id]
        request_dep_time = flight["requests"]["001"]["request_departure_time"]
        original_budget = flight["budget_constraint"]
        valuation = flight['requests']["001"]["valuation"]
        origin_destination_tuple = (flight["origin_vertiport_id"], flight['requests']["001"]["destination_vertiport_id"])

        rebased = any(flight_id == allocation[0] for allocation in rebased_allocations)
        dropouts = any(flight_id == dropout for dropout in dropout_agents)
        if rebased:
            status = "Rebased"
            agent_payment = payment[i]
        elif dropouts:
            status = "DroppedOut"
            agent_payment = 0
        else:
            status = "Allocated"  
            agent_payment = payment[i]
          
        allocated_flight = next((allocation[1] for allocation in allocations if flight_id == allocation[0]), None)
        market_results_data.append([flight_id, status, budget[i], original_budget, valuation, origin_destination_tuple, 
                        request_dep_time, allocated_flight, agent_payment])
    market_results_df = pd.DataFrame(market_results_data, columns=["Agent", "Status", "Mod. Budget", "Ori. Budget", "Valuation", "(O,D)", "Desired Departure (ts)",
                                                            "Allocation (ts)", "Payment"])

    market_results_file = os.path.join(output_folder, "market_results_table.csv")
    if not os.path.isfile(market_results_file):
        market_results_df.to_csv(market_results_file, index=False, mode='w')
    else:
        market_results_df.to_csv(market_results_file, index=False, mode='a', header=False)
    
    print("Market interval written to", market_results_file)


