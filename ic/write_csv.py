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


def write_market_data(edge_information, prices, new_prices, capacity, market_auction_time, output_folder):

    # Market data
    market_data = []
    for i, (key, value) in enumerate(edge_information.items()):
        market_data.append([market_auction_time, key, ', '.join(value), prices[i], new_prices[i], capacity[i]])
    
    # Create the DataFrame with the appropriate columns
    columns = ["Auction Time", "Edge Label", "Good", "Fisher Prices", "New Prices", "Capacity"]
    market_df = pd.DataFrame(market_data, columns=columns)
    market_file = os.path.join(output_folder, "market.csv")

    # Write to the file, ensuring the header is included only when creating the file
    if not os.path.isfile(market_file):
        market_df.to_csv(market_file, index=False, mode='w')
    else:
        market_df.to_csv(market_file, index=False, mode='a', header=False)

    


def write_output(flights, agent_constraints, edge_information, prices, new_prices, capacity, 
                 agent_allocations, agent_indices, agent_edge_information, agent_goods_lists, 
                 int_allocations, new_allocations_goods, u, budget, payment,allocations, rebased, market_auction_time, output_folder):
    """
    """
    print("Writing output to file...")
    # we need to separate this data writing later

    write_results_table(flights, allocations, budget, payment, rebased, output_folder)
    write_market_data(edge_information, prices, new_prices, capacity, market_auction_time, output_folder)



    # Agent data
    for i, flight_id in enumerate(list(flights.keys())):
        agent_data = {
            "Allocations": full_list_string(agent_allocations[i]),
            "Indices": full_list_string(agent_indices[i]),
            "Edge Information": full_list_string(agent_edge_information[i]),
            "Goods Lists": full_list_string(agent_goods_lists[i]),
            "Sample and Int Allocations": full_list_string(int_allocations[i]),
            "Deconflicted Allocations": full_list_string(new_allocations_goods[i]),
            "Utility": np.array2string(np.array(u[i]), separator=', '),
            "Budget": str(budget[i]),
            "Payment": str(payment[i])
        }
        agent_df = pd.DataFrame({k: [v] for k, v in agent_data.items()})
        agent_file = os.path.join(output_folder, f"{flight_id}.csv")

        if not os.path.isfile(agent_file):
            agent_df.to_csv(agent_file, index=False, mode='w')
        else:
            agent_df.to_csv(agent_file, index=False, mode='a', header=False)

    print("Output files written to", output_folder)

def write_results_table(flights, allocations, budget, payment, rebased_allocations, output_folder):
    """
    """

    market_results_data = []
    for i, flight_id in enumerate(list(flights.keys())):
        rebased = any(flight_id == allocation[0] for allocation in rebased_allocations)
        allocated_flight = next((allocation[1] for allocation in allocations if flight_id == allocation[0]), None)
        flight = flights[flight_id]
        request_dep_time = flight["requests"]["001"]["request_departure_time"]
        original_budget = flight["budget_constraint"]
        valuation = flight['requests']["001"]["valuation"]
        origin_destination_tuple = (flight["origin_vertiport_id"], flight['requests']["001"]["destination_vertiport_id"])
        market_results_data.append([flight_id, budget[i], original_budget, valuation, origin_destination_tuple, 
                        request_dep_time, allocated_flight, payment[i], rebased])
    market_results_df = pd.DataFrame(market_results_data, columns=["Agent", "Mod. Budget", "Ori. Budget", "Valuation", "(O,D)", "Desired Departure (ts)",
                                                            "Allocation (ts)", "Price", "Rebased Allocation"])

    market_results_file = os.path.join(output_folder, "market_results_table.csv")
    if not os.path.isfile(market_results_file):
        market_results_df.to_csv(market_results_file, index=False, mode='w')
    else:
        market_results_df.to_csv(market_results_file, index=False, mode='a', header=False)
    
    print("Market interval written to", market_results_file)


