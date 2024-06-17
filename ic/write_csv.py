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


# def write_allocated_flights(allocations, rebased_allocations, output_folder):

#     market_allocation =  
#     market_interval_file = os.path.join(output_folder, "market_interval.csv")
#     if not os.path.isfile(market_interval_file):
#         market_interval_df.to_csv(market_interval_file, index=False, mode='w')
#     else:
#         market_interval_df.to_csv(market_interval_file, index=False, mode='a', header=False)
    


def write_output(flight_ids, agent_constraints, edge_information, prices, new_prices, capacity, 
                 agent_allocations, agent_indices, agent_edge_information, agent_goods_lists, 
                 int_allocations, new_allocations_goods, u, budget, payment, output_folder):
    """
    """
    print("Writing output to file...")

    # Market data
    market_data = []
    for i, (key, value) in enumerate(edge_information.items()):
        market_data.append([key, ', '.join(value), prices[i], new_prices[i], capacity[i]])
    market_df = pd.DataFrame(market_data, columns=["Edge Label", "Good", "Fisher Prices", "New Prices", "Capacity"])
    market_file = os.path.join(output_folder, "market.csv")

    if not os.path.isfile(market_file):
        market_df.to_csv(market_file, index=False, mode='w')
    else:
        market_df.to_csv(market_file, index=False, mode='a', header=False)


    # Agent data
    for i, flight_id in enumerate(flight_ids):
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
        agent_df = pd.DataFrame(agent_data.items(), columns=['Category', 'Value'])
        agent_file = os.path.join(output_folder, f"{flight_id}.csv")

        if not os.path.isfile(agent_file):
            agent_df.to_csv(agent_file, index=False, mode='w')
        else:
            agent_df.to_csv(agent_file, index=False, mode='a', header=False)


    print("Output files written to", output_folder)




