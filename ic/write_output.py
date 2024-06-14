import numpy as np

def full_list_string(lst):
    return ', '.join([str(item) for item in lst])


def write_market_interval(auction_start, auction_end, interval_flights, output_folder):
    """
    """
    print("Writing market interval to file...")
    market_interval_file = f"{output_folder}/market_output.txt"
    with open(market_interval_file, "a") as f:
        f.write(f"Market interval: {auction_start} to {auction_end}\n")
        f.write("Flights in interval:\n")
        for flight in interval_flights:
            f.write(f"{flight}\n")
        f.write("\n")
    print("Market interval written to", market_interval_file)

def write_output(agent_constraints, edge_information, prices, new_prices, capacity, 
                 agent_allocations, agent_indices, agent_edge_information, agent_goods_lists, 
                 int_allocations, new_allocations_goods, u, budget, payment, output_folder):
    """
    """
    print("Writing output to file...")
    # mapping for easier writing of output file:
    
    num_agents = len(agent_constraints)
    
    # Convert each matrix in agent_constraints to a string and add to data_to_output
    data_to_output = []
    for i, matrix in enumerate(agent_constraints):
        data_to_output.append(f"Matrix {i+1}:\n")
        data_to_output.append(full_list_string(matrix[0]))
        data_to_output.append("\n")
        data_to_output.append(full_list_string(matrix[1]))
        data_to_output.append("\n")
    output_data = ''.join(data_to_output)


    # Set print options to avoid truncation
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    market_output_file = f"{output_folder}/market_output.txt"
    agent_output_file = f"{output_folder}/agent_output.txt"
    edge_key_file = f"{output_folder}/edge_key.txt"

    # with open(edge_key_file, "a") as f:
    #     f.write("Edge Label, Good, Fisher Prices, New Prices, Capacity\n")
    #     for i, (key, value) in enumerate(edge_information.items()):
    #         line = f"{key}: {', '.join(value)}, {prices[i]}, {new_prices[i]}, {capacity[i]}\n"
    #         f.write(line)
    #         f.write("\n")


    with open(market_output_file, "a") as f:
        f.write("Edge Label, Good, Fisher Prices, New Prices, Capacity\n")
        for i, (key, value) in enumerate(edge_information.items()):
            line = f"{key}: {', '.join(value)}, {prices[i]}, {new_prices[i]}, {capacity[i]}\n"
            f.write(line)

    with open(agent_output_file, "a") as f:
        f.write("Allocations:\n")
        for i in range(num_agents):
            f.write(f"Agent {i+1}:\n")
            f.write("Fisher: ")
            f.write(full_list_string(agent_allocations[i]))
            f.write("\n")
            f.write(full_list_string(agent_indices[i]))
            f.write("\n")
            f.write(full_list_string(agent_edge_information[i]))
            f.write("\n")
            f.write(full_list_string(agent_goods_lists[i]))
            f.write("\n")
            f.write("Sample and int: ")
            f.write(full_list_string(int_allocations[i]))
            f.write("\n")
            f.write("Deconflicted: ")
            f.write(full_list_string(new_allocations_goods[i]))
            f.write("\n")
            f.write("Utility:\n")
            f.write(np.array2string(np.array(u[i]), separator=', '))
            f.write("\n")
            f.write("Budget:\n")
            f.write(str(budget[i]))
            f.write("\n")
            f.write("Payment:\n")
            f.write(str(payment[i]))
            f.write("\n")
            # f.write("Constraints:\n")
            # f.write(output_data)


    print("Output files written to", output_folder)
