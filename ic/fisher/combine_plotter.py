import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import json
import re
from itertools import cycle

def extract_parameter_from_filename(filename, parameter):
    if parameter == "beta":
        match = re.search(r'[^_]+_[^_]+_[^_]+_(\d+\.\d+)', filename)
    elif parameter == "price_val":
        match = re.search(r'[^_]+_[^_]+_[^_]+_[^_]+_(\d+\.\d+)', filename)
    elif parameter == "default_val":
        match = re.search(r'[^_]+_[^_]+_[^_]+_[^_]+_[^_]+_(\d+\.\d+)', filename)
    elif parameter == "price_default_good":
        match = re.search(r'[^_]+_[^_]+_[^_]+_[^_]+_[^_]+_[^_]+_(\d+\.\d+)', filename)
    elif parameter == "lambda_freq":
        match = re.search(r'[^_]+_[^_]+_[^_]+_[^_]+_[^_]+_[^_]+_[^_]+_(\d+\.\d+)', filename)
    else:
        raise ValueError(f"Unknown parameter: {parameter}")

    if match:
        return float(match.group(1)), parameter
    else:
        raise ValueError(f"Could not extract {parameter} value from filename: {filename}")
    
    # # Use regex to extract the first number after the 3rd underscore

    # match = re.search(r'[^_]+_[^_]+_[^_]+_(\d+\.\d+)', filename)
    # print("beta val:", match.group(1))
    # if match:
    #     return float(match.group(1))
    # else:
    #     raise ValueError(f"Could not extract beta value from filename: {filename}")

def read_and_merge_data(files, parameter_to_evaluate):
    # Initialize empty structures for combined data
    combined_data = {
        "x_iter": [],
        "prices": [],
        "p": [],
        "overdemand": [],
        "error": [],
        "abs_error": [],
        "rebates": [],
        "agent_allocations": [],
        "market_clearing": [],
        "agent_constraints": [],
        "yplot": [],
        "social_welfare_vector": [],
        "desired_goods": [],
        "parameter_values": [],
    }

    for file in files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            data_to_plot = data["data_to_plot"]
            
            # Extract beta value from filename
            parameter_value, parameter_name = extract_parameter_from_filename(file, parameter_to_evaluate)
            combined_data["parameter_values"].append(parameter_value)

            # Update combined data with data from this file
            combined_data["x_iter"].append(data_to_plot["x_iter"])
            combined_data["prices"].append(data_to_plot["prices"])
            combined_data["overdemand"].append(data_to_plot["overdemand"])
            combined_data["error"].append(data_to_plot["error"])
            combined_data["abs_error"].append(data_to_plot["abs_error"])
            combined_data["rebates"].append(data_to_plot["rebates"])
            combined_data["agent_allocations"].append(data_to_plot["agent_allocations"])
            combined_data["market_clearing"].append(data_to_plot["market_clearing"])
            combined_data["yplot"].append(data_to_plot["yplot"])
            combined_data["social_welfare_vector"].append(data_to_plot["social_welfare_vector"])
            combined_data["desired_goods"].append(data["desired_goods"])
            # Set p, agent_constraints, and desired_goods (assume these are the same across files)
            if combined_data["p"] is None:
                combined_data["p"] = data_to_plot["p"]
            if combined_data["agent_constraints"] is None:
                combined_data["agent_constraints"] = data_to_plot["agent_constraints"]
            # if combined_data["desired_goods"] is None:
            #     combined_data["desired_goods"] = data["desired_goods"]

    return combined_data

def plotting_combined_market(files, output_folder, parameter_to_evaluate, market_auction_time=None):
    # Read and merge the data from multiple .pkl files
    data_to_plot = read_and_merge_data(files, parameter_to_evaluate)
    
    # Extract variables for plotting
    x_iter = data_to_plot["x_iter"]
    prices = data_to_plot["prices"]
    p = data_to_plot["p"]
    overdemand = data_to_plot["overdemand"]
    error = data_to_plot["error"]
    abs_error = data_to_plot["abs_error"]
    rebates = data_to_plot["rebates"]
    agent_allocations = data_to_plot["agent_allocations"]
    market_clearing = data_to_plot["market_clearing"]
    agent_constraints = data_to_plot["agent_constraints"]
    yplot = data_to_plot["yplot"]
    social_welfare = data_to_plot["social_welfare_vector"]
    desired_goods = data_to_plot["desired_goods"]
    parameter_values = data_to_plot["parameter_values"]
    parameter_name = parameter_to_evaluate

    def get_filename(base_name):
        if market_auction_time:
            return f"{output_folder}/{base_name}_a{market_auction_time}.png"
        else:
            return f"{output_folder}/{base_name}.png"
    
    # # Price evolution
    # plt.figure(figsize=(10, 5))
    # for idx, beta in enumerate(beta_values):
    #     beta = beta_values[idx]
    #     for good_index in range(len(p) - 2):
    #         plt.plot(range(1, x_iter + 1), [prices[idx][i][good_index] for i in range(len(prices[idx]))], label=f"Beta {beta}")
    # plt.plot(range(1, x_iter + 1), [prices[idx][i][-2] for i in range(len(prices[idx]))], 'b--', label="Default Good")
    # plt.plot(range(1, x_iter + 1), [prices[idx][i][-1] for i in range(len(prices[idx]))], 'r-.', label="Dropout Good")
    # plt.xlabel('x_iter')
    # plt.ylabel('Prices')
    # plt.title("Price evolution")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.savefig(get_filename("price_evolution"), bbox_inches='tight')
    # plt.close()

    # # Overdemand evolution
    # plt.figure(figsize=(10, 5))
    # for idx, beta in enumerate(beta_values):
    #     plt.plot(range(1, x_iter + 1), overdemand[idx], label=f"Beta {beta}")
    # plt.xlabel('x_iter')
    # plt.ylabel('Demand - Supply')
    # plt.title("Overdemand evolution")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.savefig(get_filename("overdemand_evolution"))
    # plt.close()

    # # Constraint error evolution
    # plt.figure(figsize=(10, 5))
    # for agent_index in range(len(agent_constraints)):
    #     for idx, beta in enumerate(beta_values):
    #         plt.plot(range(1, x_iter + 1), [error[i][agent_index] for i in range(len(error))], label=f"Beta {beta}")
    # plt.ylabel('Constraint error')
    # plt.title("Constraint error evolution $\sum ||Ax - b||^2$")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.savefig(get_filename("linear_constraint_error_evolution"))
    # plt.close()

    # # Absolute error evolution
    # plt.figure(figsize=(10, 5))
    # for agent_index in range(len(agent_constraints)):
    #     for idx, beta in enumerate(beta_values):
    #         plt.plot(range(1, x_iter + 1), [abs_error[idx][i][agent_index] for i in range(len(abs_error[idx]))], label=f"Beta {beta}")
    # plt.ylabel('Constraint error')
    # plt.title("Absolute error evolution $\sum ||x_i - y_i||^2$")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.savefig(get_filename("x-y_error_evolution"))
    # plt.close()

    # # Rebate evolution
    # plt.figure(figsize=(10, 5))
    # for constraint_index in range(len(rebates[0][0])):
    #     for idx, beta in enumerate(beta_values):
    #         plt.plot(range(1, x_iter + 1), [rebates[idx][i][constraint_index] for i in range(len(rebates[idx]))], label=f"Beta {beta}")
    # plt.xlabel('x_iter')
    # plt.ylabel('rebate')
    # plt.title("Rebate evolution")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.savefig(get_filename("rebate_evolution"))
    # plt.close()

    # # Payment evolution
    # plt.figure(figsize=(10, 5))
    # for agent_index in range(len(agent_allocations[0][0])):
    #     for idx, beta in enumerate(beta_values):
    #         plt.plot(range(1, x_iter + 1), [prices[idx][i] @ agent_allocations[idx][i][agent_index] for i in range(len(prices[idx]))], label=f"Beta {beta}")
    # plt.xlabel('x_iter')
    # plt.title("Payment evolution")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.savefig(get_filename("payment"), bbox_inches='tight')
    # plt.close()

    # Desired goods evolution
    plt.figure(figsize=(10, 5))
    # Define a list of 20+ colors
    colors = [
        "#000000",  # Black
        "#FF5733",  # Red-Orange
        "#33FF57",  # Green
        "#3357FF",  # Blue
        "#FF33A1",  # Pink
        "#FFDA33",  # Yellow
        "#33FFD6",  # Cyan
        "#DA33FF",  # Purple
        "#FFD433",  # Gold
        "#FF8333",  # Orange
        "#33FF9E",  # Light Green
        "#FF3388",  # Light Pink
        "#33B2FF",  # Sky Blue
        "#A733FF",  # Violet
        "#FF5733",  # Coral
        "#8CFF33",  # Lime Green
        "#FFBB33",  # Amber
        "#33FFC4",  # Turquoise
        "#FF3333",  # Crimson
        "#FF8C33",  # Apricot
        "#6B33FF",  # Indigo
        "#33FF57"   # Mint Green
    ]
    for id, parameter in enumerate(parameter_values):
        agents = desired_goods[id]
        # agents_of_interest = ["AC002", "AC009"]
        i = 0
        for agent in enumerate(agents):
            agent_id = agent[0]
            agent_name = agent[1]
            if agent_name == "AC003" or agent_name == "AC004":       
                # dep_index = desired_goods[agent_name]["desired_good_dep"]
                # arr_index = desired_goods[agent_name]["desired_good_arr"]
                # label = f"Flight:{agent_name}, {desired_goods[id][agent_name]['desired_edge']}" 
                label = f"Flight:{agent_name}, {parameter_name}:{parameter}"
                print(label, colors[id])
                # plt.plot(range(1, x_iter + 1), [agent_allocations[i][agent_id][dep_index] for i in range(len(agent_allocations))], '-', label=f"{agent_name}_dep good")
                dep_to_arr_index = desired_goods[id][agent_name]["desired_good_dep_to_arr"]
                if i % 2 == 0:
                    plt.plot(range(1, x_iter[id] + 1), [agent_allocations[id][i][agent_id][dep_to_arr_index] for i in range(len(agent_allocations[id]))], color=colors[id], label=label)
                else:
                    plt.plot(range(1, x_iter[id] + 1), [agent_allocations[id][i][agent_id][dep_to_arr_index] for i in range(len(agent_allocations[id]))], color=colors[id], linestyle='--', label=label)
                print("plotted:", agent_name)
                i += 1


    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('$x_{iter}$')
    plt.ylabel('Allocated Edge')
    plt.xlim(0, 200)
    plt.title("Contested Goods Beta Analysis")
    plt.savefig(get_filename(f"combined_desired_goods_beta_analysis_{parameter_to_evaluate}"), bbox_inches='tight')
    plt.close()

    # # Market Clearing Error
    # plt.figure(figsize=(10, 5))
    # for idx, beta in enumerate(beta_values):
    #     plt.plot(range(1, x_iter + 1), market_clearing[idx], label=f"Beta {beta}")
    # plt.xlabel('x_iter')
    # plt.title("Market Clearing Error")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.savefig(get_filename("market_clearing_error"))
    # plt.close()

    # # Y-values
    # plt.figure(figsize=(10, 5))
    # for agent_index in range(len(yplot[0][0])):
    #     for idx, beta in enumerate(beta_values):
    #         plt.plot(range(1, x_iter + 1), [yplot[idx][i][agent_index][:-2] for i in range(len(yplot[idx]))], label=f"Beta {beta}")
    # plt.xlabel('x_iter')
    # plt.title("Y-values")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.savefig(get_filename("y-values"), bbox_inches='tight')
    # plt.close()

    # # Social Welfare
    # plt.figure(figsize=(10, 5))
    # for idx, beta in enumerate(beta_values):
    #     plt.plot(range(1, x_iter + 1), social_welfare[idx], label=f"Beta {beta}")
    # plt.xlabel('x_iter')
    # plt.ylabel('Social Welfare')
    # plt.title("Social Welfare")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.savefig(get_filename("social_welfare"))
    # plt.close()

if __name__ == "__main__":
    # Example usage
    # files = ["/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/ic/results/casef_20240917_081204_1.0_1.0_1.0_10.0_2.0/fisher_data_1.pkl", 
    #         "/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/ic/results/casef_20240917_081204_1.0_1.0_1.0_10.0_10.0/fisher_data_1.pkl",
    #         "/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/ic/results/casef_20240917_081204_5.0_1.0_1.0_10.0_2.0/fisher_data_1.pkl",
    #         "/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/ic/results/casef_20240917_081204_5.0_1.0_1.0_10.0_10.0/fisher_data_1.pkl",
    #         "/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/ic/results/casef_20240917_081204_10.0_1.0_1.0_10.0_2.0/fisher_data_1.pkl",
    #         "/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/ic/results/casef_20240917_081204_10.0_1.0_1.0_10.0_10.0/fisher_data_1.pkl",
    #         "/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/ic/results/casef_20240917_081204_50.0_1.0_1.0_10.0_2.0/fisher_data_1.pkl",
    #         "/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/ic/results/casef_20240917_081204_50.0_1.0_1.0_10.0_10.0/fisher_data_1.pkl",
    #         "/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/ic/results/casef_20240917_081204_100.0_1.0_1.0_10.0_2.0/fisher_data_1.pkl",
    #         "/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/ic/results/casef_20240917_081204_100.0_1.0_1.0_10.0_10.0/fisher_data_1.pkl",
    #         ]

    # files = [
    #         "/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/ic/results/casef_20240917_081204_1.0_1.0_1.0_10.0_10.0/fisher_data_1.pkl",
    #         "/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/ic/results/casef_20240917_081204_5.0_1.0_1.0_10.0_10.0/fisher_data_1.pkl",
    #         "/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/ic/results/casef_20240917_081204_10.0_1.0_1.0_10.0_10.0/fisher_data_1.pkl",
    #         "/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/ic/results/casef_20240917_081204_50.0_1.0_1.0_10.0_10.0/fisher_data_1.pkl",
    #         "/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/ic/results/casef_20240917_081204_100.0_1.0_1.0_10.0_10.0/fisher_data_1.pkl",
    #         ]
    files = [
            "/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/ic/results/casef_20240925_175552short_5.0_1.0_1.0_0.1_10.0/fisher_data_1.pkl",
            "/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/ic/results/casef_20240925_175552short_5.0_1.0_1.0_0.2_10.0/fisher_data_1.pkl",
            "/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/ic/results/casef_20240925_175552short_5.0_1.0_1.0_0.5_10.0/fisher_data_1.pkl",
            "/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/ic/results/casef_20240925_175552short_5.0_1.0_1.0_1.0_10.0/fisher_data_1.pkl",
            # "/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/ic/results/casef_20240925_175552short_5.0_1.0_1.0_10.0_2.0/fisher_data_1.pkl",
            # "/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/ic/results/casef_20240925_175552short_1.0_1.0_1.0_10.0_2.0/fisher_data_1.pkl",
            ]
    output_folder = "/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/ic/plots/"
    # options "beta","price_val", "default_val", "price_default_good","lambda_freq"
    parameter_to_evaluate = "price_default_good"    
    market_auction_time = None  # Adjust if needed

    plotting_combined_market(files, output_folder, parameter_to_evaluate, market_auction_time)



