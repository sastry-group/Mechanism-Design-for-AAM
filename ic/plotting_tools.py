from matplotlib import pyplot as plt
import numpy as np
from utils import compute_utilites
import logging



logger = logging.getLogger("global_logger")



def plotting_market(data_to_plot, desired_goods, output_folder, market_auction_time=None, lambda_frequency=1):

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
    fixed_point_errors = data_to_plot["fixed_point_error"] 




    # x_iter, prices, p, overdemand, error, abs_error, rebates, agent_allocations, market_clearing, agent_constraints, yplot, social_welfare, desired_goods = data_to_plot
    def get_filename(base_name):
        case_name = output_folder.split("/")[-1]
        if market_auction_time:
            return f"{output_folder}/plots/{base_name}_a{market_auction_time}_{case_name}.png"
        else:
            return f"{output_folder}/plots/{base_name}_{case_name}.png"
    
    # Price evolution
    plt.figure(figsize=(10, 5))
    for good_index in range(len(p) - 2):
        plt.plot(range(1, x_iter + 1), [prices[i][good_index] for i in range(len(prices))])
    plt.plot(range(1, x_iter + 1), [prices[i][-2] for i in range(len(prices))], 'b--', label="Default Good")
    plt.plot(range(1, x_iter + 1), [prices[i][-1] for i in range(len(prices))], 'r-.', label="Dropout Good")
    plt.xlabel('x_iter')
    plt.ylabel('Prices')
    plt.title("Price evolution")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(get_filename("price_evolution"),  bbox_inches='tight')
    plt.close()

    # Overdemand evolution
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, x_iter + 1), overdemand)
    plt.xlabel('x_iter')
    plt.ylabel('Demand - Supply')
    plt.title("Overdemand evolution")
    plt.savefig(get_filename("overdemand_evolution"))
    plt.close()

    # Constraint error evolution
    plt.figure(figsize=(10, 5))
    for agent_index in range(len(agent_constraints)):
        plt.plot(range(1, x_iter + 1), error[agent_index])
    
    plt.ylabel('Constraint error')
    plt.title("Constraint error evolution $\sum ||Ax - b||^2$")
    plt.savefig(get_filename("linear_constraint_error_evolution"))
    plt.close()

    # Absolute error evolution
    plt.figure(figsize=(10, 5))
    for agent_index in range(len(agent_constraints)):
        plt.plot(range(1, x_iter + 1), abs_error[agent_index])
    plt.ylabel('Constraint error')
    plt.title("Absolute error evolution $\sum ||x_i - y_i||^2$")
    plt.savefig(get_filename("x-y_error_evolution"))
    plt.close()

    # Rebate evolution
    plt.figure(figsize=(10, 5))
    for constraint_index in range(len(rebates[0])):
        plt.plot(range(1, x_iter + 1), [rebates[i][constraint_index] for i in range(len(rebates))])
    plt.xlabel('x_iter')
    plt.ylabel('rebate')
    plt.title("Rebate evolution")
    plt.savefig(get_filename("rebate_evolution"))
    plt.close()

    # Agent allocation evolution
    # plt.figure(figsize=(10, 5))
    # for agent_index in range(len(agent_allocations[0])):
    #     plt.plot(range(1, x_iter + 1), [agent_allocations[i][agent_index][:-2] for i in range(len(agent_allocations))])
    #     plt.plot(range(1, x_iter + 1), [agent_allocations[i][agent_index][-2] for i in range(len(agent_allocations))], 'b--', label=f"{agent_index} - Default Good")
    #     plt.plot(range(1, x_iter + 1), [agent_allocations[i][agent_index][-1] for i in range(len(agent_allocations))], 'r-.', label=f"{agent_index} - Dropout Good")
    # plt.legend()
    # plt.xlabel('x_iter')
    # plt.title("Agent allocation evolution")
    # plt.savefig(get_filename("agent_allocation_evolution"))
    # plt.close()

    # Payment 
    
    plt.figure(figsize=(10, 5))
    # agent allocations 
    for agent_index in range(len(agent_allocations[0])):
        # payment = prices[agent_index] @ agent_allocations[agent_index][0]
        label = f"Flight:{agent_index}" 
        plt.plot(range(1, x_iter + 1), [prices[i] @ agent_allocations[i][agent_index] for i in range(len(prices))], label=label)
    plt.xlabel('x_iter')
    plt.title("Payment evolution")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(get_filename("payment"), bbox_inches='tight')
    plt.close()


    # Desired goods evolution
    plt.figure(figsize=(10, 5))
    # print(f"Allocations: {agent_allocations}")
    agent_desired_goods_list = []
    for agent in enumerate(desired_goods):
        agent_id = agent[0]
        agent_name = agent[1]       
        # dep_index = desired_goods[agent_name]["desired_good_dep"]
        # arr_index = desired_goods[agent_name]["desired_good_arr"]
        label = f"Flight:{agent_name}, {desired_goods[agent_name]['desired_dep_edge']}" 
        # plt.plot(range(1, x_iter + 1), [agent_allocations[i][agent_id][dep_index] for i in range(len(agent_allocations))], '-', label=f"{agent_name}_dep good")
        dep_index = desired_goods[agent_name]["desired_dep_edge_idx"]
        agent_desired_goods = [agent_allocations[i][agent_id][dep_index] for i in range(len(agent_allocations))]
        agent_desired_goods_list.append(agent_desired_goods)
        plt.plot(range(1, x_iter + 1), agent_desired_goods, '--', label=label)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('x_iter')
    plt.title("Desired Goods Agent allocation evolution")
    plt.savefig(get_filename("desired_goods_allocation_evolution"), bbox_inches='tight')
    plt.close()
    # print(f"Final Desired Goods Allocation: {[desired_goods[-1] for desired_goods in agent_desired_goods_list]}")
    logger.debug(f"Final Desired Goods Allocation: {[desired_goods[-1] for desired_goods in agent_desired_goods_list]}")

    # Market Clearing Error
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, x_iter + 1), market_clearing)
    plt.xlabel('x_iter')
    plt.title("Market Clearing Error")
    plt.savefig(get_filename("market_clearing_error"))
    plt.close()

    # Fixed Point Error
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(fixed_point_errors)+1), fixed_point_errors, label='Fixed Point Error')
    # plt.axhline(y=1e-3, color='r', linestyle='--', label='Tolerance')
    plt.xlabel('Iteration')
    plt.ylabel('Fixed Point Error')
    plt.legend()
    plt.title(f"Fixed-Point Convergence")
    plt.savefig(get_filename("fixed_point_error_plot"), bbox_inches='tight')
    plt.close()

    # # y
    # plt.figure(figsize=(10, 5))
    # for agent_index in range(len(yplot[0])):
    #     plt.plot(range(1, x_iter + 1), [yplot[i][agent_index][:-2] for i in range(len(yplot))])
    #     # plt.plot(range(1, x_iter + 1), [yplot[i][agent_index][-2] for i in range(len(yplot))], 'b--', label="Default Good")
    #     # plt.plot(range(1, x_iter + 1), [yplot[i][agent_index][-1] for i in range(len(yplot))], 'r-.', label="Dropout Good")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.xlabel('x_iter')
    # plt.title("Y-values")
    # plt.savefig(get_filename("y-values"), bbox_inches='tight')
    # plt.close()


    # # Social Welfare
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(1, x_iter + 1), social_welfare)
    # plt.xlabel('x_iter')
    # plt.ylabel('Social Welfare')
    # plt.title("Social Welfare")
    # plt.savefig(get_filename("social_welfare"))
    # plt.close()

    # # Rebate error
    # plt.figure(figsize=(10, 5))
    # # print(rebates)
    # # print(f"Rebate frequency: {lambda_frequency}")
    # rebate_error = [[rebates[i][j][0] - rebates[i - i % int(lambda_frequency)][j][0] for j in range(len(rebates[0]))] for i in range(len(rebates))]
    # plt.plot(range(1, x_iter + 1), rebate_error)
    # plt.xlabel('x_iter')
    # plt.ylabel('Rebate error')
    # plt.title("Rebate error")
    # plt.savefig(get_filename("rebate_error"))
    # plt.close()

def plot_utility_functions(agents_data_dict, market_data_dict, output_folder):
    agent_names = []
    max_utilities = []
    end_utilities = []
    ascending_utilities = []  # Placeholder for Ascending Auction utilities
    total_max_utility = 0
    total_fisher_utility = 0
    
    for agent_name, agent in agents_data_dict.items():
        agent_names.append(agent_name)
        agent_fu, agent_max_fu = compute_utilites(agent)
        max_utilities.append(agent_max_fu)
        end_utilities.append(agent_fu)
        total_max_utility += agent_max_fu
        total_fisher_utility += agent_fu
        ascending_utilities.append(agent_fu - 30)  # Placeholder logic for ascending auction
    
    market_data_dict['total_max_utility'] = total_max_utility
    market_data_dict['total_fisher_utility'] = total_fisher_utility

    bar_width = 0.25  # Narrower bar width since we have three bars
    indices = np.arange(len(agent_names))

    plt.figure(figsize=(12, 8))

    # Plot Max Utility (light blue)
    plt.bar(indices - bar_width, max_utilities, bar_width, 
            color='lightblue', label='Max Utility')

    # Plot Fisher Market (blue)
    plt.bar(indices, end_utilities, bar_width, 
            color='blue', label='Fisher Market')

    # Plot Ascending Auction (light green)
    plt.bar(indices + bar_width, ascending_utilities, bar_width, 
            color='lightgreen', label='Ascending Auction')

    # Add labels, title, and legend
    plt.xlabel('Agents')
    plt.ylabel('Utility')
    plt.xticks(indices, agent_names, rotation=45)

    # Simplified color-coded legend
    plt.legend(loc='upper right')

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{output_folder}/plots/utility_distribution.png")
    plt.close()
    return market_data_dict