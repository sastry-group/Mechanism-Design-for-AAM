import sys
import os
from pathlib import Path
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json
from pathlib import Path
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console
from rich.table import Table
import logging
from rich.logging import RichHandler
import time
import traceback

logger = logging.getLogger("global_logger")

# Add the bluesky package to the path
top_level_path = Path(__file__).resolve().parent.parent
# print(str(top_level_path))
sys.path.append(str(top_level_path))

from VertiportStatus import VertiportStatus
from fisher.fisher_int_optimization import agent_allocation_selection, map_goodslist_to_agent_goods, track_delayed_goods
from fisher.FisherGraphBuilder import FisherGraphBuilder
from write_csv import write_output, save_data 
from utils import store_agent_data, rank_allocations, store_market_data, get_next_auction_data, build_edge_information
from market import construct_market, run_market
from ic.plotting_tools import plotting_market, plot_utility_functions

INTEGRAL_APPROACH = False
UPDATED_APPROACH = True
TOL_ERROR = 1e-3
MAX_NUM_ITERATIONS = 10000



def load_json(file=None):
    """
    Load a case file for a fisher market test case from a JSON file.
    """
    if file is None:
        return None
    assert Path(file).is_file(), f"File {file} does not exist."

    # Load the JSON file
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        # print(f"Opened file {file}")
        logger.info(f"Opened file {file}")
    return data



def track_desired_goods(flights, goods_list):
    "return the index of the desired goods for each flight"
    desired_goods = {}
    for i, flight_id in enumerate(flights.keys()):
        flight = flights[flight_id]
        appearance_time = flight["appearance_time"]
        desired_request = flight["requests"]["001"]
        origin_vertiport = flight["origin_vertiport_id"]
        desired_dep_time = desired_request["request_departure_time"]
        desired_vertiport = desired_request["destination_vertiport_id"]
        desired_arrival_time = desired_request["request_arrival_time"]
        desired_transit_edges = []
        desired_dep_edge = (f"{origin_vertiport}_{desired_dep_time}", f"{origin_vertiport}_{desired_dep_time}_dep")
        flights_desired_goods = [desired_dep_edge]
        for i in range(appearance_time, desired_dep_time):
            flights_desired_goods.append((f"{origin_vertiport}_{i}", f"{origin_vertiport}_{i+1}"))
        flights_desired_goods.append((f"{origin_vertiport}_{desired_dep_time}_dep", f"{desired_request['sector_path'][0]}_{desired_request['sector_times'][0]}"))
        desired_transit_edges.append((f"{origin_vertiport}_{desired_dep_time}_dep", f"{desired_request['sector_path'][0]}_{desired_request['sector_times'][0]}"))
        for i in range(len(desired_request["sector_path"])):
            sector = desired_request["sector_path"][i]
            start_time = desired_request["sector_times"][i]
            end_time = desired_request["sector_times"][i+1]
            for sector_time in range(start_time, end_time):
                flights_desired_goods.append((f"{sector}_{sector_time}", f"{sector}_{sector_time+1}"))

            if i < len(desired_request["sector_path"]) - 1:
                next_sector = desired_request["sector_path"][i+1]
                flights_desired_goods.append((f"{sector}_{end_time}", f"{next_sector}_{end_time}"))
        if desired_vertiport is not None:
            desired_arr_edge = (f"{sector}_{end_time}", f"{desired_vertiport}_{desired_arrival_time}_arr")
            flights_desired_goods.append((f"{sector}_{end_time}", f"{desired_vertiport}_{desired_arrival_time}_arr"))
            flights_desired_goods.append((f"{desired_vertiport}_{desired_arrival_time}_arr", f"{desired_vertiport}_{desired_arrival_time}"))
            # desired_good_dep_to_arr = (f"{origin_vertiport}_{desired_dep_time}_dep", f"{desired_vertiport}_{desired_arrival_time}_arr")
            # good_id_arr = goods_list.index(desired_good_arr)
            # good_id_dep_to_arr = goods_list.index(desired_good_dep_to_arr)
            # desired_goods[flight_id]["desired_good_dep_to_arr"] = good_id_dep_to_arr
        # good_id_dep = goods_list.index(desired_good_dep)
        # desired_goods[flight_id] = {"desired_good_arr": good_id_arr, "desired_good_dep": good_id_dep, "desired_good_dep_to_arr": good_id_dep_to_arr}
        # desired_goods[flight_id]["desired_good_dep"] = good_id_dep
        # print(f"Desired goods for flight {flight_id}: {flights_desired_goods}")
        index_list = []
        for good in flights_desired_goods:
            index_list.append(goods_list.index(good))
        desired_goods[flight_id] = {"good_indices": index_list, "desired_dep_edge_idx": goods_list.index(desired_dep_edge), 
                                    "desired_dep_edge": desired_dep_edge, "desired_paths": flights_desired_goods, 
                                    "desired_arr_edge": desired_arr_edge, "desired_transit_edges": desired_transit_edges, 
                                    "desired_arr_edge_idx": goods_list.index(desired_arr_edge), 
                                    "desired_transit_edges_idx": [goods_list.index(desired_transit_edges[i]) for i in range(len(desired_transit_edges))]}

    return desired_goods

def map_previous_prices(previous_price_data, new_goods_list):
    """
    Maps previous prices to new goods, ensuring only relevant prices are carried forward.

    Parameters:
    - previous_price_data (dict): Dictionary with {old_good: old_price}.
    - new_goods_list (list): List of goods in the new auction.

    Returns:
    - np.array: Updated price array for the new auction.
    """

    new_prices = np.zeros(len(new_goods_list)) 


    for i, good in enumerate(new_goods_list):
        if good in previous_price_data:
            new_prices[i] = previous_price_data[good]

    return new_prices

def fisher_allocation_and_payment(vertiport_usage, flights, timing_info, sectors_data, vertiports, overcapacitated_goods,
                                  output_folder=None, save_file=None, initial_allocation=True, design_parameters=None, previous_price_data=None, auction=1,
                                  tol_error_to_check=None):

    logger = logging.getLogger("global_logger")
    logger.info("Starting Fisher Allocation and Payment Process.")
    start_market_time = time.time()

    market_auction_time=timing_info["auction_start"]
    price_default_good = design_parameters["price_default_good"]
    default_good_valuation = design_parameters["default_good_valuation"]
    dropout_good_valuation = design_parameters["dropout_good_valuation"]
    BETA = design_parameters["beta"]
    # BETA = design_parameters["beta"] * (round(market_auction_time / 20) + 1)
    lambda_frequency = design_parameters["lambda_frequency"]
    price_upper_bound = design_parameters["price_upper_bound"]
    save_pkl_files = design_parameters["save_pkl_files"]       
    
    logger.info("Constructing market...")
    agent_information, market_information, bookkeeping, updated_flight_info = construct_market(flights, timing_info, sectors_data, vertiport_usage, output_folder,
                                                                          overcapacitated_goods,
                                                                          default_good_valuation=default_good_valuation, 
                                                                          dropout_good_valuation=dropout_good_valuation, BETA=BETA)
    
    # Run market
    updated_flights, flights_to_rebase = updated_flight_info
    goods_list = bookkeeping
    num_goods, num_agents = len(goods_list), len(updated_flights)
    u, agent_constraints, agent_goods_lists = agent_information
    _ , capacity, _ = market_information
    agent_indices = map_goodslist_to_agent_goods(goods_list, agent_goods_lists)
    agent_information = (*agent_information, agent_indices)
    # Sparse Representation
    sparse_agent_x_inds = []
    sparse_agent_y_inds = []
    x_start = 0
    y_start = 0
    y_agent_indices = []
    for inds in agent_indices:
        x_end = len(inds) + x_start
        y_end = len(inds) - 2 + y_start
        sparse_agent_x_inds.append(list(np.arange(x_start, x_end)))
        sparse_agent_y_inds.append(list(np.arange(y_start, y_end)))
        y_agent_indices.append(inds[:-2])
        x_start = x_end
        y_start = y_end
    y_sparse_array = np.concatenate(y_agent_indices)
    x_sparse_array = np.concatenate(agent_indices)
    y_sum_matrix = np.array([[1 if elem == i else 0 for elem in y_sparse_array] for i in range(num_goods - 2)])

    # y = np.random.rand(num_agents, num_goods-2)*10
    y = np.zeros((num_agents, num_goods - 2))
    dense_y = np.zeros((num_agents, num_goods - 2))
    # y = np.zeros((len(sparse_goods_representation), 1))
    desired_goods = track_desired_goods(updated_flights, goods_list)
    for i, agent_id in enumerate(desired_goods):
        # dept_id = desired_goods[agent_ids]["desired_good_arr"]
        # arr_id = desired_goods[agent_ids]["desired_good_dep"] 
        # dept_to_arr_id = desired_goods[agent_ids]["desired_good_dep_to_arr"]
        # y[i][dept_id]= 1
        # y[i][arr_id] = 1
        # y[i][desired_goods[agent_id]["good_indices"][0]] = 1
        for good_idx in desired_goods[agent_id]["good_indices"]:
            y[i][good_idx] = 1
            dense_y[i][good_idx] = 1
        # print(f"Initial allocation for agent {i}: {y[i]}")
        # print(f"Goods: {agent_goods_lists[i]}")
        # ybar = np.array([y[i, goods_list.index(good)] for good in agent_goods_lists[i][:-2]] + [0,0])
        # print(f"y bar: {ybar}")
        # agent_goods = agent_goods_lists[i]
        # print(f"Agent goods: {agent_goods}")
        # print(f"ybar is 1: {[agent_goods[ind] for ind in np.where(ybar == 1)[0]]}")
        # print(f"Ax - b: {agent_constraints[i][0] @ ybar - agent_constraints[i][1]}")
        # invalid_constraints = np.where(agent_constraints[i][0] @ ybar - agent_constraints[i][1] == 1)
        # print(f"A: {agent_constraints[i][0]}")
        # assert all(agent_constraints[i][0] @ ybar - agent_constraints[i][1] == 0), f"Initial allocation for agent {i} does not satisfy constraints for agent {i}"
    # y = np.random.rand(num_agents, num_goods)
    # start the prices witht the preiovus prices 
    # remove them overcapacity 
    y = np.concatenate([[agent_y[ind] for ind in y_sparse_array[inds]] for agent_y, inds in zip(dense_y, sparse_agent_y_inds)])

    if market_auction_time == 0 or previous_price_data is None:
        p = np.zeros(num_goods)
    else:
        p =  map_previous_prices(previous_price_data, goods_list)

    p[-2] = price_default_good 
    p[-1] = 0 # dropout good
    # r = [np.zeros(len(agent_constraints[i][1])) for i in range(num_agents)]
    r = np.zeros(num_agents)
    # x, p, r, overdemand = run_market((y,p,r), agent_information, market_information, bookkeeping, plotting=True, rational=False)
    logger.info("Running market...")


    try:
        x, prices, r, overdemand, agent_constraints, adjusted_budgets, data_to_plot = run_market((y,p,r), agent_information, market_information, 
                                                                bookkeeping, (x_sparse_array, y_sparse_array, sparse_agent_x_inds, sparse_agent_y_inds, y_sum_matrix),
                                                                rational=False, price_default_good=price_default_good, 
                                                                lambda_frequency=lambda_frequency, price_upper_bound=price_upper_bound, auction=auction, tol_error_to_check=tol_error_to_check)
    except Exception as e:
        logger.error(f"Error in run_market at auction time {market_auction_time}:\n{traceback.format_exc()}")
        return None, None, None, None, None  # Avoid crashing, return safe values

    logger.info("Market run complete.")
    end_fisher_time =  time.time() - start_market_time
    console = Console(force_terminal=True)
    console.print(f"[bold green]Fisher Algorithm runtime {end_fisher_time}...[/bold green]")

    # print("---FINAL ALLOCATION---")
    # logger.debug("---FINAL ALLOCATION---")
    # for agent_x, desired_good in zip(x, desired_goods):
    #     # print(f"Agent allocation of goods: {[goods_list[i] for i in np.where(x[0] > 0.1)[0]]}")
    #     # print(f"Partially allocated good values: {[x[0][i] for i in np.where(x[0] > 0.1)[0]]}")
    #     logger.debug(f"Agent allocation of goods: {[goods_list[i] for i in np.where(x[0] > 0.1)[0]]}")
    #     logger.debug(f"Partially allocated good values: {[x[0][i] for i in np.where(x[0]> 0.1)[0]]}")
    # print(f"Partial allocation for 0th agent: {[goods_list[i] for i in np.where(x[0] > 0.1)[0]]}")
    # print(f"Partial allocation for 1st agent: {[goods_list[i] for i in np.where(x[1] > 0.1)[0]]}")
    # print(f"Partially allocated good values: {[x[0][i] for i in np.where(x[0] > 0.1)[0]]}")
    # print(f"Partially allocated good values: {[x[1][i] for i in np.where(x[1] > 0.1)[0]]}")
    # print(f"Full agent allocations: {x[2][-2]}")
    # print(f"Goods list: {goods_list}")
    # print(f"{np.where(np.array(goods_list) == ('S001_37', 'S002_37'))}")
    # print(f"Weird allocation: {[x[2][ind] for ind in np.where(goods_list == ('S001_37', 'S002_37'))[0]]}")
    # print(f"Weird allocation: {[x[2][ind] for ind in np.where(goods_list == ('S002_37', 'S002_38'))[0]]}")
        # allocated_goods = [agent_goods_list[i] for i in np.where(row > 0.9)[0]]
        # print(f"{row}")
    # print(f"Final allocation: {x}")
    logger.info("Processing results...")
    extra_data = {
    'x_prob': x,
    'prices': prices,
    'rebates': r,
    'agent_constraints': agent_constraints,
    'adjusted_budgets': adjusted_budgets,
    'desired_goods': desired_goods,
    'goods_list': goods_list,
    'capacity': capacity,
    'data_to_plot': data_to_plot,
    'agent_goods_lists': agent_goods_lists,
    'num_agents': num_agents,
    'num_goods': num_goods,
    }

    if save_pkl_files:
        save_data(output_folder, "fisher_data", market_auction_time, **extra_data)
    # save_data(output_folder, "fisher_data", market_auction_time, **extra_data)
    plotting_market(data_to_plot, desired_goods, output_folder, market_auction_time)
    
    # Building edge information for mapping - move this to separate function
    # move this part to a different function
    edge_information = build_edge_information(goods_list)


    agents_data_dict = store_agent_data(updated_flights, x, agent_information, adjusted_budgets, desired_goods, agent_goods_lists, edge_information)
    market_data_dict = store_market_data(extra_data, design_parameters, market_auction_time)
    price_map = {goods_list[i]: prices[i] for i in range(len(goods_list)) if prices[i] > 0.01}
    agents_data_dict = track_delayed_goods(agents_data_dict, market_data_dict)
    # Rank agents based on their allocation and settling any contested goods
    logger.info("Ranking allocations")
    sorted_agent_dict, ranked_list, market_data_dict = rank_allocations(agents_data_dict, market_data_dict)
    logger.info("Processing allocations")
    agents_data_dict, market_data_dict= agent_allocation_selection(ranked_list, sorted_agent_dict, agents_data_dict, market_data_dict)
    valuations = {key: agents_data_dict[key]["valuation"] for key in agents_data_dict.keys()}

    overcapacitated_goods = [good for good, cap in zip(market_data_dict["goods_list"], market_data_dict["capacity"]) if cap == 0]
    market_data_dict.setdefault("overcapacitated_goods", []).append(overcapacitated_goods)
    # Getting data for next auction
    logger.info("Getting next auction data")
    allocation, rebased, dropped, = get_next_auction_data(agents_data_dict, market_data_dict)

    console.print(f"[bold green] Algorithm 1 & 2 runtime {time.time() - start_market_time}...[/bold green]")

    market_data_dict = plot_utility_functions(agents_data_dict, market_data_dict, output_folder)

    # print(f"Allocation: {allocation}")

    output_data = {"market_data":market_data_dict, "agents_data":agents_data_dict, "ranked_list":ranked_list, "valuations":valuations}
    
    if save_pkl_files:
        save_data(output_folder, "fisher_data_after", market_auction_time, **output_data)


    write_output(updated_flights, edge_information, market_data_dict, 
                agents_data_dict, market_auction_time, output_folder)

    return allocation, rebased, dropped, valuations, price_map, overcapacitated_goods
    



if __name__ == "__main__":
    pass

