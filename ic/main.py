"""
Incentive Compatible Fleet Level Strategic Allocation
"""

import argparse
import json
import os
import sys
from pathlib import Path
import time
import math
import numpy as np
import re
import copy
from datetime import datetime
from logging_config import setup_logger
import logging
import traceback

def initialize_logger(log_folder):
    logger = setup_logger(
        logger_name="global_logger",
        log_file_name="simulation_log.txt",
        log_folder=log_folder,
        log_level=logging.DEBUG  # Capture DEBUG and above levels
    )
    logger.info("Logger initialized successfully.")
    return logger


# Add the bluesky package to the path
top_level_path = Path(__file__).resolve().parent.parent
# print(str(top_level_path))
sys.path.append(str(top_level_path))

import bluesky as bs
from ic.VertiportStatus import VertiportStatus, draw_graph
#from ic.allocation import allocation_and_payment
from ic.fisher.fisher_allocation import fisher_allocation_and_payment
from ic.ascending_auc.asc_auc_allocation import  ascending_auc_allocation_and_payment
from ic.write_csv import write_market_interval
from ic.vcg_allocation import vcg_allocation_and_payment
from ic.ff_allocation import ff_allocation_and_payment

# Bluesky settings
T_STEP = 10000
MANEUVER = True
VISUALIZE = False
LOG = True
SIMDT = 1
EQUITABLE_FLEETS = True


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    elif value.lower() in ("false", "0", "no"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")
    
parser = argparse.ArgumentParser(description="Process a true/false argument.")
parser.add_argument("--gui", action="store_true", help="Flag for running with gui.")
parser.add_argument("--output_bsky", type=str2bool, nargs="?", const=True, default=False,
                    help="Set to True to output Bluesky scenario file.")
parser.add_argument(
    "--file", type=str, required=True, help="The path to the test case json file."
)
parser.add_argument(
    "--scn_folder", type=str, help="The folder in which scenario files are saved."
)
parser.add_argument(
    "--force_overwrite",
    action="store_true",
    help="Flag for overwriting the scenario file(s).",
)
parser.add_argument(
    "--method", type=str, help="The method for allocation and payment (vcg or ff or ascending auction).."
)

##### Running from configurations

# parser = argparse.ArgumentParser()
# parser.add_argument('--file', type=str, required=True)
# parser.add_argument('--method', type=str, default='fisher')
# parser.add_argument('--force_overwrite', action='store_true')  
parser.add_argument('--BETA', type=float, default=1)
parser.add_argument('--dropout_good_valuation', type=float, default=1)
parser.add_argument('--default_good_valuation', type=float, default=1)
parser.add_argument('--price_default_good', type=float, default=10)
parser.add_argument('--lambda_frequency', type=float, default=1)
parser.add_argument('--price_upper_bound', type=float, default=50)
parser.add_argument('--num_agents_to_run', type=int, default=None)
parser.add_argument('--run_up_to_auction', type=float, default=10)
parser.add_argument('--use_AADMM', type=str2bool, nargs="?", const=True, default=False)
parser.add_argument('--save_pkl_files', type=str2bool, nargs="?", const=True, default=True)
parser.add_argument("--tol_error_to_check", nargs="+", type=float, default=None, help="List of tolerances for experiments")
parser.add_argument(
    "--beta_adjustment_method",
    type=str,
    choices=["none", "errorbased", "excessdemand", "normalizederror", "pidcontrol", "adjustedlearning"],
    default="none",
    help="Method to adjust beta dynamically. Options: none, errorbased, excessdemand, normalizederror, pidcontrol, adjustedlearning."
)
parser.add_argument("--alpha", type=float, default=1, help="Alpha value for tolerance.")
args = parser.parse_args()


def load_json(file=None):
    """
    Load a case file for a bluesky simulation from a JSON file.
    """
    if file is None:
        return None
    assert Path(file).is_file(), f"File {file} does not exist."

    # Load the JSON file
    
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        logger.info(f"Opened file {file}")
        # print(f"Opened file {file}")
    return data


def get_vehicle_info(flight, lat1, lon1, lat2, lon2):
    """
    Get the vehicle information for a given flight.

    Args:
        flight (dict): The flight information.

    Returns:
        str: The vehicle type.
        str: The altitude.
        int: The speed.
        int: The heading.
    """
    # Assuming zero magnetic declination
    true_heading = calculate_bearing(lat1, lon1, lat2, lon2) % 360
    

    # Predefined placeholders as constants for now, the initial speed must be 0 to gete average of 90ish in flight
    return "B744", "FL250", 0, true_heading


def get_lat_lon(vertiport):
    """
    Get the latitude and longitude of a vertiport.

    Args:
        vertiport (dict): The vertiport information.

    Returns:
        float: The latitude of the vertiport.
        float: The longitude of the vertiport.
    """
    return vertiport["latitude"], vertiport["longitude"]

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the initial bearing between two points 
    to determine the orientation of the strategic region
    with respect to the trajectory.

    input: lat1, lon1, lat2, lon2
    output: initial bearing
    """
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    
    delta_lon = lon2 - lon1
    
    y = math.sin(delta_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    
    initial_bearing = math.atan2(y, x)
    
    initial_bearing = math.degrees(initial_bearing)
    initial_bearing = (initial_bearing + 360) % 360
    
    return initial_bearing

def create_allocated_area(lat1, lon1, lat2, lon2, width):
    """
    Create a rectangular shape surrounding a trajectory given two points and width.

    input: lat1, lon1, lat2, lon2, width (in kilometers)
    ourtput: string of coordinates for the polygon
    """
    bearing = calculate_bearing(lat1, lon1, lat2, lon2)
    perpendicular_bearing = (bearing + 90) % 360
    
    # Convert width from kilometers to degrees (approximate)
    width_degrees = width / 111.32
    
    lat_delta = math.cos(math.radians(perpendicular_bearing)) * width_degrees
    lon_delta = math.sin(math.radians(perpendicular_bearing)) * width_degrees
    
    lat3 = lat1 + lat_delta
    lon3 = lon1 + lon_delta
    lat4 = lat1 - lat_delta
    lon4 = lon1 - lon_delta
    lat5 = lat2 + lat_delta
    lon5 = lon2 + lon_delta
    lat6 = lat2 - lat_delta
    lon6 = lon2 - lon_delta
    
    poly_string = f"{lat3},{lon3},{lat4},{lon4},{lat6},{lon6},{lat5},{lon5},{lat3},{lon3}"
    
    return poly_string

def add_commands_for_flight(
    flight_id, flight, request, origin_vertiport, destination_vertiport, stack_commands
):
    """
    Add the necessary stack commands for a given allocated request to the stack commands list.

    Args:
        flight_id (str): The flight ID.
        flight (dict): The flight information.
        request (dict): The request information.
        origin_vertiport (dict): The origin vertiport information.
        destination_vertiport (dict): The destination vertiport information.
        stack_commands (list): The list of stack commands to add to.
    """
    # Get vertiport information
    or_lat, or_lon = get_lat_lon(origin_vertiport)
    des_lat, des_lon = get_lat_lon(destination_vertiport)

    # Get vehicle information
    veh_type, alt, spd, head = get_vehicle_info(flight, or_lat, or_lon, des_lat, des_lon)
    # print(request)

    # Timestamps
    time_stamp = convert_time(request["request_departure_time"]*60)
    arrival_time_stamp = convert_time(request["request_arrival_time"]*60)

    # Object name to represent the strategic deconfliction area
    poly_name = f"{flight_id}_AREA"
    strategic_area_string = create_allocated_area(or_lat, or_lon, des_lat, des_lon, 3)

    stack_commands.extend(
        [
            f"{time_stamp}>CRE {flight_id} {veh_type} {or_lat} {or_lon} {head} {alt} {spd}\n",
            f"{time_stamp}>DEST {flight_id} {des_lat}, {des_lon}\n",
            f"{time_stamp}>SCHEDULE {arrival_time_stamp}, DEL {flight_id}\n",
            # f"{time_stamp}>POLY {poly_name},{strategic_area_string}\n",
            # f"{time_stamp}>AREA, {poly_name}\n",
            # f"{time_stamp}>SCHEDULE {arrival_time_stamp}, DEL {poly_name}\n",
        ]
    
    )

def step_simulation(
    vertiport_usage, vertiports, flights, allocated_flights, stack_commands
):
    """
    Step the simulation forward based on the allocated flights.

    Args:
        vertiport_usage (VertiportStatus): The current status of the vertiports.
        vertiports (dict): The vertiports information.
        flights (dict): The flights information.
        allocated_flights (list): The list of allocated flights.
        stack_commands (list): The list of stack commands to add to.

    """
    if allocated_flights is None:
        return vertiport_usage
    
    for flight_id, request_id in allocated_flights:
        # Pull flight and allocated request
        flight = flights[flight_id]
        print(f"Request id: {request_id}")
        request = flight["requests"][request_id]

        # Move aircraft in VertiportStatus
        vertiport_usage.move_aircraft(flight["origin_vertiport_id"], request)

        # Add movement to stack commands
        origin_vertiport = vertiports[flight["origin_vertiport_id"]]
        destination_vertiport = vertiports[request["destination_vertiport_id"]]
        add_commands_for_flight(
            flight_id,
            flight,
            request,
            origin_vertiport,
            destination_vertiport,
            stack_commands,
        )

    return vertiport_usage

def step_simulation_delay(
    vertiport_usage, vertiports, flights, allocated_flights, stack_commands
):
    """
    Step the simulation forward based on the allocated flights.

    Args:
        vertiport_usage (VertiportStatus): The current status of the vertiports.
        vertiports (dict): The vertiports information.
        flights (dict): The flights information.
        allocated_flights (list): The list of allocated flights.
        stack_commands (list): The list of stack commands to add to.
    """
    if allocated_flights is None:
        return vertiport_usage
    
    for flight_id, request_id, delay, bid, d, a in allocated_flights:
        # Pull flight and allocated request
        flight = flights[flight_id]
        request = flight["requests"][request_id]

        request["request_departure_time"] += delay
        request["request_arrival_time"] += delay
        request["bid"] = bid

        # Move aircraft in VertiportStatus
        # vertiport_usage.move_aircraft(flight["origin_vertiport_id"], request)

        # Add movement to stack commands
        # origin_vertiport = vertiports[flight["origin_vertiport_id"]]
        # if request["destination_vertiport_id"] is None:
        #     continue
        # destination_vertiport = vertiports[request["destination_vertiport_id"]]
        # add_commands_for_flight(
        #     flight_id,
        #     flight,
        #     request,
        #     origin_vertiport,
        #     destination_vertiport,
        #     stack_commands,
        # )

    return vertiport_usage

def step_simulation_delay_fisher(
    vertiport_usage, vertiports, flights, allocated_flights, stack_commands, auction_period
):
    """
    Step the simulation forward based on the allocated flights.

    Args:
        vertiport_usage (VertiportStatus): The current status of the vertiports.
        vertiports (dict): The vertiports information.
        flights (dict): The flights information.
        allocated_flights (list): The list of allocated flights.
        stack_commands (list): The list of stack commands to add to.
        auction_period (int): The length of each auction
    """
    if allocated_flights is None:
        return vertiport_usage
    
    for flight_id, request in allocated_flights:
        # Pull flight and allocated request
        flight = flights[flight_id]

        # Move aircraft in VertiportStatus
        vertiport_usage.allocate_aircraft(flight["origin_vertiport_id"], flight, request, auction_period)
    
    return vertiport_usage


def adjust_interval_flights(allocated_flights, flights):
    """
    This function adjusts the allocated flights based on the departure time of the flight 
    and adds a new request to the flight dictionary to tracj on vertiport status.

    Args:
        allocated_flights (list): The list of allocated flights ids and their departure and arrival good tuple
        flights (dict): The flights information.

    Returns:
        list: The adjusted list of allocated flights, dictionary with f
    """
    adjusted_flights = []
    for i, (flight_id, dept_arr_tuple) in enumerate(allocated_flights):
        flight = flights[flight_id]
        allocated_dep_time = int(re.search(r'_(\d+)_', dept_arr_tuple[0]).group(1)) 
        allocated_arr_time = int(re.search(r'_(\d+)_', dept_arr_tuple[1]).group(1))
        requested_dep_time = flight["requests"]['001']['request_departure_time']
        if requested_dep_time == allocated_dep_time:
            adjusted_flights.append((flight_id, '001'))
        else:
            adjusted_flights.append((flight_id, '002'))
            # change the valuatin, create a new function for it
            decay = flights[flight_id]["decay_factor"]
            flight["requests"]['002'] = {
                "destination_vertiport_id": flight["requests"]['001']['destination_vertiport_id'],
                "request_departure_time": allocated_dep_time,
                "request_arrival_time": allocated_arr_time,
                'valuation': flight["requests"]['001']["valuation"] * decay**i
            }

    return adjusted_flights, flights

def adjust_rebased_flights(rebased_flights, flights, arrival_time, depart_time, max_time):
    for i, flight_id in enumerate(rebased_flights):
        if flights[flight_id]["rebase_count"] >= 2:
            flights[flight_id]["rebase_count"] += 1
            continue
        else:
            flights[flight_id]["appearance_time"] = arrival_time
            # print(f"Flight {flight_id} appearance time: {auction_start}")
            valuation = flights[flight_id]["requests"]['001']["valuation"]
            decay = flights[flight_id]["decay_factor"]
            travel_time = flights[flight_id]["requests"]['001']['request_arrival_time'] - flights[flight_id]["requests"]['001']['request_departure_time']
            new_requested_dep_time = depart_time
            delay = depart_time - flights[flight_id]["requests"]['001']['request_departure_time']
            new_sector_times = [sector_time + delay for sector_time in flights[flight_id]["requests"]['001']["sector_times"]]
            if new_requested_dep_time + travel_time > max_time:
                # Do not rebase if the flight will not arrive before the simulation ends
                continue
            flights[flight_id]["requests"]['001']['request_arrival_time'] = new_requested_dep_time + travel_time
            flights[flight_id]["requests"]['001']['request_departure_time'] = new_requested_dep_time
            flights[flight_id]["requests"]['001']["sector_times"] = new_sector_times
            flights[flight_id]['valuation']= valuation/2 #change decay
            flights[flight_id]['budget_constraint'] +=  flights[flight_id]['original_budget']
            flights[flight_id]["rebase_count"] += 1

    return flights




def create_output_folder(design_parameters, file_path, method, base_dir="ic/results"):
    """
    Creates a main folder for storing simulation outputs, including subfolders for logs, results, and plots.
    The folder name incorporates design parameters for easy identification.
    """
    file_name = file_path.split("/")[-1].split(".")[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_agents = design_parameters["num_agents_to_run"]
    if n_agents is None:
        n_agents = "all"
    folder_name = (
        f"{file_name}_{method}_b-{design_parameters['beta']}_"
        f"agents{n_agents}_"
        f"dval{design_parameters['dropout_good_valuation']}_"
        f"outval{design_parameters['default_good_valuation']}_"
        f"pout{design_parameters['price_default_good']}_"
        f"freq{design_parameters['lambda_frequency']}_"
        f"pbound{design_parameters['price_upper_bound']}_"
        f"beta-method-{design_parameters['beta_adjustment_method']}_"
        f"alpha-{design_parameters['alpha']}_"
        f"use_AADMM-{design_parameters['use_AADMM']}_"
        f"receding_{timestamp}"
    )
    main_output_folder = os.path.join(base_dir, folder_name)
    # Path(output_folder).mkdir(parents=True, exist_ok=True)
    os.makedirs(main_output_folder, exist_ok=True)
    subfolders = ["log", "results", "plots"]
    for subfolder in subfolders:
        os.makedirs(os.path.join(main_output_folder, subfolder), exist_ok=True)

    return main_output_folder

def run_scenario(data, scenario_path, scenario_name, output_folder, method, design_parameters=None, save_scenario = True, payment_calc = True):
    """
    Create and run a scenario based on the given data. Save it to the specified path.

    Args:
        data (dict): The data containing information about flights, vertiports, routes, timing, etc.
        scenario_path (str): The path where the scenario file will be saved.
        scenario_name (str): The name of the scenario file.
        method (str): Allocation and payment calculation method to use.

    Returns:
        str: The path to the created scenario file.
        results (list): List of tuples containing allocated flights, payments, valuation, and congestion costs.
    """
    # added by Gaby, creating save folder path


    # data = load_json(file_path)
    # 
    # output_folder = f"ic/results/{file_name}_{method}_b-{design_parameters['beta']}_dval{design_parameters['dropout_good_valuation']}_outval{design_parameters['default_good_valuation']}_pout{design_parameters['price_default_good']}_freq{design_parameters['lambda_frequency']}_pbound{design_parameters['price_upper_bound']}_receding_{time.time()}"
    logger = logging.getLogger("global_logger")
    logger.info(f"Starting scenario run with method: {method}")

    flights = data["flights"]
    vertiports = data["vertiports"]
    timing_info = data["timing_info"]
    auction_freq = timing_info["auction_frequency"]
    # routes_data = data["routes"]
    sectors_data = data["sectors"]

    fleets = data["fleets"]
    if method == "vcg":
        congestion_params = data["congestion_params"]

        def C(vertiport, q):
            """
            Congestion cost function for a vertiport.
            
            Args:
                vertiport (str): The vertiport id.
                q (int): The number of aircraft in the hold.
            """
            assert float(q).is_integer() and q >= 0 and q < len(congestion_params["C"]), "q must be a non-negative integer."
            return congestion_params["C"][q]
    
        #add delays for vcg
        max_delay = 10
        if(method == 'vcg'):
            for fl in data["flights"]:
                dr = {}
                for j, r in enumerate(data["flights"][fl]["requests"]):
                    if(r=='000'):
                        dr['000'] = copy.deepcopy(data["flights"][fl]["requests"][r])
                        dr['000']['delay'] = 0
                        continue
                    for i in range(max_delay + 1):
                        k = copy.deepcopy(data["flights"][fl]["requests"][r])
                        k["request_departure_time"] += i
                        k["request_arrival_time"] += i
                        k["bid"] *= pow(0.95,i)
                        k["valuation"] *= pow(0.95,i)
                        k["delay"] = i
                        id_ = str(1 + (j-1)*(max_delay+1) + i)
                        s = '0' * (3 - len(id_)) + str(id_)
                        dr[s] = k
                data["flights"][fl]["requests"] = dr

        congestion_info = {"lambda": congestion_params["lambda"], "C": C}
            # Add fleet weighting information to flights
        for fleet_id, fleet in fleets.items():
            for flight_id in fleet["members"]:
                if EQUITABLE_FLEETS:
                    flights[flight_id]["rho"] = fleet["rho"]
                else:
                    flights[flight_id]["rho"] = 1

    # # Create vertiport graph and add starting aircraft positions
    vertiport_usage = VertiportStatus(vertiports, sectors_data, timing_info)
    # vertiport_usage.add_aircraft(flights)


    start_time = timing_info["start_time"]
    end_time = timing_info["end_time"]
    auction_intervals = list(range(start_time, end_time, auction_freq))

    max_travel_time = 6
    last_auction =  end_time - max_travel_time - auction_freq


    

    # Sort arriving flights by appearance time
    ordered_flights = {}
    for flight_id, flight in flights.items():
        appearance_time = flight["appearance_time"]
        if appearance_time not in ordered_flights:
            ordered_flights[appearance_time] = [flight_id]
        else:
            ordered_flights[appearance_time].append(flight_id)

    # Initialize number of rebases to 0
    for flight_id, flight in flights.items():
        flight["rebase_count"] = 0

    max_travel_time = 6
    last_auction =  end_time - max(max_travel_time,auction_freq)
    auction_times = list(np.arange(0, last_auction+1, auction_freq))
    # print(f"Last auction: {last_auction}")
    logger.info(f"Last auction: {last_auction}")

    # Initialize stack commands
    stack_commands = ["00:00:00.00>TRAILS OFF\n00:00:00.00>PAN CCR\n00:00:00.00>ZOOM 1\n00:00:00.00>CDMETHOD STATEBASED\n00:00:00.00>DTMULT 30\n"]
    colors = ["blue", "red", "green", "yellow", "pink", "orange", "cyan", "magenta", "white"]
    for i, vertiport_id in enumerate(vertiports.keys()):
        vertiport_lat, vertiport_lon = get_lat_lon(vertiports[vertiport_id])
        stack_commands.append(f"00:00:00.00>CIRCLE {vertiport_id},{vertiport_lat},{vertiport_lon},1,\n") # in nm
        stack_commands.append(f"00:00:00.00>COLOR {vertiport_id} {colors[i]}\n") # in nm 

    simulation_start_time = time.time()
    initial_allocation = True
    rebased_flights = None

    # Iterate through each time flights appear
    results = []
    prev_auction_prices = None 
    overcapacitated_goods = []
    auction = 1
    logger.info(f"Auction times: {auction_times}")

    for prev_auction_time, auction_time in zip(auction_times[:-1], auction_times[1:]):
        # Get the current flights
        # current_flight_ids = ordered_flights[appearance_time]
        
        if prev_auction_time > design_parameters["run_up_to_auction"]:
            break

        # This is to ensure it doest not rebase the flights beyond simulation end time
        if rebased_flights and auction_time <= last_auction + 1:
        #    print("Rebasing flights")
            logger.info("Rebasing flights")
            flights = adjust_rebased_flights(rebased_flights, flights, prev_auction_time, auction_time, end_time)
        
        # Sort arriving flights by appearance time
        ordered_flights = {}
        for flight_id, flight in flights.items():
            appearance_time = flight["appearance_time"]
            if appearance_time not in ordered_flights:
                ordered_flights[appearance_time] = [flight_id]
            else:
                ordered_flights[appearance_time].append(flight_id)

        if design_parameters["num_agents_to_run"] is not None:
            num_agents = design_parameters["num_agents_to_run"]
            current_flight_ids = []
            relevant_appearances = []
            
            for appearance_time in sorted(ordered_flights.keys()):
                relevant_appearances.append(appearance_time)
                current_flight_ids.extend(ordered_flights[appearance_time])
                
                # Stop if we have enough agents
                if len(current_flight_ids) >= num_agents:
                    break

            current_flight_ids = current_flight_ids[:num_agents]
        else:         
            relevant_appearances = [key for key in ordered_flights.keys() if key >= prev_auction_time and key < auction_time]
            current_flight_ids = sum([ordered_flights[appearance_time] for appearance_time in relevant_appearances], [])
        


        # print("Current flight ids: ", current_flight_ids)
        
        logger.debug(f"Current flight ids: {current_flight_ids}")
        if len(current_flight_ids) == 0:
            continue
        current_flights = {
            flight_id: flights[flight_id] for flight_id in current_flight_ids
        }
        

        unique_vertiport_ids = set()
        interval_sectors = set()
        for flight in current_flights.values():
            origin = flight['origin_vertiport_id']
            unique_vertiport_ids.add(origin)
            # Assuming there's also a destination_vertiport_id in the flight data
            for request in flight['requests'].values():
                destination = request['destination_vertiport_id']
                unique_vertiport_ids.add(destination)
                if request["request_departure_time"] != 0 and request["request_departure_time"] != -1:
                    interval_sectors.update(request["sector_path"])
                # interval_routes.add((origin, destination))

        filtered_vertiports = {vid: vertiports[vid] for vid in unique_vertiport_ids}
        filtered_sectors = {sid: sectors_data[sid] for sid in interval_sectors}
    
        # Create vertiport graph and add starting aircraft positions
        # filtered_vertiport_usage = VertiportStatus(filtered_vertiports, filtered_sectors, timing_info)
        # filtered_vertiport_usage.add_aircraft(interval_flights)

        # print("Performing auction for interval: ", prev_auction_time, " to ", auction_time) 
        logger.info(f"Performing auction for interval: {prev_auction_time} to {auction_time}")
        write_market_interval(prev_auction_time, auction_time, current_flights, output_folder)

        if not current_flights:
            continue

        # print("Method: ", method)
        logger.info(f"Method: {method}")
        # Determine flight allocation and payment
        current_timing_info = {
            "start_time" : timing_info["start_time"],
            "current_time" : appearance_time,
            "end_time": timing_info["end_time"],
            "auction_start": prev_auction_time,
            "auction_end": auction_time,
            "auction_frequency": timing_info["auction_frequency"],
            "time_step": timing_info["time_step"]
        }
        if method == "vcg":

            allocated_flights, payments, sw = vcg_allocation_and_payment(
                vertiport_usage, current_flights, current_timing_info, congestion_info, fleets, save_file=scenario_name, initial_allocation=initial_allocation, payment_calc=payment_calc, save=save_scenario
            )
            # Update system status based on allocation
            #print(allocated_flights)

            # DUMP INTO JSON HERE

            vertiport_usage = step_simulation(
                vertiport_usage, vertiports, flights, allocated_flights, stack_commands
            )
        elif method == "fisher":
            tol_error_to_check = design_parameters["tol_error_to_check"]
            allocated_flights, rebased_flights, payments, valuations, prices, overcapacitated_goods = fisher_allocation_and_payment(
                vertiport_usage, current_flights, current_timing_info, filtered_sectors, filtered_vertiports, overcapacitated_goods,
                output_folder, save_file=scenario_name, initial_allocation=initial_allocation, 
                design_parameters=design_parameters, previous_price_data=prev_auction_prices, auction=auction,
                tol_error_to_check=tol_error_to_check)
            # print(f"Allocated flights: {allocated_flights}")
            # print(f"Rebased flights: {rebased_flights}")
            # print(f"Social welfare: {sum([val for val in valuations.values()])}")
            prev_auction_prices = prices
            logger.debug(f"Allocated flights: {allocated_flights}")
            logger.debug(f"Rebased flights: {rebased_flights}")
            # logger.debug(f"Social welfare: {sum([val for val in valuations.values()])}")
            allocated_requests = []
            for flight_id, allocated_dep in allocated_flights:
                dep_time = int(allocated_dep[0].split("_")[1])
                flight = current_flights[flight_id]
                allocated_request = None
                for delay in range(5):
                    if flight["requests"]["001"]["request_departure_time"] + delay != dep_time:
                        continue
                    allocated_request = flight["requests"]["001"]
                    allocated_request["request_departure_time"] += delay
                    allocated_request["request_arrival_time"] += delay
                    allocated_request["sector_times"] = [sector_time + delay for sector_time in allocated_request["sector_times"]]
                    allocated_request["valuation"] = allocated_request["valuation"]*flight["decay_factor"]**delay
                    allocated_requests.append((flight_id, allocated_request))
                    break
            assert len(allocated_requests) == len(allocated_flights), "Not all flight requests were read."
            # print(f"Allocated requests: {allocated_requests}")
            logger.debug(f"Allocated requests: {allocated_requests}")
            vertiport_usage = step_simulation_delay_fisher(
                vertiport_usage, vertiports, flights, allocated_requests, stack_commands, auction_freq
            )
        elif method == "ascending-auction-budgetbased":
            allocated_flights, rebased_flights, payments = ascending_auc_allocation_and_payment(
                    vertiport_usage, current_flights, current_timing_info, filtered_sectors, "budget",
                    save_file=scenario_name, initial_allocation=initial_allocation, design_parameters=design_parameters
                )
            #print(allocated_flights)
            #print(payments)

            allocated_requests = []
            for flight_id, allocated_dep in allocated_flights:
                dep_time = int(allocated_dep[0].split("_")[1])
                flight = current_flights[flight_id]
                allocated_request = None
                for delay in range(5):
                    if flight["requests"]["001"]["request_departure_time"] + delay != dep_time:
                        continue
                    allocated_request = flight["requests"]["001"]
                    allocated_request["request_departure_time"] += delay
                    allocated_request["request_arrival_time"] += delay
                    allocated_request["sector_times"] = [sector_time + delay for sector_time in allocated_request["sector_times"]]
                    allocated_request["valuation"] = allocated_request["valuation"]*flight["decay_factor"]**delay
                    allocated_requests.append((flight_id, allocated_request))
                    break
            assert len(allocated_requests) == len(allocated_flights), "Not all flight requests were read."
            print(f"Allocated requests: {allocated_requests}")
            vertiport_usage = step_simulation_delay_fisher(
                vertiport_usage, vertiports, flights, allocated_requests, stack_commands, auction_freq
            )

            # # print("ALLOCATED FLIGHTS")
            # logger.debug("ALLOCATED FLIGHTS")
            # for af in allocated_flights:
            #     # print("flight id: ", af[0], "request id: ", af[1]," delay: ", af[2],"value: ", af[3], )
            #     logger.debug(f"flight id: {af[0]}, request id: {af[1]}, delay: {af[2]}, value: {af[3]}")    
            # logger.debug("---------")
            # # print('---------')
            # # print(f"Social welfare: {[sum(af[3] for af in allocated_flights)]}")
            # logger.debug(f"Social welfare: {[sum(af[3] for af in allocated_flights)]}")
            # allocated_flights = [i[0:2] for i in allocated_flights]
        
        elif method == "ascending-auction-profitbased":
            allocated_flights, payments = ascending_auc_allocation_and_payment(
                    vertiport_usage, current_flights, current_timing_info, filtered_sectors, "profit",
                    save_file=scenario_name, initial_allocation=initial_allocation, design_parameters=design_parameters
                )
            #print(allocated_flights)
            #print(payments)

            vertiport_usage = step_simulation_delay(
                vertiport_usage, vertiports, flights, allocated_flights, stack_commands
            )

            # print("ALLOCATED FLIGHTS")
            for af in allocated_flights:
                print("flight id: ", af[0], "request id: ", af[1]," delay: ", af[2],"value: ", af[3], )
            # print('---------')
            # print(f"Social welfare: {[sum(af[3] for af in allocated_flights)]}")
            allocated_flights = [i[0:2] for i in allocated_flights]
        elif method == "ff":
            allocated_flights, payments = ff_allocation_and_payment(
                vertiport_usage, current_flights, current_timing_info, save_file=scenario_name, initial_allocation=initial_allocation
            )
            # System status updated as part of allocation
        if initial_allocation:
            initial_allocation = False

        simulation_end_time = time.time()
        elapsed_time = simulation_end_time - simulation_start_time
        # print(f"Elapsed time: {elapsed_time} seconds")

        # Evaluate the allocation
        
        valuation = False
        if valuation:
            allocated_valuation = {flight_id: current_flights[flight_id]["requests"][request_id]["valuation"] for flight_id, request_id in allocated_flights}
            parked_valuation = {flight_id: flight["requests"]["000"]["valuation"] for flight_id, flight in current_flights.items() if flight_id not in [flight_id for flight_id, _ in allocated_flights]}
            valuation = sum([current_flights[flight_id]["rho"] * val for flight_id, val in allocated_valuation.items()]) + sum([current_flights[flight_id]["rho"] * val for flight_id, val in parked_valuation.items()])


            #[print('Nodes: -----')]
            #for n in vertiport_usage.nodes:
            #    print("node: ", n)
            #    print("cost: ", C(vertiport_usage.nodes[n]["vertiport_id"], vertiport_usage.nodes[n]["hold_usage"]))
            #    print('----')
            congestion_costs = congestion_info["lambda"] * sum([C(vertiport_usage.nodes[node]["vertiport_id"], vertiport_usage.nodes[node]["hold_usage"]) for node in vertiport_usage.nodes])
            if method == "vcg":
                assert sw - (valuation - congestion_costs) <= 0.01, "Social welfare calculation incorrect."
            # print(f"Social welfare: {valuation - congestion_costs}")
            results.append((allocated_flights, payments, valuation, congestion_costs))
        auction += 1

    # Write the scenario to a file
    if save_scenario:
        path_to_written_file = write_scenario(scenario_path, scenario_name, stack_commands)
    else:
        path_to_written_file = None
    
    # print("AUCTION DONE")
    # print(results)

    # Visualize the graph
    #if VISUALIZE:
    #    draw_graph(filtered_vertiport_usage)

    return path_to_written_file, results


def evaluate_scenario(path_to_scenario_file, run_gui=False):
    """
    Evaluate the scenario by running the BlueSky simulation.

    Args:
        path_to_scenario_file (str): The path to the scenario file to run.
        run_gui (bool): Flag for running the simulation with the GUI (default is False)
    """
    # Create the BlueSky simulation
    if not run_gui:
        bs.init(mode="sim", detached=True)
    else:
        bs.init(mode="sim")
        bs.net.connect()

    bs.stack.stack("IC " + path_to_scenario_file)
    bs.stack.stack("DT 1; FF")
    # if LOG:
    #     bs.stack.stack(f"CRELOG rb 1")
    #     bs.stack.stack(f"rb  ADD id, lat, lon, alt, tas, vs, hdg")
    #     bs.stack.stack(f"rb  ON 1  ")
    #     bs.stack.stack(f"CRE {aircraft_id} {vehicle_type} {or_lat} {or_lon} {hdg} {alt} {spd}\n")
    # bs.stack.stack(f"DEST {aircraft_id} {des_lat}, {des_lon}")
    # bs.stack.stack("OP")


def convert_time(time):
    """
    Convert a time in seconds to a timestamp in the format HH:MM:SS.SS.

    Args:
        time (int): The time in seconds.
    """
    total_seconds = time

    # Calculate hours, minutes, and seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = (total_seconds % 3600) % 60

    timestamp = f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
    return timestamp
    # Create the BlueSky simulation
    # if not run_gui:
    #     bs.init(mode="sim", detached=True)
    # else:
    #     bs.init(mode="sim")
    #     bs.net.connect()


def write_scenario(scenario_folder, scenario_name, stack_commands):
    """
    Write the stack commands to a scenario file.

    Args:
        scenario_folder (str): The folder where the scenario file will be saved.
        scenario_name (str): The desired name of the scenario file.
        stack_commands (list): A list of stack commands to write to the scenario file.
    """
    text = "".join(stack_commands)

    # Create directory if it doesn't exist
    directory = f"{scenario_folder}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write the text to the scenario file
    path_to_file = f"{directory}/{scenario_name}.scn"
    with open(path_to_file, "w", encoding="utf-8") as file:
        file.write(text)

    return path_to_file



if __name__ == "__main__":
    # Example call:
    # python3 main.py --file /path/to/test_case.json
    # python3 ic/main.py --file test_cases/case1.json --scn_folder /scenario/TEST_IC

    
    # Extract design parameters
    # BETA = args.BETA
    # dropout_good_valuation = args.dropout_good_valuation
    # default_good_valuation = args.default_good_valuation
    # price_default_good = args.price_default_good
    # lambda_frequency = args.lambda_frequency
    # price_upper_bound = args.price_upper_bound
    design_parameters = {
        "beta": args.BETA,
        "dropout_good_valuation": args.dropout_good_valuation,
        "default_good_valuation": args.default_good_valuation,
        "price_default_good": args.price_default_good,
        "lambda_frequency": args.lambda_frequency,
        "price_upper_bound": args.price_upper_bound,
        "num_agents_to_run": args.num_agents_to_run,
        "run_up_to_auction": args.run_up_to_auction,
        "save_pkl_files": args.save_pkl_files,
        "tol_error_to_check": args.tol_error_to_check,
        "beta_adjustment_method": args.beta_adjustment_method,
        "use_AADMM": args.use_AADMM,
        "alpha": args.alpha

        }
    method = args.method    
    file_path = args.file 
    assert Path(file_path).is_file(), f"File at {file_path} does not exist."

    output_folder = create_output_folder(design_parameters, file_path, method)
    log_folder = os.path.join(output_folder, "log")
    logger = initialize_logger(log_folder)  # Initialize logger here


    test_case_data = load_json(file_path)
    file_name = Path(file_path).name

    
    # Create the scenario
    if args.scn_folder is not None:
        SCN_FOLDER = str(top_level_path) + args.scn_folder
    else:
        # print(str(top_level_path))
        SCN_FOLDER = str(top_level_path) + "/scenario/TEST_IC"
        # print(SCN_FOLDER)
    SCN_NAME = file_name.split(".")[0]
    path = f"{SCN_FOLDER}/{SCN_NAME}.scn"




    if os.path.exists(path):
        # Directly proceed if force overwrite is enabled; else, prompt the user
        if (not args.force_overwrite and input(
                "The scenario file already exists. Do you want to overwrite it? (y/n): "
            ).lower()
            != "y"
        ):
            logger.info("File not overwritten. Exiting...")
            sys.exit(0)

    # Run scenario and handle errors
    try:
        path_to_scn_file, results = run_scenario(test_case_data, SCN_FOLDER, SCN_NAME, output_folder, method, design_parameters)
        logger.info(f"Scenario file written to: {path_to_scn_file}")
    except Exception as e:
        logger.error(f"Error while running the scenario: {e}\n{traceback.format_exc()}")
        sys.exit()

    # BLUESKY SIM 
    if args.output_bsky:
        # run_from_json(file_path, run_gui=True)
        # Always call as false because the gui does not currently work
        if args.gui:
            evaluate_scenario(path_to_scn_file, run_gui=False)
        else:
            evaluate_scenario(path_to_scn_file)
