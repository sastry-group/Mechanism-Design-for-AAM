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


# Add the bluesky package to the path
top_level_path = Path(__file__).resolve().parent.parent
print(str(top_level_path))
sys.path.append(str(top_level_path))

import bluesky as bs
from ic.VertiportStatus import VertiportStatus, draw_graph
from ic.allocation import allocation_and_payment
from ic.fisher.fisher_allocation import fisher_allocation_and_payment

# Bluesky settings
T_STEP = 10000
MANEUVER = True
VISUALIZE = False
LOG = True
SIMDT = 1

parser = argparse.ArgumentParser(description="Process a true/false argument.")
parser.add_argument("--gui", action="store_true", help="Flag for running with gui.")
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
    "--method", type=str, help="The method used to allocate flights."
)
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
        print(f"Opened file {file}")
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
    

    # Predefined placeholders as constants for now
    return "B744", "FL250", 200, true_heading


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
    print(request)

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
            # f"{time_stamp}>SCHEDULE {arrival_time_stamp}, DEL {flight_id}\n",
            f"{time_stamp}>POLY {poly_name},{strategic_area_string}\n",
            f"{time_stamp}>AREA, {poly_name}\n",
            f"{time_stamp}>SCHEDULE {arrival_time_stamp}, DEL {poly_name}\n",
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
    for flight_id, request_id in allocated_flights:
        # Pull flight and allocated request
        flight = flights[flight_id]
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


def run_scenario(data, scenario_path, scenario_name, file_path, method="fisher"):
    """
    Create and run a scenario based on the given data. Save it to the specified path.

    Args:
        data (dict): The data containing information about flights, vertiports, routes, timing, etc.
        scenario_path (str): The path where the scenario file will be saved.
        scenario_name (str): The name of the scenario file.

    Returns:
        str: The path to the created scenario file.
    """
    # added by Gaby, creating save folder path

    file_name = file_path.split("/")[-1].split(".")[0]
    data = load_json(file_path)
    output_folder = f"ic/results/{file_name}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    flights = data["flights"]
    vertiports = data["vertiports"]
    timing_info = data["timing_info"]

    # Create vertiport graph and add starting aircraft positions
    vertiport_usage = VertiportStatus(vertiports, data["routes"], timing_info)
    vertiport_usage.add_aircraft(flights)

    # Sort arriving flights by appearance time
    ordered_flights = {}
    for flight_id, flight in flights.items():
        appearance_time = flight["appearance_time"]
        if appearance_time not in ordered_flights:
            ordered_flights[appearance_time] = [flight_id]
        else:
            ordered_flights[appearance_time].append(flight_id)

    # Initialize stack commands
    stack_commands = ["00:00:00.00>TRAILS ON\n00:00:00.00>PAN OAK\n00:00:00.00>ZOOM 1\n00:00:00.00>CDMETHOD STATEBASED\n00:00:00.00>DTMULT 30\n"]
    
    start_time = time.time()
    initial_allocation = True
    # Iterate through each time flights appear
    # I think we should change this to running the simulatin every n time steps and specify the
    # frequency of fisher market run 
    for appearance_time in sorted(ordered_flights.keys()):
        # Get the current flights
        current_flight_ids = ordered_flights[appearance_time]
        current_flights = {
            flight_id: flights[flight_id] for flight_id in current_flight_ids
        }

        # Determine flight allocation and payment
        current_timing_info = {
            "start_time" : timing_info["start_time"],
            "current_time" : appearance_time,
            "end_time": timing_info["end_time"],
            "time_step": timing_info["time_step"]
        }
        if method == "fisher":
            allocated_flights, payments = fisher_allocation_and_payment(
                vertiport_usage, current_flights, current_timing_info, output_folder, save_file=scenario_name, initial_allocation=initial_allocation
            )
        elif method == "vcg":
            allocated_flights, payments = allocation_and_payment(
                vertiport_usage, current_flights, current_timing_info, save_file=scenario_name, initial_allocation=initial_allocation
            )
        if initial_allocation:
            initial_allocation = False

        # Update system status based on allocation
        vertiport_usage = step_simulation(
            vertiport_usage, vertiports, flights, allocated_flights, stack_commands
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")

    # Write the scenario to a file
    path_to_written_file = write_scenario(scenario_path, scenario_name, stack_commands)

    # Visualize the graph
    if VISUALIZE:
        draw_graph(vertiport_usage)

    return path_to_written_file


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
    file_path = args.file
    assert Path(file_path).is_file(), f"File at {file_path} does not exist."
    test_case_data = load_json(file_path)
    file_name = Path(file_path).name

    # Create the scenario
    if args.scn_folder is not None:
        SCN_FOLDER = str(top_level_path) + args.scn_folder
    else:
        print(str(top_level_path))
        SCN_FOLDER = str(top_level_path) + "/scenario/TEST_IC"
        print(SCN_FOLDER)
    SCN_NAME = file_name.split(".")[0]
    path = f"{SCN_FOLDER}/{SCN_NAME}.scn"

    print(SCN_NAME)

    # Check if the path exists and if the user wants to overwrite
    if os.path.exists(path):
        # Directly proceed if force overwrite is enabled; else, prompt the user
        if (
            not args.force_overwrite
            and input(
                "The scenario file already exists. Do you want to overwrite it? (y/n): "
            ).lower()
            != "y"
        ):
            print("File not overwritten. Exiting...")
            sys.exit()

    # Create the scenario file and double check the correct path was used
    # run_scenario(data, scenario_path, scenario_name, file_path, method="fisher")
    path_to_scn_file = run_scenario(test_case_data, SCN_FOLDER, SCN_NAME, file_path, args.method)
    print(path_to_scn_file)
    assert path == path_to_scn_file, "An error occured while writing the scenario file."

    # Evaluate scenario
    if args.gui:
        # run_from_json(file_path, run_gui=True)
        # Always call as false because the gui does not currently work
        evaluate_scenario(path_to_scn_file, run_gui=False)
    else:
        evaluate_scenario(path_to_scn_file)
