"""
Incentive Compatible Fleet Level Strategic Allocation
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add the bluesky package to the path
top_level_path = Path(__file__).resolve().parent.parent
print(str(top_level_path))
sys.path.append(str(top_level_path))

import bluesky as bs
from ic.VertiportStatus import VertiportStatus, draw_graph
from ic.allocation import allocation_and_payment

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
parser.add_argument("--scn_name", type=str, help="The name of the scenario file.")
parser.add_argument(
    "--force_overwrite",
    action="store_true",
    help="Flag for overwriting the scenario file(s).",
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
    with open(file, "r", encoding='utf-8') as f:
        data = json.load(f)
        print(f"Opened file {file}")
    return data


def create_scenario(data, scenario_path, scenario_name):
    """
    Create and run a scenario based on the given data. Save it to the specified path.

    Args:
        data (dict): The data containing information about flights, vertiports, routes, timing, etc.
        scenario_path (str): The path where the scenario file will be saved.
        scenario_name (str): The name of the scenario file.

    Returns:
        str: The path to the created scenario file.

    """
    stack_commands = []
    init_settings = "00:00:00.00>TRAILS ON\n00:00:00.00>PAN OAK\n"
    stack_commands.append(init_settings)
    flights = data["flights"]  # Let's get a name to refer to these vehicles
    vertiports = data["vertiports"]

    # Create vertiport graph and add starting aircraft positions
    vertiport_usage = VertiportStatus(vertiports, data["routes"], data["timing_info"])
    vertiport_usage.add_aircraft(flights)

    # Determine allocation
    start_time = data["timing_info"]["start_time"]
    end_time = data["timing_info"]["end_time"]
    time_step = data["timing_info"]["time_step"]
    allocated_flights, payments = allocation_and_payment(
        vertiport_usage, flights, start_time, end_time, time_step
    )

    # Allocate all flights and move them
    for flight_id, request_id in allocated_flights:
        flight = flights[flight_id]
        request = flight["requests"][request_id]
        vertiport_usage.move_aircraft(flight["origin_vertiport_id"], request)

        origin_vertiport_id = flight["origin_vertiport_id"]
        origin_vertiport = vertiports[origin_vertiport_id]
        destination_vertiport_id = request["destination_vertiport_id"]
        destination_vertiport = vertiports[destination_vertiport_id]
        request_departure_time = request["request_departure_time"]
        or_lat = origin_vertiport["latitude"]
        or_lon = origin_vertiport["longitude"]
        des_lat = destination_vertiport["latitude"]
        des_lon = destination_vertiport["longitude"]
        type = "B744"  # placeholder
        alt = "FL250"  # placeholder
        spd = 200  # placeholder
        hdg = 0  # placeholder
        time_stamp = convert_time(request_departure_time)
        stack_commands.append(
            f"{time_stamp}>CRE {flight_id} {type} {or_lat} {or_lon} {hdg} {alt} {spd}\n"
        )
        stack_commands.append(f"{time_stamp}>DEST {flight_id} {des_lat}, {des_lon}\n")

    path_to_scn_file = write_scenario(scenario_path, scenario_name, stack_commands)

    # Visualize the graph
    if VISUALIZE:
        draw_graph(vertiport_usage)

    return path_to_scn_file


def evaluate_scenario(path_to_scenario_file, run_gui=False):
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
    #     bs.stack.stack(f"CRE {aircraft_id} {type} {or_lat} {or_lon} {hdg} {alt} {spd}\n")
    # bs.stack.stack(f"DEST {aircraft_id} {des_lat}, {des_lon}")
    # bs.stack.stack("OP")


def convert_time(time):
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
    text = "".join(stack_commands)

    # Create directory if it doesn't exist
    directory = f"{scenario_folder}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write the text to the scenario file
    path_to_scn_file = f"{directory}/{scenario_name}.scn"
    with open(path_to_scn_file, "w", encoding='utf-8') as file:
        file.write(text)

    return path_to_scn_file


# def create_uav(data):
#     flights = data['flights'] # Let's get a name to refer to these vehicles
#     vertiports = data['vertiports']
#     for flight in flights:
#         aircraft_id = flight['aircraft_id']
#         origin_vertiport_id = flight['origin_vertiport_id']
#         origin_vertiport = vertiports[origin_vertiport_id]
#         appearance_time = flight['appearance_time']
#     return


if __name__ == "__main__":
    # Example call:
    # python3 main.py --file /path/to/test_case.json
    # python3 ic/main.py --file test_cases/case1.json --scn_folder ./scenario/TEST_IC --scn_name test-ic
    file_name = args.file
    assert Path(file_name).is_file(), f"File {file_name} does not exist."
    test_case_data = load_json(file_name)

    # Create the scenario
    if args.scn_folder is not None:
        SCN_FOLDER = str(top_level_path) + args.scn_folder
    else:
        SCN_FOLDER = str(top_level_path) + "/scenario/TEST_IC"
        print(SCN_FOLDER)
    if args.scn_name is not None:
        SCN_NAME = args.scn_name
        if SCN_NAME.endswith(".scn"):
            SCN_NAME = SCN_NAME[:-4]
    else:
        SCN_NAME = "test-ic"
    path = f"{SCN_FOLDER}/{SCN_NAME}.scn"

    # Check if the path exists and if the user wants to overwrite
    if os.path.exists(path):
        if args.force_overwrite:
            OVERWRITE = "y"
        else:
            OVERWRITE = input(
                "The scenario file already exists. Do you want to overwrite it? (y/n): "
            )
        if OVERWRITE.lower() != "y":
            print("File not overwritten. Exiting...")
            sys.exit()

    # Create the scenario file and double check the correct path was used
    path_to_scn_file = create_scenario(test_case_data, SCN_FOLDER, SCN_NAME)
    assert path == path_to_scn_file, "An error occured while writing the scenario file."

    # Evaluate scenario
    if args.gui:
        # run_from_json(file_name, run_gui=True)
        # Always call as false because the gui does not currently work
        evaluate_scenario(path_to_scn_file, run_gui=False)
    else:
        evaluate_scenario(path_to_scn_file)
