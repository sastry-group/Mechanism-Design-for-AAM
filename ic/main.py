"""
Incentive Compatible Fleet Level Strategic Allocation
"""

import argparse
import json
import sys
from pathlib import Path
import os

# Add the bluesky package to the path
top_level_path = Path(__file__).resolve().parent.parent
print(str(top_level_path))
sys.path.append(str(top_level_path))
import bluesky as bs


# Bluesky settings
T_STEP = 10000
MANEUVER = True
GUI = False
LOG = True
SIMDT = 1

parser = argparse.ArgumentParser(description='Process a true/false argument.')
parser.add_argument('--gui', action='store_true', help='Flag for running with gui.')
parser.add_argument('--file', type=str, required=True, help='The path to the test case json file.')
args = parser.parse_args()


def run_from_json(file = None):
    """
    Run the bluesky simulation from a JSON file.
    """
    if file is None:
        return None
    assert Path(file).is_file(), f"File {file} does not exist."
    
    # Load the JSON file
    with open(file, 'r') as f:
        data = json.load(f)
        print(f"Opened file {file}")
    return data


def create_scenario(data, scene_path, scene_name, run_gui = False):

    stack_commands = [] 
    init_settings = "00:00:00.00>TRAILS ON\n00:00:00.00>PAN OAK\n"
    stack_commands.append(init_settings)
    flights = data['flights'] # Let's get a name to refer to these vehicles
    vertiports = data['vertiports']


    for flight in flights:
        aircraft_id = flight['aircraft_id']
        origin_vertiport_id = flight['origin_vertiport_id']
        origin_vertiport = vertiports[origin_vertiport_id]
        appearance_time = flight['appearance_time']

        for request in flight['requests']:
            destination_vertiport_id = request['destination_vertiport_id']
            destination_vertiport = vertiports[destination_vertiport_id]
            request_departure_time = request['request_departure_time']
            or_lat = origin_vertiport['latitude']
            or_lon = origin_vertiport['longitude']
            des_lat = destination_vertiport['latitude']
            des_lon = destination_vertiport['longitude']
            type = 'B744' #placeholder
            alt = 'FL250'  # placeholder
            spd = 200  # placeholder
            hdg = 0  # placeholder
            time_stamp = convert_time(request_departure_time)
            stack_commands.append(f"{time_stamp}>CRE {aircraft_id} {type} {or_lat} {or_lon} {hdg} {alt} {spd}\n")
            stack_commands.append(f"{time_stamp}>DEST {aircraft_id} {des_lat}, {des_lon}\n")
            
        write_scenario(scene_path, scene_name, stack_commands)

    return


def evaluate_scenario(scn_file, run_gui=False):
        # Create the BlueSky simulation
    if not run_gui:
        bs.init(mode="sim", detached=True)
    else:
        bs.init(mode="sim")
        bs.net.connect()
    
    bs.stack.stack("IC " + scn_file)
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

def write_scenario(SCN_PATH, SCN_NAME, stack_commands):
    text = ''.join(stack_commands)
    
    # Create directory if it doesn't exist
    directory = f'{SCN_PATH}/{SCN_NAME}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Write the text to the scenario file
    with open(f'{directory}/{SCN_NAME}.scn', "w") as file:
        file.write(text)

              

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
    # python3 main.py --file ./../test_cases/case1.json
    file_name = args.file
    assert Path(file_name).is_file(), f"File {file_name} does not exist."
    data = run_from_json(file_name)
    # The two below will become args in bash file
    SCN_PATH ='/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/scenario'
    SCN_NAME = 'TEST_IC'
    scn_file = f'{SCN_PATH}/{SCN_NAME}.scn'
    create_scenario(data, SCN_PATH, SCN_NAME)
    if args.gui:
        # run_from_json(file_name, run_gui=True)
        # Always call as false because the gui does not currently work
        evaluate_scenario(scn_file, run_gui=False)
    else:
        evaluate_scenario(scn_file)