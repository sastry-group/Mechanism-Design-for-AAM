"""
Incentive Compatible Fleet Level Strategic Allocation
"""

import argparse
import json
import sys
from pathlib import Path
import time

# Add the bluesky package to the path
top_level_path = Path(__file__).resolve().parent.parent
print(str(top_level_path))
sys.path.append(str(top_level_path))
import bluesky as bs

parser = argparse.ArgumentParser(description='Process a true/false argument.')
parser.add_argument('--gui', action='store_true', help='Flag for running with gui.')
parser.add_argument('--file', type=str, required=True, help='The path to the test case json file.')
args = parser.parse_args()


def run_from_json(file = None, run_gui = False):
    """
    Run the bluesky simulation from a JSON file.
    """
    if file is None:
        return None
    assert Path(file).is_file(), f"File {file} does not exist."
    
    # Load the JSON file
    with open(file) as f:
        data = json.load(f)
        print(f"Opened file {file}")

    # Create the BlueSky simulation
    if not run_gui:
        bs.init(mode="sim", detached=True)
    else:
        bs.init(mode="sim")
        bs.net.connect()
    
    return data


def create_uav(data):
    flights = data['flights'] # Let's get a name to refer to these vehicles
    vertiports = data['vertiports']
    stack_commands = []
    for flight in flights:
        aircraft_id = flight['aircraft_id']
        origin_vertiport_id = flight['origin_vertiport_id']
        origin_vertiport = vertiports[origin_vertiport_id]
        appearance_time = flight['appearance_time']

    
        bs.stack.stack("PAN OAK")
        SCN_PATH ='/home/gaby/Documents/UCB/AAM/GIT/bluesky-IC/scenario'
        SCN_NAME = 'TEST_IC'
        bs.stack.stack(f"IC {SCN_PATH}/{SCN_NAME}")
        bs.stack.stack("DT 1; FF")
        # Generate stack commands for each request
        for request in flight['requests']:
            destination_vertiport_id = request['destination_vertiport_id']
            destination_vertiport = vertiports[destination_vertiport_id]
            request_departure_time = request['request_departure_time']
            origin_vertiport_id = request['destination_vertiport_id']
            origin_vertiport = vertiports[origin_vertiport_id]

            # Calculate values for stack command
            or_lat = origin_vertiport['latitude']
            or_lon = origin_vertiport['longitude']
            des_lat = destination_vertiport['latitude']
            des_lon = destination_vertiport['longitude']
            type = 'B744' #placeholder
            alt = 'FL250'  # placeholder
            spd = 200  # placeholder
            hdg = 0  # placeholder

            # Generate stack command for destination vertiport
            # print(f"CRE {aircraft_id}, {type},{lat}, {lon}, {hdg}, {alt}, {spd}\n")

            bs.stack.stack(f"CRE {aircraft_id} {type} {or_lat} {or_lon} {hdg} {alt} {spd}\n")
            bs.stack.stack(f"ADDWPT {aircraft_id} {des_lat}, {des_lon}")
            bs.stack.stack("TRAIL ON")
        
        bs.stack.stack("OP")

        
        # time.sleep(10)
        # bs.stack.stack("QUIT")




if __name__ == "__main__":
    # Example call:
    # python3 main.py --file /path/to/test_case.json
    # python3 main.py --file ./../test_cases/case1.json
    file_name = args.file
    assert Path(file_name).is_file(), f"File {file_name} does not exist."
    if args.gui:
        # run_from_json(file_name, run_gui=True)
        # Always call as false because the gui does not currently work
        data = run_from_json(file_name, run_gui=False)
    else:
        data = run_from_json(file_name)
        create_uav(data)