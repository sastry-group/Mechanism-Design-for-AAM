"""
Incentive Compatible Fleet Level Strategic Allocation
"""

import argparse
import json
import sys
from pathlib import Path

# Add the bluesky package to the path
top_level_path = Path(__file__).resolve().parent.parent
print(str(top_level_path))
sys.path.append(str(top_level_path))
import bluesky as bs
from ic.VertiportStatus import VertiportStatus

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
    # if not run_gui:
    #     bs.init(mode="sim", detached=True)
    # else:
    #     bs.init(mode="sim")
    #     bs.net.connect()

    # Create vertiport graph and add starting aircraft positions
    vertiport_usage = VertiportStatus(data["vertiports"], data["edges"], data["time_horizon"])
    vertiport_usage.add_aircraft(data["flights"])

    # Allocate all flights and move them
    for flight in data["flights"]:
        vertiport_usage.move_aircraft(flight["origin_vertiport_id"], flight["requests"][0])


if __name__ == "__main__":
    # Example call:
    # python3 main.py --file test_case.json
    file_name = args.file
    assert Path(file_name).is_file(), f"File {file_name} does not exist."
    if args.gui:
        # run_from_json(file_name, run_gui=True)
        # Always call as false because the gui does not currently work
        run_from_json(file_name, run_gui=False)
    else:
        run_from_json(file_name)