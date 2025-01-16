import json
import random
from datetime import datetime, timedelta
import numpy as np
import sys
import os
import math
from math import radians, sin, cos, sqrt, atan2

# Simulation time settings
START_TIME = 1 # multiple of timestep
END_TIME = 100
TIME_STEP = 1
AUCTION_DT = 10 # AUCTION FREQUENCY
SPEED = 90 # knots

# Case study settings
# N_FLIGHTS = random.randint(50, 60)
N_FLIGHTS = 6
NUM_FLEETS = 2

# change the request 000 for always be 0 - done
# routes must match travel time, arrival time not random, match the travel time + startime

# List of vertiports
# Project data: https://earth.google.com/earth/d/1bqXr8pgmjtshu5UKfT1zkq092Af36bQ0?usp=sharing
# V001: UCSF Medical Center Helipad
# V002: Helipad Hospital, Oakland
# V003: Pyron Heli Pad, SF
# V004: San Rafael Private Heliport
# V005: Santa Clara Towers Heliport
# V006: Random Flat location around Perscadero, Big Sur
# V007: Random Flat Location in Sacramento 
# Eventually we could extend the functionaility to read the .kml file from google earth to get the coordinates
vertiports = {
    "V001": {"latitude": 37.766699, "longitude": -122.3903664, "landing_capacity": random.randint(1, 3), "takeoff_capacity": random.randint(1, 5), "hold_capacity": 8},
    "V002": {"latitude": 37.8361761, "longitude": -122.2668028, "landing_capacity": random.randint(1, 3), "takeoff_capacity": random.randint(1, 5), "hold_capacity": 8},
    "V003": {"latitude": 37.7835538, "longitude": -122.5067642, "landing_capacity": random.randint(1, 3), "takeoff_capacity": random.randint(1, 5), "hold_capacity": 8},
    "V004": {"latitude": 37.9472484, "longitude": -122.4880737, "landing_capacity": random.randint(1, 3), "takeoff_capacity": random.randint(1, 5), "hold_capacity": 8},
    "V005": {"latitude": 37.38556649999999, "longitude": -121.9723564, "landing_capacity": random.randint(1, 3), "takeoff_capacity": random.randint(1, 5), "hold_capacity": 8},
    "V006": {"latitude": 37.25214395119753, "longitude": -122.4066509403772, "landing_capacity": random.randint(1, 3), "takeoff_capacity": random.randint(1, 5), "hold_capacity": 8},
    "V007": {"latitude": 38.58856301092047, "longitude": -121.5627454937505, "landing_capacity": random.randint(1, 3), "takeoff_capacity": random.randint(1, 5), "hold_capacity": 8},
}



total_capacity = sum(vertiport["hold_capacity"] for vertiport in vertiports.values())
# Assert that total holding capacity is greater than or equal to the number of flights
assert total_capacity >= N_FLIGHTS, f"Total holding capacity ({total_capacity}) must be greater than or equal to the number of flights ({N_FLIGHTS})"


def calculate_sector_times(origin, destination, sector_path, request_departure_time, request_arrival_time):
    """
    Calculate the times at which a flight will cross each sector along its path, based on the speed.

    Args:
    - origin (dict): Origin vertiport data with latitude and longitude.
    - destination (dict): Destination vertiport data with latitude and longitude.
    - sector_path (list): Ordered list of sectors along the path.
    - request_departure_time (int): Departure time of the flight.

    Returns:
    - list: Times at which the flight crosses each sector.
    """
    # sector_distances = [calculate_distance(origin, sectors[sector_path[0]])]
    sector_distances = []

    for i in range(len(sector_path) - 1):
        sector_distances.append(calculate_distance(sectors[sector_path[i]], sectors[sector_path[i + 1]]))

    sector_distances.append(calculate_distance(sectors[sector_path[-1]], destination))


    sector_times = [request_departure_time]

    for distance in sector_distances:
        if distance == 0:
            travel_time = 0
        else:
            # assuming a 10 km radius for the sector
            sector_entrance_distance = distance - 10 
            travel_time = math.ceil(sector_entrance_distance  * 0.5399568 * 60  / SPEED)
        sector_times.append(sector_times[-1] + travel_time)
    
    sector_times = sector_times[:-1] # Exclude the last element (arrival at destination)
    sector_times.append(request_arrival_time)

    return sector_times  

def get_sectors_along_path(origin_id, destination_id, vertiports, proximity_miles=3):
    """
    Identify sectors along the path between origin and destination vertiports based on proximity.
    
    Args:
    - origin_id (str): Origin vertiport ID.
    - destination_id (str): Destination vertiport ID.
    - vertiports (dict): Dictionary of vertiports with their latitude and longitude.
    - proximity_miles (float): Proximity threshold in miles to consider a sector along the path.
    
    Returns:
    - list: Ordered list of sectors along the path.
    """
    origin = vertiports[origin_id]
    destination = vertiports[destination_id]
    sectors_along_path = []

    for sector_id, sector_data in sectors.items():
        # Calculate distance from the sector to the path endpoints
        distance_to_origin = calculate_distance(origin, sector_data) * 0.621371  # Convert km to miles
        distance_to_destination = calculate_distance(destination, sector_data) * 0.621371  # Convert km to miles

        # If the sector is within proximity, add it to the path
        if distance_to_origin <= proximity_miles or distance_to_destination <= proximity_miles:
            sectors_along_path.append(sector_id)

    # Ensure sectors are ordered based on their proximity to the origin
    sectors_along_path = sorted(sectors_along_path, key=lambda s: calculate_distance(origin, sectors[s]))
    return sectors_along_path

def generate_flights():
    flights = {}
    vertiports_list = list(vertiports.keys())
    allowed_origin_vertiport = [
        vertiport_id
        for vertiport_id in vertiports_list
        for _ in range(vertiports[vertiport_id]["hold_capacity"])
    ]
    routes = generate_routes(vertiports)
    route_dict = {
        (route["origin_vertiport_id"], route["destination_vertiport_id"]): route["travel_time"]
        for route in routes
    }

    max_travel_time = route_dict[max(route_dict, key=route_dict.get)]
    last_auction = END_TIME - max_travel_time - AUCTION_DT
    auction_intervals = list(range(START_TIME, END_TIME, AUCTION_DT))

    for i in range(N_FLIGHTS):
        flight_id = f"AC{i + 1:03d}"
        auction_interval = random.choice(
            auction_intervals[: (np.abs(np.array(auction_intervals) - last_auction)).argmin()])
        
        appearance_time = random.randint(1, 9) # CHANGE FOR A LONGER RECEDING HORIZON

        origin_vertiport_id = random.choice(allowed_origin_vertiport)
        allowed_origin_vertiport.remove(origin_vertiport_id)

        destination_vertiport_id = random.choice(vertiports_list)
        while destination_vertiport_id == origin_vertiport_id:
            destination_vertiport_id = random.choice(vertiports_list)

        # request_departure_time = random.randint(
        #     auction_interval + AUCTION_DT, auction_interval + 2 * AUCTION_DT)
        
        request_departure_time = random.randint(appearance_time +   AUCTION_DT, appearance_time + 2 * AUCTION_DT)
        
        travel_time = route_dict.get((origin_vertiport_id, destination_vertiport_id), None)
        request_arrival_time = request_departure_time + travel_time

        valuation = random.randint(100, 200)
        budget_constraint = random.randint(50, 200)

        # Determine the sectors along the path
        sector_path = get_sectors_along_path(
            origin_vertiport_id, destination_vertiport_id, vertiports
        )

        # Calculate sector times
        sector_times = calculate_sector_times(
            vertiports[origin_vertiport_id],
            vertiports[destination_vertiport_id],
            sector_path,
            request_departure_time, request_arrival_time)

        flight_info = {
            "appearance_time": appearance_time,
            "origin_vertiport_id": origin_vertiport_id,
            "budget_constraint": budget_constraint,
            "decay_factor": 0.95,
            "requests": {
                "000": {
                    "bid": 1,
                    "valuation": 1,
                    "request_departure_time": 0,
                    "request_arrival_time": 0,
                    "destination_vertiport_id": origin_vertiport_id,
                },
                "001": {
                    "bid": random.randint(100, 200),
                    "valuation": valuation,
                    "sector_path": sector_path,
                    "sector_times": sector_times,
                    "destination_vertiport_id": destination_vertiport_id,
                    "request_departure_time": request_departure_time,
                    "request_arrival_time": request_arrival_time,
                },
            },
        }
        flights[flight_id] = flight_info
    return flights, routes



def generate_sectors(vertiports, num_sectors_per_vertiport=1):
    """
    Generates sectors based on vertiports. Initially, each sector will have the same 
    latitude and longitude as its associated vertiport. This can be modified later for customization.
    
    Args:
    - vertiports (dict): Dictionary of vertiports with latitude and longitude.
    - num_sectors_per_vertiport (int): Number of sectors to create per vertiport.
    
    Returns:
    - dict: Dictionary of sectors.
    """
    sectors = {}
    sector_id = 1  # Unique sector ID
    for vertiport_id, vertiport_data in vertiports.items():
        for i in range(num_sectors_per_vertiport):
            sector_key = f"S{sector_id:03d}"
            sectors[sector_key] = {
                "latitude": vertiport_data["latitude"],  
                "longitude": vertiport_data["longitude"],  
                "hold_capacity": random.randint(5, 10) 
            }
            sector_id += 1
    return sectors



def calculate_distance(origin, destination):
    """
    This function calculates distance from two vertiports
    input:
    output: distance in km
    """
    R = 6371.0  # Radius of the Earth in kilometers

    # Convert latitude and longitude from degrees to radians
    lat1 = radians(origin["latitude"])
    lon1 = radians(origin["longitude"])
    lat2 = radians(destination["latitude"])
    lon2 = radians(destination["longitude"])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance




def generate_fleets(flights_data):
    fleets = {}
    # flights = list(generate_flights().keys())
    flights = flights_data
    # print(flights)
    random.shuffle(flights)
    split = len(flights) // NUM_FLEETS
    for i in range(NUM_FLEETS):
        fleet_id = f"F{i+1:03d}"
        fleet_flights = flights[i * split: (i + 1) * split]
        fleets[fleet_id] = fleet_flights
    return fleets

def generate_routes(vertiports):
    # Generate routes that connect all vertiports
    routes = []
    for origin_id, origin_data in vertiports.items():
        for destination_id, destination_data in vertiports.items():
            if origin_id != destination_id:
                # this could be bad code practice below to input unformatted data, might change later
                # this is also somthing that will be moved to each agent's bid
                distance = calculate_distance(origin_data, destination_data) # km
                travel_time = math.ceil(distance * 0.5399568 * 60  / SPEED)   # cover distance from km to naut.miles then hr to min
                route = {
                    "origin_vertiport_id": origin_id, 
                    "destination_vertiport_id": destination_id, 
                    "travel_time": travel_time,
                    "capacity": random.randint(2, 5),
                    }
                routes.append(route)
    return routes



sectors = generate_sectors(vertiports)
flights, routes = generate_flights()
fleets = generate_fleets(list(flights.keys()))


json_data = {
    "timing_info": {
        "start_time": START_TIME,
        "end_time": END_TIME,
        "time_step": TIME_STEP,
        "auction_frequency": AUCTION_DT,
    },
    "congestion_params": {
        "lambda": 0.1,
        "C": {
        vertiport: [round(value, 2) for value in list(
            np.array([0, 0.1, 0.3, 0.6, 1, 1.5, 2.1, 2.8, 3.6, 4.5, 5.5])
            * random.randint(1, 4))] for vertiport in vertiports.keys()},
    },
    "fleets": fleets,
    "flights": flights,
    "vertiports": vertiports,
    "routes": routes,
    "sectors": sectors,
}


current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

current_directory = os.getcwd()
test_cases_directory = os.path.join(current_directory, 'test_cases')
if not os.path.exists(test_cases_directory):
    os.makedirs(test_cases_directory)

file_path = os.path.join(test_cases_directory, f'casef_{formatted_datetime}.json')
with open(file_path, "w") as f:
    json.dump(json_data, f, indent=4)

print(f"Flight data has been generated and saved to '{file_path}'")

