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
AUCTION_DT = 10 # every 15 timesteps there is an auction

# Case study settings
N_FLIGHTS = random.randint(10, 15)
NUM_FLEETS = 10

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

# This is for testing of case #3, similarly in generate_routes function set capacity = 0 to test convergence
# vertiports = {
#     "V001": {"latitude": 37.766699, "longitude": -122.3903664, "landing_capacity": 0, "takeoff_capacity": 0, "hold_capacity": 5},
#     "V002": {"latitude": 37.8361761, "longitude": -122.2668028, "landing_capacity": 0, "takeoff_capacity": 0, "hold_capacity": 5},
#     "V003": {"latitude": 37.7835538, "longitude": -122.5067642, "landing_capacity": 0, "takeoff_capacity": 0, "hold_capacity": 5},
#     "V004": {"latitude": 37.9472484, "longitude": -122.4880737, "landing_capacity": 0, "takeoff_capacity": 0, "hold_capacity": 5},
#     "V005": {"latitude": 37.38556649999999, "longitude": -121.9723564, "landing_capacity": 0, "takeoff_capacity": 0, "hold_capacity": 5},
#     "V006": {"latitude": 37.25214395119753, "longitude": -122.4066509403772, "landing_capacity": 0, "takeoff_capacity": 0, "hold_capacity": 5},
#     "V007": {"latitude": 38.58856301092047, "longitude": -121.5627454937505, "landing_capacity": 0, "takeoff_capacity": 0, "hold_capacity": 5},
# }



total_capacity = sum(vertiport["hold_capacity"] for vertiport in vertiports.values())
# Assert that total holding capacity is greater than or equal to the number of flights
assert total_capacity >= N_FLIGHTS, f"Total holding capacity ({total_capacity}) must be greater than or equal to the number of flights ({N_FLIGHTS})"


# Function to generate random flights
def generate_flights():
    flights = {}
    vertiports_list = list(vertiports.keys())
    allowed_origin_vertiport = [vertiport_id for vertiport_id in vertiports_list for _ in range(vertiports[vertiport_id]["hold_capacity"])]
    # appearance_time = 0
    routes = generate_routes(vertiports)
    route_dict = {(route["origin_vertiport_id"], route["destination_vertiport_id"]): route["travel_time"] for route in routes}

    max_travel_time = route_dict[max(route_dict, key=route_dict.get)]
    last_auction =  END_TIME - max_travel_time - AUCTION_DT
    auction_intervals = list(range(START_TIME, END_TIME, AUCTION_DT))

    for i in range(N_FLIGHTS):  
        flight_id = f"AC{i+1:03d}"
        
        # Select a random auction interval for the appearance time
        auction_interval = random.choice(auction_intervals[:(np.abs(np.array(auction_intervals) - last_auction)).argmin()])
        appearance_time = random.randint(auction_interval, auction_interval + AUCTION_DT) # to avoid flights appearing after the last auction, this is also constraint by the maximu travel time for node creation
        # appearance_time = random.randint(1, 50) #needs to be changes using end time variable

        # Choose origin vertiport
        origin_vertiport_id = random.choice(allowed_origin_vertiport)
        allowed_origin_vertiport.remove(origin_vertiport_id)
        
        destination_vertiport_id = random.choice(vertiports_list)
        while destination_vertiport_id == origin_vertiport_id:
            destination_vertiport_id = random.choice(vertiports_list)

        # request_departure_time = appearance_time + random.randint(5, 10)
        request_departure_time = random.randint(auction_interval + AUCTION_DT, auction_interval + 2*AUCTION_DT)
        # delay = random.randint(1, 5)
        # second_departure_time = request_departure_time + delay
        travel_time = route_dict.get((origin_vertiport_id , destination_vertiport_id), None)
        request_arrival_time = request_departure_time + travel_time
        # second_arrival_time = request_arrival_time + delay

        valuation = random.randint(70, 200)
        budget_constraint = random.randint(50, 200)
        # second_valuation = valuation - random.randint(5,10)
        flight_info = { # change the request to be parking or move if nonalloc
            "appearance_time": appearance_time,
            "origin_vertiport_id": origin_vertiport_id,
            "budget_constraint": budget_constraint,
            "decay_factor": 0.5,
            "requests": {
                "000": {
                    "destination_vertiport_id": origin_vertiport_id,
                    "request_departure_time": 0,
                    "request_arrival_time": 0,
                    "valuation": 30,
                },
                "001": {
                    "destination_vertiport_id": destination_vertiport_id,
                    "request_departure_time": request_departure_time,
                    "request_arrival_time": request_arrival_time,
                    "valuation": valuation,
                },
                # "002": {

                #     "destination_vertiport_id": destination_vertiport_id,
                #     "request_departure_time": second_departure_time,
                #     "request_arrival_time": second_arrival_time,
                #     "valuation": second_valuation,  
                # }
            }
        }
        flights[flight_id] = flight_info
    return flights, routes



# Function to calculate distance between two points using Haversine formula
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

    # Calculate the change in coordinates
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Apply Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Calculate distance
    distance = R * c
    return distance



# Generate fleets
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
                speed = 90 # placeholder, we need to add specific vehicle speed in knots (Wisk)
                travel_time = math.ceil(distance * 0.5399568 * 60  / speed)   # cover distance from km to naut.miles then hr to min
                route = {
                    "origin_vertiport_id": origin_id, 
                    "destination_vertiport_id": destination_id, 
                    "travel_time": travel_time,
                    "capacity": random.randint(2, 5),
                    }
                routes.append(route)
    return routes

# Write JSON data to dictionary
flights, routes = generate_flights()
fleets = generate_fleets(list(flights.keys()))
json_data = {
    "timing_info": {"start_time": START_TIME, "end_time": END_TIME, "time_step": TIME_STEP, "auction_frequency": AUCTION_DT},
    "fleets": fleets,
    "flights": flights,
    "vertiports": vertiports,
    "routes": routes
}

# Write flight data to JSON file
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

