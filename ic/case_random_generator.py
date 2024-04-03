import json
import random
from datetime import datetime, timedelta


# Simulation time settings
START_TIME = 1
END_TIME = 300
TIME_STEP = 10

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
    "V001": {"latitude": 37.766699, "longitude": -122.3903664, "landing_capacity": 1, "takeoff_capacity": 1, "hold_capacity": 2},
    "V002": {"latitude": 37.8361761, "longitude": -122.2668028, "landing_capacity": 1, "takeoff_capacity": 1, "hold_capacity": 2},
    "V003": {"latitude": 37.7835538, "longitude": -122.5067642, "landing_capacity": 1, "takeoff_capacity": 1, "hold_capacity": 2},
    "V004": {"latitude": 37.9472484, "longitude": -122.4880737, "landing_capacity": 1, "takeoff_capacity": 1, "hold_capacity": 2},
    "V005": {"latitude": 37.38556649999999, "longitude": -121.9723564, "landing_capacity": 1, "takeoff_capacity": 1, "hold_capacity": 2},
    "V006": {"latitude": 37.25214395119753, "longitude": -122.4066509403772, "landing_capacity": 1, "takeoff_capacity": 1, "hold_capacity": 2},
    "V007": {"latitude": 38.58856301092047, "longitude": -121.5627454937505, "landing_capacity": 1, "takeoff_capacity": 1, "hold_capacity": 2},

}


# Function to generate random flights
def generate_flights():
    flights = {}
    for i in range(random.randint(100, 200)):  # Random number of flights between 100 to 200
        flight_id = f"AC{i+1:03d}"
        appearance_time = random.randint(10, 30)
        origin_vertiport_id = random.choice(list(vertiports.keys()))
        destination_vertiport_id = random.choice(list(vertiports.keys()))
        while destination_vertiport_id == origin_vertiport_id:
            destination_vertiport_id = random.choice(list(vertiports.keys()))
        request_departure_time = appearance_time
        request_arrival_time = request_departure_time + random.randint(5, 20)
        valuation = random.randint(50, 200)
        bid = valuation
        flight_info = {
            "appearance_time": appearance_time,
            "origin_vertiport_id": origin_vertiport_id,
            "requests": {
                "000": {
                    "destination_vertiport_id": destination_vertiport_id,
                    "request_departure_time": request_departure_time,
                    "request_arrival_time": request_arrival_time,
                    "valuation": valuation,
                    "bid": bid
                }
            }
        }
        flights[flight_id] = flight_info
    return flights

# Generate fleets
fleets = {
    "F001": list(generate_flights().keys()),
    "F002": list(generate_flights().keys())
}

# Generate routes
routes = [{
    "origin_vertiport_id": "V001",
    "destination_vertiport_id": "V002",
    "travel_time": 10
}]

# Write JSON data to dictionary
json_data = {
    "timing_info": {"start_time": START_TIME, "end_time": END_TIME, "time_step": TIME_STEP},
    "fleets": fleets,
    "flights": generate_flights(),
    "vertiports": vertiports,
    "routes": routes
}

# Write flight data to JSON file
with open("flight_data.json", "w") as f:
    json.dump(json_data, f, indent=4)

print("Flight data has been generated and saved to 'flight_data.json'")