import json
import random

DISCRETIZATION_STEP = 15

def is_vertiport(id):
    return (id == "V001") or (id == "V002") or (id == "V003") or (id == "V004")

def this_round(input_time):
    return round(input_time / DISCRETIZATION_STEP)

# Load the original JSON data from a file (for example)
with open("toulouse_case.json", "r") as f:
    data = json.load(f)
flights = data["flights"]

empty_flights = []
# Iterate over the flights and modify the "requests" section
for flight_id, flight_data in flights.items():
    requests = flight_data["requests"]
    if len(requests) == 0:
        empty_flights.append(flight_id)
        continue
    request_start = int(flight_data["appearance_time"])
    sampled_appearance_time = random.randint(max(request_start - 600, 0), max(request_start - 120, 0))
    rounded_appearance_time = this_round(sampled_appearance_time)
    
    # Create the sector_path and sector_times
    old_to_new_sectors = {"V001": "V001", "V002": "V002", "V003": "V003", "V004": "V004",
                          "V005": "S001", "V006": "S002", "V007": "S003", "V008": "S004", "V009": "S005"}
    sector_path = [req["destination_vertiport_id"] for req in requests]
    fixed_sector_path = [old_to_new_sectors[sector] for sector in sector_path]
    sector_times = [this_round(int(req["request_departure_time"])) for req in requests]

    if len(requests) > 0:
        sector_times.append(this_round(int(requests[-1]["request_arrival_time"])))  # Add the last arrival time
        destination = requests[-1]["destination_vertiport_id"]
        if is_vertiport(requests[-1]["destination_vertiport_id"]):
            destination_vertiport_id = requests[-1]["destination_vertiport_id"]
        else:
            destination_vertiport_id = None
    else:
        destination_vertiport_id = None
    
    if len(sector_times) > 0:
        departure_time = sector_times[0]
        arrival_time = sector_times[-1]
    else:    
        departure_time = 0
        arrival_time = 0
    # Replace the requests section with the new format
    flight_data["appearance_time"] = rounded_appearance_time
    flight_data["budget_constraint"] = random.randint(1, 300)
    flight_data["requests"] = {"000": {
            "bid": 1,
            "valuation": 1,
            "request_departure_time": 0,
            "request_arrival_time": 0,
            "destination_vertiport_id": flight_data["origin_vertiport_id"]
        },
        "001": {
            "bid": random.randint(1, 300),
            "valuation": random.randint(1, 300),
            "sector_path": fixed_sector_path,
            "sector_times": sector_times,
            "destination_vertiport_id": destination_vertiport_id,
            "request_departure_time": departure_time,
            "request_arrival_time": arrival_time        
        }
    }
for flight_id in empty_flights:
    del flights[flight_id]
data["flights"] = flights
data["timing_info"] = {
    "start_time": 1,
    "end_time": 20,
    "time_step": 1,
    "auction_frequency": 10
}
data["congestion_params"] = {
    "lambda": 0.1,
    "C": {
        "V001": [
            0.0,
            0.2,
            0.6,
            1.2,
            2.0,
            3.0,
            4.2,
            5.6,
            7.2,
            9.0,
            11.0
        ]
    }
}

# Save the modified data back to a file (or use the data as needed)
with open("modified_toulouse_case.json", "w") as f:
    json.dump(data, f, indent=4)