import json
import re

filename = "test_cases/toulouse_case_cap7_updated_60.json"

# Load your original JSON file
with open(filename, 'r') as infile:
    data = json.load(infile)

length_tau = 5
original_ts_length = 15  
new_ts_length = length_tau  

factor = original_ts_length / new_ts_length 

auction_val = re.search(r'updated_(\d+)', filename)
if auction_val:
    value = int(auction_val.group(1))
else:
    value = None
auction_ts_length = value #or set manually here


# 15 is the number of seconds that 1 time step in the original case represents
seconds = int(round(length_tau))

# Process the flights section to update time values
for flight in data.get('flights', {}).values():
    # Multiply appearance_time by factor
    if 'appearance_time' in flight:
        new_time = flight['appearance_time'] * factor
        flight['appearance_time'] = round(new_time)
        

    # Process each request inside the flight
    for request in flight.get('requests', {}).values():
        if 'request_departure_time' in request:
            new_time = request['request_departure_time'] * factor
            request['request_departure_time'] = round(new_time)
        if 'request_arrival_time' in request:
            new_time = request['request_arrival_time'] * factor
            request['request_arrival_time'] = round(new_time)
        if 'sector_times' in request:
            new_times = [round(sector_time * factor) for sector_time in request['sector_times']]
            request['sector_times'] = new_times

new_time = data.get('timing_info', {})['end_time'] * factor
data.get('timing_info', {})['end_time'] = round(new_time)
new_time = data.get('timing_info', {})['auction_frequency'] * factor
data.get('timing_info', {})['auction_frequency'] = round(new_time)

# Save the updated JSON to a new file
with open(f"{filename[:-7]}{int(auction_ts_length)}stepauction_{seconds}sectau.json", 'w') as outfile:
    json.dump(data, outfile, indent=4)
# with open(f"{filename[:-7]}{int(20*factor)}stepauction_{seconds}sectau.json", 'w') as outfile:
#     json.dump(data, outfile, indent=4)
