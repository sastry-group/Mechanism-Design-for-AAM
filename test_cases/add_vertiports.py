import json

filename = "modified_toulouse_case3_withC_cap5"
with open(f"{filename}.json", "r") as f:
    data = json.load(f)

flights = data["flights"]
for flight_id, flight in flights.items():
    flight["requests"]["001"]["destination_vertiport_id"] = flight["origin_vertiport_id"]

with open(f"modified_toulouse_case3_withC_cap5_withReturn.json", "w") as f:
    json.dump(data, f, indent=4)