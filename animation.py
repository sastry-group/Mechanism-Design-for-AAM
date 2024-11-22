import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import numpy as np

with open('test_cases/casef_20240925_175552_mod2.json', 'r') as file:
    data = json.load(file)

granularity = 3

lats = []
lons = []
for i in data["vertiports"].keys():
    lats += [data["vertiports"][i]["latitude"]]
    lons += [data["vertiports"][i]["longitude"]]

minlat = min(data["gridBounds"][0]["lat"],data["gridBounds"][1]["lat"])
minlon = min(data["gridBounds"][0]["lng"],data["gridBounds"][1]["lng"])
maxlat = max(data["gridBounds"][0]["lat"],data["gridBounds"][1]["lat"])
maxlon = max(data["gridBounds"][0]["lng"],data["gridBounds"][1]["lng"])

data["flights"]["AC001"]["allocated_request"] = "001"
data["flights"]["AC002"]["allocated_request"] = "001"
data["flights"]["AC003"]["allocated_request"] = "001"
data["flights"]["AC004"]["allocated_request"] = "001"
data["flights"]["AC005"]["allocated_request"] = "001"
data["flights"]["AC006"]["allocated_request"] = "001"
data["flights"]["AC007"]["allocated_request"] = "001"
data["flights"]["AC008"]["allocated_request"] = "001"
data["flights"]["AC009"]["allocated_request"] = "001"
data["flights"]["AC010"]["allocated_request"] = "001"
data["flights"]["AC011"]["allocated_request"] = "001"
data["flights"]["AC012"]["allocated_request"] = "001"
data["flights"]["AC013"]["allocated_request"] = "001"
data["flights"]["AC014"]["allocated_request"] = "001"
data["flights"]["AC015"]["allocated_request"] = "001"
data["flights"]["AC016"]["allocated_request"] = "001"
data["flights"]["AC017"]["allocated_request"] = "001"
data["flights"]["AC018"]["allocated_request"] = "001"
data["flights"]["AC019"]["allocated_request"] = "001"
data["flights"]["AC020"]["allocated_request"] = "001"

bg = plt.imread("test_cases/pres_bg.png")

def get_latlon(frame, flight_id):
    
    origin_vp = data["flights"][flight_id]["origin_vertiport_id"]
    
    o_lat = data["vertiports"][origin_vp]["latitude"]
    o_lon = data["vertiports"][origin_vp]["longitude"]
    
    alloc = data["flights"][flight_id]["allocated_request"]
    dest_vp = data["flights"][flight_id]["requests"][alloc]["destination_vertiport_id"]
    
    d_lat = data["vertiports"][dest_vp]["latitude"]
    d_lon = data["vertiports"][dest_vp]["longitude"]
    
    dep_time = data["flights"][flight_id]["requests"][alloc]["request_departure_time"] * granularity 
    arr_time = data["flights"][flight_id]["requests"][alloc]["request_arrival_time"] * granularity
    t = arr_time - dep_time
    k = 0
    if(frame < dep_time):
        k = 0
    elif(frame>arr_time):
        k = 1
    else:
        k = (frame-dep_time) / (arr_time - dep_time)
    
    fin_lat = o_lat * (1-k) + d_lat * k
    fin_lon = o_lon * (1-k)  + d_lon * k
    return fin_lat, fin_lon


flight_lat = []
flight_lon = []
for i in data["flights"]:
    origin_vp = data["flights"][i]["origin_vertiport_id"]
    lat = data["vertiports"][origin_vp]["latitude"]
    lon = data["vertiports"][origin_vp]["longitude"]
    flight_lat += [lat]
    flight_lon += [lon]

fig, ax = plt.subplots(figsize=(5,4))
plt.tight_layout()
plt.scatter(lons, lats, color='blue', label='Vertiports', s= 80)

# Set the axis limits according to the grid bounds
plt.xlim(minlon, maxlon)
plt.ylim(minlat, maxlat)

# Add labels and a title
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('AAM Simulation')


num_flights = len(data["flights"])
colors = cm.plasma(np.linspace(0, 1, num_flights)) 

#animated_points, = ax.plot(flight_lon, flight_lat, c = colors, label='Flights', marker = '^')
ax.imshow(bg,extent=[minlon, maxlon, maxlat, minlat], origin='lower', aspect='equal')

scatter = ax.scatter(flight_lon, flight_lat, c=colors, s=60, label='Flights', marker='^')

def update(frame):
    new_lat = []
    new_lon = []
    for i in data["flights"]:
        lat_,lon_ = get_latlon(frame, i)
        new_lat += [lat_]
        new_lon += [lon_]
    scatter.set_offsets(np.c_[new_lon, new_lat])  # Update positions
    #animated_points.set_data(new_lon, new_lat)

ani = FuncAnimation(fig, update, frames=data["timing_info"]["end_time"] * granularity, interval=15, repeat = False)

ani.save("sim/aam_sim9.gif", writer="pillow", fps=60)


# Show the plot
plt.legend()
plt.show()

