import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

with open('test_cases/gui_v1.json', 'r') as file:
    data = json.load(file)

granularity = 10

lats = []
lons = []
for i in data["vertiports"].keys():
    lats += [data["vertiports"][i]["latitude"]]
    lons += [data["vertiports"][i]["longitude"]]

minlat = min(data["gridBounds"][0]["lat"],data["gridBounds"][1]["lat"])
minlon = min(data["gridBounds"][0]["lng"],data["gridBounds"][1]["lng"])
maxlat = max(data["gridBounds"][0]["lat"],data["gridBounds"][1]["lat"])
maxlon = max(data["gridBounds"][0]["lng"],data["gridBounds"][1]["lng"])

data["flights"]["AC001"]["allocated_request"] = "002"
data["flights"]["AC002"]["allocated_request"] = "002"
data["flights"]["AC003"]["allocated_request"] = "001"

bg = plt.imread("test_cases/gui_v1.png")

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

flight_lat, flight_lon

fig, ax = plt.subplots()
plt.scatter(lons, lats, color='red', label='Vertiports', s= 100)

# Set the axis limits according to the grid bounds
plt.xlim(minlon, maxlon)
plt.ylim(minlat, maxlat)

# Add labels and a title
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Latitude and Longitude Points')

animated_points, = ax.plot(flight_lon, flight_lat, 'go', label='Flights', marker = '^')
ax.imshow(bg,extent=[minlon, maxlon, maxlat, minlat], origin='lower')


def update(frame):
    new_lat = []
    new_lon = []
    for i in data["flights"]:
        lat_,lon_ = get_latlon(frame, i)
        new_lat += [lat_]
        new_lon += [lon_]
    animated_points.set_data(new_lon, new_lat)
    print(frame)

ani = FuncAnimation(fig, update, frames=data["timing_info"]["end_time"] * granularity, interval=25, repeat = False)

# Show the plot
plt.legend()
plt.show()

