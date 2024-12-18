import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import numpy as np
import copy

with open('test_cases/casef_20240925_175552_mod3.json', 'r') as file:
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

method = 'vcg'
max_delay = 10
if(method == 'vcg'):
    for fl in data["flights"]:
        dr = {}
        for j, r in enumerate(data["flights"][fl]["requests"]):
            if(r=='000'):
                dr['000'] = copy.deepcopy(data["flights"][fl]["requests"][r])
                dr['000']['delay'] = 0
                continue
            for i in range(max_delay + 1):
                k = copy.deepcopy(data["flights"][fl]["requests"][r])
                k["request_departure_time"] += i
                k["request_arrival_time"] += i
                k["bid"] *= pow(0.95,i)
                k["valuation"] *= pow(0.95,i)
                k["delay"] = i
                id_ = str(1 + (j-1)*(max_delay+1) + i)
                s = '0' * (3 - len(id_)) + str(id_)
                dr[s] = k
        data["flights"][fl]["requests"] = dr


data["flights"]["AC001"]["allocated_request"] = "001"
data["flights"]["AC002"]["allocated_request"] = "001"
data["flights"]["AC003"]["allocated_request"] = "001"
data["flights"]["AC004"]["allocated_request"] = "001"
data["flights"]["AC005"]["allocated_request"] = "002" #"001"
data["flights"]["AC006"]["allocated_request"] = "005" #"001"
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
data["flights"]["AC017"]["allocated_request"] = "002" #"001"
data["flights"]["AC018"]["allocated_request"] = "003" #"001"
data["flights"]["AC019"]["allocated_request"] = "001"
data["flights"]["AC020"]["allocated_request"] = "001"


#fisher data: Allocation: [('AC001', ('V007_16_dep', 'V002_54_arr')), ('AC007', ('V003_15_dep', 'V002_23_arr')), ('AC010', ('V001_13_dep', 'V005_34_arr')), ('AC002', ('V005_19_dep', 'V004_47_arr')), ('AC003', ('V002_19_dep', 'V001_24_arr')), ('AC004', ('V002_16_dep', 'V001_21_arr')), ('AC014', ('V001_11_dep', 'V002_24_arr')), ('AC018', ('V002_18_dep', 'V007_56_arr')), ('AC005', ('V003_11_dep', 'V002_19_arr')), ('AC017', ('V004_16_dep', 'V006_44_arr')), ('AC006', ('V005_18_dep', 'V007_68_arr')), ('AC015', ('V002_20_dep', 'V001_25_arr')), ('AC008', ('V007_12_dep', 'V001_54_arr')), ('AC019', ('V004_16_dep', 'V002_35_arr')), ('AC020', ('V003_16_dep', 'V006_38_arr')), ('AC009', ('V001_14_dep', 'V005_35_arr')), ('AC011', ('V006_20_dep', 'V004_48_arr')), ('AC013', ('V002_17_dep', 'V006_41_arr')), ('AC012', ('V005_16_dep', 'V001_37_arr')), ('AC016', ('V007_17_dep', 'V005_67_arr'))]

#payments = [('F001', 8.420000000000073), ('F002', 171.0), ('F003', 8.420000000000073), ('F004', 3.2199999999998), ('F005', 0.0), ('F006', 0.0), ('F007', 0.0), ('F008', 0.0), ('F009', 134.24000000000024), ('F010', 0.0), ('F011', 0.0), ('F012', 3.3799999999996544), ('F013', 0.0), ('F014', 0.0), ('F015', 0.0), ('F016', 0.0), ('F017', 1.819999999999709), ('F018', 6.179999999999836), ('F019', 4.320000000000164), ('F020', 0.0)]
payments_delayed = [('F001', 5.119999999999891), ('F002', 28.694043749999764), ('F003', 5.2599999999997635), ('F004', 1.900000000000091), ('F005', 0.40000000000009095), ('F006', 0.9200000000000728), ('F007', 0.05999999999994543), ('F008', 0.15999999999985448), ('F009', 6.929999999999836), ('F010', 5.741543749999892), ('F011', 0.15999999999985448), ('F012', 1.4600000000000364), ('F013', 12.624043750000055), ('F014', 1.2199999999998), ('F015', 0.1799999999998363), ('F016', 0.1799999999998363), ('F017', 0.40000000000009095), ('F018', 2.4200000000000728), ('F019', 0.3200000000001637), ('F020', 20.094043749999855)]

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
#colors = cm.greens(np.linspace(0, 1, num_flights)) 

#animated_points, = ax.plot(flight_lon, flight_lat, c = colors, label='Flights', marker = '^')
ax.imshow(bg,extent=[minlon, maxlon, maxlat, minlat], origin='lower', aspect='equal')

scatter = ax.scatter(flight_lon, flight_lat, c='green',s=60, label='Flights', marker='^')

fleet_ids = [i[0] for i in payments_delayed]
payment_vals = [str(i[1])[:4] for i in payments_delayed]
fleet_labels = [
    ax.text(lon - 0.1, lat + 0.02, flight_id, fontsize=9, color='black', ha='center', va='center')
    for flight_id, lon, lat in zip(fleet_ids, flight_lon, flight_lat)
]

payment_labels = [
    ax.text(lon + 0.1, lat, flight_id, fontsize=10, color='black', ha='center', va='center')
    for flight_id, lon, lat in zip(payment_vals, flight_lon, flight_lat)
]

def update(frame):
    new_lat = []
    new_lon = []
    for ix, i in enumerate(data["flights"]):
        lat_,lon_ = get_latlon(frame, i)
        new_lat += [lat_]
        new_lon += [lon_]
        fleet_labels[ix].set_position((lon_ - 0.1, lat_ + 0.02))
        payment_labels[ix].set_position((lon_ + 0.1, lat_))
    scatter.set_offsets(np.c_[new_lon, new_lat])  # Update positions
    #animated_points.set_data(new_lon, new_lat)

ani = FuncAnimation(fig, update, frames=data["timing_info"]["end_time"] * granularity, interval=15, repeat = False)

ani.save("sim/aam_sim9 .gif", writer="pillow", fps=60)


# Show the plot
plt.legend()
plt.show()

