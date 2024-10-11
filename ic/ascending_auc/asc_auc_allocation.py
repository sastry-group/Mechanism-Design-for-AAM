import sys
from pathlib import Path
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
import json
import math
from pathlib import Path
from multiprocessing import Pool
import logging


logging.basicConfig(filename='solver_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Add the bluesky package to the path
top_level_path = Path(__file__).resolve().parent.parent
print(str(top_level_path))
sys.path.append(str(top_level_path))

UPDATED_APPROACH = True
TOL_ERROR = 1e-3
MAX_NUM_ITERATIONS = 1000

def load_json(file=None):
    """
    Load a case file for a fisher market test case from a JSON file.
    """
    if file is None:
        return None
    assert Path(file).is_file(), f"File {file} does not exist."

    # Load the JSON file
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        print(f"Opened file {file}")
    return data


def find_dep_and_arrival_nodes(edges):
    dep_node_found = False
    arrival_node_found = False
    
    for edge in edges:
        if "dep" in edge[0]:
            dep_node_found = edge[0]
            arrival_node_found = edge[1]
            assert "arr" in arrival_node_found, f"Arrival node not found: {arrival_node_found}"
            return dep_node_found, arrival_node_found
    
    return dep_node_found, arrival_node_found


class time_step:
    flight_id = ""
    time_no = -1
    spot = ""
    price = 0
    def __init__(self, id_, time_, spot_):
        self.time_no = time_
        self.flight_id = id_
        self.spot = spot_
    def disp(self):
        print("CLASS: ", self.flight_id, self.spot, self.time_no, self.price)
    def copy(self):
        return type(self)(self.flight_id, self.time_no, self.spot)
    def raise_val(self):
        self.price += 1

class bundle:
    flight_id = ""
    time = -1
    delay = 0
    req_id = None
    budget = 0
    flight = () #allocated_requests format for step_simulation
    times = []
    value = 0
    dep_id = None
    arr_id = None
    
    def populate(self, start, end, spot):
        for i in range(start,end):
            self.times += [time_step(self.flight_id, i, spot)]

    def update_flight_path(self, depart_time, depart_port, arrive_time, arrive_port):
        #(self.flight_id, (dep_id, arr_id)) was ff output format for allocated_requests
        self.dep_id = depart_port + '_' + str(depart_time) + '_dep'
        self.arr_id = arrive_port + '_' + str(arrive_time) + '_arr'
        self.flight = (self.flight_id, self.req_id, self.delay, self.value, depart_port, arrive_port)
        return self.flight
    
    def __init__(self, f, req_id, t, v, delay, budget):
        self.flight_id = f
        self.req_id = req_id
        self.times = t
        self.value =v
        self.delay = delay
        self.budget = budget
    
    
    def findCost(self, current_time):
        tot = 0
        for i in range(current_time,len(self.times)):
            tot += self.times[i].price
        return tot
    
    def show(self):
        print(self.flight_id)
        spots = [i.spot for i in self.times]
        print(spots)


#[('AC004', ('V001_13_dep', 'V002_18_arr')), ('AC005', ('V007_17_dep', 'V002_55_arr')), ('AC008', ('V003_20_dep', 'V006_42_arr'))]

def remove_requests(all, confirmed):
    remaining = []
    for r in all:
        used = False
        for c in confirmed:
            if(r.flight_id == confirmed.flight_id):
                used = True

            else:
                remaining += [c]

def process_request(id_, req_id, depart_port, arrive_port, depart_time, arrive_time, maxBid, start_time, end_time, step, decay, budget):
    reqs = []
    if(depart_port == arrive_port):
        b = bundle(id_, req_id, [], maxBid, 0, budget)
        b.populate(start_time,end_time + 1, depart_port)
        b.update_flight_path(depart_time, depart_port, arrive_time, arrive_port)
        reqs += [b]
        return reqs
    
    curtimesarray = [time_step(id_, i, 'NA') for i in range(start_time, end_time + 1, step)]
    blub = 0
    for i in range(start_time - 1, depart_time, step): #start_time -1 so it starts at 0
        curtimesarray[i].spot =  depart_port
        if(blub<5):
            blub+=1

    curtimesarray[depart_time].spot = depart_port+'_dep'
    for i in range(depart_time + 1, arrive_time, step):
        curtimesarray[i].spot = depart_port+arrive_port

    curtimesarray[arrive_time].spot = arrive_port+'_arr'

    for i in range(arrive_time + 1, end_time, step):
        curtimesarray[i].spot = arrive_port
    

    delayed_dep_t = depart_time 
    delayed_arr_t = arrive_time
    delay = 0
    nb = bundle(id_, req_id, curtimesarray, maxBid, 0, budget)
    nb.update_flight_path(depart_time, depart_port, arrive_time, arrive_port)
    reqs += [nb]

    while(delayed_arr_t + 1 < end_time): # arrive_port + _arr on last timestep
        delay +=1
        c2 = [tm.copy() for tm in reqs[-1].times]
        c2[delayed_dep_t].spot = depart_port
        c2[delayed_dep_t + 1].spot = depart_port + '_dep'
        c2[delayed_arr_t].spot = depart_port + arrive_port
        c2[delayed_arr_t + 1].spot = arrive_port + '_arr'
        delayedBundle = bundle(id_, req_id, c2, maxBid * decay, delay, budget)
        delayedBundle.update_flight_path(delayed_dep_t, depart_port, delayed_arr_t, arrive_port)
        reqs += [delayedBundle]
        decay *= 0.95
        delayed_dep_t += 1
        delayed_arr_t += 1
    return reqs 

def multiplicitiesDict(vals): # can optimize
    #print(vals)
    k = set(vals)
    s = {}
    for i in k:
        s[i] = vals.count(i)
    return s

def run_auction(reqs, method, start_time, end_time, capacities):
    numreq = len(reqs)
    price_per_req = [0] * numreq
    final = False
    pplcnt_log = []
    maxprice_log = []
    it = 0
    while(not final):
        final = True
        it += 1
        #print('AUC ITER # ', it)
        for t in range(start_time, end_time):
            #look through all requests at a time step
            spots_reqs = []
        
            for r in reqs:
                #print(type(r.times[t]))
                #print(r.flight_id, r.req_id, r.delay, len(r.times))
                spots_reqs += [[r.times[t].spot, r.flight_id]]

            spots_reqs = [e[0] for e in list({tuple(i) for i in spots_reqs})] # ensuring same flights arent competing by creating set of used spots including flight ids
            #print('A  -----------------')
            multiplicities = multiplicitiesDict(spots_reqs)
            #print('B ----------')
            pricedOut = []
            for r_ix in range(len(reqs)):
                r = reqs[r_ix]
                #compare each request size to capacity
                #print(r.times[t].spot,multiplicities[r.times[t].spot], capacities[r.times[t].spot])
                if(multiplicities[r.times[t].spot] > capacities[r.times[t].spot]): # contested
                    #r.times[t].disp()
                    #print(r.value)
                    final = False
                     #if larger increase price of time step at each request & total flight price
                    r.times[t].raise_val()
                    price_per_req[r_ix] += 1
                    if(method == "profit"):
                        if(price_per_req[r_ix] >= r.value):
                            pricedOut += [r_ix]
                    elif(method == "budget"):
                        if(price_per_req[r_ix] >= r.budget):
                            pricedOut += [r_ix]
            for n in pricedOut[::-1]:
                reqs.pop(n)
                price_per_req.pop(n)
            #
            
            pplcnt_log += [[t, len(price_per_req)]]
            maxprice_log += [[t, max(price_per_req)]]
            numreq = len(reqs)
            

    print('     ----')
    for ri in range(len(reqs)):
        r = reqs[ri]
        print("Flight ID: ", r.flight_id, " | Request ID: ", r.req_id, " | FROM: ", r.dep_id, " | TO: ", r.arr_id, " | Delay: ",r.delay, " | Value: " ,r.value, " | Overall Price: ",price_per_req[ri], " | Profit: " ,r.value - price_per_req[ri])
    print('     ----')

    plot = False
    if(plot):

        duration = [i for i in range(start_time, end_time)]
        fig, axs = plt.subplots(2,1, figsize=(10, 5))
        for r in reqs:
            prices = [i.price for i in r.times]
            if(r.flight_id == 'AC003'):
                axs[0].plot(duration, prices, label = r.delay)
            if(r.flight_id == 'AC004'):
                axs[1].plot(duration, prices, label = r.delay)
        axs[0].legend(loc = 'upper left')
        axs[0].set_ylabel("Price")
        axs[0].set_title("AC 003, Req 001 Delay Comparison")
        axs[1].legend(loc = 'upper left')
        axs[1].set_xlabel("Simulation Time Step")
        axs[1].set_ylabel("Price")
        axs[1].set_title("AC 004, Req 001 Delay Comparison")
        plt.show()

    
    return reqs, price_per_req, pplcnt_log, maxprice_log


def define_capacities(vertiport_data, route_data):
    capacities = {}
    for i in route_data:
        dep_port = i["origin_vertiport_id"]
        arr_port = i["destination_vertiport_id"]
        capacities[dep_port+arr_port] = i["capacity"]
    for i in vertiport_data:
        capacities[i] = vertiport_data[i]["hold_capacity"]
        capacities[i+i] = vertiport_data[i]["hold_capacity"] #route to self
        capacities[i+'_dep'] = vertiport_data[i]["takeoff_capacity"]
        capacities[i+'_arr'] = vertiport_data[i]["landing_capacity"]
    return capacities

def pickHighest(requests, start_time, method):
    mxMap = {}
    mxReq = {}
    for r in requests:
        mxMap[r.flight_id] = -1
        mxReq[r.flight_id] = None
    for r in requests:
        if(method == "profit"):
            tot = r.value - r.findCost(start_time)    
        if(method == "budget"):
            tot = r.value
        if(tot>mxMap[r.flight_id]):
            mxMap[r.flight_id] = tot 
            mxReq[r.flight_id] = r
    return mxReq.values()


def ascending_auc_allocation_and_payment(vertiport_usage, flights, timing_info, routes_data, auction_method,  
                                  save_file=None, initial_allocation=True, design_parameters=None):

    market_auction_time=timing_info["start_time"]


    capacities = define_capacities(vertiport_usage.vertiports, routes_data)
    print("--- PROCESSING REQUESTS ---")
    requests = []
    for f in flights.keys():
        flight_data = flights[f]
        origin_vp = flight_data["origin_vertiport_id"]
        flight_req = flight_data["requests"]
        decay = flight_data["decay_factor"]
        budget = flight_data["budget_constraint"]
        for req_index in flight_req.keys(): #req_index = 000, 001, ...
            print(f, req_index)
            fr = flight_req[req_index]
            dest_vp = fr["destination_vertiport_id"]
            dep_time = fr["request_departure_time"]
            arr_time = fr["request_arrival_time"]
            if(origin_vp== dest_vp):
                continue

            val = fr["valuation"]
            requests += process_request(f, req_index, origin_vp, dest_vp, dep_time, arr_time, val, timing_info["start_time"], timing_info["end_time"], timing_info["time_step"], decay, budget)
    
    print("PROCESSED REQUESTS")
    for r in requests:
        print(r.flight_id, r.req_id, r.value)


    print("--- RUNNING AUCTION ---")    


    allocated_requests, final_prices_per_req, agents_left_time, price_change = run_auction(requests, auction_method, timing_info["auction_start"], timing_info["end_time"], capacities)


    



    allocated_requests = pickHighest(allocated_requests, market_auction_time, auction_method)

    print('FINAL')

    #print(allocated_requests)
    
    
    print('S - AR REQUESTS')
    print(allocated_requests)                                       
    print('E - AR REQUESTS')

    allocation = []

    allocation = [ar.flight for ar in allocated_requests]
    rebased = None

    #write_output(flights, agent_constraints, edge_information, prices, new_prices, capacity, end_capacity,
    #            agent_allocations, agent_indices, agent_edge_information, agent_goods_lists, 
    #            int_allocations, new_allocations_goods, u, adjusted_budgets, payment, end_agent_status_data, market_auction_time, output_folder)
    costs_ = [ar.value - ar.findCost(timing_info["auction_start"]) for ar in allocated_requests]
    return allocation, costs_



if __name__ == "__main__":
    pass

