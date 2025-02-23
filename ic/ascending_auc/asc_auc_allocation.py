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

# from fisher.fisher_allocation import find_capacity
from ic.market import find_capacity

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

class Good:
    def __init__(self, good, price_increment=1):
        self.good = good
        self.price = 0
        self.price_increment = price_increment
    def raise_val(self):
        self.price += self.price_increment
    def __str__(self):
        return f"({self.good[0]}, {self.good[1]})"
    def __eq__(self, other):
        return self.good == other.good
    def __hash__(self):
        return hash(self.good)
    
class bundle:
    flight_id = ""
    time = -1
    delay = 0
    req_id = None
    budget = 0
    flight = () #allocated_requests format for step_simulation
    times = []
    goods = []
    value = 0
    dep_id = None
    arr_id = None
    
    def populate(self, start, end, spot, beta):
        for i in range(start, end):
            self.goods.append(Good((f"{spot}_{i}", f"{spot}_{i+1}"), beta))
            # print(f"Populating: {spot}_{i} -> {spot}_{i+1}")
            # print(f"Intermediate goods: {self.goods}")
        # for i in range(start,end):
        #     self.times += [time_step(self.flight_id, i, spot)]

    def update_flight_path(self, depart_time, depart_port, arrive_time, arrive_port):
        #(self.flight_id, (dep_id, arr_id)) was ff output format for allocated_requests
        self.dep_id = depart_port + '_' + str(depart_time) + '_dep'
        self.arr_id = arrive_port + '_' + str(arrive_time) + '_arr'
        if arrive_port is not None:
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
        self.goods = []
    
    
    def findCost(self, current_time):
        tot = 0
        for i in range(current_time,len(self.times)):
            tot += self.times[i].price
        return tot
    
    def show(self):
        # print(self.flight_id)
        spots = [i.spot for i in self.times]
        # print(spots)


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

def process_request(id_, req_id, depart_port, arrive_port, sector_path, sector_times, appearance_time, depart_time, arrive_time, maxBid, start_time, end_time, step, auction_end, auction_period, decay, budget, beta):
    reqs = []
    if(req_id == "000"):
        b = bundle(id_, req_id, [], maxBid, -1, budget)
        b.populate(appearance_time, auction_end, depart_port, beta)
        b.update_flight_path(depart_time, depart_port, arrive_time, arrive_port)
        reqs += [b]
        return reqs
    
    for delay in range(5):
        this_decay = decay**delay
        adjusted_depart_time = depart_time + delay
        adjusted_sector_times = [i + delay for i in sector_times]
        adjusted_arrive_time = arrive_time + delay
    
        b = bundle(id_, req_id, [], maxBid*this_decay, delay, budget)
        # print(f"Iteration {delay}, b ID: {id(b)}")
        curtimesarray = [time_step(id_, i, 'NA') for i in range(start_time, end_time + 1, step)]
        # for i in range(start_time, depart_time, step): #start_time -1 so it starts at 0
            # curtimesarray[i].spot =  depart_port
        b.populate(appearance_time, adjusted_depart_time, depart_port, beta)
        # print(f"Goods: {b.goods}")
        b.goods.append(Good((depart_port + '_' + str(adjusted_depart_time), depart_port + '_' + str(adjusted_depart_time) + '_dep'), beta))
        b.goods.append(Good((depart_port + '_' + str(adjusted_depart_time) + '_dep', sector_path[0] + '_' + str(adjusted_depart_time)), beta))

        # curtimesarray[depart_time].spot = depart_port+'_dep'
        for i in range(len(sector_path)):
            # for j in range(sector_times[i], sector_times[i+1]):
            b.populate(adjusted_sector_times[i], adjusted_sector_times[i+1], sector_path[i], beta)
                # curtimesarray[j].spot = sector_path[i]
            if i != len(sector_path) - 1:
                b.goods.append(Good((sector_path[i] + '_' + str(adjusted_sector_times[i+1]), sector_path[i+1] + '_' + str(adjusted_sector_times[i+1])), beta))
                # curtimesarray[sector_times[i+1]].spot = sector_path[i] + sector_path

        if arrive_port is not None:
            b.goods.append(Good((sector_path[-1] + '_' + str(adjusted_arrive_time), arrive_port + '_' + str(adjusted_arrive_time) + '_arr'), beta))
            b.goods.append(Good((arrive_port + '_' + str(adjusted_arrive_time) + '_arr', arrive_port + '_' + str(adjusted_arrive_time)), beta))
            final_time = ((adjusted_arrive_time // auction_period) + 1) * auction_period
            # print(f"Final time: {final_time}")
            b.populate(adjusted_arrive_time, final_time, arrive_port, beta)
        # for i in range(depart_time + 1, arrive_time, step):
        #     curtimesarray[i].spot = depart_port+arrive_port

        # curtimesarray[arrive_time].spot = arrive_port+'_arr'

        # for i in range(arrive_time + 1, end_time, step):
        #     curtimesarray[i].spot = arrive_port
        

        # delayed_dep_t = depart_time 
        # delayed_arr_t = arrive_time
        # delay = 0
        # nb = bundle(id_, req_id, curtimesarray, maxBid, 0, budget)
        b.update_flight_path(adjusted_depart_time, depart_port, adjusted_arrive_time, arrive_port)
        reqs += [b]
        # print(f"Request goods: {b.goods}")

    # while(delayed_arr_t + 1 < end_time): # arrive_port + _arr on last timestep
    #     delay +=1
    #     c2 = [tm.copy() for tm in reqs[-1].times]
    #     c2[delayed_dep_t].spot = depart_port
    #     c2[delayed_dep_t + 1].spot = depart_port + '_dep'
    #     c2[delayed_arr_t].spot = depart_port + arrive_port
    #     c2[delayed_arr_t + 1].spot = arrive_port + '_arr'
    #     delayedBundle = bundle(id_, req_id, c2, maxBid * decay, delay, budget)
    #     delayedBundle.update_flight_path(delayed_dep_t, depart_port, delayed_arr_t, arrive_port)
    #     reqs += [delayedBundle]
    #     decay *= 0.95
    #     delayed_dep_t += 1
    #     delayed_arr_t += 1
    return reqs 

def multiplicitiesDict(vals): # can optimize
    #print(vals)
    k = set(vals)
    s = {}
    for i in k:
        s[i] = vals.count(i)
        # print(f"Count of {i.good}: {vals.count(i)}")
    return s

def run_auction(reqs, method, start_time, end_time, capacities, sector_data, vertiport_data):
    # print(f"All agent reqs: {reqs}")
    numreq = len(reqs)
    price_per_req = [0] * numreq
    final = False
    pplcnt_log = []
    maxprice_log = []
    it = 0
    all_prices = []
    while(not final):
        final = True
        it += 1
        print('AUC ITER # ', it)
        # for t in range(start_time, end_time):
        prices = []
        favored_reqs = []
        favored_req_inds = []
        favored_goods = []
        for agent_reqs in reqs:
            # for req in agent_reqs:
                # print(f"Agent req goods: {req.goods}")
            price_per_req = [sum([good.price for good in req.goods]) for req in agent_reqs]
            if(method == "profit"):
                _, ind, favored_req = max([(r.value - price_per_req[i], i, r) for i, r in enumerate(agent_reqs)], key = lambda x: x[0])
            elif(method == "budget"):
                filtered_inds = [i for i, r in enumerate(agent_reqs) if price_per_req[i] < r.budget]
                filtered_reqs = [agent_reqs[i] for i in filtered_inds]
                _, ind, favored_req = max([(r.value, i, r) for i, r in zip(filtered_inds, filtered_reqs)], key = lambda x: x[0])
            prices.append(price_per_req[ind])
            favored_reqs.append(favored_req)
            favored_req_inds.append(ind)
            favored_goods += favored_req.goods
        # print(f"Favored reqs: {favored_reqs}")
            #look through all requests at a time step
            # spots_reqs = []
        
            # for r in reqs:
                #print(type(r.times[t]))
                #print(r.flight_id, r.req_id, r.delay, len(r.times))
                # spots_reqs += [[r.times[t].spot, r.flight_id]]

            # spots_reqs = [e[0] for e in list({tuple(i) for i in spots_reqs})] # ensuring same flights arent competing by creating set of used spots including flight ids
            #print('A  -----------------')
            # print(spots_reqs)
        favored_good_names = [good.good for good in favored_goods]
        # print(f"Favored good names: {favored_good_names}")
        capacities = find_capacity(favored_good_names + ['dummy', 'dummy'], sector_data, vertiport_data)
        capacitiesDict = {good: capacity for good, capacity in zip(favored_goods, capacities)}
        multiplicities = multiplicitiesDict(favored_goods)
        # print(f"Multiplicities: {[mult.good for mult in multiplicities]}")
        # print(f"Capacities: {[cap.good for cap in capacitiesDict.keys()]}")

        for key, val in multiplicities.items():
            # print(f"Key: {key.good} with usage {val} and capacity {capacitiesDict[key]}")
            if(val > capacitiesDict[key]):
                print(f"Contested: {key.good} with usage {val} and capacity {capacitiesDict[key]}")
                final = False
                for agent_reqs in reqs:
                    for req in agent_reqs:
                        for good in req.goods:
                            if good == key:
                                good.raise_val()
        #print('B ----------')
        # pricedOut = []
        # for r_ix in range(len(reqs)):
        #     r = reqs[r_ix]
        #     #compare each request size to capacity
        #     #print(r.times[t].spot,multiplicities[r.times[t].spot], capacities[r.times[t].spot])
        #     if(multiplicities[r.times[t].spot] > capacities[r.times[t].spot]): # contested
        #         #r.times[t].disp()
        #         #print(r.value)
        #         final = False
        #             #if larger increase price of time step at each request & total flight price
        #         r.times[t].raise_val()
        #         price_per_req[r_ix] += 1
                # if(method == "profit"):
                #     if(price_per_req[r_ix] >= r.value):
                #         pricedOut += [r_ix]
                # elif(method == "budget"):
                #     if(price_per_req[r_ix] >= r.budget):
                #         pricedOut += [r_ix]
        # for n in pricedOut[::-1]:
        #     reqs.pop(n)
        #     price_per_req.pop(n)
        #
        
        # pplcnt_log += [[t, len(price_per_req)]]
        # maxprice_log += [[t, max(price_per_req)]]
        # numreq = len(reqs)
    pplcnt_log = []
    maxprice_log = []

    print('     ----')
    if method == "profit":
        for r, price in zip(favored_reqs, prices):
            print("Flight ID: ", r.flight_id, " | Request ID: ", r.req_id, " | FROM: ", r.dep_id, " | TO: ", r.arr_id, " | Delay: ",r.delay, " | Value: " ,r.value, " | Overall Price: ",price, " | Profit: " ,r.value - price)
    elif method == "budget":
        for r, price in zip(favored_reqs, prices):
            print("Flight ID: ", r.flight_id, " | Request ID: ", r.req_id, " | FROM: ", r.dep_id, " | TO: ", r.arr_id, " | Delay: ",r.delay, " | Value: " ,r.value, " | Overall Price: ",price, " | Remaining Budget: " ,r.budget - price)
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

    
    return favored_reqs, price_per_req, pplcnt_log, maxprice_log


def define_capacities(vertiport_data, sector_data):
    capacities = {}
    for sector, params in sector_data.items():
        # dep_port = i["origin_vertiport_id"]
        # arr_port = i["destination_vertiport_id"]
        capacities[sector] = params["hold_capacity"]
    for i in vertiport_data:
        capacities[i] = vertiport_data[i]["hold_capacity"]#route to self
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


def ascending_auc_allocation_and_payment(vertiport_usage, flights, timing_info, sector_data, auction_method,
                                  save_file=None, initial_allocation=True, design_parameters=None):

    beta = design_parameters["beta"]
    market_auction_time=timing_info["start_time"]
    auction_period = timing_info["auction_frequency"]

    capacities = define_capacities(vertiport_usage.vertiports, sector_data)
    print("--- PROCESSING REQUESTS ---")
    requests = []
    for f in flights.keys():
        agent_requests = []
        flight_data = flights[f]
        origin_vp = flight_data["origin_vertiport_id"]
        flight_req = flight_data["requests"]
        decay = flight_data["decay_factor"]
        budget = flight_data["budget_constraint"]
        appearance_time = flight_data["appearance_time"]
        for req_index in flight_req.keys(): #req_index = 000, 001, ...
            print(f, req_index)
            fr = flight_req[req_index]
            dest_vp = fr["destination_vertiport_id"]
            # if(origin_vp == dest_vp):
            #     continue
            dep_time = fr["request_departure_time"]
            arr_time = fr["request_arrival_time"]
            if req_index == "000":
                val = fr["valuation"]
                agent_requests += process_request(f, req_index, origin_vp, dest_vp, None, None, appearance_time, dep_time, arr_time, val, timing_info["start_time"], timing_info["end_time"], timing_info["time_step"], timing_info["auction_end"], auction_period, decay, budget, beta)
                continue
            sector_path = fr["sector_path"]
            sector_times = fr["sector_times"]

            val = fr["valuation"]
            agent_requests += process_request(f, req_index, origin_vp, dest_vp, sector_path, sector_times, appearance_time, dep_time, arr_time, val, timing_info["start_time"], timing_info["end_time"], timing_info["time_step"], timing_info["auction_end"], auction_period, decay, budget, beta)
        requests.append(agent_requests)
    print("PROCESSED REQUESTS")
    for agent_reqs in requests:
        for r in agent_reqs:
            print(r.flight_id, r.req_id, r.value)


    print("--- RUNNING AUCTION ---")    


    allocated_requests, final_prices_per_req, agents_left_time, price_change = run_auction(requests, auction_method, timing_info["auction_start"], timing_info["end_time"], capacities, sector_data, vertiport_usage)


    



    # allocated_requests = pickHighest(allocated_requests, market_auction_time, auction_method)

    print('FINAL')

    #print(allocated_requests)
    
    
    # print('S - AR REQUESTS')
    # print(allocated_requests)                                       
    # print('E - AR REQUESTS')

    allocation  = [(ar.flight_id, ('_'.join(ar.dep_id.split("_")[:-1]), ar.dep_id)) for ar in allocated_requests if ar.delay != -1]
    rebased = { ar.flight_id: flights[ar.flight_id] for ar in allocated_requests if ar.delay == -1}

    #write_output(flights, agent_constraints, edge_information, prices, new_prices, capacity, end_capacity,
    #            agent_allocations, agent_indices, agent_edge_information, agent_goods_lists, 
    #            int_allocations, new_allocations_goods, u, adjusted_budgets, payment, end_agent_status_data, market_auction_time, output_folder)
    costs_ = [ar.value - ar.findCost(timing_info["auction_start"]) for ar in allocated_requests]
    return allocation, rebased, costs_



if __name__ == "__main__":
    pass

