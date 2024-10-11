def determine_allocation(vertiport_usage, flights, timing_info):
    """
    Determine the allocation of flights to vertiports.

    Args:
        vertiport_usage (VertiportStatus): Usage information on all vertiports.
        flights (list): Flights making requests.

    Returns:
        dict: Flights allocated.
    """
    max_time, time_step = timing_info["end_time"], timing_info["time_step"]
    allocation = []
    # Order flights by appearance time
    ordered_flights = {}
    for flight_id, flight in flights.items():
        appearance_time = flight["appearance_time"]
        if appearance_time not in ordered_flights:
            ordered_flights[appearance_time] = [flight_id]
        else:
            ordered_flights[appearance_time].append(flight_id)
    
    # Allocate flights in order of appearance time
    for appearance_time in sorted(ordered_flights.keys()):
        for flight_id in ordered_flights[appearance_time]:
            flight = flights[flight_id]
            # Order flight requests based on bid (highest bid first)
            bids = [request["bid"] for _, request in flight["requests"].items()]
            sorted_requests = [request_id for _, request_id in sorted(zip(bids, flight["requests"]), reverse=True)]
            origin = flight["origin_vertiport_id"]

            # Check if landing, take-off, and parking capacity is available
            for request_id in sorted_requests:
                print(sorted_requests)
                request = flight["requests"][request_id]
                if request["request_departure_time"] == 0:
                    break
                # Check if vertiport has capacity to accommodate the flight
                time_extended_takeoff_id = origin + "_" + str(request["request_departure_time"])
                time_extended_landing_id = request["destination_vertiport_id"] + "_" + str(request["request_arrival_time"])
                takeoff_space = vertiport_usage.nodes[time_extended_takeoff_id]["takeoff_capacity"] - vertiport_usage.nodes[time_extended_takeoff_id]["takeoff_usage"]
                landing_space = vertiport_usage.nodes[time_extended_landing_id]["landing_capacity"] - vertiport_usage.nodes[time_extended_landing_id]["landing_usage"]
                parking_request_times = [time for time in range(request["request_arrival_time"], max_time + 1, time_step)]
                parking_request_ids = [request["destination_vertiport_id"] + "_" + str(time) for time in parking_request_times]
                hold_space = [vertiport_usage.nodes[landing_time_id]["hold_capacity"] - vertiport_usage.nodes[landing_time_id]["hold_usage"] for landing_time_id in parking_request_ids]
                print(f"Space for {takeoff_space} takeoffs and {landing_space} landings at {origin} and {request['destination_vertiport_id']} for time {request['request_departure_time']} and {request['request_arrival_time']}")
                if not (takeoff_space > 0 and landing_space > 0 and all([hold > 0 for hold in hold_space])):
                    continue
                else:
                    # If so, allocate the flight and move to next flight
                    print(f"Allocating flight {flight_id} with request {request_id}")
                    vertiport_usage.allocate_aircraft(flight["origin_vertiport_id"], request)
                    allocation.append((flight_id, request_id))
                    break
    
    return allocation


def ff_allocation_and_payment(vertiport_usage, flights, timing_info, save_file, initial_allocation): 
    """
    Allocate flights for a given time and set of requests.

    Args:
        graph (VertiportStatus): Graph from which to allocate flights.
        time (int): Time step for which to allocate flights.
        flights (list): List of flights making requests at this time step.
    """
    # Create auxiliary graph and determine allocation
    # Todo: Also determine payment here
    allocation = determine_allocation(vertiport_usage, flights, timing_info)
    # save_allocation(allocation, save_file, timing_info["current_time"], initial_allocation=initial_allocation)

    # Print outputs
    print(f"Allocation\n{allocation}")
    payment = None
    print(f"\nPayment\n{payment}")

    return allocation, payment