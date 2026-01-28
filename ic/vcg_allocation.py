import copy
from fleet_vcg_allocation import build_auxiliary, determine_allocation, save_allocation


def vcg_payment(fleets, flights_allocated, flights, fleet_members_allocated, vertiport_usage, timing_info, congestion_info, SW):
    fleet_payments = [] # Todo: Change to dict with fleet IDs as keys
    for fleet_id, fleet in fleets.items():
        # Determine which fleet members were allocated
        SW_minus_i = SW
        for flight_id in fleet["members"]:
            if flight_id in flights_allocated.keys():
                fleet_members_allocated[fleet_id].append(flight_id)
                # Find social welfare excluding the allocated members of the fleet
                allocated_request_id = flights_allocated[flight_id]
                SW_minus_i -= fleet["rho"] * flights[flight_id]["requests"][allocated_request_id]["bid"]
            else:
                print(flight_id)
                print(flights)
                SW_minus_i -= fleet["rho"] * flights[flight_id]["requests"]["000"]["bid"]

        # Get the social welfare (SW) for the allocation without each flight in the fleet
        payment = 0
        for flight_id in fleet["members"]:
            adjusted_flights = copy.deepcopy(flights)
            adjusted_flight_requests = copy.deepcopy(flights[flight_id]["requests"])
            for request_id in adjusted_flight_requests.keys():
                adjusted_flight_requests[request_id]["bid"] = 0
            adjusted_flights[flight_id]["requests"] = adjusted_flight_requests
            auxiliary_graph, unique_departure_times = build_auxiliary(vertiport_usage, adjusted_flights, timing_info, congestion_info)
            _, SW_alternate_allocation = determine_allocation(vertiport_usage, adjusted_flights, auxiliary_graph, unique_departure_times)
            assert SW_alternate_allocation >= 0, "Social welfare should be non-negative"

            # Add vehicle payment to fleet payment
            payment += SW_alternate_allocation - SW_minus_i

        # Scale and save the fleet payment
        fleet_payments.append(payment * fleet["rho"])

    print(f"\nFleet payment\n{fleet_payments}")
    return fleet_payments

def vcg_allocation_and_payment(vertiport_usage, flights, timing_info, congestion_info, fleets, save_file, initial_allocation, payment_calc=True, save=True):
    """
    Allocate flights and determine payment using standard VCG mechanism.
    """
    auxiliary_graph, unique_departure_times = build_auxiliary(vertiport_usage, flights, timing_info, congestion_info)
    allocation, SW = determine_allocation(vertiport_usage, flights, auxiliary_graph, unique_departure_times)
    flights_allocated = {flight_id: request_id for flight_id, request_id in allocation}
    fleet_members_allocated = {fleet_id: [] for fleet_id in fleets.keys()}
    # Print outputs
    print(f"Allocation\n{allocation}")
    print(f"Original Social Welfare: {SW}")

    if payment_calc:
        payment = vcg_payment(fleets, flights_allocated, flights, fleet_members_allocated, vertiport_usage, timing_info, congestion_info, SW)
    else:
        payment = []
    if save:
        save_allocation(allocation, save_file, timing_info["current_time"], initial_allocation=initial_allocation)

    return allocation, payment, SW
