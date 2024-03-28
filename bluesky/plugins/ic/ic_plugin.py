"""
    BlueSky-IC plugin template. This plugin will instantiate the UAVs,
    create a scheduler, keep track of the current state of the world
"""

from random import randint
import numpy as np
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import core, stack, traf  #, settings, navdb, sim, scr, tools
import json

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognizes this file as a plugin.
def init_plugin():
    ''' Plugin initialization function. '''
    # Instantiate our example entity
    ic_plugin = Allocation()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'EXAMPLE',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }

    # init_plugin() should always return a configuration dict.
    return config

# Probably should be in a different file
class CurrentState():
    def __init__(self):
        self.current_requests = []
        # Define other attributes
        self.aircraft_locations = {}  # Dictionary to store current locations of aircraft
        self.route_allocations = {}  # Dictionary to store allocations to specific routes

    def update_allocation(self, time):
        self.allocation = {}  # Update allocation based on current state and time
        # Update aircraft locations and route allocations based on time and other factors

class Scheduler():
    def __init__(self):
        # Initialize scheduler attributes
        pass

    def schedule(self):
        # Schedule parameters to be added
        pass

# Your existing code for Allocation and Example classes
class Allocation(core.Entity):
    ''' Example new entity object for BlueSky. '''
    def __init__(self):
        super().__init__()
        # All classes deriving from Entity can register lists and numpy arrays
        # that hold per-aircraft data. This way, their size is automatically
        # updated when aircraft are created or deleted in the simulation.
        # read json file 
        self.aircraft_data = {}  # Let's get a name to refer to these vehicles

    def create(self, n=1):
        ''' This function gets called automatically when new UAVs are created. '''
        # Don't forget to call the base class create when you reimplement this function!
        super().create(n)
        # After base creation we can change the values in our own states for the new aircraft
        #create uavs based on JSON FILE 
        # I want to change this later so that the file is specified with arguments and not hardcoded
        with open('case1.json', 'r') as file:
            self.aircraft_data = json.load(file)
        
        for _ in range(n):
            self.create_uav(self.aircraft_data)

    def create_uav(self, data):
        ''' Create UAV based on data from JSON file. '''
        # data to add: speed, ...?
        pass
 
    @core.timed_function(name='example', dt=5)
    def update(self):
        ''' Periodic update function for our example entity. ''' 
        # this function will call the next allocations

 
    def get_allocations():
        return

    def keep_track():
        pass
