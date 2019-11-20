""" BlueSky deletion area plugin. This plugin can use an area definition to
    delete aircraft that exit the area. Statistics on these flights can be
    logged with the FLSTLOG logger. """
import numpy as np
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import traf, sim  #, settings, navdb, traf, sim, scr, tools
from bluesky.tools import datalog, areafilter, \
    TrafficArrays, RegisterElementParameters
from bluesky.tools.aero import ft
from bluesky.tools.simtime import timed_function

# Log parameters for the flight statistics log
flstheader = \
    '#######################################################\n' + \
    'FLST LOG\n' + \
    'Flight Statistics\n' + \
    '#######################################################\n\n' + \
    'Parameters [Units]:\n' + \
    'Deletion Time [s], ' + \
    'Call sign [-], ' + \
    'Spawn Time [s], ' + \
    'Flight time [s], ' + \
    'Actual Distance 2D [m], ' + \
    'Actual Distance 3D [m], ' + \
    'Work Done [J], ' + \
    'Latitude [deg], ' + \
    'Longitude [deg], ' + \
    'Altitude [m], ' + \
    'TAS [m/s], ' + \
    'Vertical Speed [m/s], ' + \
    'Heading [deg], ' + \
    'Origin Lat [deg], ' + \
    'Origin Lon [deg], ' + \
    'Destination Lat [deg], ' + \
    'Destination Lon [deg], ' + \
    'ASAS Active [bool], ' + \
    'Pilot ALT [m], ' + \
    'Pilot SPD (TAS) [m/s], ' + \
    'Pilot HDG [deg], ' + \
    'Pilot VS [m/s]'  + '\n'

confheader = \
    '#######################################################\n' + \
    'CONF LOG\n' + \
    'Conflict Statistics\n' + \
    '#######################################################\n\n' + \
    'Parameters [Units]:\n' + \
    'Simulation time [s], ' + \
    'Total number of conflicts in exp area [-]\n'

# Global data
area = None

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():

    # Addtional initilisation code
    global area
    area = Area()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'AREA',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',

        # The update function is called after traffic is updated.
        'update':          area.update,

        # The reset function
        'reset':           area.reset
        }

    stackfunctions = {
        'AREA': [
            'AREA Shapename/OFF or AREA lat,lon,lat,lon,[top,bottom]',
            '[float/txt,float,float,float,alt,alt]',
            area.set_area,
            'Define deletion area (aircraft leaving area are deleted)'
        ],
        'EXP': [
            'EXP Shapename/OFF or EXP lat,lon,lat,lon,[top,bottom]',
            '[float/txt,float,float,float,alt,alt]',
            lambda *args: area.set_area(*args, exparea=True),
            'Define experiment area (area of interest)'
        ],
        'TAXI': [
            'TAXI ON/OFF [alt] : OFF auto deletes traffic below 1500 ft',
            'onoff[,alt]',
            area.set_taxi,
            'Switch on/off ground/low altitude mode, prevents auto-delete at 1500 ft'
        ]
    }
    # init_plugin() should always return these two dicts.
    return config, stackfunctions

class Area(TrafficArrays):
    ''' Traffic area: delete traffic when it leaves this area (so not when outside)'''
    def __init__(self):
        super(Area, self).__init__()
        # Parameters of area
        self.active = False
        self.delarea = ''
        self.exparea = ''
        self.swtaxi = True  # Default OFF: Doesn't do anything. See comments of set_taxi fucntion below.
        self.swtaxialt = 1500.0  # Default OFF: Doesn't do anything. See comments of set_taxi fucntion below.

        self.confinside_all = 0

        # The FLST logger
        self.flst = datalog.crelog('FLSTLOG', None, flstheader)
        self.conflog = datalog.crelog('CONFLOG', None, confheader)

        with RegisterElementParameters(self):
            self.insdel = np.array([], dtype=np.bool) # In deletion area or not
            self.insexp = np.array([], dtype=np.bool) # In experiment area or not
            self.oldalt = np.array([])
            self.distance2D = np.array([])
            self.distance3D = np.array([])
            self.dstart2D = np.array([])
            self.dstart3D = np.array([])
            self.workstart = np.array([])
            self.work = np.array([])
            self.entrytime = np.array([])

    def reset(self):
        ''' Reset area state when simulation is reset. '''
        super().reset()
        self.active = False
        self.delarea = ''
        self.exparea = ''
        self.swtaxi = True
        self.swtaxialt = 1500.0
        self.confinside_all = 0

    def create(self, n=1):
        super(Area, self).create(n)
        self.oldalt[-n:] = traf.alt[-n:]
        self.insdel[-n:] = False
        self.insexp[-n:] = False

    @timed_function('AREA', dt=1.0)
    def update(self, dt):
        ''' Update flight efficiency metrics
            2D and 3D distance [m], and work done (force*distance) [J] '''
        if self.active:
            resultantspd = np.sqrt(traf.gs * traf.gs + traf.vs * traf.vs)
            self.distance2D += dt * traf.gs
            self.distance3D += dt * resultantspd
            self.work += (traf.perf.thrust * dt * resultantspd)

            # Find out which aircraft are currently inside the experiment area, and
            # determine which aircraft need to be deleted.
            insdel = areafilter.checkInside(self.delarea, traf.lat, traf.lon, traf.alt)
            insexp = insdel if not self.exparea else \
                areafilter.checkInside(self.exparea, traf.lat, traf.lon, traf.alt)
            # Find all aircraft that were inside in the previous timestep, but no
            # longer are in the current timestep
            delidx = np.where(np.array(self.insdel) * (np.array(insdel) == False))[0]
            self.insdel = insdel

            # Count new conflicts where at least one of the aircraft is inside
            # the experiment area
            if traf.asas.confpairs_new:
                newconf_unique = {frozenset(pair)
                                  for pair in traf.asas.confpairs_new}
                ac1, ac2 = zip(*newconf_unique)
                idx1 = traf.id2idx(ac1)
                idx2 = traf.id2idx(ac2)
                newconf_inside = np.logical_or(insexp[idx1], insexp[idx2])

                nnewconf_exp = np.count_nonzero(newconf_inside)
                if nnewconf_exp:
                    self.confinside_all += nnewconf_exp
                    self.conflog.log(self.confinside_all)

            # Register distance values upon entry of experiment area
            newentries = np.logical_not(self.insexp) * insexp
            self.dstart2D[newentries] = self.distance2D[newentries]
            self.dstart3D[newentries] = self.distance3D[newentries]
            self.workstart[newentries] = self.work[newentries]
            self.entrytime[newentries] = sim.simt

            # Log flight statistics when exiting experiment area
            exits = self.insexp * np.logical_not(insexp)
            if np.any(exits):
                self.flst.log(
                    np.array(traf.id)[exits],
                    self.create_time[exits],
                    sim.simt - self.entrytime[exits],
                    self.dstart2D[exits] - self.distance2D[exits],
                    self.dstart3D[exits] - self.distance3D[exits],
                    self.workstart[exits] - self.work[exits],
                    traf.lat[exits],
                    traf.lon[exits],
                    traf.alt[exits],
                    traf.tas[exits],
                    traf.vs[exits],
                    traf.hdg[exits],
                    traf.asas.active[exits],
                    traf.pilot.alt[exits],
                    traf.pilot.tas[exits],
                    traf.pilot.vs[exits],
                    traf.pilot.hdg[exits])

            # delete all aicraft in self.delidx
            if len(delidx) > 0:
                traf.delete(delidx)

        # Autodelete for descending with swTaxi:
        if not self.swtaxi:
            delidxalt = np.where((self.oldalt >= self.swtaxialt)
                                 * (traf.alt < self.swtaxialt))[0]
            self.oldalt = traf.alt
            if len(delidxalt) > 0:
                traf.delete(list(delidxalt))

    def set_area(self, *args, exparea=False):
        ''' Set Experiment Area. Aicraft leaving the experiment area are deleted.
        Input can be exisiting shape name, or a box with optional altitude constrainsts.'''
        curname = self.exparea if exparea else self.delarea
        msgname = 'Experiment area' if exparea else 'Deletion area'
        # if all args are empty, then print out the current area status
        if not args:
            return True, f'{msgname} is currently ON (name={curname})' if self.active else \
                         f'{msgname} is currently OFF'

        # start by checking if the first argument is a string -> then it is an area name
        if isinstance(args[0], str) and len(args)==1:
            if areafilter.hasArea(args[0]):
                # switch on Area, set it to the shape name
                if exparea:
                    self.exparea = args[0]
                else:
                    self.delarea = args[0]

                self.active = True
                self.flst.start()
                self.conflog.start()
                return True, f'{msgname} is set to {args[0]}'
            if args[0][:2] =='OF':
                # switch off the area and reset the logger
                self.active = False
                return True, f'{msgname} is switched OFF'
            if args[0][:2] == 'ON':
                if not self.name:
                    return False, 'No area defined.'
                else:
                    self.active = True
                    return True, f'{msgname} switched ON (name={curname})'
            # shape name is unknown
            return False, 'Shapename unknown. ' + \
                'Please create shapename first or shapename is misspelled!'
        # if first argument is a float -> then make a box with the arguments
        if isinstance(args[0],(float, int)) and 4<=len(args)<=6:
            self.active = True
            if exparea:
                self.exparea = 'EXPAREA'
                areafilter.defineArea('EXPAREA', 'BOX', args[:4], *args[4:])
            else:
                self.delarea = 'DELAREA'
                areafilter.defineArea('DELAREA', 'BOX', args[:4], *args[4:])
            self.flst.start()
            self.conflog.start()
            return True, f'{msgname} is ON. Area name is: {"EXP" if exparea else "DEL"}AREA'

        return False,  'Incorrect arguments' + \
                       '\nAREA Shapename/OFF or\n Area lat,lon,lat,lon,[top,bottom]'

    def set_taxi(self, flag,alt=1500*ft):
        ''' Taxi ON/OFF to autodelete below a certain altitude if taxi is off'''
        self.swtaxi = flag # True =  taxi allowed, False = autodelete below swtaxialt
        self.swtaxialt = alt
