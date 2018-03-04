from bluesky import settings
# Register settings defaults
settings.set_variable_defaults(prefer_compiled=False)

if settings.prefer_compiled:
    try:
        from . import cgeo as geo
    except ImportError:
        from . import geo
else:
    from . import geo

from .trafficarrays import RegisterElementParameters, TrafficArrays
from . import cachefile
