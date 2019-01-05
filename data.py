"""Code to analyze Google location history data to get a histogram of travel distances.

The activity types:
    "EXITING_VEHICLE",
    "IN_FOUR_WHEELER_VEHICLE",
    "IN_RAIL_VEHICLE",
    "IN_ROAD_VEHICLE",
    "IN_TWO_WHEELER_VEHICLE",
    "IN_VEHICLE",
    "ON_BICYCLE",
    "ON_FOOT",
    "RUNNING",
    "STILL",
    "TILTING",
    "UNKNOWN",
    "VALUE",
    "WALKING",
"""

from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from datetime import datetime
import geopy
import geopy.distance as gd
import json
import numpy as np
import os


location_dtype = [  # array internalized from the input data
    ('latitude', 'f8'),
    ('longitude', 'f8'),
    ('timestamp', 'f8'),
    # altitude, accuracy, velocity, verticalAccuracy, activity
]

delta_dtype = [('miles', 'f8'), ('seconds', 'f8'), ('mph', 'f8')]

GEOCODER = geopy.Nominatim(user_agent='location distance histogram')


def dicts_to_array(history):
    """Convert the read-in location history to a structured numpy array."""
    locations = history['locations']  # list[dict]
    length = len(locations)
    array = np.zeros(length, dtype=location_dtype)

    for i, location in enumerate(locations):
        reverse_index = length - i - 1  # reverse the data to forwards time order
        array[reverse_index] = (
            int(location['latitudeE7']) * 1e-7,
            int(location['longitudeE7']) * 1e-7,
            int(location['timestampMs']) * 1e-3,
        )
    return array


def distance_miles(loc1, loc2):
    """Return the geo distance between two locations in miles.
    Each argument can be a (latitude, longitude) pair or an element of a location array (which prints like a tuple but
    is really a numpy view).
    """
    # gd.great_circle() and geopy.Point() accept
    #   latitude: float latitude in degrees (default 0)
    #   longitude: float longitude in degrees (default 0)
    #   altitude: altitude in km (default 0)
    # gp.distance() is 7x slower but slightly more accurate.
    return gd.great_circle((loc1[0], loc1[1]), (loc2[0], loc2[1])).miles


def compute_deltas(locations):
    """Compute the delta miles, delta seconds, and MPH array from a locations array."""
    deltas = np.zeros(len(locations), dtype=delta_dtype)
    previous = locations[0]

    for i in range(1, len(locations)):
        location = locations[i]
        miles = distance_miles(location, previous)
        seconds = location['timestamp'] - previous['timestamp']
        mph = miles / seconds * 3600.0
        deltas[i] = (miles, seconds, mph)
        previous = location

    return deltas


def find_years(locations):
    """Find the {year: start_index} for each year in locations, returning an OrderedDict with a sentinel."""
    result = OrderedDict()
    previous_year = None

    for i, location in enumerate(locations):
        year = datetime.utcfromtimestamp(location['timestamp']).year
        if year != previous_year:
            result[year] = i
            previous_year = year

    if previous_year:
        result[previous_year + 1] = len(locations)

    return result


def geocode(query):
    """Geocode a query. Return (latitude, longitude).
    The query can be a string or a dict w/keys 'street', 'city', 'county', 'state', 'country', 'postalcode'.
    NOTE: "CA" might geocode as Canada so use "CA, USA" for California.
    """
    location = GEOCODER.geocode(query)
    # Location props: address, altitude, latitude, longitude, point, raw.
    # location[0] is the formatted location string.
    # location[1] is the (latitude, longitude) pair.
    return location[1]


class HistoryData(object):
    def __init__(self):
        self.locations = None   # numpy array(dtype=location_dtype)
        self.years = None       # numpy array(dtype=delta_dtype)
        self.deltas = None      # OrderedDict(year -> start_index)

    def read(self, path=None):
        """Read in the Location History JSON file."""
        if not path:
            home = os.environ['HOME']
            path = os.path.join(home, 'Downloads', 'Takeout', 'Location History', 'Location History.json')

        with open(path, 'r') as f:
            history = json.load(f)

        self.locations = dicts_to_array(history)
        self.years = find_years(self.locations)
        self.deltas = compute_deltas(self.locations)
