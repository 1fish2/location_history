"""Code to analyze Google location history data to get a histogram of travel distances."""

from __future__ import absolute_import, division, print_function

import geopy
import geopy.distance as gd
import json
import numpy as np
import os


location_dtype = [
    ('latitudeE7', 'f8'),
    ('longitudeE7', 'f8'),
    ('timestampMs', 'i8'),
    # altitude, accuracy, velocity, verticalAccuracy, activity
]

GEOCODER = geopy.Nominatim(user_agent='location distance histogram')


def read_history():
    """Read the decompressed Location History JSON file. Return a list[dict]."""
    home = os.environ['HOME']
    path = os.path.join(home, 'Downloads', 'Takeout', 'Location History', 'Location History.json')
    with open(path, 'r') as f:
        history = json.load(f)
    return history['locations']


def dicts_to_array(locations):
    """Convert the location history from a list[dict] to a structured numpy array."""
    length = len(locations)
    array = np.zeros(length, dtype=location_dtype)
    for i, location in enumerate(locations):
        reverse_index = length - i - 1  # the input is in reverse time order
        array[reverse_index] = (
            int(location['latitudeE7']) * 1e-7,
            int(location['longitudeE7']) * 1e-7,
            int(location['timestampMs']),
        )
    return array


def read_history_to_array():
    return dicts_to_array(read_history())


def distance_miles(loc1, loc2):
    """Return the geo distance between two locations in miles.
    Each argument can be a (latitude, longitude) pair or an element of a history array (which prints like a triple but
    has type numpy.void).
    """
    # gp.distance() and geopy.Point() accept
    #   latitude: float latitude in degrees (default 0)
    #   longitude: float longitude in degrees (default 0)
    #   altitude: altitude in km (default 0)
    return gd.distance((loc1[0], loc1[1]), (loc2[0], loc2[1])).miles


def geocode(query):
    """Geocode a query. Return (latitude, longitude).
    The query can be a string or a dict w/keys 'street', 'city', 'county', 'state', 'country', 'postalcode'.
    NOTE: "CA" might geocode as Canada so use "CA, USA" for California.
    """
    location = GEOCODER.geocode(query)
    # Location props: address, altitude, latitude, longitude, point, raw.
    # location[0] is the formatted location string; location[1] is the (latitude, longitude) pair.
    return location[1]

