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
import matplotlib.pyplot as plt
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
    """Compute the adjacent point delta miles, delta seconds, and MPH array from a locations array."""
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
    """Find the year index ranges in locations, returning an OrderedDict {year: slice}."""
    starts = []
    previous_year = None

    for index, location in enumerate(locations):
        year = datetime.utcfromtimestamp(location['timestamp']).year
        if year != previous_year:
            starts.append((year, index))
            previous_year = year

    starts.append((None, len(locations)))
    result = OrderedDict()

    for i in range(len(starts) - 1):
        year, start = starts[i]
        stop = starts[i + 1][1]
        result[year] = slice(start, stop)

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


def driving_indexes(deltas):
    """Return a list of indexes into deltas[] with feasible driving leg values."""
    miles_limit = 200
    mph_limit = 100
    seconds_limit = miles_limit / mph_limit * 3600

    def test(delta):
        return delta['miles'] < miles_limit and delta['seconds'] < seconds_limit and delta['mph'] < mph_limit

    return [i for i in range(len(deltas)) if test(deltas[i])]


class HistoryData(object):
    def __init__(self):
        self.locations = None   # numpy array(dtype=location_dtype)
        self.years = None       # OrderedDict(year -> index slice)
        self.deltas = None      # numpy array(dtype=delta_dtype)

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

    def year_labels(self):
        return [str(year) for year in self.years]

    def driving_leg_deltas_by_year(self):
        """Return a list of driving leg delta arrays, one per year."""
        yearly_deltas = [self.deltas[index_range] for index_range in self.years.values()]
        data = [yd[driving_indexes(yd)] for yd in yearly_deltas]
        return data

    def home_trip_lengths(self, year, home_query):
        """Return a list of trip distances in the given year, segmented by travel from "home"."""
        home = geocode(home_query)
        fence = 5280 / 200  # miles radius for "home"
        result = []
        trip_miles = 0
        year_slice = self.years[year]

        for loc, delta in zip(self.locations[year_slice], self.deltas[year_slice]):  # TODO(jerry): Does this copy all?
            if gd.great_circle(home, (loc[0], loc[1])).miles < fence:
                if trip_miles > 0:
                    result.append(trip_miles)
                    trip_miles = 0
            else:
                trip_miles += delta['miles']

        if trip_miles > 0:
            result.append(trip_miles)

        return result


def distance_speed_histogram(history):
    """Plot a (distance, speed) histogram from the location driving data."""
    indexes = driving_indexes(history.deltas)
    d = history.deltas[indexes]
    plt.hist((d['miles'], d['mph']), label=('miles', 'MPH'), log=True, histtype='bar', linewidth=2)
    plt.legend()
    plt.title('Driving legs')
    plt.show()


def histogram_by_year(history, field='mph'):
    """Plot a histogram by year from the location driving data.
    `field` is one of {'seconds', 'miles', 'mph'}.
    """
    data = [d[field] for d in history.driving_leg_deltas_by_year()]
    labels = history.year_labels()
    plt.hist(data, label=labels, log=True, histtype='step')
    plt.legend()
    plt.title('Driving legs, ' + field)
    plt.show()


def histogram_of_trips(history, year, home_query):
    """Plot a histogram of the given trip lengths."""
    trips = history.home_trip_lengths(year, home_query)
    plt.hist(trips, label='miles', log=True, histtype='stepfilled')
    plt.legend()
    plt.title('Trip lengths in {}'.format(year))
    plt.show()


def trips_by_day(history, year):
    result = []
    year_slice = history.years[year]
    trip_miles = 0
    previous_day = None
    deltas = history.deltas[year_slice]
    drives = driving_indexes(deltas)
    filtered_deltas = deltas[drives]
    filtered_locations = history.locations[year_slice][drives]

    for loc, delta in zip(filtered_locations, filtered_deltas):
        dt = datetime.utcfromtimestamp(loc['timestamp'])
        m_d = (dt.month, dt.day)
        if m_d != previous_day:
            result.append(trip_miles)
            previous_day = m_d
            trip_miles = 0
        else:
            trip_miles += delta['miles']
    return result


def histogram_of_trips_by_day(history, year):
    """Plot a histogram of trips by day in the given year."""
    trips = trips_by_day(history, year)
    plt.hist(trips, label='miles', log=True, histtype='stepfilled')
    plt.legend()
    plt.title('Trip lengths in {}'.format(year))
    plt.show()
