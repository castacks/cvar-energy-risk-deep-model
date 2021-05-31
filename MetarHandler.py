"""
Script that scrapes data from the IEM ASOS download service for KAGC ASOS
"""
import csv
import datetime
import json
import math
import os
import sys
import time
from io import StringIO
from urllib.request import urlopen

from metar import Metar

now = datetime.datetime.now()

# Number of attempts to download data
MAX_ATTEMPTS = 6
# HTTPS here can be problematic for installs that don't have Lets Encrypt CA
SERVICE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=AGC&data=metar&tz=Etc%2FUTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=1&report_type=2"


def download_data(uri):
    """Fetch the data from the IEM
    The IEM download service has some protections in place to keep the number
    of inbound requests in check.  This function implements an exponential
    backoff to keep individual downloads from erroring.
    Args:
      uri (string): URL to fetch
    Returns:
      string data
    """
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        try:
            data = urlopen(uri, timeout=300).read().decode('utf-8')
            if data is not None and not data.startswith('ERROR'):
                return data
        except Exception as exp:
            print("download_data({}) failed with {}".format(uri, exp))
            time.sleep(5)
        attempt += 1

    print("Exhausted attempts to download, returning empty data")
    return ""


def time_in_range(a, b, x):
    if a <= b:
        return a <= x <= b
    else:
        return a <= x or x <= b


##################################################
# The next two functions calculate the air density
##################################################


def tetens(temp):
    return 611 * math.pow(10, (7.5 * temp / (temp + 237.3)))


def density(temp, dewpt, pressure):
    ps0 = tetens(dewpt)
    ps = tetens(temp)
    relative_humid = (ps0 / ps) * 100

    rho_first = 0.0034848 / (temp + 273.15)
    rho_second = pressure * 100 - 0.0037960 * relative_humid * ps

    return rho_first * rho_second


def calculate_density(dtime):
    """Calculates the mean air density for a given timestamp + 2 hours

    Arguments:
        date {str} -- The date in YYYY-MM-DD format
        time {str} -- The time (EDT) in HH:MM format

    Returns:
        float -- The mean air density
    """

    dtime = dtime + datetime.timedelta(hours=4)

    service = SERVICE

    # Set the request parameters for the day
    service += dtime.strftime('&year1=%Y&month1=%m&day1=%d')
    service += dtime.strftime('&year2=%Y&month2=%m&day2=%d&')

    # Download the data
    data = download_data(service)

    # Read the data as a CSV file
    f = StringIO(data)
    reader = csv.reader(f, delimiter=',')
    metar_data = []

    first = True
    for row in reader:
        if first:
            first = False
            continue
        # Construct the METAR Object from the data we have
        metar_data.append(
            Metar.Metar(row[2],
                        month=int(dtime.strftime('%m')),
                        year=int(dtime.strftime('%Y'))))

    metar_to_read = []

    range_b = (dtime + datetime.timedelta(hours=2)).time()

    dewpts = []
    temps = []
    pressures = []

    # Read every dewpoint, temperature and pressure for every metar in the given time + 2h
    for met in metar_data:
        if time_in_range(dtime.time(), range_b, met.time.time()):
            dewpts.append(met.dewpt.value(units="C"))
            temps.append(met.temp.value(units="C"))
            pressures.append(met.press.value(units="HPA"))

    # Calculate the mean of every value
    temp = sum(temps) / len(temps)
    dewpt = sum(dewpts) / len(dewpts)
    pressure = sum(pressures) / len(pressures)

    # Finally, calculate the density
    return density(temp, dewpt, pressure)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(
            "Please specify a time and date in YYYY-MM-DD HH:MM format (EDT)")
        quit(1)
    rho = calculate_density(sys.argv[1], sys.argv[2])
    print(rho)


