#Global
import json
import reimport math
import requests
import dns
import dns.reversename

#Approx radius of Earth in meters
Rad = 6371000

#Local
import query_google


def disance_val(ip1, ip2, provider1, provider2):
    ip1_host = dns.reversename.from_address(ip1)
    ip2_host = dns.reversename.from_address(ip2)
    
    apdict = json.loads(open('airports_dict.json', 'r').read())
    
    if provider1 == "adnxs":
        lat1, long1 = parse_adnxs(ip1_host)
    elif provider1 == "cdn77":
        lat1, long1 = parse_cdn77(ip1_host, apdict)
    elif provider1 == "cdnetworks":
        lat1, long1 = parse_cdnetworks(ip1_host, apdict)
    elif provider1 == "cloudfront":
        lat1, long1 = parse_cloudfront(ip1_host, apdict)
    elif provider1 == "google":
        lat1, long1 = parse_google(ip1_host, apdict)
    else:
        return -1
    
    if provider2 == "adnxs":
        lat2, long2 = parse_adnxs(i2_host)
    elif provider1 == "cdn77":
        lat2, long2 = parse_cdn77(ip2_host, apdict)
    elif provider1 == "cdnetworks":
        lat2, long2 = parse_cdnetworks(ip2_host, apdict)
    elif provider1 == "cloudfront":
        lat2, long2 = parse_cloudfront(ip2_host, apdict)
    elif provider1 == "google":
        lat2, long2 = parse_google(ip2_host, apdict)
    else:
        return -1
        
    return get_dist(lat1, long1, lat2, long2)

"""Querying the Google geocode API"""


KEY = 'AIzaSyDWljKnpCD1iO4mByHrhLPBVklFtKvEiKU'

def get_latlong(qstring):
    """Best-effort latitude/longitude query.  Takes any kind of query string and
    returns the (lat, long) of the first response object that contains
    location data.  Returns (None, None) for unavailable if no location data found.
    Better query strings -> better results from this function."""
    qplus = qstring.replace(' ', '+')
    qresp = requests.get('https://maps.googleapis.com/maps/api/geocode/json?address='+qplus+'&key='+KEY)
    if qresp.status_code != 200:
        return (None, None)
    qdata = qresp.json()
    lat = 0.0
    lng = 0.0
    for res in qdata['results']:
        for loc in res['geometry']['viewport'].keys():
            lat = float(res['geometry']['viewport'][loc]['lat'])
            lng = float(res['geometry']['viewport'][loc]['lng'])
            break
    return (lat, lng)
    
"""Contains functions for parsing and looking up the latitude and longitude of
each CDN in the current Fury Route set as of 4/26/16

query_google.get_latlong(string)"""

def parse_adnxs(adnxs_hostname):
    """Queries lat/lng for an adnxs hostname.
    Assumption is that 3rd to last string in dot separated hostname
    stripped of non-alphabet chars will contain an airport code

    Update: Appears adnxs not based on airport codes:
        nym = New York
        sin = Singapore
        lax = Los Angeles
    how to parse these completely? More measurement?"""
    adnxs_comps = adnxs_hostname.split('.')
    nonum =  re.compile('[^a-zA-Z]')
    acode = nonum.sub('', adnxs_comps[-3])
    print(acode)
    return query_google.get_latlong('Airport '+acode)

def parse_cdn77(cdn77_hostname, apdict):
    """CDN77 structure - first element of dot separated hostname
    has location name up until start of numbers, try three letters
    as airports"""
    location = cdn77_hostname.split('.')[0]
    locparts = location.split('-')
    querystring = locparts[0]
    for loc in locparts[1:]:
        try:
            int(loc)
            break
        except ValueError:
            querystring += '+'+loc
    if len(querystring) == 3:
        #return query_google.get_latlong('Airport '+querystring)
        try:
            return (float(apdict[querystring.upper()]['lat']), float(apdict[querystring.upper()]['lon']))
        except KeyError:
            pass
    return query_google.get_latlong(querystring)

def parse_cdnetworks(cdnetworks_hostname, apdict):
    """Cdnetworks structure - second component of second part of hostname separated by
    hyphen is airportcode"""
    acode = cdnetworks_hostname.split('.')[1].split('-')[1]
    #return query_google.get_latlong('Airport '+acode)
    return (float(apdict[acode.upper()]['lat']), float(apdict[acode.upper()]['lon']))

def parse_cloudfront(cloudfront_hostname, apdict):
    """4th last component stripped of numbers is an airport code"""
    nonum = re.compile('[^a-zA-Z]')
    acode = nonum.sub('', cloudfront_hostname.split('.')[1])
    return (float(apdict[acode.upper()]['lat']), float(apdict[acode.upper()]['lon']))

def parse_google(google_hostname, apdict):
    """first component of hostname up to the first number is an airport code
    Google uses telmex.net.ar in South America, return Argentina for those"""
    if '.ar' in google_hostname:
        return query_google.get_latlong('Argentina')
    apcontainer = google_hostname.split('.')[0]
    acode = ''
    for char in apcontainer:
        try:
            int(char)
            break
        except ValueError:
            acode += char
    if apdict[acode.upper()]['status'] != 1:
        return query_google.get_latlong('Airport '+acode)
    return (float(apdict[acode.upper()]['lat']), float(apdict[acode.upper()]['lon']))


def get_dist(lat1, long1, lat2, long2):
    """Finds the great circle distance between a pair of coordinates using the haversine formula
       :param lat1: Latitude of first point represented as a decimal number in degrees (negative denotes South)
       :param long1: Longitude of first point represented as a deciman number in degrees (negative denotes West)
       :param lat2: Latitude of second point (represented same as lat1)
       :param long2: Longitude of second point (represented same as long1)

       :return: Distance between point (lat1, long1) and point (lat2, long2) in meters
    """
    
    # Variable setup
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    delta_p = math.radians(lat2-lat1)
    delta_l = math.radians(long2-long1)

    # Setup first and last terms
    term_1 = math.sin(delta_p/2.0)
    sq_term_1 = term_1*term_1

    term_2 = math.sin(delta_l/2.0)
    sq_term_2 = term_2*term_2

    # Calculate c of haversine formula
    a = sq_term_1 + math.cos(p1)*math.cos(p2)*sq_term_2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))

    # distance is Radius of earth * calcualted c
    return Rad*c
