"""Querying the Google geocode API"""

#Global
import requests

#Local  

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

if __name__ == "__main__":
    print("google geocode queries")
    #get_latlong("ord airport")
    print(get_latlong('MSN Airport'))
