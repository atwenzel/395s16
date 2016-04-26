"""Contains functions for parsing and looking up the latitude and longitude of
each CDN in the current Fury Route set as of 4/26/16

query_google.get_latlong(string)"""

#Global
import json
import re

#Local
import query_google

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

if __name__ == "__main__":
    print("CDN location parsing")
    apdict = json.loads(open('airports_dict.json', 'r').read())
    #print(parse_adnxs('float.2046.bm-impbus.prod.nym2.adnexus.net'))
    #print(parse_cdn77('sao-paulo-18.cdn77.com'))
    #print(parse_cdn77('lax-1.cdn77.com', apdict))
    #print(parse_cdnetworks('i0-h0-s1042.p0-mia.cdngp.net', apdict))
    #print(parse_cloudfront('server-52-85-16-241.mxp4.r.cloudfront.net', apdict))
    #print(parse_cloudfront('server-54-230-228-98.waw50.r.cloudfront.net', apdict))
    #print(parse_google('hkg07s21-in-f4.1e100.net', apdict))
    print(parse_google('host25.190-221-162.telmex.net.ar', apdict))
