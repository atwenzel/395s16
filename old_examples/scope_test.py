# scope_test.py -- Explore the effects of giving google smaller subnets with
# EDNS queries

import datetime
import dns
import json
import netaddr
#import python_common.edns.edns as edns
import edns
import socket
import sys
import time

GOOGLE = "8.8.8.8"

# "google": "www.google.com",
#        "edgecast": "gp1.wac.v2cdn.net",
#        "alibaba": "img.alicdn.com",
#        #"cloudfront": "p4.qhimg.com",
#        "cloudfront": "st.deviantart.net",
#        #"microsoft": "bat.bing.com", # MISNAMED
#        #"microsoft": "aadg.windows.net.nsatc.net",
#        "cdn77": "922977808.r.cdn77.net",
#        #"chinacache": "ccna00069.c3cdn.net",
#        #"level3": "lvl3.cdnplanet.com.c.footprint.net",
#        "cdnetworks": "cdnw.cdnplanet.com.cdngc.net",
#        #"netdna": "static.cdnplanet.netdna-cdn.com",
#
#        "adnxs": "ib.adnxs.com", # NEW
#        #"gmail": "gmail.com",
#        #"level3": "img01.redtubefiles.com", #Just level 3

providers  = {
    "google": "www.google.com",
    "edgecast": "gp1.wac.v2cdn.net",
    "alibaba": "img.alicdn.com",
    "cloudfront": "st.deviantart.net",
    "cdn77": "922977808.r.cdn77.net",
    "cdnetworks": "cdnw.cdnplanet.com.cdngc.net",
    "adnxs": "ib.adnxs.com"}

"""location_NA = [
            ("132.239.17.224",  "UCSD"),
            ("129.110.125.52", "UTDallas"),
            ("128.223.8.114", "UOregon"),
            ("131.247.2.248", "USF"),
        ]

location_AS = [
            ("137.189.98.210", "CUHK"),
            ("222.197.180.139", "UESTC"),
            ("202.112.28.98",  "SJTU - Harbin"),
            ("143.248.55.129", "KAIST"),
        ]


location_EU = [
            ("188.44.50.106", "Moscow State University"),
            ("130.192.157.138", "DIT Italy"),
            ("62.108.171.76", "KUT Poland"),
            ("129.69.210.97", "UStuttgart"),
        ]


location_SA = [
            ("200.19.159.34", "UFMG - BR"),
            ("200.17.202.194", "C3SL - BR"),
            ("190.227.163.141", "ITBA - AR"),
            ("157.92.44.101", "UBA - AR"),
        ]

location_OC = [
            ("130.194.252.8", "Monash University"),
            ("156.62.231.242", "AUT University - NZ"),
            ("130.195.4.68", "Victoria University Wellington"),
            ("139.80.206.132", "UOtago - NZ"),
        ]"""

location_NA = [
            ("132.239.17.224",  "UCSD"),
            ("129.110.125.52", "UTDallas"),
            ("128.223.8.114", "UOregon"),
            ("131.247.2.248", "USF"),
            ("198.108.101.61", "EMich"),
        ]

location_AS = [
            ("137.189.98.210", "CUHK"),
            ("222.197.180.139", "UESTC"),
            ("202.112.28.98",  "SJTU - Harbin"),
            ("143.248.55.129", "KAIST"),
            ("219.243.208.62", "Tsinghua University"),
        ]

location_EU = [
            ("188.44.50.106", "Moscow State University"),
            ("130.192.157.138", "DIT Italy"),
            ("62.108.171.76", "KUT Poland"),
            ("129.69.210.97", "UStuttgart"),
            ("130.237.50.124", "KTH RIT"),
        ]


location_SA = [
            ("200.19.159.34", "UFMG - BR"),
            ("200.17.202.194", "C3SL - BR"),
            ("190.227.163.141", "ITBA - AR"),
            ("157.92.44.101", "UBA - AR"),
            ("200.0.206.169", "Redclara - BR"),

        ]

location_OC = [
            ("130.194.252.8", "Monash University"),
            ("156.62.231.242", "AUT University - NZ"),
            ("130.195.4.68", "Victoria University Wellington"),
            ("139.80.206.132", "UOtago - NZ"),
            ("130.217.77.2", "Waikato"),
        ]


#locations = [location_NA, location_AS, location_EU, location_SA, location_OC]
locations = {
    0: location_NA,
    1: location_AS,
    2: location_EU,
    3: location_SA,
    4: location_OC}

def scope_test(prov_name):
    #TODO: learn how for loops work
    times = 0
    while times != 3:
        results = {}
        scopes = [0, 4, 8, 12, 16, 20, 24, 28, 32]
        counter = 0
        for loc in locations.keys():
            for server in locations[loc]:
                for scope in scopes:
                    results[counter] = {}
                    results[counter]['location'] = loc
                    results[counter]['server'] = server[0]
                    results[counter]['scope'] = scope
                    results[counter]['response'] = edns.do_query(GOOGLE, providers[prov_name], server[0], scope, timeout=4)
                    del results[counter]['response']['dns_results']
                    counter += 1
        outfile = open(prov_name+'_scope_results_'+str(times)+'.json', 'w')
        outfile.write(json.dumps(results))
        outfile.close()
        times += 1
        time.sleep(1)

def scope_test_32(prov_name):
    #only /32 requests
    for test in range(0, 3):
        results = {}
        counter = 0
        for loc in locations.keys():
            for server in locations[loc]:
                results[counter] = {}
                results[counter]['location'] = loc
                results[counter]['server'] = server[0]
                results[counter]['scope'] = 32
                results[counter]['response'] = edns.do_query(GOOGLE, providers[prov_name], server[0], 32, timeout=4)
                del results[counter]['response']['dns_results']
                counter += 1
        outfile = open(prov_name+'_scope_results_32_'+str(test)+'.json', 'w')
        outfile.write(json.dumps(results))
        outfile.close()
        time.sleep(1)
    print(results)

def scope_test_32_tod(prov_name):
    for test in range(0, 3):
        results = {}
        counter = 0
        for loc in locations.keys():
            for server in locations[loc]:
                results[counter] = {}
                results[counter]['location'] = loc
                results[counter]['server'] = server[0]
                results[counter]['scope'] = 32
                results[counter]['timestamp'] = str(datetime.datetime.now())
                try:
                    results[counter]['response'] = edns.do_query(GOOGLE, providers[prov_name], server[0], 32, timeout=4)
                    del results[counter]['response']['dns_results']
                except dns.exception.Timeout:
                    pass
                counter += 1
        day = datetime.datetime.now().day
        month = datetime.datetime.now().month
        hour = datetime.datetime.now().hour
        outfile = open('/shared/furyroute/time_of_day_try2/'+prov_name+'_scope_results'+'_'+str(month)+'_'+str(day)+'_'+str(hour)+'__'+str(test)+'.json', 'w')
        outfile.write(json.dumps(results))
        outfile.close()
        time.sleep(1)


if __name__ == "__main__":
    scope_test('google')
    #scope_test_32('google')
    """times = 0
    prov_name = sys.argv[1]
    while times != 3:
        results = {}
        #scopes = [8, 9, 10, 11, 12, 13, 14, 15, 16]
        scopes = [0, 8, 16, 24, 32]
        #scopes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        #  ^I've made poor life decisions 
        counter = 0
        for key in locations.keys():
            for server in locations[key]:
                for scope in scopes:
                    results[counter] = {}
                    results[counter]['location'] = key
                    results[counter]['server'] = server[0]
                    results[counter]['scope'] = scope
                    #results[counter]['response'] = edns.do_query(GOOGLE, 'www.google.com', server[0], scope)
                    #prov = socket.gethostbyname(providers['google'])
                    results[counter]['response'] = edns.do_query(GOOGLE, providers[prov_name], server[0], scope, timeout=2)
                    #resp = edns.do_query(GOOGLE, providers['google'], server[0], scope, timeout=2)
                    #for rec in resp['records']:
                    #    print("SECOND EDNS", edns.do_query(GOOGLE, rec.split(' ')[2], server[0], scope, timeout=2))
                    #print(resp)
                    counter += 1
        outfile = open(prov_name+'_scope_results_'+str(times)+'.json', 'w')
        outfile.write(json.dumps(results))
        outfile.close()
        times += 1
        time.sleep(1)"""

    #print(edns.do_query(GOOGLE, 'google.com', locations[4][0][0], 0))

"""def check_name(name):
    Load a name file, check each name

    host_name = name.strip()

    location_set_list = location_scan(name)
    
    #print host_name, location_list[4][0]
    res = edns.do_query(GOOGLE, host_name, 
                        #location_set_list[4][0],
                        location_scan(name)[4][0],
                        24)


    print res
    if res['rcode'] != 0:
        return
        #raise RuntimeError("Bad DNS response! (Nonzero Rcode)")

    scope = res['client_scope']
        
    time.sleep(1)

    if scope != 0:
        #print res['records'] 

        print host_name

def location_scan(name):
    Consider the behavior of google responses by location

    location_set_list = [location_NA,
                         location_EU,
                         location_AS,
                         location_SA,
                         location_OC,]


    # For each location, build a dictionary of sets showing the differences
    # between the subnet sizes within a location and across a locaiton

    #change this
    return location_set_list


if __name__ == "__main__":

    name = sys.argv[1]

    check_name(name)"""
