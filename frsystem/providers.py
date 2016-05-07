# A file to hold a set of all the providers, by name and domain

import json
import cPickle as pickle
import sys

PROV = {
        "google": "www.google.com",
        "edgecast": "gp1.wac.v2cdn.net",
        "alibaba": "img.alicdn.com",
        "cloudfront": "p4.qhimg.com",
        "msn": "bat.bing.com", # MISNAMED
        #"aws": "a.deviantart.net", #MISNAMED
        #"adnxs": "ib.adnxs.com", # NEW
        #"gmail": "gmail.com",
       }

def load_exported_dict(file_name):
    pkl_file = open(file_name, 'rb')

    data1 = pickle.load(pkl_file)

    #data1["google"] = "www.google.com"
    global PROV

    for classic in PROV:
        # Put the old ones in the dict
        data1[classic] = PROV[classic]

    # Coerce to strings because dns library is picky
    data1 = {x: str(data1[x]) for x in data1}
    print data1

    PROV = data1

    return

def load_json_complex(file_name):

    def_prov = {
        "google": "www.google.com",
        "edgecast": "gp1.wac.v2cdn.net",
        "alibaba": "img.alicdn.com",
        #"cloudfront": "p4.qhimg.com",
        "cloudfront": "st.deviantart.net",
        #"microsoft": "bat.bing.com", # MISNAMED
        #"microsoft": "aadg.windows.net.nsatc.net",
        "cdn77": "922977808.r.cdn77.net",
        #"chinacache": "ccna00069.c3cdn.net",
        #"level3": "lvl3.cdnplanet.com.c.footprint.net",
        "cdnetworks": "cdnw.cdnplanet.com.cdngc.net",
        #"netdna": "static.cdnplanet.netdna-cdn.com",

        "adnxs": "ib.adnxs.com", # NEW
        #"gmail": "gmail.com",
        #"level3": "img01.redtubefiles.com", #Just level 3
       }

    data1 = {}

    if file_name == "None":
        data1 = def_prov

    else:
        with open(file_name) as json_file:
            raw_list = json.load(json_file)
            

            for cluster in raw_list:
                name = cluster.keys()[0]
                domain = name
                # Just name it by the domain for the moment
                data1[domain] = name
        
        # Copy the must haves
        #data1["google"] = def_prov["google"]
        for classic in def_prov:
            # Put the old ones in the dict
            data1[classic] = def_prov[classic]
    

    return (def_prov, data1)


# Old tier classifications for the providers..
narrow = ["google"]

#medium = ["edgecast", "cloudfront", "msn"]
medium = ["msn", "cloudfront"]

#wide = ["alibaba"]
wide = ["edgecast", "alibaba"]


