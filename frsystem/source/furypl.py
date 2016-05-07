"""These are the functions to be performed on PlanetLab to assist and augment the main Fury Route program"""

#Global
import commands
import json
import os.path
import random
import sys

import collections
#Local

def get_ping_parsed(src, dst):
    text_ping = get_ping(src, dst)
    try:
        stats = text_ping.split('=')[1][1:-3]
    except IndexError:
        print("get_ping_parsed: bad response received")
        print(src, dst)
        return None
    min_rtt, avg_rtt, max_rtt, mdev = stats.split('/')
    
    return (min_rtt, avg_rtt, max_rtt, mdev)

def get_ping(src, dst):
    rawping = commands.getstatusoutput('./insertions/on_node.sh '+src+' "ping -c 4 '+dst+'"')
    #rawping = commands.getstatusoutput('./insertions/on_node.sh '+src+' "ping -c 1 '+dst+'"')
    rawstats = rawping[1].split('\n')[-1]
    return rawstats

def get_traceroute(src, dst):
    rawtrace = commands.getstatusoutput('./insertions/on_node.sh '+src+' "traceroute '+dst+'"')
    print(rawtrace)

def scrambled(inl):
    outl = inl[:]
    random.shuffle(outl)
    return outl

def build_test_config(numsrc, numdst):
    """Uses the list of usable nodes to build a test subset for running Fury Route"""
    #TODO: This is gross. It probably scrambles way more than necessary. Rewrite it
    try:
        usablefile = open('data/usable_hostnames.dat', 'r')
    except IOError:
        print("Error, hostnames file not found, please build PL Data")
        sys.exit(-1)
    hosts = []
    for line in usablefile.readlines():
        hosts.append(line.strip('\n'))
    config = {}
    hosts = scrambled(hosts)
    #for i in range(0, numsrc):
    #    newsrc = hosts[i]
    #    while newsrc in config.keys():
    #        hosts = scrambled(hosts)
    #        newsrc = hosts[i]
    #    config[newsrc] = []
    for i in hosts:
        config[i] = []

    for key in config.keys():
        #for i in range(0, numdst):
        #    print(i)
        #    newdst = hosts[i]
        #    while newdst == key or newdst in config[key]:
        #        hosts = scrambled(hosts)
        #        newdst = hosts[i]
        #    config[key].append(newdst)
        for i in hosts:
            if i == key:
                continue
            config[key].append(i)

    outfile = open('data/curr_config_FULL.json', 'w')
    outfile.write(json.dumps(config))
    outfile.close()
    return config

def run_test_loop(config):
    for key in config:
        for host in config[key]:
            #do something with host as dst and key as src
            print(0)
            sys.exit(-1)

def test_ping(config):
    for origin in config:
        print ("Testing %s"%(origin))
    
        for dest in config[origin]:
            ping_mean = float(get_ping_parsed(origin, dest)[1])
            print "\t%s: %2.2f"%(dest, ping_mean)



########################################3

# Actual min ping
#
########################################
"""This file defines functions to build a test config in which, for each source,
any difference between the pings from a source to it's destinations will not be smaller
than some minimum value

Strategy:
    pick NUMSRC sources randomly
    for source
        pick NUMDST destinations randomly, keep list of hosts - chosen - source
        build a dict of all ping differences, keep average ping
        while violating ping pairs exist
            for violation
                remove pair with ping closest to average ping
                add new host and random, recalculate dict and average ping"""

def sample_list(l):
    """Takes a list and returns a random element"""
    #outl = l[:]
    #random.shuffle(outl)
    return random.sample(l, 1)[0]

def read_hosts(hpath):
    """Takes a path to a list of PlanetLab hosts and returns a list with no \n chars"""
    hlist = []
    hfile = open(hpath, 'r')
    for line in hfile.readlines():
        hlist.append(line.strip('\n'))
    return hlist

def build_config_min_ping_fixed(source_file, numdst, mindiff):
    """Performs the above algorithm.  numsrc is the number of source nodes, 
    numdst is the number of destination nodes for each source and minping is 
    the threshold such that no difference of any two pings(src, dst[n]) may be smaller
    
    Used a fixed file that has source nodes!
    """
    #read in hosts
    hosts = read_hosts('data/usable_hostnames.dat')

    # Load up the fixed source list
    source_list = read_hosts(source_file)
    conf = {}
    for host in source_list:
        conf[host] = []

    #pick initial dst nodes
    for src in conf:
        while len(conf[src]) != numdst:
            newdst = sample_list(hosts)
            while newdst == src or newdst in conf[src]:
                newdst = sample_list(hosts)
            conf[src].append(newdst)

    #now keep checking dst nodes and replacing bad ones
    for src in conf:
        violators, pings = validate_pingset(src, conf[src], mindiff, {})
        while len(violators) != 0:
            print("violating nodes: ", violators)
            for hv in violators:
                try:
                    conf[src].remove(hv)
                except ValueError:
                    print(conf[src], hv)
                    sys.exit(-1)
                try:
                    del pings[hv]
                except KeyError:
                    pass
            while len(conf[src]) != numdst:
                newdst = sample_list(hosts)
                while newdst == src or newdst in conf[src]:
                    newdst = sample_list(hosts)
                conf[src].append(newdst)
            violators, pings = validate_pingset(src, conf[src], mindiff, pings)
        print("This source is good")
        violators = []
        pings = {}
    print("Found a valid conf")
    outfile = open('data/curr_config.json', 'w')
    outfile.write(json.dumps(conf))
    outfile.close()


def build_ping_file(src):
    status, output = commands.getstatusoutput('./insertions/full_node.sh '+src+' "ping -c 4 '+src+'"')
    print "status: ", status
    #if status != 0:
    #    raise RuntimeError("Ping script failed!")
   
def read_ping_data(src):
    ping_data_name = "data/" + src + "-ping_data.dat"

    print "Reading ping data for %s..."%(src)

    # If it doesn't exist, go ahead and build it
    if not os.path.isfile(ping_data_name):
        build_ping_file(src)
    
    # Ok lets open it and parse and all that
    ping_data_file = open(ping_data_name, 'r')
    ping_data = ping_data_file.read()
    ping_data_file.close()

    out_dict = collections.defaultdict(lambda: None)
    for entry in ping_data.split("\n"):
        try:
            dest, ping_out = entry.split()  
        except ValueError:
            dest = entry
            out_dict[dest] = None
            continue
        
        ping_vals = ping_out.split('/')
        if (len(ping_vals) < 2):
            ping_vals = None

        out_dict[dest] = ping_vals
   
    print "done"

    return out_dict

def build_config_min_ping(numsrc, numdst, mindiff, host_file):
    """Performs the above algorithm.  numsrc is the number of source nodes, numdst is the number of destination nodes for each source
    and minping is the threshold such that no difference of any two pings(src, dst[n]) may be smaller"""
    #read in hosts
    src_hosts = read_hosts(host_file)
    hosts = read_hosts('data/usable_hostnames.dat')
    conf = {}
    dest_pos = {}
    #pick source nodes
    src_list = random.sample(src_hosts, numsrc)
    for newsrc in src_list:
        conf[newsrc] = []
        dest_pos[newsrc] = [x for x in hosts if x != newsrc]

    
    ping_dict = collections.defaultdict(lambda: None) # If something happens and one is missing

    for newsrc in src_list:
        # Read it, store it in the ping dict
        dest_data = read_ping_data(newsrc)
        ping_dict[newsrc] = dest_data

    #pick initial dst nodes
    for src in conf:
        conf[src] = random.sample(dest_pos[src], numdst)
    
    #now keep checking dst nodes and replacing bad ones
    for src in conf:
        violators, pings = validate_pingset(src, conf[src], mindiff, {}, ping_dict)
        while len(violators) != 0:
            print("violating nodes: ", violators)
            for hv in violators:
                try:
                    conf[src].remove(hv)
                except ValueError:
                    print(conf[src], hv)
                    sys.exit(-1)
                try:
                    del pings[hv]
                except KeyError:
                    pass
            while len(conf[src]) != numdst:
                newdst = sample_list(hosts)
                while newdst == src or newdst in conf[src]:
                    newdst = sample_list(hosts)
                conf[src].append(newdst)
            violators, pings = validate_pingset(src, conf[src], mindiff, pings, ping_dict)
        print("This source is good")
        violators = []
        pings = {}
    print("Found a valid conf")
    outfile = open('data/curr_config.json', 'w')
    outfile.write(json.dumps(conf))
    outfile.close()

def validate_pingset(src, ldst, mindiff, pings, ping_dict):
    """Takes a source node and a list of dst nodes and returns pairs of dst nodes that violate mindiff"""
    #pings = {}
    violators = []
    for dst in ldst:
        if dst not in pings.keys():
            #rawping = get_ping_parsed(src, dst)
            rawping = ping_dict[src][dst]
            if rawping == None:
                return [dst], pings
            pings[dst] = rawping[1]
    print(pings)
    for dst1 in pings.keys():
        for dst2 in pings.keys():
            if dst1 != dst2 and abs(float(pings[dst1]) - float(pings[dst2])) < mindiff and (dst1 not in violators and dst2 not in violators):
                violators.append(dst1)
    print("Found "+str(len(violators))+" violators")
    return list(set(violators)), pings

if __name__ == "__main__":
    minping(5, 5, 20)

