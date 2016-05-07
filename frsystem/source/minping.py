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

#Global
import json
import random
import sys

#Local
import furypl

def sample_list(l):
    """Takes a list and returns a random element"""
    outl = l[:]
    random.shuffle(outl)
    return outl[0]

def read_hosts(hpath):
    """Takes a path to a list of PlanetLab hosts and returns a list with no \n chars"""
    hlist = []
    hfile = open(hpath, 'r')
    for line in hfile.readlines():
        hlist.append(line.strip('\n'))
    return hlist

def minping(numsrc, numdst, mindiff):
    """Performs the above algorithm.  numsrc is the number of source nodes, numdst is the number of destination nodes for each source
    and minping is the threshold such that no difference of any two pings(src, dst[n]) may be smaller"""
    #read in hosts
    hosts = read_hosts('data/usable_hostnames.dat')
    conf = {}
    #pick source nodes
    while len(conf) != numsrc:
        newsrc = sample_list(hosts)
        while newsrc in conf.keys():
            newsrc = sample_list(hosts)
        conf[newsrc] = []
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

def validate_pingset(src, ldst, mindiff, pings):
    """Takes a source node and a list of dst nodes and returns pairs of dst nodes that violate mindiff"""
    #pings = {}
    violators = []
    for dst in ldst:
        if dst not in pings.keys():
            rawping = furypl.get_ping_parsed(src, dst)
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
