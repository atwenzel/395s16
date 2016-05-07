"""This is a Planet Lab management tool for Fury Route.
It's goals are, given any sequence of node pairs,
    1) Determine the physical distance between nodes,
    2) Determine ping between nodes, and
    3) Execute Fury Route with this set of nodes.

Please see README.txt in this directory for more information and usage instructions."""

#Global
import errno
import functools
import json
import os
import signal
import subprocess
import sys
import time
import xmlrpclib

#Local
import source.authenticate as plauth
import source.utils as utils
import source.plapi as api
import source.pldata as pldata
import source.furypl as furypl

if __name__ == "__main__":
    conf = utils.read_config('config.conf')
    api_server, auth = plauth.authenticate(conf['server'], conf['username'], conf['password'])
    
    #parse args
    if sys.argv[1] == "--pull":
        print("Refreshing all data from PlanetLab...")
        pldata.parse_pldata(pldata.pull_info(conf, api_server, auth))
        pldata.usable_hostnames()
        print("Done")
    elif sys.argv[1] == "--usable":
        print("Building list of usable hostnames from existing data in data/usable_hostnames.dat...")
        pldata.usable_hostnames()
        print("Done")
    elif sys.argv[1] == "--runtest":
        print("Running a single test with "+sys.argv[2]+" as source and "+sys.argv[3]+" as destination")
        ping = furypl.get_ping_parsed(sys.argv[2], sys.argv[3])
        print(ping)
        #trace = furypl.get_traceroute(sys.argv[2], sys.argv[3])
        #print(trace)
    elif sys.argv[1] == "--build":
        print("Building a config for multiple tests...")
        #newconf = furypl.build_test_config(10, 10)
        if len(sys.argv) == 4:
            newconf = furypl.build_config_min_ping(57, 7, int(sys.argv[2]), sys.argv[3])
        else:
            newconf = furypl.build_test_config(10, 6)
        print(newconf)
    elif sys.argv[1] == "--build_fixed":
        newconf = furypl.build_config_min_ping_fixed(sys.argv[3], 6, int(sys.argv[2]))

    elif sys.argv[1] == "--runloop":
        print("Looping tests...")
        try:
            testconf = json.loads(open('data/curr_config.json', 'r').read())
        except IOError:
            print("error, data/curr_config doesn't exist.  Use --build to create a new config")
            sys.exit(-1)
        furypl.run_test_loop(testconf)
    elif sys.argv[1] == "--testping":
        print ("Testing ping between destinations...")
        try:
            testconf = json.loads(open('data/curr_config.json', 'r').read())
        except IOError:
            print("error, data/curr_config doesn't exist.  Use --build to create a new config")
            sys.exit(-1)
        furypl.test_ping(testconf) 

    else:
        print("This is the PlanetLab manager for Fury Route.  Please see README.txt for usage info")
