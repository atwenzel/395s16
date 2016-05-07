"""Functions to gather data from PlanetLab"""


#Global
import errno
import functools
import json
import os
import signal
import socket
import subprocess
import sys
import time
import xmlrpclib

#Local
import source.authenticate as plauth
import source.utils as utils
import source.plapi as api

FNULL = open(os.devnull, 'w')

#Timeout functions
"""Code for running timeouts with a 10 second timeout.  Based on
code from http://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish"""

#NOTE Original code used TimeoutError, which is in >python3.x.x.  Replaced with RuntimeError for >2.7.x

BLACKLIST = [ ]
#             "planet-lab2.itba.edu.ar",
#             "planetlab1.pop-mg.rnp.br",
#             "planetlab1.cs.otago.ac.nz",
#             "planetlab1.aut.ac.nz",
#             "planetlab1.ecs.vuw.ac.nz",
#             "planetlab-n1.wand.net.nz",
#             "planetlab2-buenosaires.lan.redclara.net",
#            ]

class timeout:
    def __init__(self, seconds=10, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise RuntimeError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

#End timeout functions

def get_bootstate(hostname):

    if hostname in BLACKLIST:
        print (hostname+" blacklisted!")
        return 0

    start = time.time()
    try:
        with timeout(seconds=6):
            try:
                if subprocess.check_output(["./insertions/check_nodes.sh", hostname], stderr=FNULL).strip('\n') == hostname:
                    end = time.time()
                    print(hostname+str(" confirmed valid after "+str(end-start)+" seconds"))
                    return 1
                else:
                    print(hostname+" invalid - some other reason besides timeout")
                    return 0
            except(subprocess.CalledProcessError):
                print(hostname+" invalid - timeout")
                return 0
    except(RuntimeError):
        return 0

def pull_info(conf, api_server, auth):
    node_ids = api.pl_GetSlices(api_server, auth, conf, ['node_ids'])['node_ids']
    ndata = {}
    for nid in node_ids:
        #can use the pl api to get hosts
        ndata[nid] = api.pl_GetNodes(api_server, auth, nid, ['hostname', 'site_id'])
        #pl api is not correct for bootstate as of 12/13/15. Using the old Oak ssh trick for now  
        ndata[nid]['boot_state'] = get_bootstate(ndata[nid]['hostname'])
        ndata[nid]['location'] = api.pl_GetSites(api_server, auth, ndata[nid]['site_id'], ['latitude', 'longitude'])
        #revision 12/15: add IP to retrieved info
        try:
            ndata[nid]['IP'] = socket.gethostbyname(ndata[nid]['hostname'])
        except socket.gaierror:
            print("Couldn't find IP for "+ndata[nid]['hostname'])
            ndata[nid]['IP'] = 'N/A'
    return ndata

def parse_pldata(rawdict):
    pldata = {}
    #rawdict = json.loads(open('rawpldata.txt', 'r').read())
    for key in rawdict.keys():
        if rawdict[key]['boot_state'] == 1:
            hostname = rawdict[key]['hostname']
            pldata[hostname] = {}
            pldata[hostname]['latitude'] = rawdict[key]['location']['latitude']
            pldata[hostname]['longitude'] = rawdict[key]['location']['longitude']
            pldata[hostname]['IP'] = rawdict[key]['IP']
        else:
            continue
    plfile = open('data/pldata.dat', 'w')
    plfile.write(json.dumps(pldata))
    print(len(pldata.keys()))
    return


def usable_hostnames():
    hfile = open('data/usable_hostnames.dat', 'w')
    pldata = json.loads(open('data/pldata.dat', 'r').read())
    hosts = pldata.keys()
    for host in hosts:
        hfile.write(host+'\n')
    hfile.close()
