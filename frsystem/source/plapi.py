"""Scripts for returning data from the Planet Lab API"""

#Global
import json
import xmlrpclib

#Local
import authenticate as plauth
import utils

def pl_GetSlices(api_server, auth, conf, retfields):
    """Takes the auth object, a conf object (for slice name) and fields to return"""
    rawres = api_server.GetSlices(auth, conf['slice'], retfields)
    return json.loads(utils.fix_json(str(rawres[0])))

def pl_GetNodes(api_server, auth, ids, nfilter):
    rawres = api_server.GetNodes(auth, ids, nfilter)
    return json.loads(utils.fix_json(str(rawres[0])))

def pl_GetSites(api_server, auth, ids, sfilter):
    rawres = api_server.GetSites(auth, ids, sfilter)
    return json.loads(utils.fix_json(str(rawres[0])))

if __name__ == "__main__":
    print("put some tests here")
