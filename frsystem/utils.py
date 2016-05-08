
import netaddr

import networkx as nx
import python_common.edns.edns as edns

import dns.exception
import dns.reversename
import dns.resolver

GOOGLE = "8.8.8.8"
NU = "129.105.5.98"

#############################
#                           #
# Graph Building utils      #
#                           #
#############################

def compute_cost(provider, value):

    if provider == "google":
        value = value
    elif provider == "cloudfront":
        value = value
    elif provider == "cdn77":
        value = value
        if value == 0:
            value = 1
    elif provider == "cdnetworks":
        value = value * 3

    return value

def find_link(node, target_set, granularity):
    
    ip = netaddr.IPAddress(node)
    for t_node in target_set:
        subnet = netaddr.IPNetwork("%s/%d"%(t_node, granularity))
        if ip in subnet:
            link = t_node
            break

    return link

def get_response_set(host, provider, scope=32):
    """ Get the set of responses from the DNS """

    #print "Query to: %s"%(provider)

    #try:
    res = edns.do_query(GOOGLE, provider, host, scope, timeout=1.75)
    #except dns.exception.Timeout as e:
    #    #print "We got hit here!"
    #    raise RuntimeError("DNS Timeout on %s"%host)
    
    
    if res['rcode'] != 0:
        # We didn't get a good reply!
        print "RES DUMP:"
        print res, provider

        # If we are here, bad things have happened. We should remove that provider
        # TODO: Probably this should only raise for rcode == 2?
        #
        return -1, []
        #raise RuntimeError("Bad DNS response!")
    
    res_scope = res['client_scope']

    # Did it get an edns response?
    if res_scope == 0:
        # No edns ecs data!
        #print "RES DUMP:"
        #print res
        #raise RuntimeError("Got 0 scope in the DNS response!")
        
        # No good answer...
        return 0, []

    ## Invert it!
    #res_scope = (32 - res_scope) + 1
    #print "Res scope:", res_scope

    # Ok, so we have a real response 
    raw_record_list = res['records']
    record_ip_list = [x.split()[2] for x in raw_record_list if x.split()[1] == '1']

    # ok now let's return all that
    return res_scope, record_ip_list

#############################
#                           #
# Reporting Tools           #
#                           #
#############################

def output(text, string=""):
    if string == "":
        print text
    else:
        string = string + "\n" + text
    return string


def print_path(graph, path, finish=None, target_set=None, string=""):
    """ Prints out a path, but in a pretty fashion so we can see what's going
    on, also computes the length and things
    """

    if finish != None:
        print "**************************************" 

    total_cost = 0
    for index, elt in enumerate(path):
        if index == 0:
            # no need to worry about cost
            cost = 0
        else:
            # Compute the cost to get here
            prev = path[index - 1]
            cost = graph.edge[prev][elt]['prov_weight']  
            #total_cost += compute_cost(graph.node[elt]['provider'], cost)
            total_cost += cost  

        # Lets get the reverse name
        rev = dns.reversename.from_address(elt)
        #try:
        #    rev_name = str(dns.resolver.query(rev, "PTR")[0])
        #except (dns.resolver.NoNameservers, dns.resolver.NXDOMAIN, dns.exception.Timeout):
        string = output("%s %s + %d (%s)"%(elt, graph.node[elt]['provider'], cost, rev), string)
        #else:
        #    string = output("%s %s + %d (%s)"%(elt, graph.node[elt]['provider'], cost, rev_name), string)

    # Ok at the end, we've computed the cost to an element of the target set,
    # but now we need to consider the cost to that first hop from the finish
    # NOTE: Reverse step!
   
    if target_set == None and finish != None:
        raise RuntimeError("Need a target set with that finisher!")

    
    if finish != None:
        string = output("+++++++++++", string)
        backwards = list(reversed(finish))
        for index, elt in enumerate(backwards[:-1]):
            #if index == 0: # TODO: THis is not entirely true
            #    # no need to worry about cost
            #    cost = 0
            #else:
            # Compute the cost to get here
            next_el = backwards[index + 1]
            cost = graph.edge[next_el][elt]['prov_weight']  
            total_cost += cost 
            #total_cost += compute_cost(graph.node[elt]['provider'], cost)

            # Lets get the reverse name
            rev = dns.reversename.from_address(next_el)
            #try:
            #    rev_name = str(dns.resolver.query(rev, "PTR")[0])
            #except (dns.resolver.NoNameservers, dns.resolver.NXDOMAIN, dns.exception.Timeout):
            string = output("%s %s + %d (%s)"%(next_el, graph.node[elt]['provider'], cost, rev), string)
            #else:
            #    string = output("%s %s + %d (%s)"%(next_el, graph.node[elt]['provider'], cost, rev_name), string)


    return total_cost, string

def annotated_path(path, graph, end_path=None):
 

    annotated_path = []
    for index, elt in enumerate(path):
        if index == 0:
            cost = 0
        else:
            # Compute the cost to get here
            prev = path[index - 1]
            cost = graph.edge[prev][elt]['weight']  

        # Lets get the reverse name
        rev = dns.reversename.from_address(elt)
        try:
            rev_name = str(dns.resolver.query(rev, "PTR")[0])
        except (dns.resolver.NoNameservers, dns.resolver.NXDOMAIN, dns.exception.Timeout):
            rev_name = str(rev)

        annotated_path.append((elt, cost, rev_name))

    if end_path != None:
        backwards = list(reversed(end_path))
        for index, elt in enumerate(backwards[:-1]):
            next_el = backwards[index + 1]
            cost = graph.edge[next_el][elt]['weight']  

            # Lets get the reverse name
            rev = dns.reversename.from_address(next_el)
            try:
                rev_name = str(dns.resolver.query(rev, "PTR")[0])
            except (dns.resolver.NoNameservers, dns.resolver.NXDOMAIN, dns.exception.Timeout):
                rev_name = str(rev)

            annotated_path.append((next_el, cost, rev_name))

    return annotated_path
        


