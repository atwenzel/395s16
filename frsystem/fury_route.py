# The Route graph structure built totally out of networkx graphs

import argparse
import dns 
import itertools 
import json
import logging
import netaddr
import networkx as nx
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import os.path
import python_common.edns.edns as edns
import utils
import random
import socket
import sys
import time

# Project
import providers

import geo_location


##################################
# Globals to keep an eye on rate #
##################################

# Keep the rate low
g_time = None
g_queries = 0
g_prev_queries = 0
max_qps = 100

# some parameters:
GRANULARITY = 24

def init_qps_watch():
    global g_time
    global g_queries
    global g_prev_queries

    g_time = time.time()
    g_queries = 0
    g_prev_queries = 0

    return

def qps_throttle():
   
    global g_time
    global g_queries
    global g_prev_queries

    now = time.time()
    
    # Compute the rate based on change since last check
    t_delta = now - g_time
    q_delta = g_queries - g_prev_queries
    rate = float(q_delta)/t_delta

    # If we went too fast...
    print "\tRATE", rate
    if rate > max_qps:
        print "Exceeded %d QPS! Sleeping..."%(max_qps)
        sleep(60)
  
    # Update everything...
    g_prev_queries = g_queries
    g_time = time.time()

def qps_update_count():
    global g_queries

    g_queries += 1

##################################

class furyGraph(object):
    def __init__(self, start, finish=None, match_tier=2, debug=False, 
                 candidate_max=None, vote_sample=1.0, prov_file="None",
                 no_opp_finding=False,):
        """
            start - ip of start locaiton

            finish - ip of end location
        """
        self.start = start
        self.finish = finish # May be None

        # A target set for now...
        self.target_list = [finish,]
        self.match_tier = match_tier


        self.graph = nx.DiGraph()

        # Add the start and finish
        self.graph.add_node(start, provider=['root',],
                                   location=None,
                                   prefix=32,
                                   addr_list=[start,],
                                   scanned_set = set())
    
        self.graph.add_node(finish, provider=['root',],
                                    location=None,
                                    prefix=32,
                                    addr_list=[finish,],
                                    scanned_set = set())
        # Current path
        self.current_path = []

        # Accounting
        self.query_total = 0

        self.complete = False
   
        self.debug = debug

        # Keep a list of nodes that are known to cause timeout problems 
        self.blacklist = set()
        self.remove = set()

        self.prov_file = prov_file
        default_prov, self.providers = providers.load_json_complex(prov_file)

        #self.candidate_prov = ['google', 'edgecast', 'alibaba', 'cloudfront', 'msn']
        self.candidate_prov = default_prov.keys()
        self.candidate_prov.remove("edgecast")
        self.candidate_prov.remove("alibaba")
        self.candidate_prov.remove("adnxs")

        # Candidate Sampling
        self.candidate_max = candidate_max
        self.vote_sample = vote_sample

        # IMPORTANT FLAGS!
        self.pp_voting = False
        self.pp_candidate = False
        self.pp_target = False

        # Find opportunistic paths when you hit the limit?
        self.no_opp_finding = no_opp_finding

    def reset_dest(self, dest, src=None):
        """ Reset and get things ready for a run on the same graph """
        self.finish = dest

        self.target_list = [dest,]

        if dest not in self.graph.node:
            self.graph.add_node(dest, provider=['root',],
                                        location=None,
                                        prefix=32,
                                        addr_list=[dest,],
                                        scanned_set = set())

        self.current_path = []
        self.complete = False

        if src != None:
            self.start = src
            
            if src not in self.graph.node:
                raise RuntimeError("What did you do...")
             

    def dump_params(self):
        params = {}

        params['granularity'] = GRANULARITY
        params['tiers'] = self.match_tier
        params['provider_set'] = self.candidate_prov
        params['pp_voting'] = int(self.pp_voting)
        params['pp_candidate'] = int(self.pp_candidate)
        params['pp_target'] = int(self.pp_target)
        params['prov_file'] = self.prov_file

        return params

    def clear_scans(self):
        for node in self.graph.node:
            self.graph.node[node]['scanned_set'] = set()
    
    ######################################################
    #                                                    #
    #                                                    #
    # Display and Reporting Functions                    #
    #                                                    #
    #                                                    #
    ######################################################

    def plot(self, full=0, path=None):

        # Node parameters, to keep it consistent
        node_size = 500
        node_alpha = 1

        # Some list choices to keep colors consistent
        colors = cm.rainbow(np.linspace(0, 1, len(self.providers)))
        p_list = self.providers.keys()
        p_index = {x: p_list.index(x) for x in p_list}

        # Determine a position for everything
        pos = nx.spring_layout(self.graph, weight='spring', iterations=50)

        # First, draw the roots
        nx.draw_networkx_nodes(self.graph, pos, nodelist=[self.start, self.finish],
                               node_size=node_size, alpha=node_alpha, node_color='red')
        root_labels = dict([(x, x+"/"+str(self.graph.node[x]['prefix'])) for x in [self.start, self.finish]])
        nx.draw_networkx_labels(self.graph, pos, labels=root_labels)

        draw_set = []
        if full == 0 and path == None:
            # just draw the current path
            draw_set.extend(self.current_path)
            # ALso draw the target set...
            draw_set.extend(self.target_list)
        elif full == 0 and path != None:
            # Draw a specific path instead of the main current one
            draw_set.extend(path)
            # ALso draw the target set...
            draw_set.extend(self.target_list)
        elif full == 1:
            draw_set.extend(self.graph.nodes())
        else:
            raise RuntimeError("Bad draw set: %d"%full)

        # Now draw everything along the determined path..., sorting by provider
        for provider in p_index:
            # It will plot as if prov. 0
            # TODO: This is a fucking nightmare
            provider_path = [colors[p_index[list(self.graph.node[x]['provider'])[0]]] for x in draw_set if provider in self.graph.node[x]['provider']]
        

            # Don't call the draw on an empty list
            if provider_path == []:
                continue

            # Figure out the nodes for that provider
            provider_nodes = [x for x in draw_set if provider in self.graph.node[x]['provider']]
            # Actually draw it 
            nx.draw_networkx_nodes(self.graph, pos, nodelist=provider_nodes,
                               node_size=node_size, alpha=node_alpha,
                               node_color=colors[p_index[provider]],
                               label=provider
                               )

        # Draw the edges
        # First, figure out which edges we care about
        total_edge_list = self.graph.edges() 
        edge_list = []
        for edge in total_edge_list:
            if (edge[0] in draw_set) and (edge[1] in draw_set):
                edge_list.append(edge) 

        nx.draw_networkx_edges(self.graph, pos, edgelist=edge_list)


        # Draw the labels
        # A recuded lable that just shows subnet size
        node_labels = dict([(x,"/"+str(self.graph.node[x]['prefix'])) for x in draw_set])
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels)

        edge_labels=dict([((u,v,),d['weight']) for u,v,d in self.graph.edges(data=True)
                         if ((u in draw_set) and (v in draw_set))])
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        plt.legend(numpoints=1)

        plt.show()

    def dump_node_info(self, node, cost=0):
        """ Prints some info about a node""" 
        provider = self.graph.node[node]['provider']
          
        rev = dns.reversename.from_address(node)

        try:
            rev_name = str(dns.resolver.query(rev, "PTR")[0])
        except (dns.resolver.NoNameservers, dns.resolver.NXDOMAIN, dns.exception.Timeout) as e:
            self.fr_print("\t%s %s %d (%s)"%(node, self.graph.node[node]['provider'], cost, rev))
        else:
            # If lookup of the nodes are failing, we should consider blacklisting.
            self.fr_print("\t%s %s %d (%s)"%(node, self.graph.node[node]['provider'], cost, rev_name))

        return

    def fr_print(self, string):
        """ Only print if debug is activated """
        if self.debug == True:
            print string

    def generate_report(self):
        """ Output a parseable measure of system performance so we can run
        larger scale experiments """

        report = "REPORT: %s->%s\n"%(self.start, self.finish)
        report += "Queries: %d"%(self.query_total)

        # Check and see if there is any path at all
        if self.no_opp_finding == False:
            for node in self.target_list:  
                if nx.has_path(self.graph, self.start, node):
                    # it exists!
                    if self.complete != True:
                        # We didn't know about it before
                        print "Complete but didn't know it!"
                        self.complete = True
                        self.current_path = [node]
                    else:
                        # We did know about it!
                        if node == self.current_path[-1]:
                            continue
                        else:
                            #print "Found a different path!"
                            # Its a different one! Is it better?
                            old_p = nx.shortest_path_length(self.graph, self.start, self.current_path[-1],
                                                 weight='prov_weight')
                            new_p = nx.shortest_path_length(self.graph, self.start, node,
                                                 weight='prov_weight')

                            # Also the second halves...
                            old_lin = utils.find_link(self.current_path[-1], self.target_list, GRANULARITY)
                            old_t = nx.shortest_path_length(self.graph, self.finish, old_lin,
                                                 weight='prov_weight')
                            new_t = nx.shortest_path_length(self.graph, self.finish, node,
                                                 weight='prov_weight')

                            if (new_p + new_t) < (old_p + old_t):
                                print "Found a shorter path, taking it..."
                                print old_p, old_t
                                print new_p, new_t
                                self.current_path = [node]
                             

        # If we finished
        total_path = []
        if self.complete:
            last_node = self.current_path[-1]
            shortest_path = nx.shortest_path(self.graph, self.start, last_node,
                                             weight='prov_weight')

            # Lets figure out the link to the final set
            # This better return, else we wouldnt be complete
            last_ip = netaddr.IPAddress(last_node)
            for target_node in self.target_list:
                subnet = netaddr.IPNetwork("%s/%d"%(target_node, GRANULARITY))
                if last_ip in subnet:
                    link = target_node
            target_path = nx.shortest_path(self.graph, self.finish, link,
                                           weight='prov_weight')

            #utils.print_path(self.graph, shortest_path, self.finish, self.target_list)
            dist, report = utils.print_path(self.graph, shortest_path, target_path, self.target_list,report)
            #self.plot() 
            #a_path = utils.annotated_path(shortest_path, self.graph, end_path=target_path)
            a_path = [] # BAD BAD BAD XXX XXX XXX

        else:
            

            report += "\nINCOMPLETE!\n"
            last_node = self.current_path[-1]
            dist, report = utils.print_path(self.graph, self.current_path, string=report) 
            
            #a_path = utils.annotated_path(self.current_path, self.graph)
            a_path = [] # BAD BAD BAD XXX XXX XXX
            dist = -1

        return dist, report, a_path
    
    ######################################################
    #                                                    #
    #                                                    #
    # Operational Functions                              #
    #                                                    #
    #                                                    #
    ######################################################

    def scan_neighbors(self, node, prefix=32, provider_set=None, rounds=1):
        """ Resolve from all providers, add the new nodes to the graph 
            node is just an IP string here
        """

        if provider_set == None:
            provider_set = self.providers.keys() # take the whole list



        # Don't scan yourself
        provider_set = [x for x in provider_set if x not in self.graph.node[node]['provider']]
        

        scanned_set = self.graph.node[node]['scanned_set']
        

        for curr_round in range(rounds):
            for provider in provider_set: 
                # Check to see if we've hit this node before
                #print node, provider, scanned_set
                if provider in scanned_set:
                    continue

                # Otherwise, go ahead and scan it

                host = self.providers[provider] 
                # DO the query, assume 32
                #print "scanning %s for %s"%(node, host)
                try:
                    scope, res_set = utils.get_response_set(node, host, prefix)
                except dns.exception.Timeout:
                    # Ok well if this one failed just... keep going
                    print "Timeout from %s"%(host)
                    scope = -1
                    #continue

                # Check for errror cases...
                if scope == -1:
                    # There was a DNS failure.

                    # First timer or a repeated failure?
                    if host in self.blacklist:
                        # Ok, this is a second failure, let's go ahead
                        # and yank this server from the provider set...
                        self.remove.add(host)

                    else:
                        # Remember this failure in case it happens again...
                        self.blacklist.add(host)

                    # There is no data here, so let's not do anything else 
                    continue
                        

                #weight = 2**(32-scope)
                weight = 32 - scope
                self.query_total += 1 
                qps_update_count()
  
                #TODO: compute geographic distance here
                #threshold = 100000
                #provider2 = self.graph.node[node]['provider']
                #dist = geo_location.distance_val(node1, node2, provider, provider2?)
                #if dist < threshold:
                #   weight = 0

                prov_weight = utils.compute_cost(provider, weight)
                if prov_weight == 0:
                    prov_weight = 8


                # Let's turn them into ip objects
                ip_list = [netaddr.IPAddress(x) for x in res_set]
                
                # Now we want to compute the spanning subnet, but we want the
                # subnets to have a maximum of /24, to keep our scan relatively
                # granular. 
                subnet_list = netaddr.cidr_merge(ip_list)

                #Alex
                """Idea: make dictionary {subnet: [IPs]}, in subnet loop, 
                if distance to node and any IP is < threshold, weight becomes 1"""

                #sn2ip = {}
                #for sn in subnet_list:
                #    sn2ip[str(sn)] = []
                #    for ip in ip_list:
                #        if netaddr.IPAddress(ip) in sn:
                #            sn2ip[str(sn)].append(str(ip))
                #print("****Alex debug*****")
                #print(sn2ip)

                """don't actually need this, subnets are /32"""
                            
                airportdict = json.loads(open('airports_dict.json', 'r').read())

                for ret_subnet in subnet_list:
                    # For each subnet in that new set, let's compare it against
                    # all the existing nodes
                    breakout = False
                    for existing_node in self.graph.nodes():
                        try:
                            prefix = self.graph.node[existing_node]['prefix']
                        except KeyError:
                            print "Caught that key error with prefix!"
                            print self.graph.node[existing_node]
                            self.dump_node_info(self.graph.node[existing_node]) 
                            raise

                        existing_subnet = netaddr.IPNetwork(existing_node+"/"+str(prefix))

                        #Alex
                        """Distance goes here:
                            ip1 = node
                            prov1 = self.graph.node[node]['provider']
                            ip2 = ret_subnet.network
                            prov2 = provider"""
                        #do distance, change weights if necessary
                        """save_weight = weight  #save this to restore later
                        ip1 = node
                        prov1 = self.graph.node[node]['provider']
                        ip2 = str(ret_subnet.network).split('/')[0]
                        prov2 = provider
                        apdict = json.loads(open('airports_dict.json', 'r').read())
                        dist = geo_location.distance_val(ip1, ip2, prov1, prov2, apdict)
                        if dist < 500: #threshold is 500 miles
                            #print("-*-*-*-*-*THRESHOLD MET, SETTING WEIGHT TO 1 -*-*-*-*-*-*-*")
                            weight = 1"""
                           
                        # Is it in that subnet?
                        if ret_subnet in existing_subnet:
                            # Its inside an existing subnet, add the edges,
                            # but first check if the edge exists already and if
                            # it already had a better weight
                            if self.graph.has_edge(node, existing_node):
                                # It exists, what's the weight
                                old_weight = self.graph.edge[node][existing_node]['weight']
                                if scope > old_weight:
                                    self.graph.add_edge(node, existing_node, 
                                                        weight=weight, 
                                                        spring=scope,
                                                        prov_weight=prov_weight)
                                #else do nothing, keep data we have
                            else:
                                # It didn't have the edge already, go ahead and add it
                                self.graph.add_edge(node, existing_node,
                                                    weight=weight,
                                                    spring=scope,
                                                    prov_weight=prov_weight)
                            # Update the provider list
                            self.graph.node[existing_node]['provider'].add(provider)
    
                            breakout = True 
                            break
                        else:
                            #weight = save_weight #Alex
                            continue
                    # Double break!
                    if breakout:
                        break
        
                    # Ok so, if we are here, the subnet is not anywhere in the
                    # graph, so we should add it the old fashioned way
                    prefixlen = ret_subnet.prefixlen
                    ip = ret_subnet.network
                    #calculate location for new node to be added
                    if prefixlen < 32:
                        print("trying a subnet")
                        sub = netaddr.IPNetwork(str(ip)+'/'+str(prefixlen))
                        try:
                            list(sub)
                        except TypeError:
                            print(ip)
                        for ipaddr in list(sub):
                            loc = geo_location.get_location(str(ipaddr), provider, airportdict)
                            if loc != None:
                                break
                    else:
                        loc = geo_location.get_location(str(ip), provider, airportdict)
                    if loc == (0.0, 0.0): #assume this is a punt
                        loc = None
                    self.graph.add_node(str(ip),
                                        location=loc,
                                        provider=set([provider]), # Current prov.
                                        prefix=prefixlen,
                                        addr_list = ip_list,
                                        scanned_set=set())
                    print("added a new node at ip: "+str(ip)+" which is at "+str(loc))
                    # Add a link to the one who got sent here...
                    dist = None
                    if self.graph.node[node]['location'] != None and self.graph.node[str(ip)]['location'] != None:
                        dist = geo_location.get_dist(self.graph.node[node]['location'], self.graph.node[str(ip)]['location'])
                    if dist != None and dist < 100000: #THRESHOLD
                        print("SETTING WEIGHT TO 1 BECAUSE GEOGRAPHIC DISTANCE!!!!!!!!!!!!!!!!!!")
                        self.graph.add_edge(node, str(ip), distance=dist, weight=1, spring=scope, prov_weight=prov_weight)
                    else:
                        self.graph.add_edge(node, str(ip), distance=dist, weight=weight, spring=scope,
                                        prov_weight=prov_weight)
                    print("added an edge between "+str(node)+" and "+str(ip)+" which are "+str(dist)+" meters apart")

                # Add it to the scanned set
                self.graph.node[node]['scanned_set'].add(provider)

        # Once we have added everything we should coalesce
        #start = time.time()
        #self.coalesce_graph(node)
        #stop = time.time()
        #timer = stop - start
        #print "COALESCE TIME: %2.2f"%(timer)


        # Go ahead and delete any failing providers
        #for bad_prov in remove:
        #    print "Removing", bad_prov
        #    del self.providers[bad_prov]

        #print "Exiting scan_neighbors"
        
        return

    def coalesce_graph(self, node):
        """Check the destinations from a particular node and see if any of them
        can be merged """


        destinations = nx.descendants(self.graph, node)
  
        changes = False
        #print "Checking for a merge on %s..."%(node)

        print "LEN:", len(destinations)

        for i, node_a in enumerate(destinations):
            node_a_len = self.graph.node[node_a]['prefix']

            for j, node_b in enumerate(destinations):
                # Ignore yourself
                if i == j:
                    continue
                node_b_len = self.graph.node[node_b]['prefix']
                net_a = netaddr.IPNetwork(node_a + "/" + str(node_a_len))
                net_b = netaddr.IPNetwork(node_b + "/" + str(node_b_len))

                # See what happens when you try to merg
                merge_list = netaddr.cidr_merge([net_a, net_b])
                if len(merge_list) == 1:
                    # They merged!
                    print "We have a merge!"
                    changes = True

                    # Update info in the node
                    new_len = merge_list[0].prefixlen
                    self.graph.node[node_a]['prefix'] = new_len
                    print "\t new length of ", new_len
                    
                    # Update the edges
                    for u, v in self.graph.out_edges([node_b]):
                        # Did that edge already exist? If so, take the min weight
                        if node_a == v:
                            # Points to yourself, just skip it
                            continue
                        if (node_a, v) in self.graph.edges():
                            new_weight = self.graph.edge[node_a][v]['weight'] 
                            old_weight = self.graph.edge[u][v]['weight'] 

                            min_weight = max(old_weight, new_weight)
                            
                            self.graph.edge[node_a][v]['weight'] = min_weight
                            self.graph.edge[node_a][v]['spring'] = min_weight
                        else:
                            old_weight = self.graph.edge[u][v]['weight'] 
                            self.graph.add_edge(node_a, v, weight=old_weight,
                                                spring=old_weight)
                    
                      
                    for u, v in self.graph.in_edges([node_b]):
                        if node_a == u:
                            # Points to you, skip it
                            continue
                        if (u, node_a) in self.graph.edges():
                            new_weight = self.graph.edge[u][node_a]['weight'] 
                            old_weight = self.graph.edge[u][v]['weight'] 

                            min_weight = max(old_weight, new_weight)
                            
                            self.graph.edge[u][node_a]['weight'] = min_weight
                            self.graph.edge[u][node_a]['spring'] = min_weight
                        else:
                            old_weight = self.graph.edge[u][v]['weight'] 
                            self.graph.add_edge(u, node_a, weight=old_weight,
                                                spring=old_weight)


                    # Remove the old node
                    self.graph.remove_node(node_b) 

                    break
                # Inside for loop
            if changes == True:
                break

        if changes == True:
            # We updated, should make another pass
            self.coalesce_graph(node)
        else:
            return

    def expand_target(self, tiers=2):
        """ Increase the set of nodes that makeup the target node by seeing what
        the finish would see """
        self.fr_print("Expanding target...")
       
        # Determine which providers should go in the target set. This should be
        # *hand picked* to avoid providers who have large matching sets (ie
        # edgecast) in the area.

        #target_providers = self.candidate_prov
        target_providers = []
        for provider in self.candidate_prov:
            #if provider == "edgecast":
            #    continue
            #if provider == "alibaba":
            #    continue
            #if provider == "level3":
            #    continue
            #if provider == "netdna":
            #    continue
            #if provider == "chinacache":
            #    continue
            #if provider == "adnxs":
            #    continue
            #print provider
            target_providers.append(provider)

        tier_dict = {
            1: target_providers,
            2: ["google"],
                }

        #self.scan_neighbors(self.finish, 
        #                    provider_set=target_providers)    

        fresh_list = []
        prev_list = []
        total_list = [self.finish]

        tier = 1

        prev_list = [self.finish,]
        while tier <= tiers:
            #print "processing tier %d"%(tier)
            for node in prev_list:
                #print "NODE: %s"%(node)
                
                # One at a time so we can filter
                #for provider in target_providers:
                for provider in tier_dict[tier]:
                    if (self.pp_target == False) and (provider in self.graph.node[node]['provider']):
                        continue

                    if (tier == 2) and (self.graph.node[node]['provider'] != "google"):
                        continue

                    self.scan_neighbors(node, provider_set=[provider])

                #self.scan_neighbors(node, 
                #                    provider_set=target_providers)    
                possible_adds = self.graph.neighbors(node)
                final_adds = []
                for node in possible_adds:
                    if list(self.graph.node[node]['provider'])[0] in target_providers:
                        final_adds.append(node)

                #fresh_list.extend(self.graph.neighbors(node))
                fresh_list.extend(final_adds)

            #print fresh_list
            # Update the total list
            total_list.extend(fresh_list)
            # Advance all the loop state
            prev_list = fresh_list
            fresh_list = []
            tier += 1


        self.target_list = list(set(total_list))

#        # Add all the stuff that you can get to from the target 
#        for node in self.graph.neighbors(self.finish):
#        #    network = node
#        #    prefixlen = str(self.graph.node[node]['prefix'])
#        #    subnet = netaddr.IPNetwork(network + "/" + prefixlen)
#        #    # Get the actual addrs and add them to our list
#        #    #addr_list = [str(x) for x in list(subnet)]
#        #    #print addr_list
#        #    #self.target_list.extend(addr_list)             
#            self.target_list.append(node)
        

        # Print them

        if self.debug == True:
            self.fr_print("Target List:")
            for target in self.target_list:
                #cost = self.graph.edge[self.finish][target]['prov_weight']  
                self.dump_node_info(target,)#cost)  

        
    def choose_close(self, node, targets):
        """ Looks at a node and a list of connected points, returns the
        closest/best provider"""
        seen = -1

        self.fr_print("[TIE] Choosing close from %d nodes..."%(len(targets)))
  
        # Generally speaking, if we are here, all the candidates received equal
        # weights (ie votes, currently). So pick the one with the lowest cost
        for target in targets:
            cost = self.graph.edge[node][target]['prov_weight']

            if seen != -1:
                if cost < seen:
                    seen = cost
                    best = target
                #else stick with what  we have
            else:
                seen = cost
                bast = target
                
        return target
        ################################################
        # Choose a google one
        #for target in targets:
        #    if self.graph.node[target]['provider'] == 'google':
        #        return target
        # Just choose a random one...
        #return random.sample(targets, 1)[0]
        ###############################################


    def provider_match_test(self, candidate, provider):
        """ Check to see if a provider gives a matching node for a host """
        
        try:
            self.scan_neighbors(candidate, provider_set=[provider])
        except KeyError:
            print "It's in the scan!"
            raise

        # Look downstream from the candidate
        dec = self.graph.successors(candidate)
        # Reduce to the currently considered provider
        dec = [x for x in dec if provider in self.graph.node[x]['provider']]
       
        # Generate a list of the target descendents
        target_dec = []
        for target in self.target_list:
            # Can't vote based on your own target
            if (self.pp_voting == False) and (provider in self.graph.node[target]['provider']):
                continue
            # Look at target neighbors
            t_dec = self.graph.successors(target)
            # Restrict provider
            t_dec = [x for x in t_dec if provider in self.graph.node[x]['provider']] 
            # Extend the target neighbor set
            target_dec.extend(t_dec)
       
        # Find the intersection
        intersect = set(dec).intersection(target_dec)
        
        return intersect

    def count_votes(self, vote_dict, candidate):
        """ Given a candidate and a vote dict, count the unique votes """

        # NOTE:##############################
        # 
        # This is not the best way to do this. In fact, I think it might be 
        # kind of bad. Consider:
        #
        # P1: S1, S2
        # P2: S1
        #
        # If P2 is first in the list, this will count as two votes for 
        # this candidate, if P1 comes first, it will (correctly) consider
        # it as 1.
        #
        # For the sake of time, I've dedcided to leave it like this for now,
        # as I think the occurances of these scenarios will be low, but we will
        # see...
        # NOTE #################################


        #print " #################################"
        #print "DUMPING VOTES:"
        #for x in vote_dict:
        #    print x, vote_dict[x]
        #print " #################################"


        p_block_list = [vote_dict[x] for x in vote_dict]

        # Loop over the provider blocks
        for index, p_block in enumerate(p_block_list):
            # Loop over the nodes in that block
            for node in p_block:
                # Loop over the other blocks...
                for q_block in p_block_list[index + 1:]:
                    # Found a repeat! Delete it forward
                    
                    # Lets try a /24 instead?
                    removal_list = []
                    for q_b in q_block:
                        subnet = netaddr.IPNetwork("%s/%d"%(q_b, GRANULARITY))
                        node_addr = netaddr.IPAddress(node)
                        if node_addr in subnet:
                            print "Forward removal!"
                            removal_list.append(q_b)
                    for x in removal_list:
                        q_block.remove(x)
                            

                    #if node in q_block:
                    #    print "Forward removal!"
                    #    q_block.remove(node)
   

        # Count the non_empties            
        votes = 0
        for p_block in p_block_list:
            if len(p_block) != 0:
                # It had a uniqueish server
                votes += 1

        #######################################
        #                                     #
        # XXX Penalty for CDNetworks        ###
        #                                     #
        #######################################
        if "cdnetworks" in self.graph.node[candidate]['provider']:
            if votes > 0:
                votes = 1   

        #######################################


        return votes

    def is_complete(self, current_node): 
        """ Is there a path from the start to the target set? """

        # First, let's figure out what /24 we have in the target set
        subnet_set = [netaddr.IPNetwork("%s/%d"%(x, GRANULARITY)) for x in self.target_list]
   
        current_ip = netaddr.IPAddress(current_node) 

        # Does it live in one of those?
        for subnet in subnet_set:
            if current_ip in subnet:
                return True
         
        #for node in self.target_list:
        #    if nx.has_path(self.graph, self.start, node):
        #        return True

        return False

    def scan_vote_history(self, target_votes):
        """ Scan the vote stack, return the first node that satisfies the target vote
        requirement """
        # Basically the idea here is that we are going to keep
        # spinning backwards over the path until we find a node 
        # that seems good. When we find it, we set selected, the 
        # vote, set selected to true and break

        found = False
        # self.vote_stack = (node, votes, [(child, votes), (child2, votes2), ...]

        # This is the "ephemeral" blacklist, so we don't pick something we already have
        # right here
        last_choice = self.vote_stack[-1][0] #current_node
        last_choice_list = [last_choice]

        for index, choice in enumerate(reversed(self.vote_stack)):
            # Check each child
            for nb in choice[2]:
                if nb[0] in last_choice_list:
                    continue
                # Does it have the number of votes we want?
                if nb[1] == target_votes:
                    # Check to see if its in the list already
                    if nb[0] in self.current_path:
                        continue

                    if nb[0] in self.old_branches:
                        if target_votes >= self.old_branches[nb[0]]:
                            #print "Skipping old branch (target:", target_votes, ")"
                            continue

                    found = True
                    selected = nb[0]
                    prev_votes = nb[1]
                    break
            if found == True:
                break
            last_choice_list.append(choice[0]) # Ie the one we took to get to the deeper branch
           
        if found == False:
            # We didn't find anything
            return None, None, None

        return selected, prev_votes, index

    def connection_scheme(self):
        """ Try and connect the start and finish """
        # Keep an eye on the rate...
        init_qps_watch()
            
        # Scan the end point, so we can get the target set
        self.expand_target(tiers=self.match_tier)
        #self.plot()

        path_status = False

        # The path that we are trying to use to break through
        self.current_path = [self.start]
        # Keep track of the vote stack
        self.vote_stack = []
        self.old_branches = {}

        current_node = self.current_path[-1]

        # How many time through the loop.
        passes = 0
        # How many votes did the previous iteration of the loop get?
        prev_votes = 0
      
        print "Beggining loop..."

        #while current_node not in self.target_list:
        while not self.is_complete(current_node):
            # Too far?
            if (len(self.current_path) > 75) or (passes >= 25):
                print "\tToo long!"
                print "==========================================="
                return -1
                
            passes += 1
            curr_prov = self.graph.node[current_node]['provider']
            print "Current node is: %s (%s)"%(current_node, curr_prov)
            
            qps_throttle()

            # Go ahead and see what we can see from here
            print "\tScanning current node..."
            self.scan_neighbors(current_node)

            # Ok, now we have to choose somewhere to move to next
            neighbors = self.graph.successors(current_node)
     
            candidates = []
            
            # This loops through the neightbors, if a node is not unique, we
            # skip it. The idea being that if node i matches j for some i<j,
            # then if we throw out i, j will become unique, so we will only
            # count each once.

            for index, node in enumerate(neighbors):
                
                #XXX XXX XXX XXX#
                # Double check what components are activated when!
    
                #self.dump_node_info(node)
                # Go ahead and make sure its a candidate provider...
                if list(self.graph.node[node]['provider'])[0] not in self.candidate_prov:
                    continue
               
                # No provider to provider?
                if (self.pp_candidate == False) and (self.graph.node[current_node]['provider'] == self.graph.node[node]['provider']):
                    continue 
                
                # First we should only consider neighors who are in unique /24s,
                # reducing the probing requests we have to do
                base_net = netaddr.IPNetwork(node+"/%d"%(GRANULARITY))
                # Check everything forward
                unique = True
                for other_node in neighbors[index+1:]:
                    node_addr = netaddr.IPAddress(other_node)
                    if node_addr in base_net:
                        unique = False
                        break

                # Was it in a /24 for something we had already?
                for old_node in self.current_path:
                    old_addr = netaddr.IPAddress(old_node)
                    if old_addr in base_net:
                        unique = False
                        break
                    
                # If it wasn't in any subnet, keep it
                if unique == True:
                    candidates.append(node)
            #print "COLLAPSED:", candidates

            # Eliminate previous choices from the set of neightbors... 
            candidates = [x for x in candidates if x not in self.current_path]


            # Ok, sample the max number
            total_candidates = candidates
            #if self.candidate_max != None:
            #    size = min(len(candidates), self.candidate_max)
            #    candidates = random.sample(candidates, size)

            # We should make sure we still have any candidates...
            if candidates != []:
                # We should check for the trivial case first, if one of the candidates is
                # in the target set, take it right away. In particular, this takes the 
                # closest one
                min_dist = -1
                target_match = None
                for candidate in candidates:
                    #if candidate in self.target_list:
                    if self.is_complete(candidate): # Shortcut to checking subnets
                        distance = self.graph.edge[current_node][candidate]['weight']
                        if (distance < min_dist) or (min_dist == -1):
                            target_match = candidate
               
                # 
                #if target_match != None:
                #    # NOTE: this messes up the voting stack and all that, but if
                #    # we had a target match, we are done anyway
                #    current_node = target_match
                #    self.current_path.append(current_node)
                #    self.fr_print("Hit a target node -- cutting out now! (%s)"%current_node)
                #    continue
                    


                ###########
                # NOTE: We actually defer exploration to the last minute here, then only 
                # XXX: It might be that this reduces the quality of the connections and drives
                # XXX up the cost of the shortest path, but eliminating potential mid-way
                # XXX paths!
                # XXX
                #
                ## First, make sure each is fully explored
                for index, candidate in enumerate(candidates):
                    self.scan_neighbors(candidate)
                ###########


                # Make sure targets are scanned too -- should happen early on
                print "\tScanning target..."
                for target in self.target_list:
                    self.scan_neighbors(target)



                # Lets look at each candidate and determine how many providers  vote for each.
                vote_dict = {}
                # Which providers get to vote this round?
                voter_count = self.vote_sample * len(self.providers)
                if len(self.providers) <= 10:
                    voter_count = len(self.providers)
                else:
                    voter_count = int(max(10, voter_count))
                print "\tVOTER COUNT:", voter_count
                sampled_voters = random.sample(self.providers.keys(), voter_count)
    
                #if self.debug == True:
                #    print "CAND DUMP"
                #    for candidate in candidates:
                #        self.dump_node_info(candidate)
    
                print "\tConsidering %d candidates..."%(len(candidates))
                for candidate in candidates:
                    if self.debug == True:
                        cand_cost = self.graph.edge[current_node][candidate]['prov_weight']  
                        self.dump_node_info(candidate, cand_cost)
                    vote_dict[candidate] = {} 
                    #for provider in self.providers:
                    for provider in sampled_voters:
                        vote = self.provider_match_test(candidate, provider)
                        #if len(vote)  != 0:
                        #    print "\t\t%s"%(provider), vote
                        vote_dict[candidate][provider] = vote
                print "\tDone voting..."
                # Let's see who has the most votes
                count_list = []
                for candidate in vote_dict:
                    vote_set = vote_dict[candidate]
                    #count = sum(vote_set[x] for x in vote_set)
                    count  = self.count_votes(vote_dict[candidate], candidate)
                    count_list.append((candidate, count))
        
    
                # Let's see how many votes the winner(s) had 
                max_vote_pair = max(count_list, key=lambda item:item[1])
                max_vote = max_vote_pair[1]
                max_cand = [x[0] for x in count_list if x[1] == max_vote] 
                if self.debug == True:
                    print "\tMax votes:", max_vote_pair, "(", max_cand, ")"
    
                # Let's record the voting information for the current candidate
                self.vote_stack.append((current_node, prev_votes, count_list)) 
                if self.debug == True:
                    print '\t\t', count_list
    
                # There are a few conditions for a winner to be considered:
                # 1) It had enough votes (ie more than previous)
                # 2) It had not been explored and eliminated previously
    
                # max_vote -- Number of votes seen by most popular
                # prev_vote -- Number of votes current_node got to get here
    
                # Ok, so lets see, if the max vote is enough, check those
                # candidates
                found = False
                target_votes = prev_votes
            
            # Else: No candidates! We should force it to step back
            else:
                print "No candidates!"
                self.vote_stack.append((current_node, prev_votes, [])) 
                max_vote = 0
                
                found = False
                target_votes = prev_votes



            if max_vote >= target_votes:
                selected = None
                for choice in max_cand:
                    # Had we visted it as a branch?
                    if choice in self.old_branches:
                        # Ok, but maybe our goal is lower?
                        if target_votes < self.old_branches[choice]:
                            # its an ok choice
                            selected = choice
                            break
                    else:
                        selected = choice
                        break
                # Did we take something from the max candidate set?
                if selected != None:
                    prev_votes = max_vote
                    # Check the state
                    print "\tSELECTED: %s (%s)"%(selected, self.graph.node[selected]['provider'])
                    current_node = selected
                    self.current_path.append(current_node)
                    # Lets print the path...
                    if self.debug == True:
                        utils.print_path(self.graph, self.current_path) 
                    print "\tQUERIES: %d"%(self.query_total)

                    # Set the flag to skip over the next loop
                    found = True

            # If we enter this loop, we have no new candidates with enough votes
            if found != True:
                # it was too little!
                print "\tToo few votes! No Confidence (max: %d, prev: %d)"%(max_vote, prev_votes)

            while found != True:
                # Now we need to climb back up the stack until we find a better choice
                    
                selected, prev_votes, index = self.scan_vote_history(target_votes)
                # Made it through... decrease expectations...
                if selected == None:
                    target_votes -= 1

                    if target_votes == 0:
                        #print "VOTE STACK:\n"
                        #for elt in self.vote_stack:
                        #    print "\t", elt
                        #print "OLD PATHS:"
                        #for old in self.old_branches:
                        #    print "\t%s: %d"%(old, self.old_branches[old])
                        raise RuntimeError("Target votes pushed to 0!")
                    print "\tNow settling for %d"%target_votes
                    continue
                else:
                    found = True

                # If we leave that loop the following must be true:
                # selected - has been set to some node
                # prev_votes - indicates the vote that landed that spot
                # index - indicates where that node lives

                # Ok, first,lets chop the path
                chop_index = len(self.vote_stack) - index
                # First, we have to remember who we skipped
                # There is a length check here because maybe we didnt skip branches.
                # just lowered our requirements
                if chop_index < len(self.vote_stack):
                    for step in self.vote_stack[chop_index:]:
                        old_branch = step[0]
                        self.old_branches[old_branch] = prev_votes

                self.current_path = self.current_path[:chop_index] 
                # Pop all the crap off the vote stack
                self.vote_stack = self.vote_stack[:chop_index]
            
                # Now let's staple the new guy on there
                #print "\t**********************************"
                print "\t[reverse] SELECTED: %s (%s)"%(selected, self.graph.node[selected]['provider'])
                #print "\tChop Index:", chop_index
                #print "\tBRANCH: ", self.vote_stack[-1]
                #print "**********************************"
                current_node = selected
                self.current_path.append(current_node)
                # Lets print the path...
                if self.debug == True:
                    utils.print_path(self.graph, self.current_path) 
                print "\tQUERIES: %d"%(self.query_total)

            # Let's go ahead and clean out the providers list if we need
            # Also reset the blacklist, so it is a little conservative in zapping
            for bad_prov in self.remove:
                print "Removing %s"%bad_prov
                try:
                    del self.providers[bad_prov]
                except KeyError:
                    continue
                self.blacklist = set()


            if self.debug:
                raw_input("Waiting...")
            print "Looping around!"
            self.remove = set()

            
        # OK so we know if we made it here that the path now contains an element
        # in the target set. In particular, the last one
        # Go ahead and get the path:
        #print "Before print path, the path was:"
        #print self.current_path

        print "====> Path complete! <===="
        if self.debug:
            raw_input("Done!...")

        self.complete = True

def process_run_wrapper(loc_a, loc_b, loc_a_ip, loc_b_ip,
                        tiers, candidate_max, vote_sample,
                        prov_file,
                        debug=False,
                        dump=False,
                        graph=None):

    print loc_a, loc_b

    # Instantiate the object
    if graph == None:
        g = furyGraph(loc_a_ip, loc_b_ip, tiers, 
                  candidate_max=candidate_max,
                  vote_sample=vote_sample,
                  debug=debug, prov_file=prov_file)
    else:
        g = graph
        g.reset_dest(loc_b_ip, src=loc_a_ip)


    params = g.dump_params()
    # Get the logging stuff ready
    out_dict = {}
    
    # Actually run it...
    start_time = time.time()
    #try:
    ret = g.connection_scheme()
    """except (RuntimeError, dns.exception.Timeout, edns.ednsException) as e:
        # SOmething bad happend...
        print(e)
        report = ("%s->%s:%d\n"%(loc_a, loc_b, -1))
        report += str(e)
        out_dict['distance'] = -1
        out_dict['error'] = str(e)
        out_dict['error_type'] = str(type(e))
        out_dict['report'] = report
        out_dict['path'] = None
    except Exception as e: # Catchall
        print "Hit the catchall! %s"%(e)
        print type(e)
        print(e)
        # Some other exception occured...
        report = ("%s->%s:%d\n"%(loc_a, loc_b, -1))
        report += str(e)
        out_dict['distance'] = -1
        out_dict['error'] = str(e)
        out_dict['error_type'] = str(type(e))
        out_dict['report'] = report
        out_dict['path'] = None

    else:"""
    dist, report, a_path = g.generate_report()
    #if ret == -1:
    #    dist = -1

    out_dict['distance'] = dist

    if dist == 0:
        raise RuntimeError("Got a 0 length path!")



    if dist == -1:
        out_dict['error'] = "Exceeded Length!"
        out_dict['error_type'] = "length error"
    else:
        out_dict['error'] = None
        out_dict['error_type'] = None
    out_dict['report'] = report
    out_dict['path'] = a_path
    
    # Who cares why it ended
    end_time = time.time()
    total_time = end_time - start_time
    out_dict['total_time'] = total_time
    out_dict['start_time'] = start_time

    # Go ahead and write the other features that should be logged
    out_dict['tiers'] = tiers
    out_dict['candidate_max'] = candidate_max
    out_dict['vote_sample'] = vote_sample

    # Log some stuff for everyone
    out_dict['queries'] = g.query_total

    if dump:
        return params, report, out_dict
    else:
        return report, out_dict

def generate_ip(num):

    for x in xrange(num):
        a = random.randint(0,255)
        b = random.randint(0,255)
        c = random.randint(0,255)
        d = random.randint(0,255)
        addr = "%d.%d.%d.%d"%(a, b, c, d)

        yield addr
    

def repeated_scan(origin, tier, prov_file, debug, outdir):

    #target_list = ["192.33.90.69", "192.33.90.69"]
    #t_list = ['194.254.215.12', '133.9.81.164', '133.69.32.131', '151.97.9.224', '149.43.80.20', '216.48.80.12', '128.10.18.53', '160.80.221.37', '129.107.35.132', '129.22.150.29', '206.12.16.154', '155.225.2.72', '206.23.240.28', '198.83.85.45', '165.230.49.114']
    
    # North American t_list
    #t_list = ['164.107.127.12', '129.10.120.193', '129.63.159.102', '129.97.74.14', '128.227.150.11', '198.133.224.147', '131.247.2.248', '156.56.250.226', '128.223.8.114', '160.36.57.172', '170.140.119.70', '216.48.80.12', '148.206.185.34', '129.32.84.160', '155.225.2.72']
    
    # Build a list of targets
    #host_file = open("data/usable_hostnames.dat")
    #host_file = open("data/usable_nm.dat")
    #host_list = host_file.read().strip().split('\n')
    #       
    #host_ip_list = [socket.gethostbyname(x) for x in host_list]

    #t_list = random.sample(host_ip_list, 20)
    #t_list = [host_ip_list[7]] * 10

    t_list = generate_ip(200)



    #target_iters = itertools.permutations(t_list) 
    #picked_targets = random.sample(list(target_iters), 10)

    #picked_targets = [random.sample(t_list, len(t_list)) for x in range(10)]

    master_list = []

    
    query_count = []

    # Instantiate the object
    #g = furyGraph(origin, target_list[0], tier, 
    #              debug=debug, prov_file=prov_file)
   
    q_offset = 0  
    
    full_dict = {}

    q_list = []

    for index, dest in enumerate(t_list): 
        out_dict = {}
        if index != 0:
            g.reset_dest(dest)
        else:
            g = furyGraph(origin, dest, tier, 
                  debug=debug, prov_file=prov_file)
    
        # Actually run it...
        start_time = time.time()
        try:
            ret = g.connection_scheme()
        except Exception as e:
            print "FATAL ERROR IN: %s->%s"%(origin, dest)
            #raise e
            # XXX: Just keep going for now -- will look into it
            ret = -1
        end_time = time.time()
        
        #dist, report, a_path = g.generate_report()

        if ret == -1:
            dist = -1

        #out_dict['distance'] = dist

        if ret == -1:
            out_dict['error'] = "Exceeded Length!"
            out_dict['error_type'] = "length error"
        else:
            out_dict['error'] = None
            out_dict['error_type'] = None
        #out_dict['report'] = report
        #out_dict['path'] = a_path
        
        # Who cares why it ended
        total_time = end_time - start_time
        out_dict['total_time'] = total_time
        out_dict['start_time'] = start_time

        # Go ahead and write the other features that should be logged
        #out_dict['tiers'] = tier
        #out_dict['candidate_max'] = candidate_max
        #out_dict['vote_sample'] = vote_sample

        # Log some stuff for everyone
        out_dict['queries'] = g.query_total
       
        print "TOTAL Q:", out_dict["queries"]
        
        query_count.append(out_dict['queries'] - q_offset)

        # Set the offset for the next one
        q_offset = out_dict['queries']

        full_dict[dest] = out_dict

        # Dump as you go
        d_file = open(outdir, 'w')
        json.dump(query_count, d_file)
        d_file.close()


    print query_count
    print out_dict['queries']

    master_list.append(query_count)


    # Dump the output
    d_file = open(outdir, 'w')
    json.dump(query_count, d_file)
    d_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fury Route - Network Distance Estimator")
    # Parameters
    parser.add_argument("--single_scan", action="store_true",
                        help="Attempt to connect just the input pairs")
    parser.add_argument("--loc_a", type=str, default=None,
                        help="Start location for single scan")
    parser.add_argument("--loc_b", type=str, default=None,
                        help="Final location for single scan")
    parser.add_argument("--provider_file", type=str, default="None",
                        help="Provider file (complex json format)")
    parser.add_argument("--tier", type=int, default=1,
                        help="Number of tiers to extend the target")
    parser.add_argument("--debug", action="store_true",
                        help="Print full debug outputs")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Directory to store json reports")
    parser.add_argument("--candidate_max", type=int, default=None,
                        help="Max number of candidates to sample. Default takes all.")
    parser.add_argument("--vote_sample", type=float, default=1.0,
                        help="Fraction of providers to use for each vote")
    parser.add_argument("--cache_scan", action="store_true",
                        help="perform queries from the same origin, measure query dec.")
    parser.add_argument("--reverse", action="store_true",
                        help="switch loc_a and loc_b in a single scan")
    #parser.add_argument("--no_opp", action="store_true",
    #                    help="disable opportunistic pathfinding")

    args = parser.parse_args()

    loc_d = {
        "ucsd": "132.239.17.224",
        "ucla": "131.179.150.72",
        "rutgers": "165.230.49.114",
        "berk": "69.229.50.3",
        "eth": "192.33.90.69",
        "nagoya": "133.68.253.243",
        "arizona": "206.207.248.35",
        "ufl": "128.227.150.11",
        "nu": "165.124.184.65",
    }



    # Manually entered locations
    #if len(sys.argv) == 4:
    if args.single_scan == True:

        if args.loc_a == None or args.loc_b == None:
            print "Must Specify locations!"
            sys.exit(-1)

        if args.loc_a not in loc_d:
            loc_1 = socket.gethostbyname(args.loc_a)
        else:
            loc_1 = loc_d[args.loc_a]

        if args.loc_b not in loc_d:
            loc_2 = socket.gethostbyname(args.loc_b)
        else:
            loc_2 = loc_d[args.loc_b]

        tier = args.tier
        outdir = args.outdir

        # Load the provider file...
        prov_file = args.provider_file
        if prov_file != "None":
            #providers.load_exported_dict(prov_file)  
            #providers.load_json_complex(prov_file)  
            out_name = os.path.basename(prov_file).split('.')[0]
        else:
            out_name = "none"

        report_json = "%s/run-report-%s.json"%(outdir, out_name)
        report_json_f = open(report_json, 'w')

        if args.reverse != True:
            report, pair_dict = process_run_wrapper(args.loc_a, args.loc_b,
                                                loc_1, loc_2,
                                                args.tier, args.candidate_max,
                                                args.vote_sample,
                                                prov_file,
                                                args.debug,)
        else:
            report, pair_dict = process_run_wrapper(args.loc_b, args.loc_a,
                                                loc_2, loc_1,
                                                args.tier, args.candidate_max,
                                                args.vote_sample,
                                                prov_file,
                                                args.debug,)


        #g = furyGraph(loc_1, loc_2, tier, debug=args.debug)
        pair_key = "%s,%s"%(args.loc_a, args.loc_b)
        wrapper = {}
        wrapper[pair_key] = pair_dict
        #g.connection_scheme()
        #dist, report, a_path = g.generate_report()

        print report

        print len(pair_dict['path']), pair_dict['distance']

        json.dump(wrapper, report_json_f)
        report_json_f.close()

        sys.exit(0)
    elif args.cache_scan == True:
        if args.loc_a not in loc_d:
            loc_1 = socket.gethostbyname(args.loc_a)
        else:
            loc_1 = loc_d[args.loc_a]

        repeated_scan(loc_1, args.tier, args.provider_file, args.debug, args.outdir) 

        sys.exit(0)

    else:
        prov_file = args.provider_file
        if prov_file != "None":
            #providers.load_json_complex(prov_file)  
            #out_name = os.path.basename(prov_file).split('.')[0]
            out_name = os.path.basename(prov_file)[:-4] #incase there is another . 
        else:
            out_name = "none"

        if args.outdir == None:
            print "Need to specify --outdir!"
            sys.exit(-1)

        outdir = args.outdir

        report_file = "%s/report-%s.dat"%(outdir, out_name)
        report_json = "%s/report-%s.json"%(outdir, out_name)
        report_f = open(report_file, 'w')
        report_json_f = open(report_json, 'w')

        loc_d = {
            "ucsd": "132.239.17.224",
            "ucla": "131.179.150.72",
            "rutgers": "165.230.49.114",
            #"berk": "69.229.50.3",
            "eth": "192.33.90.69",
            "nagoya": "133.68.253.243",
            "arizona": "206.207.248.35",
            "ufl": "128.227.150.11",
            #"nu": "165.124.184.65",
        }

        out_list = []
        out_dict = {}

        for loc_a in loc_d:
            for loc_b in loc_d:
                if loc_a == loc_b:
                    continue
                report_f.write("**********************\n")
                pair_key = "%s,%s"%(loc_a, loc_b)
                report, pair_dict = process_run_wrapper(loc_a, loc_b, loc_d[loc_a], loc_d[loc_b],
                                    args.tier, args.candidate_max, args.vote_sample,
                                    prov_file)

                out_list.append("%s->%s:%d"%(loc_a, loc_b, pair_dict['distance']))

                out_dict[pair_key] = pair_dict

                report_f.write(report)
                report_f.write("\n**********************\n")

                # Pace slightly
                time.sleep(1)

        json.dump(out_dict, report_json_f)

        for pair in out_list:
            print pair

        report_f.close()
        report_json_f.close()

        sys.exit(0)


