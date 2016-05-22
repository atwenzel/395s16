#
#   performance.py -- The main script for running the evaluation scripts
#

#Global
import glob
import json
import os.path
import os
import random
import socket
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

#Local
import fury_route
import plotting
import source.authenticate as plauth
import source.utils as utils
import source.plapi as api
import source.pldata as pldata
import source.furypl as furypl

import scipy.stats

BLACKLIST = True
#blacklist = ['planetlab2.cti.espol.edu.ec', u'planetlab-1.sjtu.edu.cn',]# 'planetlab02.cs.washington.edu']
#blacklist = ['planetlab1.cs.otago.ac.nz', u'planet-lab2.itba.edu.ar', u'planetlab2.cti.espol.edu.ec']
blacklist = [
'planetlab1.cs.otago.ac.nz',
u'planetlab2.cti.espol.edu.ec',
#'planetlab2.unr.edu', 
'planet-lab2.itba.edu.ar',
'planetlab-1.sjtu.edu.cn',
'planetlab2-buenosaires.lan.redclara.net',
#u'planetlab4.goto.info.waseda.ac.jp',
#'pl1.eng.monash.edu.au',
#'ple2.ait.ac.th',
u'pl2.pku.edu.cn',
]
#'planetlab2.utt.fr', 'cs-planetlab3.cs.surrey.sfu.ca']
#blacklist = ['cs-planetlab3.cs.surrey.sfu.ca']
#blacklist = []

def save_score_dict(save_file, score_dict):

    outfile = open(save_file, 'w')
    json.dump(score_dict, outfile)
    outfile.close()

    return

def load_score_dict(save_file):
    savefile = open(save_file, 'r')
    data = json.load(savefile)
    savefile.close()

    return data


def load_config(file_name):
    # First, go ahead and load the config file...
    try:
        testconf = json.loads(open('data/curr_config_FULL.json', 'r').read())
    except IOError:
        print("error, data/curr_config doesn't exist.  Use --build to create a new config")
        sys.exit(-1)

    return testconf

def create_fr(origin, dest):
    origin_ip = socket.gethostbyname(origin)
    dest_ip = socket.gethostbyname(dest)

    g = fury_route.furyGraph(origin_ip, dest_ip, 1, 
                    candidate_max=None,
                    vote_sample=1.0,
                    debug=False, prov_file="None")

    return g


def run_fr(origin, dest, reverse=False, graph=None):
    """ Run fury route from origin to dest.
    If reverse=True, a failure will cause it to try the other direction """

    try:
        origin_ip = socket.gethostbyname(origin)
        dest_ip = socket.gethostbyname(dest)
    except socket.gaierror:
        return -1, -1, {'distance': -1}

    out_info = []

    params, report, fr_out = fury_route.process_run_wrapper(origin, dest, 
                                                    origin_ip, dest_ip,
                                                    1,  #tiers
                                                    None, # Candidate max
                                                    1.0, # vote sample
                                                    "None", # provider file
                                                    debug=False, dump=True,
                                                    graph=graph)

    out_info.append(fr_out)

    if (reverse == True) and fr_out['distance'] == -1:
        params2, report2, fr_out2 = fury_route.process_run_wrapper(dest, origin, 
                                                    dest_ip, origin_ip,
                                                    1,  #tiers
                                                    None, # Candidate max
                                                    1.0, # vote sample
                                                    "None", # provider file
                                                    debug=False, dump=True,
                                                    graph=graph)


        out_info.append(fr_out2)




    return params, report, out_info

def measure_graph_diff(data_dir):
   
    
    origin_pickles = glob.glob(data_dir + "/*-12h.g")

    changes = []

    for origin in origin_pickles:
        # Load the original
        original = origin[:-6] + ".g"
        with open(original) as original_f:
            og_g = pickle.load(original_f)
        # Load the new one
        with open(origin) as new_f:
            new_g = pickle.load(new_f)
           
        # Compute the size difference. Currently this just shows 
        old_size = len(og_g.graph.node)
        new_size = len(new_g.graph.node)
        
        diff = new_size - old_size
        changes.append((float(new_size)/old_size) - 1)
        print diff, float(new_size)/old_size

    print np.mean(changes)

    return
    
def measure_perf_diff(data_dir1, data_dir2):
    
    # Let's load up the new data
    new_files = glob.glob(data_dir1+"/*.json")  
   
    diff_list = []
    big_diff = []

    for origin in new_files:
        print "***********************************"
        print "Origin: %s"%(origin)
        if not os.path.isfile(origin):
            continue
       
        # Load the new data
        data_dict = load_score_dict(origin)
        score1, spear1, count1 = compute_perf(data_dict)

        # Figure out the file name for the new data
        base = os.path.basename(origin)
        old_file = os.path.join(data_dir2, base)
   
        data_dict2 = load_score_dict(old_file)
        score2, spear2, count2 = compute_perf(data_dict2)

        diff = score2 - score1  

        if abs(diff) > .1:   
            big_diff.append(origin)
        
        diff_list.append(diff)


    print big_diff
       
    print diff_list
    print np.mean(diff_list)
    print np.median(diff_list)
    print np.std(diff_list)

############################################################
#
#   Compute performance for the mainstream Result          #  
#
############################################################

#def compare_perf(origin, dest_dict, prov_file):
def compare_perf(origin, dest_dict):
    
    testconf = dest_dict

    # Now loop over the config
    data_log = {}
    # Now loop over each destination
    print "Testing %s..."%(origin)

    # NOTE: We could sample testconf...
    #testconf = random.sample(testconf, 30)
    for index, dest in enumerate(testconf):
        ping = furypl.get_ping_parsed(origin, dest)
        tries = 0
        while ping == None:
            if tries > 5:
                #raise RuntimeError("Pinged out!")
                print "Pinged out!"
                ping = [-1,-1]
                break
            print "Retrying ping..."
            ping = furypl.get_ping_parsed(origin, dest)
            tries += 1
        #ping = [0, 10+index]
        #print origin, dest, ping
      

        if index == 0:
            # Instantiate the graph the first time through 
            
            #g = create_fr(origin, dest)
            #with open("pickles/%s.g"%(origin), 'r') as pfile:
            #    g = pickle.load(pfile)
            #    g.clear_scans()
            g = None
            

        params, report, fr_out = run_fr(origin, dest, reverse=True, graph=g)

        #print "-->Distance: %d"%(fr_out['distance'])
        
        data_log[dest] = (ping, fr_out) #avg ping
        #data_log[dest] = (ping, fr_out, fr_out2) #avg ping

    # Go ahead and pickle the graph
    with open("pickles/%s-12h.g"%(origin), 'w') as pfile:
        pickle.dump(g, pfile)

    return data_log, params


def comp_distance(fr_data):
    """ Compute the distance from the data in a generic way """

    if type(fr_data) == type({}):
        # Its just a regular dict  
        distance = int(fr_data['distance'])

    elif type(fr_data) == type([]):
        # Just return the first
        #return fr_data[0]['distance']

        # Its a list, combine them fanc
        dist_list = []
        for entry in fr_data:
            dist_list.append(int(entry['distance']))
        # scrub out the -1
        dist_list = [x for x in dist_list if x != -1]
        # If nothing, just return 
        if dist_list == []:
            return -1
        # Otherwise...
        if len(dist_list) == 1:
            return dist_list[0]



        #distance = np.mean(dist_list)
        #distance = np.amin(dist_list)
        if dist_list[0] != -1:
            return dist_list[0]
        else:
            return dist_list[1]

    return distance


def compute_perf(dest_dict, bl=False, min_ping=0, max_ping=1000):
    score_dict = {}

    # Ok, we have all the destinations for this origin, let's compute the score 
    dest_list = dest_dict.keys()
    match = 0
    count = 0
    ping_list = []
    dist_list = []
    # Loop over unique pairs
    for index, dest in enumerate(dest_list):
        # Skip things if analysis explicitly removed them
        if bl and dest in blacklist:
            continue
        # Get the distance (whatever version we are using), skip if incomplete
        distance = comp_distance(dest_dict[dest][1])
        if distance == -1:
            continue
        # Ditch it if the ping data came back bad
        ping = float(dest_dict[dest][0][1])
        if ping == -1:
            continue

        #if ping > 250:
        #    continue

        # Log them so we can use them for statistics later
        ping_list.append(ping)
        dist_list.append(distance)

        # Loop through everything else
        for ind2, dest2 in enumerate(dest_list[index+1:]):
            # BLACKLIST STUFF 
            if bl and dest2 in blacklist:
                continue
            # Get the other distnace
            fr2 = comp_distance(dest_dict[dest2][1])
            # Bail if it didn't finish
            if fr2 ==  -1:
                continue
            # Get the other ping, bail if fail
            ping2 = float(dest_dict[dest2][0][1])
            if ping2 == -1:
                continue

            #if ping > 250:
            #    continue
            #if ping > 200 and ping2 > 200:
            #    continue
            max_ping = 1000
            if bl== True and( (abs(ping - ping2) < min_ping) or (abs(ping - ping2) > max_ping)):
                print "\tskipping:", ping, ping2, dest, dest2
                continue

            count += 1

            # Actually look at all the casses, increment accordingly
            if ping < ping2:
                if distance < fr2:
                    match += 1
                else:
                    print "Missed: %2.2f < %2.2f as %2.2f, %2.2f (%s, %s)"%(ping, ping2, distance, fr2, dest, dest2)
            else:
                if distance > fr2:
                    match += 1

                else:
                    print "Missed: %2.2f > %2.2f as %2.2f, %2.2f (%s, %s)"%(ping, ping2, distance, fr2, dest, dest2)
    #total = (len(dest_list) * (len(dest_list) - 1)) / 2
    total = count
    print total

    # If we don't have enough, just skip it
    if count <= 3:
        return -1, -1, -1

    score = float(match) / total

    spear = scipy.stats.spearmanr(ping_list, dist_list)

    return score, spear[0], total

def compare_adjacent(sorted_list):

    print "Comparing adjacent hosts..."

    match = 0
    for index, destination in enumerate(sorted_list):
        if index == 0:
            continue
        if destination[2] > sorted_list[index-1][2]:
            match += 1

    total = float(match)/(len(sorted_list) - 1)

    print "\t%2.2f match"%total

    return

#def run_total_exp(host_dict, out_dir, prov_file):
def run_total_exp(host_dict, out_dir):

    score_list = []

    for index, origin in enumerate(host_dict):
        # Ping and build a chain
        #out_data, params = compare_perf(origin, host_dict[origin], prov_file) 
        out_data, params = compare_perf(origin, host_dict[origin])
        # Log that as a json 
        origin_file = os.path.join(out_dir, origin+".json")
        save_score_dict(origin_file, out_data)
        # Save the parameters too
        save_score_dict(origin_file+"-p", params)

        # go ahead and score it
        score, spear, count = compute_perf(out_data)
        if score == -1:
            continue
        score_list.append(score)

    #print score_list
    #print spear_list
    #comparisons_matched_cdf(score_list)

def performance_output(data_dir):

    files = glob.glob(data_dir+"/*.json")

    score_list = []
    spear_list = []

    comp_count = 0
    total_count = 0


    for origin in files:
        print "***********************************"
        print "Origin: %s"%(origin)
        if not os.path.isfile(origin):
            continue

        #if BLACKLIST:
        #    flagged = False
        #    for name in blacklist:
        #        if name in origin:
        #            flagged = True
        #    if flagged == True:
        #        continue

        data_dict = load_score_dict(origin)

        score, spear, count = compute_perf(data_dict)
        if score == -1:
            continue


        comp_count += len([x for x in data_dict if comp_distance(data_dict[x][1]) != -1])

        total_count += len(data_dict)
        print score

        score_list.append(score)
        spear_list.append(spear)


    print "Comp Count: %d"%(comp_count)
    print "Total Count: %d"%(total_count)
    comp_rate = float(comp_count)/total_count
    print "Comp Rate: %2.2f"%(comp_rate)

    # Dump the params too
    params = load_score_dict(origin+"-p")
    print "Parameters:"
    for param in params:
        print "\t", param, ":", params[param]
    print "\n"

    # Some notes on total finished
    print len(score_list)
    

    print "Matches:"
    print "\t", score_list
    print "\t", np.mean(score_list)
    print "Correlation:"
    print "\t", spear_list
    print "\t", np.mean(spear_list)

def count_total_queries(data_dict):
    count = []

    for dest in data_dict:
        fr_data = data_dict[dest][1]
        if type(fr_data) == type({}):
            length = fr_data['queries']
            count.append(length)
        elif type(fr_data) == type([]):
            # We need to look at the distance values to see what happened
            q_list = []
            dist_list = []
            for entry in fr_data:
                q_list.append(entry['queries'])
                dist_list.append(entry['distance'])
            # Ok now consider...
            if dist_list[0] != -1:
                # Forward direction was successfull
                length = q_list[0]
            else:
                # If we cant get there skip
                if dist_list[1] == -1:
                    continue
                length = q_list[0] + q_list[1]
            count.append(length)

    return count

def compute_rank_change(pair_list):
    """ Given a list of values, computes the rank change """
    # Lets sort it by ping
    pair_list.sort(key=lambda x: x[2])

    complete = len(pair_list)

    # Add the ping rank to the front
    new_list = [(y, x[0], x[1], x[2], x[3]) for y, x in enumerate(pair_list)]
    #new_list = [(y, x[0], x[1], x[2], x[3], x[4]) for y, x in enumerate(pair_list)]

    # Sort by the distance value
    new_list.sort(key=lambda x: x[4])

    # compute the change in rank
    score_list = []
    diff_list = []
    for rank, chain in enumerate(new_list):
        diff = abs(rank - chain[0])   
        score = float(diff)/complete
        score_list.append(score)
        diff_list.append((score, chain[1], chain[2], chain[3], chain[4]))


    # Sort by the second ping
    #new_list.sort(key=lambda x: x[5])
    #
    # compute the change in rank
    #score_list2 = []
    #for rank, chain in enumerate(new_list):
    #    diff = abs(rank - chain[0])   
    #    score = float(diff)/complete
    #    score_list2.append(score)

    return diff_list

def total_rank_change(data_dir):
    """ Looks at change in rank across all origins"""

    pair_list = []

    total = 0
    complete = 0

    # Load the files
    files = glob.glob(data_dir+"/*.json")
    for origin_file in files:
        data_dict = load_score_dict(origin_file) 

        origin = os.path.basename(origin_file)[:-5]
        for dest in data_dict:
            total += 1
            #print origin, dest

            #if data_dict[dest][2] == None:
            #    continue

            ping = float(data_dict[dest][0][1])
            #fr = float(data_dict[dest][1]['distance'])
            fr = comp_distance(data_dict[dest][1])
            #ping2 = float(data_dict[dest][2][1])

            if fr == -1:
                continue

            complete += 1

            pair_list.append((origin, dest, ping, fr))
            #pair_list.append((origin, dest, ping, fr, ping2))

    print "TOTAL: %d"%(total)
    print "COMPL: %d"%(complete)

    # Try and eliminate the worst offenders:
    total_removals = []
    while True:
        print "Pair list len: %d"%(len(pair_list))
        diff_list = compute_rank_change(pair_list)
        score_list = [x[0] for x in diff_list]

        diff_list.sort(key=lambda x:x[0])
        #for elt in diff_list:
        #    print elt

        #random test
        x = range(1000)
        random.shuffle(x)
        rand_score = []
        for rank, chain in enumerate(x):
            diff = abs(rank - chain)
            rand_score.append(float(diff)/1000)
        score_list2 = rand_score

        #plotting.total_rank_change_cdf(score_list, rand_score)
        plotting.total_rank_change_cdf(score_list, score_list2)

        # Determine the worst violators:
        worst = diff_list[-15:]
        #worst = [x for x in diff_list if x[0] > .6]
        # Count how many of each we saw in worst
        for elt in worst:
            print elt

        offender_count = {}
        remove_list = []
        for pair in worst:
            origin = pair[1]
            dest = pair[2]

            if origin in offender_count:
                offender_count[origin] += 1
            else:
                offender_count[origin] = 1

            if dest in offender_count:
                offender_count[dest] += 1
            else:
                offender_count[dest] = 1

        for offender in offender_count:
            print offender, offender_count[offender]

            # If they had 3 or more, remove them   
            if offender_count[offender] >= 3:
                remove_list.append(offender)

        # update the pair list
        print "Removing:", remove_list
        total_removals.extend(remove_list)
        pair_list = [x for x in pair_list if ((x[0] not in remove_list) and (x[1] not in remove_list))] 
        print "Total Removed:", total_removals
        if remove_list == []:
            break
        #raw_input("Waiting...")


############################################################
#
#   Random                                                 #  
#
############################################################

def random_compare(data_dir):
    """ Just pick random pairs and see how they do """
    
    # read in all the hostnames
    with open('data/usable_nm.dat', 'r') as hostfile:
        host_list = hostfile.read().split('\n') # its just a line seperated list

    host_list =  host_list[:-1]

    selected_sets = []
    # Select a set of pairs
    for index in range(100):
        selected_sets.append(random.sample(host_list, 3))
 
    perf_list = []
    for (origin, dest1, dest2) in selected_sets:
        # Check the ping distances 
        try:
            ping1 = float(furypl.get_ping_parsed(origin, dest1)[1])
        except TypeError:
            continue
        
        try:
            ping2 = float(furypl.get_ping_parsed(origin, dest2)[1])
        except TypeError:
            continue

        if abs(ping1 - ping2) < 20:
            continue

        if ping1 > 200 and ping2 > 200:
            continue

        params1, report1, fr_out1 = run_fr(origin, dest1, reverse=True)

        params2, report2, fr_out2 = run_fr(origin, dest2, reverse=True)
       
        fr1 = fr_out1['distance']
        fr2 = fr_out2['distance']

        # Log it
        perf_list.append((ping1, ping2, fr1, fr2))
        # Overwrites evertime, but lets me see progress
        save_score_dict(data_dir, perf_list)    

    return

def random_check(data_dir):
    data = load_score_dict(data_dir)    

    total = 0
    compl = 0
    match = 0
    for (p1, p2, f1, f2) in data:
        if p1 == -1 or p2 == -2:
            print "Ping fail!"
            continue
        
        if f1 == -1 or f2 == -2:
            print "Connect fail!"
            continue

        compl += 1
        # did they match? 
        if p1 < p2:
            if f1 < f2:
                match += 1
        elif p2 < p1:
            if f2 < f1:
                match += 1

    # Compute
    comp_rate = float(compl)/len(data)
    match_rate = float(match)/compl

    print "Total: %d"%(len(data))
    print "Complete: %d"%(compl)
    print "Match: %d"%(match)
    print "Completion rate: %2.2f"%(comp_rate)
    print "Match Rate:%2.2f"%(match_rate)
            
    return

############################################################
#
# Plot the completion rate by origin...                    # 
#
############################################################

def completion_rate(data_dir):

    files = glob.glob(data_dir+"/*.json")

    completion_list = []

    count_dict = {}

    for origin in files:
        data_dict = load_score_dict(origin)

        total = float(len(data_dict))

        complete = 0
        for dest in data_dict:
            #if data_dict[dest][1]['distance'] != -1:
            if comp_distance(data_dict[dest][1]) != -1:
                complete += 1
            else:
                if dest in count_dict:
                    count_dict[dest] += 1
                else:
                    count_dict[dest] = 1

        rate = float(complete)/total

        completion_list.append(rate)


    total = 0
    for host in count_dict:
        total += count_dict[host]
        print host, count_dict[host]

    print total


    # Plot it
    #print completion_list
    #plotting.completion_cdf(completion_list)


############################################################
#
# Load the query cache data and crunch some numbers        #
#
############################################################

def query_cache_load(data_file):

    query_data = load_score_dict(data_file)

    length = len(query_data[0])
    
    avg_list = []
    std_list = []
    for index in range(length):
        ind_data = [x[index] for x in query_data]  
       
        avg_list.append(np.mean(ind_data))
        std_list.append(np.std(ind_data))

   
    plotting.query_cache(avg_list, std_list)


############################################################
#
# Absolute measurement experiments                         #
#
############################################################

def absolute_experiment(outdir):
    # read in all the hostnames
    with open('data/usable_hostnames.dat', 'r') as hostfile:
        host_list = hostfile.read().split('\n') # its just a line seperated list

    host_list =  host_list[:-1]

    selected_pairs = []
    # Select a set of pairs
    for index in range(30):
        selected_pairs.append(random.sample(host_list, 2))

    distance_list = []
    # For each pair...
    for pair in selected_pairs:
        # Do a ping between them
        ping = furypl.get_ping_parsed(pair[0], pair[1])
        tries = 0
        # A little forgiving on the ping
        while ping == None:
            if tries > 3:
                #raise RuntimeError("Pinged out!")
                print "Pinged out!"
                break
            print "Retrying ping..."
            ping = furypl.get_ping_parsed(pair[0], pair[1])
            tries += 1
        # If bad things happen, just skip that pair
        if ping == None:
            print "Skipping a pair!"
            continue

        # Build a fury route chain between them
        origin_ip = socket.gethostbyname(pair[0])
        dest_ip = socket.gethostbyname(pair[1])
        
        params, report, fr_out = fury_route.process_run_wrapper(pair[0], pair[1], 
                                                        origin_ip, dest_ip,
                                                        1,  #tiers
                                                        None, # Candidate max
                                                        1.0, # vote sample
                                                        "None", # provider file
                                                        debug=False, dump=True)


        # Put them in a big list 
        distance_list.append((float(ping[1]), float(fr_out['distance'])))
        

    # Write the list out to json
    absolute_file = os.path.join(outdir, "absolute.json")
    save_score_dict(absolute_file, distance_list)

def absolute_data_proc(absolute_file):

    abs_data = load_score_dict(absolute_file)

    abs_data = [x for x in abs_data if x[1] != -1]

    abs_data.sort(key=lambda x: x[0])
    
    print abs_data

    ping_data = [x[0] for x in abs_data]
    fr_data = [x[1] for x in abs_data]

    # NOTE: maybe better to set these to a specific value
    max_ping = max(ping_data)
    max_fr = max(fr_data)

    norm_ping = [x/max_ping for x in ping_data]
    norm_fr = [y/max_fr for y in fr_data]

    plotting.absolute_plot(norm_ping, norm_fr)    
  


############################################################
#
# Sensitivity experiments                                  #
#
############################################################

def sensitivity_experiment(outdir):

    # Read in the hostnames that have ping data
    ping_data_list = glob.glob("data/*ping_data.dat") 

    ping_dict = {}

    for ping_data_obj in ping_data_list:  
        host = os.path.basename(ping_data_obj)[:-14]  
        ping_file = open(ping_data_obj, 'r')
        ping_data = ping_file.read()
        ping_file.close()

        ping_dict[host] = {}
        for entry in ping_data.split("\n"):
            # Skip blanks
            if entry == "":
                continue

            split = entry.split()
            if len(split) == 1:
                # Ok we had no data
                ping_dict[host][split[0]] = None
            else:
                #print split
                dest = split[0]
                ping_avg = split[1].split('/')[1]
                ping_dict[host][dest] = float(ping_avg)

    # Ok so now we have a big dictionary of these...
    #print ping_dict
    #print len(ping_dict)

    TESTNUM = 10
    SRCNUM = 5
    # Select sources
    source_list = random.sample(ping_dict.keys(), SRCNUM) # 10 a good number?
    
    diff_dict = {}

    bin_edges = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    # Compute the differences, put them in bins
    for source in source_list:
        diff_dict[source] = {}
        for edge in bin_edges:
            diff_dict[source][edge] = []

        for loc_a in ping_dict[source]:
            # Skip missing data
            loc_a_ping = ping_dict[source][loc_a]
            if loc_a_ping == None:
                continue
            for loc_b in ping_dict[source]: 
                # Skip missing data
                loc_b_ping = ping_dict[source][loc_b]
                if loc_b_ping == None:
                    continue
                # Skip things that get too far
                if min(loc_a_ping, loc_b_ping) > 50: # Adjust if necessary
                    continue
                # Skip the diagonal
                if loc_b == loc_a:
                    continue
                # skip the a<b
                if loc_a_ping < loc_b_ping:
                    continue

                diff = loc_a_ping - loc_b_ping
                
                for index, edge in enumerate(bin_edges):
                    if index == 0: 
                        continue
                    
                    if (diff > bin_edges[index-1]) and (diff < edge):
                        diff_dict[source][edge].append((loc_a, loc_b, diff))
                        break
        
        #for edge in sorted(diff_dict[source].keys()):
        #    print "%d: "%(edge), diff_dict[source][edge]
        #    print len(diff_dict[source][edge])

        #break


    # Great, we have this big pile, choose 10 of each type
    diffs = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    diff_choices = {}

    #source = diff_dict.keys()[0]

    final_dict = {}

    for source in source_list:
        print "Processing source; %s"%(source) 
        for size in diffs:
            print len(diff_dict[source][size])
            try:
                diff_choices[size] = random.sample(diff_dict[source][size], TESTNUM)# 10 for now?
            except ValueError:
                print "SKIPPED!"
                diff_choices[size] = []

        # Ok, now we are going to go ahead and run the actual experiment...
        final_dict[source] = []
        for size in diffs:
            print "Processing diff %d..."%(size)
            total = 0
            correct = 0
            for pair in diff_choices[size]:
                print "\t%s, %s"%(pair[0], pair[1])

                origin = source
                dest1 = pair[0]
                dest2 = pair[1]

                params1, report1, fr_out1 = run_fr(origin, dest1)
                params2, report2, fr_out2 = run_fr(origin, dest2)
            
                dist1 = comp_distance(fr_out1)
                dist2 = comp_distance(fr_out2)
        
                # Ignore incompletes
                if (dist1 == -1) or (dist2 == -1): 
                    continue

                if dist1 < dist2:
                    correct += 1

                total += 1 
            
            final_dict[source].append((correct, total))
        
            # Save it all
            sensitivity_file = os.path.join(outdir, "sensitivity.json")
            save_score_dict(sensitivity_file, final_dict)

def sensitivity_data_proc(absolute_file):

    comp_data = load_score_dict(absolute_file)
    
    diffs = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    rate_dict = {}
    for source in comp_data:
        rate_dict[source] = []   
        for correct, total in comp_data[source]:
            if total == 0:
                rate = 0
            else:
                rate = float(correct)/total
            rate_dict[source].append(rate)

    avg = []
    std = []

    for i in range(len(diffs)):
        trans = [rate_dict[x][i] for x in rate_dict]

        avg.append(np.mean(trans))
        std.append(np.std(trans))

    plotting.sensitivity_plot(avg, std) 
    
 
############################################################
#
# Loads all the data and computes the whole suite of plots #
#
############################################################


def determine_path(out_info):

    if type(out_info) == type({}):
        path = out_info['path']

    elif type(out_info) == type([]):
        if out_info[0]['distance'] == -1:
            if out_info[1]['distance'] == -1:
                path = []
            else:
                path = out_info[1]['path']
        else:
           path = out_info[0]['path'] 
    
    return path

def overtime_experiment(outdir):
    """ Complete a set of chains ever hour, see how their lenghts change over time """
    
    selected_pairs = []
    # read in all the hostnames
    with open('data/usable_hostnames.dat', 'r') as hostfile:
        host_list = hostfile.read().split('\n') # its just a line seperated list
    
    # Select a set of pairs
    for index in range(5):
        selected_pairs.append(random.sample(host_list, 2))

    master_list = [] 


    time_step_list = []

    finish  = time.time() + (24 * 60 * 60 * 7)

    while time.time() < finish:
        start = time.time()
        time_step_list = []
        for loc_a, loc_b in selected_pairs:
            params, report, out_info = run_fr(loc_a, loc_b, reverse=True) 

            path = determine_path(out_info)

            time_step_list.append(path)

        # Update and write out
        master_list.append(time_step_list)

        overtime_file = os.path.join(outdir, "overtime.json")
        save_score_dict(overtime_file, master_list)
        

        # Sleep off the remainder:
        stop = time.time()
    
        #sleep_time = 3600 - (stop - start)
        sleep_time = 100 - (stop - start)
#
        print "Sleeping for %2.2fs"%(sleep_time)
        time.sleep(sleep_time)
 
def overtime_proc(datafile):
    over_data = load_score_dict(datafile)

    day_0 = over_data[0]
    
    black_list = set()
    
    for index, day in enumerate(day_0):
        if day == -1:
            black_list.add(index)

    for day in over_data:
        print day

    #print len([x for x in day_0 if x == -1])
    #print len(day_0)

    #print day_0

    avg_sim = []
    median_sim = []
    std_sim = []
    tenth_sim = []
    nintieth_sim = []


    for index, day in enumerate(over_data):
        if index == 0:
            continue

        rel_size_list = []
        for pair_ind, chain in enumerate(day):

            #if pair_ind in black_list:
            #    continue

            # Compare against yesterday
            previous = over_data[index - 1]
            prev_chain = previous[pair_ind]
            if prev_chain == -1:
                continue


            if chain == -1:
                continue

            rel_size = float(chain)/float(prev_chain)       

            rel_size_list.append(rel_size)
        print rel_size_list
        
        # Ok we have it for all pairs here
        avg_sim.append(np.mean(rel_size_list))
        std_sim.append(np.std(rel_size_list))

        median_sim.append(np.median(rel_size_list))
        tenth_sim.append(np.percentile(rel_size_list, 10))
        nintieth_sim.append(np.percentile(rel_size_list, 90))

    plotting.overtime_plot(avg_sim, std_sim)  
    #plotting.overtime_plot(median_sim, tenth_sim, nintieth_sim)  



############################################################
#
# Loads all the data and computes the whole suite of plots #
#
############################################################

def generate_plot(data_dir):
    files = glob.glob(data_dir+"/*.json")

    score_list = []
    spear_list = []
    query_list = []

    score_list2 = []
    score_list3 = []
    score_list4 = []


    spear_list2 = []
    spear_list3 = []
    spear_list4 = []

    #temp_bad = ['pl1.sos.info.hiroshima-cu.ac.jp']
    #temp_bad = ['planetlab1.fit.vutbr.cz', 'pl1.sos.info.hiroshima-cu.ac.jp', ]
    temp_bad = []

    for origin in files:
        print "***********************************"
        print "Origin: %s"%(origin)
        if not os.path.isfile(origin):
            continue

        #if BLACKLIST:
        #    flagged = False
        #    for name in temp_bad:
        #        if name in origin:
        #            flagged = True
        #    if flagged == True:
        #        continue


        data_dict = load_score_dict(origin)
        score, spear, count = compute_perf(data_dict, bl=True)
        score2, spear2, count2 = compute_perf(data_dict, bl=True, min_ping=25,
                                              max_ping=50)
        score3, spear3, count3 = compute_perf(data_dict, bl=True, min_ping=50,
                                              max_ping=100)
        score4, spear4, count4 = compute_perf(data_dict, bl=True, min_ping=100)
        if score == -1:
            continue
        score_list.append(score)
        spear_list.append(spear)

        score_list2.append(score2)
        score_list3.append(score3)
        score_list4.append(score4)

        spear_list2.append(spear2)
        spear_list3.append(spear3)
        spear_list4.append(spear4)

        query_list.extend(count_total_queries(data_dict))

    # Dump the params too
    params = load_score_dict(origin+"-p")
    print "Parameters:"
    for param in params:
        print "\t", param, ":", params[param]
    print "\n"

    print "Matches:"
    print "\t", np.mean(score_list)
    print "Correlation:"
    print "\t", np.mean(spear_list)

    #print query_list
    print len(score_list)
    print score_list
    print max(score_list)

    plotting.comparisons_matched_cdf(score_list, score_list2, score_list3, score_list4)
    plotting.correlations_cdf(spear_list, spear_list2, spear_list3, spear_list4)
    plotting.query_cdf(query_list)


if __name__  == "__main__":
   
    command = sys.argv[1]

    if command == "--run":
        # The host config file is first arg
        filename = sys.argv[2]
        # the output dir is second
        outdir = sys.argv[3]
        # the provider file
        #prov_file = sys.argv[4]

        # Load the config file 
        host_dict = load_config(filename)
   
        # Run the whole thing
        #run_total_exp(host_dict, outdir, prov_file)
        run_total_exp(host_dict, outdir)

    elif command == "--check":
        data_dir = sys.argv[2]
        performance_output(data_dir)
        #completion_rate(data_dir)

    elif command == "--abs":
        data_dir = sys.argv[2]
        absolute_experiment(data_dir)
    
    elif command == "--sensitivity":
        data_dir = sys.argv[2]
        sensitivity_experiment(data_dir)
    
    elif command == "--overtime":
        data_dir = sys.argv[2]
        overtime_experiment(data_dir)

    elif command == "--total_rank":
        data_dir = sys.argv[2]
        total_rank_change(data_dir) 

    elif command == "--random_match":
        data_dir = sys.argv[2]
        random_compare(data_dir) 

    elif command == "--random_check":
        data_dir = sys.argv[2]
        random_check(data_dir) 
        
    elif command == "--graph_check":
        data_dir = sys.argv[2]
        measure_graph_diff(data_dir)

    elif command == "--run_comp":
        data_dir1 = sys.argv[2]
        data_dir2 = sys.argv[3]
        measure_perf_diff(data_dir1, data_dir2)
    
        
    elif command == "--plot":
        data_dir = sys.argv[2]
        generate_plot(data_dir)
        #completion_rate(data_dir)
        #query_cache_load(data_dir)
        #absolute_data_proc(data_dir)
        #sensitivity_data_proc(data_dir)
        #overtime_proc(data_dir)
    else:
        print "Unknown command!"
