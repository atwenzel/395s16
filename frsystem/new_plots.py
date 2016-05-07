import glob
import sys

import numpy as np
import matplotlib.pyplot as plt

import performance
import plotting

def plot_completion(data_file):

    data = performance.load_score_dict(data_file)

    plotting.completion_cdf(data) 

def plot_cache(data_match):
    
    data_list = glob.glob(data_match+"*")

    master_list = []

    for data_file in data_list:

        data = performance.load_score_dict(data_file)
    
        total = 0
        cum_list = []
        for entry in data:
            total += (entry)
            cum_list.append(entry)
            
        master_list.append(cum_list)

        print len(cum_list)

    avg_list = []
    std_list = []

    cum_len = min([len(x) for x in master_list])
    cum_len = min(200, cum_len)
    

    for i in range(cum_len):
        avg_list.append(np.mean([x[i] for x in master_list]))
        std_list.append(np.std([x[i] for x in master_list]))
    print avg_list

    plotting.query_cache(avg_list, std_list) 

def plot_overtime(data_file):
    data = performance.load_score_dict(data_file)

    avg_sim = []
    std_sim = []

    # Lets compute the average fraction of matching paths for each case
    for index, time_step in enumerate(data):
        if index == 0:
            continue
        prev_step = data[index - 1] 


        sim_list = []

        
        for pair_index, pair in enumerate(time_step):

            curr_chain = set([x[0] for x in pair])
            print curr_chain
            prev_chain = set([x[0] for x in prev_step[pair_index]])

            if len(curr_chain) == 0 or len(prev_chain) == 0:
                continue

            sim = float(len(curr_chain & prev_chain)) / len(curr_chain)
            
            sim_list.append(sim)

        avg_sim.append(np.mean(sim_list)) 
        std_sim.append(np.std(sim_list))

        print "Next Time Step!"

    plotting.overtime_plot(avg_sim, std_sim)  

if __name__ == "__main__":
    command = sys.argv[1]
    data_file = sys.argv[2]
    
    if command == "--complete":
        plot_completion(data_file)
    elif command == "--cache":
        plot_cache(data_file)
    elif command == "--overtime":
        plot_overtime(data_file)


