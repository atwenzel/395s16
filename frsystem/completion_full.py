import performance

import json
import sys

def select_list():
    """ Generates a list of IPs to connect"""
    # Take the list of PL nodes
    with open('data/usable_hostnames.dat', 'r') as hostfile:
        host_list = hostfile.read().split('\n') # its just a line seperated list
     
    host_list =  host_list[:-1]

    return host_list

def connect_points(ip_list, out_dir):
    """ Actually run the connection """

    host_list = []
    # Loop over the list n^2
    for host1 in ip_list:
        total = 0
        comp = 0
        for host2 in ip_list:
            # Skip the diagonal
            if host1 == host2:
                continue
            print host1, host2
           
            params, report, out_info = performance.run_fr(host1, host2, reverse=True) 
            
            total += 1
            
            dist = performance.comp_distance(out_info)
            
            if dist == -1:
                continue
            else:
                comp += 1
        
        host_rate = float(comp)/total       
        host_list.append(host_rate)  
        performance.save_score_dict(out_dir, host_list) 
     


if __name__ == "__main__":

    outdir = sys.argv[1]
    
    ip_list = select_list()
    connect_points(ip_list, outdir)
