#
#   rank_analyze.py --  Look at output ranks
#

import performance
import sys

def analyze(out_file):
    
    perf_data = performance.load_score_dict(out_file)

    data_list = []

    print "Origin:", out_file

    # build the list
    for dest in perf_data:
        ping = float(perf_data[dest][0][1])
        distance = float(perf_data[dest][1]['distance'])
        data_list.append((dest, ping, distance))

    # Sort the list according to ping
    data_list.sort(key=lambda x: x[1])


    # output the pretty list
    print "Ping\tFR\tTarget"
    for entry in data_list:
        print entry[1], "\t", entry[2], '\t\t', entry[0]

    performance.compare_adjacent(data_list)

def print_path(origin):
    perf_data = performance.load_score_dict(origin)
   
    print "Processing %s"%(origin)
    for dest in perf_data:
        print "=============================="
        print "report from %s"%(dest) 
        report = perf_data[dest][1]['report']
        print report
        print perf_data[dest][1]['distance']
    
if __name__ == "__main__":
    perf_file = sys.argv[1]
    analyze(perf_file)
    #print_path(perf_file)
