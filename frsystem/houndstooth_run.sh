#!/bin/bash
in_data=/shared/furyroute/18-dec/clusters
out_data=/shared/furyroute/18-dec/combo-full
out_data2=/shared/furyroute/18-dec/combo-10

python2.7 fury_route.py --outdir $out_data

#python2.7 fury_route.py --provider_file $in_data/20-at-.10.dat --outdir /shared/furyroute/18-dec/sampled-20 --candidate_max 10 --vote_sample .5

#python2.7 fury_route.py --provider_file $in_data/10-at-.50.dat --outdir $out_data2 --candidate_max 10 
#python2.7 fury_route.py --provider_file $in_data/10-at-.10.dat --outdir $out_data2 --candidate_max 10 
#python2.7 fury_route.py --provider_file $in_data/10-at-.05.dat --outdir $out_data2 --candidate_max 10 
#python2.7 fury_route.py --provider_file $in_data/10-at-.01.dat --outdir $out_data2 --candidate_max 10 
#python2.7 fury_route.py --provider_file $in_data/10-at-.50.dat --outdir $out_data
python2.7 fury_route.py --provider_file $in_data/10-at-.10.dat --outdir $out_data 
#python2.7 fury_route.py --provider_file $in_data/10-at-.05.dat --outdir $out_data 
#python2.7 fury_route.py --provider_file $in_data/10-at-.01.dat --outdir $out_data 
