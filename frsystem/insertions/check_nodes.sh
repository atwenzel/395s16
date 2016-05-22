#!/bin/bash
#DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)
#sites=()
#echo $DIR
#rm $DIR'/../activenodes.dat'
#while IFS='' read -r line || [[ -n $line ]]; do
#    if ! ([[ $line == *"#"* ]])
#    then
#        sites+=($line)
#    fi
#done < $DIR"/../allnodes.dat"
#
#usr="northwestern_oak@"
#
#for ((i=0; i<${#sites[@]}; i++)); do
#    #ssh -T $usr${sites[i]} hostname >> $DIR'/activenodes.dat' &
#    echo ${sites[i]}
#    ssh -T -oStrictHostKeyChecking=no $usr${sites[i]} < $DIR'/check_on_client.sh' >> $DIR'/../activenodes.dat' &
#    pid=$!
#    sleep 10
#    kill $pid
#    sleep 2
#done

ping -c 2 $1 > /dev/null 2>&1
ssh -T -oStrictHostKeyChecking=no "northwestern_audit@"$1 < insertions/check_on_client.sh
