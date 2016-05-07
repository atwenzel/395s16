#!/bin/bash

echo "opening file..."
host=`hostname`
rm ping_data.dat
while read line; do
    p_data=`ping -c 3 $line | tail -1 | awk '{print $4}'`
    echo "$line $p_data" >> ping_data.dat
done < usable_hostnames.dat
