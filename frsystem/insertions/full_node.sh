# copy the file
scp -oStrictHostKeyChecking=no ./insertions/ping_all.sh "northwestern_oak@"$1:~/
scp -oStrictHostKeyChecking=no ./data/usable_hostnames.dat "northwestern_oak@"$1:~/

# Run the script
ssh -T -oStrictHostKeyChecking=no "northwestern_oak@"$1 ./ping_all.sh

# copy the file back
scp -oStrictHostKeyChecking=no "northwestern_oak@"$1:~/ping_data.dat ./data/$1-ping_data.dat

return $?
