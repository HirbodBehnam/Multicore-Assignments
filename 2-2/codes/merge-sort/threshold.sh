#!/bin/bash
echo "threshold,time" > threshold.csv
for i in {1..20}; do
    threshold=$(bc <<< "2 ^ $i")
    running_time=$(./parallel.out "$threshold" | cut -d$' ' -f2)
    echo "$i,$running_time" >> threshold.csv
    sleep 0.5
done