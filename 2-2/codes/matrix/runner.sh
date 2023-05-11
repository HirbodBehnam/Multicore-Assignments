#!/bin/bash
true > result.txt
for executable in *.out; do
    RUNS=""
    for i in {1..10}; do
        RUNS+=$(eval "./$executable")
        RUNS+=$'\n'
        sleep 0.1
    done
    echo "Status for $executable: $RUNS"
    echo -n "$executable: " >> result.txt
    awk '{ sum += $2; n++ } END { print sum / n }' <<< "$RUNS" >> result.txt
done