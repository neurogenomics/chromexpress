#!/bin/bash

echo "Average bigwig tracks starting"
now="$(date)"
printf "Current date and time %s\n" "$now"

# run in parallel for speed
avg_tracks () {
    local bigwig_file=$1
    #Average all tracks to larger bin
    bin_size=25
    echo "$bigwig_file"
    #R script to average track at specified size
    Rscript ./bin/avg_bigwig_tracks.R $bigwig_file $bin_size
}
export -f avg_tracks

parallel --jobs 4 avg_tracks ::: `find ./data/ -name '*.bigWig'`
