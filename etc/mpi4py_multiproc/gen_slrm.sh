#!/bin/bash

template=$1

for nodes in 1 2 4 8; do
    cores=$(( nodes * 24 ))
    outfile="run_${cores}.slrm"
    cat ${template} | sed -e "s/@NODES@/${nodes}/g" -e "s/@CORES@/${cores}/g" > ${outfile}
done

