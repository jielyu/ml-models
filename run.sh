#!/bin/bash
set -ex

# set phase
phase="evaluate"
if [ $# -ge 1 ]; then
    phase=$1
fi
echo "set phase=${phase}"

# set flag to determine whether run process on background
bg="no"
if [ $# -ge 2 ]; then
    bg=$2
fi
echo "set background=${bg}"

# set batch size
bs=64
if [ $# -ge 3 ]; then
    bs=$3
fi
echo "set batch_size=${bs}"
#exit
# modify this variable to start other process
process_name="keras_exp.train_vgg_on_celeba"
if [ ${bg} = "yes" ]; then
    python3 -m ${process_name} --phase=${phase} --batch-size=${bs} >out.${phase}.log 2>&1 &
else
    python3 -m ${process_name} --phase=${phase} --batch-size=${bs} 
fi