#!/bin/sh
export TF_CPP_MIN_LOG_LEVEL=3

TMP=/scratch/gluo/tmp_file
NUM_GPUS=2
GPU_IDS="2,3"
IFS=', ' read -a array <<< $GPU_IDS
FILELIST="/home/gluo/github/spreco/validation_volume"

config_file=/home/gluo/workspace/sampling_posterior/revision/fastmri.yaml
workspace=/home/gluo/workspace/sampling_posterior/revision/8x
h5dir=/home/ague/data/gluo/nyu_dataset/multicoil_val

job()
{
    python recon_fastmri.py --config=$config_file --workspace=$workspace --h5path=$1 --gpu_id=$2
}

pop_lines()
{
    # usage pop the first $1 lines from the file $2 to the file $3
    head -n$1 $2 > $3
    sed -i -e "1,${1}d" $2
}

while pop_lines $NUM_GPUS $FILELIST $TMP && test -s $TMP
do
    i=0
    while read -r line
    do
        job $h5dir/$line ${array[$i]} &
        i=$((i+1))
    done < "$TMP"
    wait
done