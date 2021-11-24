#!/usr/bin/env bash

stage=0
stop_stage=100
dataset="coswara" # coswara or coughvid
decompress=false

# arguments for stage 1:
model="att-rnn"
split_type="random"
semi_supervised=false


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage0: dataset preparation"
    if $decompress  && [ ${dataset} = "coswara" ]; then
        data_root="datasets/Coswara-Data"
        for subdir in $(find $data_root -name "202*" -type d); do
            echo ${subdir}
            cat ${subdir}/*.tar.gz.* > ${subdir}/combined_file.tar.gz
            tar -xvf ${subdir}/combined_file.tar.gz -C ${data_root} || exit 1;
            rm ${subdir}/*.tar.gz.*
            rm ${subdir}/combined_file.tar.gz
        done
    fi
    python data_preprocessing.py -d ${dataset} || exit 1;

fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage1: model training"
    python train.py -d ${dataset} -m ${model} -s ${semi_supervised} --split_type ${split_type}
