#!/usr/bin/env bash

echo "Convert files in coughvid dataset to wav format by ffmpeg"    
data_root="datasets/virufy-cdf-coughvid/virufy-cdf-coughvid"
for file in $(find $data_root -name "*.webm" -type f); do
    newfile=${file%.*}.wav
    echo ${newfile}
    ffmpeg -i $file $newfile || exit 1;
    rm $file
done
for file in $(find $data_root -name "*.ogg" -type f); do
    newfile=${file%.*}.wav
    echo ${newfile}
    ffmpeg -i $file $newfile || exit 1;
    rm $file
done
