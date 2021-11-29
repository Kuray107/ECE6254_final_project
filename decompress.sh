#!/usr/bin/env bash


echo "Decompress coswara dataset..."     
data_root="datasets/Coswara-Data"
for subdir in $(find $data_root -name "202*" -type d); do
    echo ${subdir}
    cat ${subdir}/*.tar.gz.* > ${subdir}/combined_file.tar.gz
    tar -xvf ${subdir}/combined_file.tar.gz -C ${data_root} || exit 1;
    rm ${subdir}/*.tar.gz.*
    rm ${subdir}/combined_file.tar.gz
done
