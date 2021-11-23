from sklearn.model_selection import train_test_split
from shutil import copyfile
import pandas as pd
import numpy as np
import argparse
import datetime
import json
import copy
import glob
import os

def creare_coswara_cdf(coswara_dir, coswara_metadata):
    # utility functions for reformating dates/times
    break_up_date = lambda s: (s[0:4], s[4:6], s[6:])
    to_datestring = lambda Y,M,D : "{}-{}-{}".format(Y, M, D)

    # every folder here represents a date we need
    # since we want to know when every sample was collected
    coswara_folders = [f for f in os.listdir(coswara_dir) if os.path.isdir(coswara_dir+f) and f[0]=='2']
    datestrings = {}
    for folder in coswara_folders:
        Y,M,D = break_up_date(folder)
        datesting = to_datestring(Y,M,D)
        for uid in os.listdir(coswara_dir+'/'+folder):
             datestrings[uid]=datesting

    coswara_cdf = []
    for idx, row in coswara_metadata.iterrows():
         #get patient info
         patient_info = {}
         patient_info['source'] = 'coswara'
         patient_info['patient_id'] = row['id']
         patient_info['cough_detected'] = np.nan
         patient_info['age']=row['a']
         patient_info['biological_sex']=row['g']
         patient_info['submission_date']= datestrings[row['id']]
         patient_info['pcr_test_date'] = np.nan
         patient_info['pcr_result_date'] = np.nan
    

def preprocess_coswara():
    coswara_dir = "datasets/Coswara-Data/"
    coswara_metadata = pd.read_csv(coswara_dir+'combined_data.csv')
    #print(coswara_metadata)
    #print(set(coswara_metadata["covid_status"]))

    coswara_cdf = create_coswara_cdf(coswara_dir, coswara_metadata)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="coswara");
    args = parser.parse_args()
    
    if args.dataset == "coswara":
       preprocess_coswara()
    
