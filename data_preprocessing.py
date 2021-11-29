import pandas as pd
import numpy as np
import argparse
import json
import os

from audio_utils import wav_to_feature




def create_coswara_json(coswara_dir, coswara_metadata):
    
    # every folder here represents a date we need
    coswara_folders = [f for f in os.listdir(coswara_dir) if os.path.isdir(coswara_dir+f) and f[0]=='2']
    datestrings = {}
    for folder in coswara_folders:
        for uid in os.listdir(coswara_dir+'/'+folder):
             datestrings[uid]=folder

    coswara_json = {}
    p_count = 0
    n_count = 0
    u_count = 0
    pd_count = 0
    nd_count = 0
    ud_count = 0
    for idx, row in coswara_metadata.iterrows():
        # get patient info
        # Note that we only need sound, patient_id and their pcr_test results 
        # for our experiments

        patient_info = {}
        patient_folder = os.path.join(coswara_dir, datestrings[row['id']], row['id'])
        wav_paths = [os.path.join(patient_folder, f) for f in os.listdir(patient_folder) if '.wav' in f and f[:2] != "._"]
        patient_info['feature_paths'] = wav_to_feature(wav_paths, store=True)
        num_of_paths = len(patient_info['feature_paths'])
        if num_of_paths < 5:
            continue


        status= row['covid_status']
        if status in {'positive_mild', 'positive_moderate', 'positive_asymp'}:
            patient_info['pcr_test_result'] = 'positive'
            p_count += 1
            pd_count += num_of_paths
        elif status in {'healthy'}:
            patient_info['pcr_test_result'] = 'negative'
            n_count += 1
            nd_count += num_of_paths
        else:
            patient_info['pcr_test_result'] = 'untested'
            u_count += 1
            ud_count += num_of_paths

        coswara_json[row['id']] = patient_info

    print(p_count, n_count, u_count)
    print(pd_count, nd_count, ud_count)
    return coswara_json
 
def create_coughvid_json(coughvid_dir, coughvid_metadata, threshold=0.7):
    coughvid_json = {}
    for idx, row in coughvid_metadata.iterrows():
        if row['cough_detected'] < threshold:
            continue
        ID = row['patient_id']
        if ID not in coughvid_json:
            patient_info = {}
            patient_info['pcr_test_result'] = row['pcr_test_result_inferred']
            feature_paths = wav_to_feature([os.path.join(coughvid_dir, row['cough_path'])])
            if len(feature_paths) == 0:
                continue
            else:
                patient_info['feature_paths'] = feature_paths
                coughvid_json[ID] = patient_info
        else:
            print('warning: detect the same speaker: {}!'.format(ID))
            assert (coughvid_json[ID]['pcr_test_result'] == row['pcr_test_result'])
            feature_paths = wav_to_feature([os.path.join(coughvid_dir, row['cough_path'])])
            if len(feature_paths) == 0:
                continue
            else:
                coughvid_json[ID]['feature_paths'] += feature_paths

    print(len(coughvid_json))
    return coughvid_json


def preprocess_coswara():
    coswara_dir = "datasets/Coswara-Data/"
    coswara_metadata = pd.read_csv(coswara_dir+'combined_data.csv')
    coswara_json = create_coswara_json(coswara_dir, coswara_metadata)
    coswara_json = json.dumps(coswara_json, indent=4)
    with open("datasets/coswara.json", "w") as outfile:
        outfile.write(coswara_json)



def preprocess_coughvid():
    coughvid_dir = 'datasets/virufy-cdf-coughvid'
    coughvid_metadata = pd.read_csv(os.path.join(coughvid_dir, 'virufy-cdf-coughvid.csv'))
    coughvid_json = create_coughvid_json(coughvid_dir, coughvid_metadata)
    coughvid_json = json.dumps(coughvid_json, indent=4)
    with open('datasets/coughvid.json', 'w') as outfile:
        outfile.write(coughvid_json)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="coswara", 
            choices=["coswara", "coughvid"]);
    args = parser.parse_args()
    if args.dataset == "coswara":
        preprocess_coswara()
    elif args.dataset == 'coughvid':
        preprocess_coughvid()
    
