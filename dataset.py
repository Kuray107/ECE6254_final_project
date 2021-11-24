import random

import torch
import numpy as np
from torch.utils.data import Dataset


# seed to initializa the random engine
seed = 1234

def data_split(data_dict, 
        semi=False, 
        split_type='random', 
        positive_patient_num=90, 
        negative_patient_num=90
    ):

    random.seed(seed)

    type_lists = {"positive": [], "negative": [], "untested": []}
    for patient, info in data_dict.items():
        result = info["pcr_test_result"]
        type_lists[result].append(patient)
    random.shuffle(type_lists["positive"])
    random.shuffle(type_lists["negative"])
    random.shuffle(type_lists["untested"])

    if positive_patient_num < 0 or positive_patient_num > len(type_lists["positive"]):
        positive_patient_num = len(type_lists["positive"])
    if negative_patient_num < 0 or negative_patient_num > len(type_lists["negative"]):
        negative_patient_num = len(type_lists["negative"])

    # resize positive and negative datasets according to the specified number
    type_lists["positive"] = type_lists["positive"][:positive_patient_num]
    type_lists["negative"] = type_lists["negative"][:negative_patient_num]
          
    train_list = []
    valid_list = []
    test_list = []

    if split_type == "speaker":
        # Split the dataset according to speaker (split by a 7speaker:1speaker:1speaker manner)

        positive_index1, positive_index2 = (positive_patient_num//9)*7, (positive_patient_num//9)*8
        for i, patient in enumerate(type_lists["positive"]):
            info = data_dict[patient]
            for wav_path in info["wav_paths"]:
                if i < positive_index1:
                    train_list.append([wav_path, 1]) # the label for positive is 1
                elif i >= positive_index1 and i < positive_index2:
                    valid_list.append([wav_path, 1])
                else:
                    test_list.append([wav_path, 1])

        
        negative_index1, negative_index2 = (negative_patient_num//9)*7, (negative_patient_num//9)*8
        for i, patient in enumerate(type_lists["negative"]):
            info = data_dict[patient]
            for wav_path in info["wav_paths"]:
                if i < negative_index1:
                    train_list.append([wav_path, 0]) # the label for negative is 1
                elif i >= negative_index1 and i < negative_index2:
                    valid_list.append([wav_path, 0])
                else:
                    test_list.append([wav_path, 0])

    elif split_type == "7-1-1":
        # Each patient/speaker has 9 audio file. We randomly pick 1 for valid, 1 for test
        for patient in type_lists["positive"]:
            info = data_dict[patient]
            random.shuffle(info["wav_paths"])
            for i, wav_path in enumerate(info["wav_paths"]):
                if i == 7:
                    valid_list.append([wav_path, 1])
                elif i == 8:
                    test_list.append([wav_path, 1])
                else:
                    train_list.append([wav_path, 1])

        for patient in type_lists["negative"]:
            info = data_dict[patient]
            random.shuffle(info["wav_paths"])
            for i, wav_path in enumerate(info["wav_paths"]):
                if i == 7:
                    valid_list.append([wav_path, 0])
                elif i == 8:
                    test_list.append([wav_path, 0])
                elif i <= 6:
                    train_list.append([wav_path, 0])


    elif split_type == "random":
        total_list = []
        for patient in type_lists["positive"]:
            info = data_dict[patient]
            for wav_path in info["wav_paths"]:
                total_list.append([wav_path, 1])
        for patient in type_lists["negative"]:
            info = data_dict[patient]
            for wav_path in info["wav_paths"]:
                total_list.append([wav_path, 0])


        random.shuffle(total_list)
        index1, index2 = (len(total_list)//9)*7, (len(total_list)//9)*8

        train_list = total_list[:index1]
        valid_list = total_list[index1:index2]
        test_list = total_list[index2:]


    if semi:
        unlabeled_list = []
        for patient in type_lists["untested"]:
            info = data_dict[patient]
            for wav_path in info["wav_paths"]:
                unlabeled_list.append([wav_path, -1])
        train_set = semi_supervsied_dataset(train_list, unlabeled_list)
    else:
        train_set = supervsied_dataset(train_list)
    valid_set = supervised_dataset(valid_list)
    test_set = supervised_dataset(test_list)

    return train_set, valid_set, test_set


class supervised_dataset(Dataset):
    def __init__(self, datalist):



    def __getitem__(self, idx):


    def __len__(self):
        return (len(data_list)):



class semi_supervised_dataset(Dataset):
    def __init__(self, labeled_datalist, unlabeled_datalist):


    def __getitem__(self, idx):


    def __len__(self):
        

    
