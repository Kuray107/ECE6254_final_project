import random
from sklearn.preprocessing import StandardScaler

import torch
import numpy as np
from torch.utils.data import Dataset

# seed to initializa the random engine
#seed = 1234

def data_split(data_dict, 
        semi=False, 
        split_type='random', 
        positive_patient_num=20, 
        negative_patient_num=20,
        seed=1234
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
            for feature_path in info["feature_paths"]:
                if i < positive_index1:
                    train_list.append([feature_path, 1]) # the label for positive is 1
                elif i >= positive_index1 and i < positive_index2:
                    valid_list.append([feature_path, 1])
                else:
                    test_list.append([feature_path, 1])

        
        negative_index1, negative_index2 = (negative_patient_num//9)*7, (negative_patient_num//9)*8
        for i, patient in enumerate(type_lists["negative"]):
            info = data_dict[patient]
            for feature_path in info["feature_paths"]:
                if i < negative_index1:
                    train_list.append([feature_path, 0]) # the label for negative is 0
                elif i >= negative_index1 and i < negative_index2:
                    valid_list.append([feature_path, 0])
                else:
                    test_list.append([feature_path, 0])

    elif split_type == "7-1-1":
        # Each patient/speaker has 9 audio file. We randomly pick 1 for valid, 1 for test
        for patient in type_lists["positive"]:
            info = data_dict[patient]
            random.shuffle(info["feature_paths"])
            for i, feature_path in enumerate(info["feature_paths"]):
                if i == 1:
                    valid_list.append([feature_path, 1])
                elif i == 2:
                    test_list.append([feature_path, 1])
                else:
                    train_list.append([feature_path, 1])

        for patient in type_lists["negative"]:
            info = data_dict[patient]
            random.shuffle(info["feature_paths"])
            for i, feature_path in enumerate(info["feature_paths"]):
                if i == 1:
                    valid_list.append([feature_path, 0])
                elif i == 2:
                    test_list.append([feature_path, 0])
                else:
                    train_list.append([feature_path, 0])


    elif split_type == "random":
        total_list = []
        for patient in type_lists["positive"]:
            info = data_dict[patient]
            for feature_path in info["feature_paths"]:
                total_list.append([feature_path, 1])
        for patient in type_lists["negative"]:
            info = data_dict[patient]
            for feature_path in info["feature_paths"]:
                total_list.append([feature_path, 0])


        random.shuffle(total_list)
        index1, index2 = (len(total_list)//9)*7, (len(total_list)//9)*8

        train_list = total_list[:index1]
        valid_list = total_list[index1:index2]
        test_list = total_list[index2:]


    if semi:
        unlabeled_list = []
        for patient in type_lists["untested"]:
            info = data_dict[patient]
            for feature_path in info["feature_paths"]:
                unlabeled_list.append([feature_path, -1])
        train_set = supervised_dataset(train_list)
    else:
        train_set = supervised_dataset(train_list)
    valid_set = supervised_dataset(valid_list)
    test_set = supervised_dataset(test_list)

    return train_set, valid_set, test_set

    
def data_split2(data_dict, 
        semi=False, 
        split_type='random', 
        positive_patient_num=20, 
        negative_patient_num=20,
        seed=1234
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
          
    seen_speaker_train_list = []
    unseen_speaker_train_list = []
    test_list = []


    ## First we selected a half of speakers. Each of the speakers will contribute 
    ## exactly one testing data. The remaining data will be our seen-speaker training data
    for i, patient in enumerate(type_lists["positive"][:(positive_patient_num//2)]):
        info = data_dict[patient]
        random.shuffle(info['feature_paths'])
        for i, feature_path in enumerate(info["feature_paths"]):
            if i == 0:
                test_list.append([feature_path, 1])
            else:
                seen_speaker_train_list.append([feature_path, 1])

    for i, patient in enumerate(type_lists["negative"][:(negative_patient_num//2)]):
        info = data_dict[patient]
        random.shuffle(info['feature_paths'])
        for i, feature_path in enumerate(info["feature_paths"]):
            if i == 0:
                test_list.append([feature_path, 0])
            else:
                seen_speaker_train_list.append([feature_path, 0])


    # Gather training data from unseen speakers 
    for i, patient in enumerate(type_lists["positive"][(positive_patient_num//2):]):
        info = data_dict[patient]
        for feature_path in info["feature_paths"]:
            unseen_speaker_train_list.append([feature_path, 1]) # the label for positive is 1

    for i, patient in enumerate(type_lists["negative"][(negative_patient_num//2):]):
        info = data_dict[patient]
        for feature_path in info["feature_paths"]:
            unseen_speaker_train_list.append([feature_path, 0]) # the label for negative is 0


    # Make sure all three type of data spliting method 
    # will produce exactly the same number of training data

    num_of_valid_data = len(seen_speaker_train_list)//8
    num_of_train_data = len(seen_speaker_train_list) - num_of_valid_data

    if split_type == "speaker":
        random.shuffle(unseen_speaker_train_list)
        valid_list = unseen_speaker_train_list[:num_of_valid_data]
        train_list = unseen_speaker_train_list[num_of_valid_data:num_of_valid_data+num_of_train_data]

    elif split_type == "7-1-1":
        random.shuffle(seen_speaker_train_list)
        valid_list = seen_speaker_train_list[:num_of_valid_data]
        train_list = seen_speaker_train_list[num_of_valid_data:]


    elif split_type == "random":
        combine_list = seen_speaker_train_list + unseen_speaker_train_list
        random.shuffle(combine_list)
        valid_list = combine_list[:num_of_valid_data]
        train_list = combine_list[num_of_valid_data:num_of_valid_data+num_of_train_data]


    if semi:
        unlabeled_list = []
        for patient in type_lists["untested"]:
            info = data_dict[patient]
            for feature_path in info["feature_paths"]:
                unlabeled_list.append([feature_path, -1])
        train_set = supervised_dataset(train_list)
    else:
        train_set = supervised_dataset(train_list)
    valid_set = supervised_dataset(valid_list)
    test_set = supervised_dataset(test_list)

    return train_set, valid_set, test_set


class supervised_dataset(Dataset):
    def __init__(self, datalist):
        self.features = []
        self.labels = []
        self.scaler = StandardScaler()
        for path, label in datalist:
            feature = np.load(path)
            self.features.append(np.load(path))
            self.labels.append(label)

        all_features = np.concatenate(self.features, axis=1)
        self.mean = all_features.mean(axis=1, keepdims=True)
        self.std = all_features.std(axis=1, keepdims=True)
        #print(self.mean, self.std)

    def __getitem__(self, idx):
        D, T = self.features[idx].shape
        mean = np.repeat(self.mean, T, axis=1)
        std = np.repeat(self.std, T, axis=1)
        return (self.features[idx]-mean)/std, self.labels[idx]

    def __len__(self):
        return (len(self.features))

class supervised_collate_fn():
    def __init__(self, num_of_frame=150, add_noise=True):
        self.num_of_frame = num_of_frame
        self.add_noise = add_noise

    def __call__(self, batch):
        features = []
        labels = []
        for feature, label in batch:
            _, length = feature.shape
            if length < self.num_of_frame:
                feature = np.concatenate(
                        [np.tile(feature, self.num_of_frame // length), feature[:, :self.num_of_frame % length]], 
                        axis=1)
            elif length > self.num_of_frame and length <= self.num_of_frame+32:
                start = (length - self.num_of_frame)//2
                feature = feature[:, start:start+self.num_of_frame]
            elif length > self.num_of_frame+32:
                start = random.choice(range(16, length-self.num_of_frame-16))
                feature = feature[:, start:start+self.num_of_frame]
            
            features.append(np.expand_dims(feature, axis=0))
            labels.append(label)

        features = torch.from_numpy(np.concatenate(features, axis=0))
        if self.add_noise:
            noise = torch.randn(features.size())/10
            features = features + noise
        labels = torch.FloatTensor(labels)

        return features, labels

#class semi_supervised_dataset(Dataset):
#    def __init__(self, labeled_datalist, unlabeled_datalist):
#
#
#    def __getitem__(self, idx):
#
#
#    def __len__(self):