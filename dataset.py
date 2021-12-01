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



    ### This part is for coughvid dataset only.
    if split_type == "coughvid":
        total_list = []
        for i, patient in enumerate(type_list["posotive"]):
            info = data_dict[patient]
            for feature_path in info["feature_paths"]:
                total_list.append([feature_path, 1])

        for i, patient in enumerate(type_list["negative"]):
            info = data_dict[patient]
            for feature_path in info["feature_paths"]:
                total_list.append([feature_path, 0])

        split_index1, split_index2 =  len(total_list)//9*7, len(total_list)//9*8
        train_list = total_list[:split_index1]
        valid_list = total_list[split_index1:split_index2]
        test_list = total_list[split_index2:]

        if semi:
            unlabeled_list = []
            for patient in type_lists["untested"]:
                info = data_dict[patient]
                for feature_path in info["feature_paths"]:
                    unlabeled_list.append([feature_path, -1])

            return train_list, valid_list, test_list, unlabeled_list

    ### The following part is for coswara dataset only.
          
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

    num_of_train_data = len(seen_speaker_train_list)

    if split_type == "speaker":
        random.shuffle(unseen_speaker_train_list)
        train_list = unseen_speaker_train_list[:num_of_train_data]

    elif split_type == "7-1-1":
        random.shuffle(seen_speaker_train_list)
        train_list = seen_speaker_train_list


    elif split_type == "random":
        combine_list = seen_speaker_train_list + unseen_speaker_train_list
        random.shuffle(combine_list)
        train_list = combine_list[:num_of_train_data]

    valid_split_index = len(train_list) // 8
    valid_list = train_list[:valid_split_index]
    train_list = train_list[valid_split_index:]

    if semi:
        unlabeled_list = []
        for patient in type_lists["untested"]:
            info = data_dict[patient]
            for feature_path in info["feature_paths"]:
                unlabeled_list.append([feature_path, -1])

        return train_list, valid_list, test_list, unlabeled_list
    else:

        return train_list, valid_list, test_list




# For training dataset, we don't fix the random frame at first.
class supervised_train_dataset(Dataset):
    def __init__(self, datalist):
        self.features = []
        self.labels = []
        for path, label in datalist:
            feature = np.load(path)
            self.features.append(feature)
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

# For test/validation dataset, we fix the random frame at first,
# so the testing result can be consistent.
class supervised_valid_dataset(Dataset):
    def __init__(self, datalist, num_of_frame, seed=1234):
        self.features = []
        self.labels = []
        self.num_of_frame = num_of_frame
        random.seed(seed)
        for path, label in datalist:

            feature = np.load(path)
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

            self.features.append(feature)
            self.labels.append(label)

        all_features = np.concatenate(self.features, axis=1)
        self.mean = all_features.mean(axis=1, keepdims=True)
        self.std = all_features.std(axis=1, keepdims=True)
        #print(self.mean, self.std)

    def __getitem__(self, idx):
        mean = np.repeat(self.mean, self.num_of_frame, axis=1)
        std = np.repeat(self.std, self.num_of_frame, axis=1)
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
