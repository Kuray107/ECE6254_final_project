import os
import sys
import json
import torch
import argparse
import numpy as np

from dataset import data_split



def train(args):
    json_file = open(os.path.join('datasets', args.dataset+'.json'), 'r')
    data_dict = json.load(json_file)
    train_set, valid_set, test_set = data_split(data_dict, 
            semi=args.semi,
            split_type=args.split_type)
  

    if args.semi:
        # semi-supervised learning algorithm
        pass
    else:
        # supervsied learning algorithm

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="coswara",
            choices=["coswara", "coughvid"]);
    parser.add_argument("-m", "--model", type=str, default="att-rnn",
            choices=["att-rnn", "VGGish"])
    parser.add_argument("-s", "--semi", type=bool, default=False)
    parser.add_argument("--split_type", type=str, default="random", 
            choices=["speaker", "7-1-1", "random"])
    
    args = parser.parse_args()
    # split_type of "speaker" or "7-1-1" are not supported for coughvid dataset
    if args.dataset == "coughvid" and args.split_type != "random":
        print("Error: split_type of \"speaker\" or \"7-1-1\" are not supported for coughvid dataset")
        sys.exit()

    else:
        train(args)




