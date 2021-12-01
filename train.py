import os
import sys
import json
import torch
import argparse
import numpy as np

from torch.nn import BCELoss, CrossEntropyLoss
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

import hparams
from models import AttRNN, AttCNN
from utils import SaveBest, calculate_F1_score, get_auc_score, delete_outlier
from dataset import data_split, supervised_train_dataset, supervised_valid_dataset, supervised_collate_fn


def validate(model, valid_loader, criterion):
    all_predictions = []
    all_labels = []
    avg_loss = 0.0
    for batch in valid_loader:
        features, labels = batch
        features = features.cuda()
        labels = labels.cuda()
        preds = model(features)
        avg_loss += torch.mean(criterion(preds, labels)).item()*len(labels)

        all_labels.append(labels.detach().cpu())
        all_predictions.append(preds.detach().cpu())

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    auc_score = get_auc_score(all_predictions, all_labels)
    F1_score, precision, recall, acc = calculate_F1_score(torch.gt(all_predictions, hparams.prob_threshold), all_labels)

    return avg_loss/len(all_labels), F1_score, acc, precision, recall, auc_score
        

def train(args, random_seed):
    json_file = open(os.path.join('datasets', args.dataset+'.json'), 'r')
    data_dict = json.load(json_file)
    if not args.semi:
        train_list, valid_list, test_list = data_split(data_dict, 
                semi=False,
                split_type=args.split_type,
                positive_patient_num=hparams.positive_patient_num,
                negative_patient_num=hparams.negative_patient_num,
                seed=random_seed)
    else:
        train_list, valid_list, test_list, unlabeled_list = data_split(data_dict, 
                semi=True,
                split_type=args.split_type,
                positive_patient_num=hparams.positive_patient_num,
                negative_patient_num=hparams.negative_patient_num,
                seed=random_seed)

    print(len(train_list), len(valid_list), len(test_list))
    # Model stucture
    if hparams.model_type == "AttRNN":
        model = AttRNN().cuda()
    else:
        model = AttCNN().cuda()

    # Loss function and optimizer
    criterion = BCELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), 
            lr=hparams.learning_rate,
            weight_decay=hparams.weight_decay)
    
    # Early stopping 
    save_best_cp = SaveBest("sup")

    if args.semi:
        # semi-supervised learning algorithm
        pass
    else:
        # supervsied learning algorithm
        collate_fn = supervised_collate_fn(num_of_frame=hparams.num_of_frame)
        valid_collate_fn = supervised_collate_fn(num_of_frame=hparams.num_of_frame, add_noise=False)

        train_set = supervised_train_dataset(train_list)
        valid_set = supervised_valid_dataset(valid_list, num_of_frame=hparams.num_of_frame, seed=random_seed)
        test_set = supervised_valid_dataset(test_list, num_of_frame=hparams.num_of_frame, seed=random_seed)

        train_loader = DataLoader(train_set, 
                num_workers=hparams.num_workers, shuffle=True,
                batch_size=hparams.batch_size,
                collate_fn=collate_fn)
        
        valid_loader = DataLoader(valid_set, 
                num_workers=hparams.num_workers, shuffle=False,
                batch_size=hparams.batch_size,
                collate_fn=valid_collate_fn)

        test_loader = DataLoader(test_set, 
                num_workers=hparams.num_workers, shuffle=False,
                batch_size=hparams.batch_size,
                collate_fn=valid_collate_fn)

        for epoch in range(hparams.num_epochs):
            if epoch >= 10:
                num = 0.9**(epoch - 9)
                lr = max(hparams.learning_rate*num, hparams.learning_rate_min)
                for g in optimizer.param_groups:
                    g['lr'] = lr
                print("Epoch: {}, lr: {:.4f}".format(epoch, lr))
                
            for batch in train_loader:
                features, labels = batch
                features = features.cuda()
                labels = labels.cuda()
                model.zero_grad()
                preds = model(features)
                loss = criterion(preds, labels)
                loss_weight = (labels*(hparams.positive_negative_loss_ratio-1) + 1)
                loss = torch.mean(loss*loss_weight)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)
                optimizer.step()

            model.eval()
            train_loss, train_F1, train_acc, train_precision, train_recall, train_auc = validate(model, train_loader, criterion)
            print("Epoch: {}. train_loss: {:.3f}, train_F1: {:.3f}, train_acc: {:.3f}, train_auc: {:.3f}".format(
                    epoch, train_loss, train_F1, train_acc, train_auc))
            val_loss, val_F1, val_acc, val_precision, val_recall, val_auc = validate(model, valid_loader, criterion)
            print("Epoch: {}. valid_loss: {:.3f}, valid_F1: {:.3f}, valid_acc: {:.3f}. valid_auc: {:.3f}".format(
                    epoch, val_loss, val_F1, val_acc, val_auc))
            

            if save_best_cp.apply(val_F1): #
                print("saving best model...")
                model_fname = os.path.join('results', args.split_type,  "best_model.ckpt")
                torch.save(model.state_dict(), model_fname)
             
            model.train()


        ### Finishing training. Do testing.
        model.eval()
        test_loss, test_F1, test_acc, test_precision, test_recall, test_auc = validate(model, test_loader, criterion)
        print("Last Epoch,  test_loss: {:.3f}, test_F1: {:.3f}, test_acc: {:.3f}, test_auc: {:.3f}".format(
            test_loss, test_F1, test_acc, test_auc))

        model.load_state_dict(torch.load(os.path.join('results', args.split_type, 'best_model.ckpt')))
        model.eval()
        test_loss, test_F1, test_acc, test_precision, test_recall, test_auc = validate(model, test_loader, criterion)
        print("Best Epoch: {},  val_F1: {:.3f}, test_loss: {:.3f}, test_F1 {:.3f}, test_acc: {}, test_auc: {:.3f}".format(
            save_best_cp.best_epoch, save_best_cp.best_val, test_loss, test_F1, test_acc, test_auc))

        return test_F1, test_acc, test_precision, test_recall, test_auc



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="coswara",
            choices=["coswara", "coughvid"]);
    parser.add_argument("-s", "--semi", type=bool, default=False)
    parser.add_argument("--split_type", type=str, default="random", 
            choices=["speaker", "7-1-1", "random"])
    
    args = parser.parse_args()
    # split_type of "speaker" or "7-1-1" are not supported for coughvid dataset
    if args.dataset == "coughvid" and args.split_type != "random":
        print("Error: split_type of \"speaker\" or \"7-1-1\" are not supported for coughvid dataset")
        sys.exit()

    F1_list = []
    acc_list = []
    precision_list = []
    recall_list = []
    auc_list = []

    for seed in hparams.random_seeds:
        test_F1, test_acc, test_precision, test_recall, test_auc = train(args, seed)
        F1_list.append(test_F1)
        acc_list.append(test_acc)
        precision_list.append(test_precision)
        recall_list.append(test_recall)
        auc_list.append(test_auc)


    F1_list = np.asarray(F1_list)
    acc_list =  np.asarray(acc_list)
    precision_list = np.asarray(precision_list)
    recall_list = np.asarray(recall_list)
    auc_list = np.asarray(auc_list)
    ### Delete outlier:
    
    print(F1_list.mean(), acc_list.mean(), precision_list.mean(), recall_list.mean(), auc_list.mean())
    print(F1_list.std(), acc_list.std(), precision_list.std(), recall_list.std(), auc_list.std())

