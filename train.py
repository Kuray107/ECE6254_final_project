import os
import sys
import json
import torch
import argparse
import numpy as np

from torch.nn import BCELoss, CrossEntropyLoss
from torch.utils.data import DataLoader

import hparams
from models import AttRNN
from utils import SaveBest
from dataset import data_split, supervised_collate_fn


def validate(model, valid_loader, criterion):
    num_correct = 0
    num_total = 0
    avg_loss = 0.0
    for batch in valid_loader:
        features, labels = batch
        features = features.cuda()
        labels = labels.cuda()
        preds = model(features)
        num_correct += sum(torch.eq(labels, torch.argmax(preds, dim=1)))
        num_total += len(labels)
        avg_loss += criterion(preds, labels).item()*len(labels)

    return avg_loss/num_total, num_correct/num_total
        

def train(args):
    json_file = open(os.path.join('datasets', args.dataset+'.json'), 'r')
    data_dict = json.load(json_file)
    train_set, valid_set, test_set = data_split(data_dict, 
            semi=args.semi,
            split_type=args.split_type,
            positive_patient_num=hparams.positive_patient_num,
            negative_patient_num=hparams.negative_patient_num)
  
    model = AttRNN().cuda()
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 
            lr=hparams.learning_rate,
            weight_decay=hparams.weight_decay)
    
    save_best_cp = SaveBest("sup")

    if args.semi:
        # semi-supervised learning algorithm
        pass
    else:
        # supervsied learning algorithm
        collate_fn = supervised_collate_fn(num_of_frame=hparams.num_of_frame)

        train_loader = DataLoader(train_set, 
                num_workers=hparams.num_workers, shuffle=True,
                batch_size=hparams.batch_size,
                collate_fn=collate_fn)
        
        valid_loader = DataLoader(valid_set, 
                num_workers=hparams.num_workers, shuffle=False,
                batch_size=hparams.batch_size,
                collate_fn=collate_fn)

        test_loader = DataLoader(test_set, 
                num_workers=hparams.num_workers, shuffle=False,
                batch_size=hparams.batch_size,
                collate_fn=collate_fn)

        global_steps = 0
        for epoch in range(hparams.num_epochs):
            print("Epoch: {}".format(epoch))
            for i, batch in enumerate(train_loader):
                features, labels = batch
                features = features.cuda()
                labels = labels.cuda()
                model.zero_grad()
                preds = model(features)
                loss = criterion(preds, labels)
                reduced_loss = loss.item()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

                optimizer.step()
                print("global_step: {}. train_loss: {}".format(global_steps, reduced_loss))

                global_steps += 1

            model.eval()
            val_loss, val_acc = validate(model, valid_loader, criterion)
            print("Epoch: {}. valid_loss: {}, valid_acc: {}".format(epoch, val_loss, val_acc))

            if save_best_cp.apply(val_acc):
                print("saving best model...")
                model_fname = os.path.join('test', 'model_save_dir', "best_model.ckpt")
                torch.save(model.state_dict(), model_fname)

            elif epoch == (hparams.num_epochs-1):
                print("saving last model...")
                model_fname = os.path.join('test', 'model_save_dir', "last_model.ckpt")
                torch.save(model.state_dict(), model_fname)
             

            model.train()


        model.load_state_dict(torch.load('test/model_save_dir/best_model.ckpt'))
        model.eval()
        test_loss, test_acc = validate(model, test_loader, criterion)
        print("Best Epoch: {}. test_loss: {}, test_acc: {}".format(save_best_cp.best_epoch, test_loss, test_acc))
        model.load_state_dict(torch.load('test/model_save_dir/last_model.ckpt'))
        model.eval()
        test_loss, test_acc = validate(model, test_loader, criterion)
        print("Last Epoch: {}. test_loss: {}, test_acc: {}".format(save_best_cp.best_epoch, test_loss, test_acc))



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




