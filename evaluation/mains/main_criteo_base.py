#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
    @project:RefineCTR
'''
import os
import sys
import time

import torch
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append("../..")
from FRCTR.model_zoo import FMFrn
from FRCTR.module_zoo import ALLFrn_OPS
from FRCTR.utils.earlystoping import EarlyStopping
from FRCTR.utils import setup_seed
from FRCTR.data import get_criteo_811

from sklearn.metrics import log_loss, roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# print("=======================", DEVICE)
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name())


def get_model(
        name,
        field_dims,
        embed_dim=20,
        frn_name="skip",
        att_size=16,
        mlp_layers=(400, 400, 400)):
    if name == "fm":
        return FMFrn(field_dims, embed_dim, FRN=ALLFrn_OPS[frn_name](len(field_dims), embed_dim))

    else:
        raise ValueError("No valid model name.")


def count_params(model):
    params = sum(param.numel() for param in model.parameters())
    return params


def train(model,
          optimizer,
          data_loader,
          criterion,
          device="cuda:0",
          log_interval=50000, ):
    model.train()
    pred = list()
    target = list()
    total_loss = 0
    for i, (user_item, label) in enumerate(tqdm.tqdm(data_loader)):
        label = label.float()
        user_item = user_item.long()

        user_item = user_item.to(DEVICE)  # [B,F]
        label = label.to(DEVICE)  # [B]

        model.zero_grad()
        pred_y = torch.sigmoid(model(user_item).squeeze(1))
        loss = criterion(pred_y, label)
        loss.backward()
        optimizer.step()

        pred.extend(pred_y.tolist())
        target.extend(label.tolist())
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            print('train_loss:', total_loss / (i + 1))
    loss2 = total_loss / (i + 1)
    return loss2


def test_roc(model, data_loader):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(
                data_loader, smoothing=0, mininterval=1.0):
            fields = fields.long()
            target = target.float()
            # fields, target = fields.cuda(), target.cuda()
            fields, target = fields.to(DEVICE), target.to(DEVICE)
            y = torch.sigmoid(model(fields).squeeze(1))

            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts), log_loss(targets, predicts)


def main(dataset_name,
         model_name,
         frn_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         save_dir,
         path,
         repeat=1,
         hint=""):
    field_dims, trainLoader, validLoader, testLoader = \
        get_criteo_811(train_path="/train.txt", batch_size=batch_size)
    print(field_dims)
    print(sum(field_dims))

    time_fix = time.strftime("%m%d%H%M%S", time.localtime())

    for K in [16]:
        paths = os.path.join(save_dir, str(K), model_name, frn_name)
        if not os.path.exists(paths):
            os.makedirs(paths)

        with open(
                paths + f"/{model_name}_{frn_name}logs2_{K}_{batch_size}_{learning_rate}_{weight_decay}_{time_fix}.p",
                "a+") as fout:
            fout.write("model_name:{}\tfrn_name:{},Batch_size:{}\tlearning_rate:{}\t"
                       "StartTime:{}\tweight_decay:{}\n"
                       .format(model_name, frn_name, batch_size, learning_rate,
                               time.strftime("%d%H%M%S", time.localtime()), weight_decay))
            print("Start train -- K : {}".format(K))
            criterion = torch.nn.BCELoss()
            model = get_model(
                name=model_name,
                field_dims=field_dims,
                embed_dim=K,
                frn_name=frn_name).to(DEVICE)

            params = count_params(model)
            fout.write("count_params:{}\n".format(params))
            print(params)

            optimizer = torch.optim.Adam(
                params=model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay)

            early_stopping = EarlyStopping(patience=8, verbose=True, prefix=path)

            val_auc_best = 0
            auc_index_record = ""

            val_loss_best = 1000
            loss_index_record = ""

            # scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=4)
            scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=4)
            for epoch_i in range(epoch):
                print(__file__, model_name, frn_name, batch_size, repeat, K, learning_rate, weight_decay, epoch_i, "/",
                      epoch)

                start = time.time()

                train_loss = train(model, optimizer, trainLoader, criterion)
                val_auc, val_loss = test_roc(model, validLoader)
                test_auc, test_loss = test_roc(model, testLoader)

                scheduler.step(val_auc)

                end = time.time()
                if val_auc > val_auc_best:
                    torch.save(model, paths + f"/{model_name}_best_auc_{K}_{time_fix}.pkl")

                if val_auc > val_auc_best:
                    val_auc_best = val_auc
                    auc_index_record = "epoch_i:{}\t{:.6f}\t{:.6f}".format(epoch_i, test_auc, test_loss)

                if val_loss < val_loss_best:
                    val_loss_best = val_loss
                    loss_index_record = "epoch_i:{}\t{:.6f}\t{:.6f}".format(epoch_i, test_auc, test_loss)

                print(
                    "Train  K:{}\tEpoch:{}\ttrain_loss:{:.6f}\tval_loss:{:.6f}\tval_auc:{:.6f}\ttime:{:.6f}\ttest_auc and test_loss: {:.6f}\t{:.6f}\n"
                    .format(K, epoch_i, train_loss, val_loss, val_auc, end - start, test_auc, test_loss))

                fout.write(
                    "Train  K:{}\tEpoch:{}\ttrain_loss:{:.6f}\tval_loss:{:.6f}\tval_auc:{:.6f}\ttime:{:.6f}\ttest_auc and test_loss: {:.6f}\t{:.6f}\n"
                    .format(K, epoch_i, train_loss, val_loss, val_auc, end - start, test_auc, test_loss))

                early_stopping(val_auc)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            print("Test:{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n"
                  .format(K, val_auc, val_auc_best, val_loss, val_loss_best, test_loss, test_auc))

            fout.write("Test:{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n"
                       .format(K, val_auc, val_auc_best, val_loss, val_loss_best, test_loss, test_auc))

            fout.write("auc_best:\t{}\nloss_best:\t{}".format(auc_index_record, loss_index_record))


if __name__ == '__main__':
    setup_seed(2022)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='frappe')
    parser.add_argument('--save_dir', default='../chkpts/frappe811')
    parser.add_argument('--path', default="../data", help="")
    parser.add_argument('--model_name', default='fm')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda:0', help="cuda:0")
    parser.add_argument('--model', default=0, type=int)
    parser.add_argument('--module', default=0, type=int)
    parser.add_argument('--repeats', default=10, type=int)
    parser.add_argument('--hint', default="Insert FR module to CTR models.")
    args = parser.parse_args()

    model_names = ["fm"]
    frn_names = ["skip"]
    if args.model == 0:
        model_names = ["fm"]
        # model_names = ["deepfmp", "fm", "deepfm", "dcnv2p", "cn2", "dcnv2", "dcn", "cn", "dcnp", "afnp2", "afn", "afnp"]

    if args.module == 0:
        frn_names = ["skip", "senet", "non", "drm", "gate_v", "gate_b", "selfatt", "tce"]

    print(model_names)
    print(frn_names)
    for bs in [2000, 5000, 1000]:
        for lr in [0.1, 0.01, 0.001]:
            for weight_decay in [1e-5]:
                for i in range(args.repeats):
                    for name in model_names:
                        for frn_name in frn_names:
                            args.learning_rate = lr
                            args.batch_size = bs
                            args.weight_decay = weight_decay

                            main(dataset_name=args.dataset_name,
                                 model_name=name,
                                 frn_name=frn_name,
                                 epoch=args.epoch,
                                 learning_rate=args.learning_rate,
                                 batch_size=args.batch_size,
                                 weight_decay=args.weight_decay,
                                 save_dir=args.save_dir,
                                 path=args.path,
                                 repeat=i + 1,
                                 hint=args.hint, )
