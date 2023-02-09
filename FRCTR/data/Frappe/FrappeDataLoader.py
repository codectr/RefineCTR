#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Author:wangfy
@project:DL_recommend
@Time:2020/4/29 9:42 上午
'''

import os
import pickle

import pandas as pd
import torch
import tqdm


class LoadData811():
    def __init__(self, path="./Data/", dataset="frappe", loss_type="square_loss"):
        self.dataset = dataset
        self.loss_type = loss_type
        self.path = path + dataset + "/"
        self.trainfile = self.path + dataset + ".train.libfm"
        self.testfile = self.path + dataset + ".test.libfm"
        self.validationfile = self.path + dataset + ".validation.libfm"
        self.features_M = {}
        self.construct_df()

    def construct_df(self):
        self.data_train = pd.read_table(self.trainfile, sep=" ", header=None, engine='python')
        self.data_test = pd.read_table(self.testfile, sep=" ", header=None, engine="python")
        self.data_valid = pd.read_table(self.validationfile, sep=" ", header=None, engine="python")

        for i in self.data_test.columns[1:]:
            self.data_test[i] = self.data_test[i].apply(lambda x: int(x.split(":")[0]))
            self.data_train[i] = self.data_train[i].apply(lambda x: int(x.split(":")[0]))
            self.data_valid[i] = self.data_valid[i].apply(lambda x: int(x.split(":")[0]))

        self.all_data = pd.concat([self.data_train, self.data_test, self.data_valid])
        self.field_dims = []

        for i in self.all_data.columns[1:]:
            maps = {val: k for k, val in enumerate(set(self.all_data[i]))}
            # self.data_test[i] = self.data_test[i].map(maps)
            # self.data_train[i] = self.data_train[i].map(maps)
            # self.data_valid[i] = self.data_valid[i].map(maps)
            self.all_data[i] = self.all_data[i].map(maps)
            self.features_M[i] = maps
            self.field_dims.append(len(set(self.all_data[i])))

        self.all_data[0] = self.all_data[0].apply(lambda x: max(x, 0))
        # self.data_test[0] = self.data_test[0].apply(lambda x: max(x, 0))
        # self.data_train[0] = self.data_train[0].apply(lambda x: max(x, 0))
        # self.data_valid[0] = self.data_valid[0].apply(lambda x: max(x, 0))


class LoadData():
    def __init__(self, path="./Data/", dataset="frappe", loss_type="square_loss"):
        self.dataset = dataset
        self.loss_type = loss_type
        self.path = path + dataset + "/"
        self.trainfile = self.path + dataset + ".train.libfm"
        self.testfile = self.path + dataset + ".test.libfm"
        self.validationfile = self.path + dataset + ".validation.libfm"
        self.features_M = {}
        self.construct_df()

    def construct_df(self):
        self.data_train = pd.read_table(self.trainfile, sep=" ", header=None, engine='python')
        self.data_test = pd.read_table(self.testfile, sep=" ", header=None, engine="python")
        self.data_valid = pd.read_table(self.validationfile, sep=" ", header=None, engine="python")

        for i in self.data_test.columns[1:]:
            self.data_test[i] = self.data_test[i].apply(lambda x: int(x.split(":")[0]))
            self.data_train[i] = self.data_train[i].apply(lambda x: int(x.split(":")[0]))
            self.data_valid[i] = self.data_valid[i].apply(lambda x: int(x.split(":")[0]))

        self.all_data = pd.concat([self.data_train, self.data_test, self.data_valid])
        self.field_dims = []

        for i in self.all_data.columns[1:]:
            maps = {val: k for k, val in enumerate(set(self.all_data[i]))}
            self.data_test[i] = self.data_test[i].map(maps)
            self.data_train[i] = self.data_train[i].map(maps)
            self.data_valid[i] = self.data_valid[i].map(maps)
            self.features_M[i] = maps
            self.field_dims.append(len(set(self.all_data[i])))
        self.data_test[0] = self.data_test[0].apply(lambda x: max(x, 0))
        self.data_train[0] = self.data_train[0].apply(lambda x: max(x, 0))
        self.data_valid[0] = self.data_valid[0].apply(lambda x: max(x, 0))


class LoadDataMSE():
    def __init__(self, path="./Data/", dataset="frappe", loss_type="square_loss"):
        self.dataset = dataset
        self.loss_type = loss_type
        self.path = path + dataset + "/"
        self.trainfile = self.path + dataset + ".train.libfm"
        self.testfile = self.path + dataset + ".test.libfm"
        self.validationfile = self.path + dataset + ".valid.libfm"
        self.features_M = {}
        self.construct_df()

    #         self.Train_data, self.Validation_data, self.Test_data = self.construct_data( loss_type )

    def construct_df(self):
        self.data_train = pd.read_table(self.trainfile, sep=" ", header=None, engine='python')
        self.data_test = pd.read_table(self.testfile, sep=" ", header=None, engine="python")
        self.data_valid = pd.read_table(self.validationfile, sep=" ", header=None, engine="python")
        #       第一列是标签，y

        for i in self.data_test.columns[1:]:
            self.data_test[i] = self.data_test[i].apply(lambda x: int(x.split(":")[0]))
            self.data_train[i] = self.data_train[i].apply(lambda x: int(x.split(":")[0]))
            self.data_valid[i] = self.data_valid[i].apply(lambda x: int(x.split(":")[0]))

        self.all_data = pd.concat([self.data_train, self.data_test, self.data_valid])
        self.field_dims = []

        for i in self.all_data.columns[1:]:
            # if self.dataset != "frappe":
            # maps = {}
            maps = {val: k for k, val in enumerate(set(self.all_data[i]))}
            self.data_test[i] = self.data_test[i].map(maps)
            self.data_train[i] = self.data_train[i].map(maps)
            self.data_valid[i] = self.data_valid[i].map(maps)
            self.features_M[i] = maps
            self.field_dims.append(len(set(self.all_data[i])))
        # -1 改成 0
        # self.data_test[0] = self.data_test[0].apply(lambda x: max(x, 0))
        # self.data_train[0] = self.data_train[0].apply(lambda x: max(x, 0))
        # self.data_valid[0] = self.data_valid[0].apply(lambda x: max(x, 0))


class RecData():
    def __init__(self, all_data):
        self.data_df = all_data

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        x = self.data_df.iloc[idx].values[1:]
        y1 = self.data_df.iloc[idx].values[0]
        return x, y1


def getfrappe_loader811(path="../data/", dataset="frappe", num_ng=4, batch_size=256):
    print("start load frappe dataset")
    AllDataF = LoadData811(path=path, dataset=dataset)
    all_dataset = RecData(AllDataF.all_data)

    train_size = int(0.9 * len(all_dataset))
    test_size = len(all_dataset) - train_size
    # 8:1:1
    train_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size])
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size - test_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=4, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=4)
    print(len(train_loader))
    print(len(valid_loader))
    print(len(test_loader))
    return AllDataF.field_dims, train_loader, valid_loader, test_loader


def getdataloader_frappe(path="../data/", dataset="frappe", num_ng=4, batch_size=256):
    print(os.getcwd())
    print("start load frappe dataset")
    DataF = LoadData(path=path, dataset=dataset)
    # 7:2:1
    datatest = RecData(DataF.data_test)
    datatrain = RecData(DataF.data_train)
    datavalid = RecData(DataF.data_valid)
    print("datatest", len(datatest))
    print("datatrain", len(datatrain))
    print("datavalid", len(datavalid))
    trainLoader = torch.utils.data.DataLoader(datatrain, batch_size=batch_size, shuffle=True, num_workers=8,
                                              pin_memory=True, drop_last=True)
    validLoader = torch.utils.data.DataLoader(datavalid, batch_size=batch_size, shuffle=False, num_workers=4,
                                              pin_memory=True)
    testLoader = torch.utils.data.DataLoader(datatest, batch_size=batch_size, shuffle=False, num_workers=4,
                                             pin_memory=True)
    return DataF.field_dims, trainLoader, validLoader, testLoader


if __name__ == '__main__':

    field_dims, trainLoader, validLoader, testLoader = getfrappe_loader811(path="../", batch_size=256)
    for _ in tqdm.tqdm(trainLoader):
        pass
    it = iter(trainLoader)
    print(next(it)[0])
    print(field_dims)
