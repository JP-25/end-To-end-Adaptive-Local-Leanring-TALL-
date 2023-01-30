from time import time
import numpy as np
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
from scipy.sparse import csr_matrix
from multiprocessing import Process, Queue, Pool, Manager

from scipy.sparse import coo_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import mode
from experiment import EarlyStop, train_model
import sys
import pickle
from collections import OrderedDict

import torch.multiprocessing as multiprocessing

import utils.Constant as CONSTANT
from dataloader import UIRTDatset
from evaluation import Evaluator
from utils import Config, Logger, ResultTable, make_log_dir
import copy

import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

class AutoEncoder(nn.Module):
    def __init__(self, args, dataset, device):
        super(AutoEncoder, self).__init__()

        self.num_item = dataset.num_items
        self.num_user = dataset.num_users

        self.train_matrix = dataset.train_matrix

        self.enc_dims = [self.num_item, args.hidden]
        self.dec_dims = self.enc_dims[::-1]
        self.dims = self.enc_dims + self.dec_dims[1:]

        self.batch_size = args.bs
        self.epoch = args.epoch
        self.lr = args.lr  # learning rate
        self.reg = args.reg  # regularization term trade-off
        self.dropout = args.dropout
        self.anneal = args.anneal

        self.display = args.display

        self.device = device

        self.build_graph()

    def build_graph(self):
        # self.encoder = nn.ModuleList()
        # for i, (d_in, d_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:])):
        #     if i == len(self.enc_dims[:-1]) - 1:
        #         d_out *= 2
        #     self.encoder.append(nn.Linear(d_in, d_out))
        #     if i != len(self.enc_dims[:-1]) - 1:
        #         self.encoder.append(nn.Tanh())
        #
        # self.decoder = nn.ModuleList()
        # for i, (d_in, d_out) in enumerate(zip(self.dec_dims[:-1], self.dec_dims[1:])):
        #     self.decoder.append(nn.Linear(d_in, d_out))
        #     if i != len(self.dec_dims[:-1]) - 1:
        #         self.decoder.append(nn.Tanh())
        self.encoder = nn.ModuleList()
        e_in = self.enc_dims[0]
        e_out = self.enc_dims[1]
        self.encoder.append(nn.Linear(e_in, e_out))
        # one linear layer finished, the output will contain some negative numbers.
        self.encoder.append(nn.Softmax()) # average
        # self.encoder.append(nn.Sigmoid()) # average
        # self.encoder.append(nn.ReLU())
        # self.encoder.append(nn.Tanh())

        self.decoder = nn.ModuleList()
        d_in = self.dec_dims[0]
        d_out = self.dec_dims[1]
        self.decoder.append(nn.Linear(d_in, d_out))

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)

        # Send model to device (cpu or gpu)
        self.to(self.device)

    def forward(self, x):
        # encoder
        h = x
        for layer in self.encoder:
            h = layer(h)

        output = h
        hidden = h
        for layer in self.decoder:
            output = layer(output)

        return hidden, output

    def train_model(self):

        best_result = float('inf')
        best_epoch = -1
        ae_dir = os.path.join('gate_AE')
        if not os.path.exists(ae_dir):
            os.mkdir(ae_dir)

        train_matrix = self.train_matrix.toarray()
        train_matrix = torch.FloatTensor(train_matrix).to(self.device)

        for epoch in range(1, self.epoch + 1):
            if epoch - best_epoch > 10:
                break

            self.train()

            # ======================== Train ========================
            epoch_loss = 0.0
            # num_batch = int(self.num_user / self.batch_size) + 1
            # random_idx = np.random.permutation(self.num_user)
            epoch_train_start = time()
            # for i in tqdm(range(num_batch)):
            #     batch_idx = random_idx[(i * self.batch_size):min(self.num_user, (i + 1) * self.batch_size)]
            #     batch_matrix = self.train_matrix[batch_idx].toarray()
            #     batch_matrix = torch.FloatTensor(batch_matrix).to(self.device)
            #
            #     batch_loss, hidden = self.train_model_per_batch(batch_matrix)
            #     epoch_loss += batch_loss
            epoch_loss, hidden = self.train_model_helper(train_matrix)

            epoch_train_time = time() - epoch_train_start
            print("Training //", "Epoch %d //" % epoch, " Total loss = {:.5f}".format(epoch_loss),
                  " Total training time = {:.2f}s".format(epoch_train_time))
            # ======================== Evaluate ========================
            if epoch % self.display == 0:
                epoch_eval_start = time()

                if epoch_loss < best_result:
                    best_epoch = epoch
                    best_result = epoch_loss

                    with open(os.path.join(ae_dir, 'AE_gate.npy'), "wb") as f:
                        np.save(f, hidden.detach().cpu().numpy())

                    print("Save the best AE")
                    print("-" * 100)

                print("Testing time : %.2fs" % (time() - epoch_eval_start))

    def train_model_helper(self, matrix):
        # zero grad
        self.optimizer.zero_grad()

        # model forwrad
        hidden, output = self.forward(matrix)

        # loss
        # mse_loss = nn.MSELoss()
        loss = (((output - matrix) ** 2) * (matrix * 10 + 1 - matrix)).sum()


        # backward
        loss.backward()

        # step
        self.optimizer.step()
        return loss, hidden


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--display', type=int, default=1, help='evaluate mode every X epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--reg', type=float, default=0.01, help='regularization')
    parser.add_argument('--dropout', type=float, default=0, help='dropout')
    parser.add_argument('--anneal', type=float, default=0.2, help='anneal')
    parser.add_argument('--hidden', type=int, default=100, help='latent dimension')
    parser.add_argument('--bs', type=int, default=512, help='batch size')


    args = parser.parse_args()
    print(args)

    gpu = 0
    gpu = str(gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "data/recsys_data"
    dataset = "ml-1m"
    min_user_per_item = 1
    min_item_per_user = 10

    # leave-k-out
    leave_k = 5
    popularity_order = True

    dataset = UIRTDatset(data_dir, dataset, min_user_per_item, min_item_per_user, leave_k, popularity_order)

    # read configs
    config = Config(main_conf_path='./', model_conf_path='model_config')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_users, num_items = dataset.num_users, dataset.num_items

    test_eval_pos, test_eval_target, vali_target, eval_neg_candidates = dataset.test_data()

    print('!' * 100)

    model = AutoEncoder(args, dataset, device)
    model.train_model()
