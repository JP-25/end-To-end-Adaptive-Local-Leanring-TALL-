import os
import math
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import linalg as LA

from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher
from utils import Tool
from utils import MP_Utility

import copy
from past.builtins import range

import pickle
import argparse
import pandas as pd
from scipy.sparse import csr_matrix, rand as sprand
from tqdm import tqdm

from evaluation.backend import HoldoutEvaluator

# np.random.seed(0)
# torch.manual_seed(0)


class MF(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(MF, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.display_step = model_conf['display_step']
        self.hidden_neuron = model_conf['hidden_neuron']
        # self.lambda_value = model_conf['lambda_value']
        # self.neg_once = model_conf['neg_once']
        self.neg_sample_rate = model_conf['neg_sample_rate']
        # self.train_epoch = model_conf['train_epoch']

        self.batch_size = model_conf['batch_size']
        # self.test_batch_size = model_conf['test_batch_size']
        self.regularization = model_conf['reg']
        self.lr = model_conf['lr']
        self.train_df = dataset.train_df
        # self.train_like = dataset.train_like
        # self.test_like = dataset.test_like
        self.user_list, self.item_list, self.label_list = MP_Utility.negative_sampling(self.num_users, self.num_items,
                                                                                    self.train_df[0],
                                                                                    self.train_df[1],
                                                                                    self.neg_sample_rate)
        print('******************** MF ********************')
        self.user_factors = torch.nn.Embedding(self.num_users, self.hidden_neuron)  # , sparse=True
        self.item_factors = torch.nn.Embedding(self.num_items, self.hidden_neuron)  # , sparse=True
        print('P: ', self.user_factors)
        print('Q: ', self.item_factors)
        self.regularization_term = self.regularization * (LA.norm(self.user_factors.weight.data, 'fro').item() +
                                                          LA.norm(self.item_factors.weight.data, 'fro').item())

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)

        print('********************* MF Initialization Done *********************')

    def forward(self, user, item):
        # Get the dot product per row
        u = self.user_factors(user)
        v = self.item_factors(item)
        x = (u * v).sum(1)
        return x



    def train_model(self, dataset, evaluator, early_stop, logger, config):
        exp_config = config['Experiment']
        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        log_dir = logger.log_dir
        # users = np.arange(dataset.num_user_mf)

        start = time()
        for epoch_itr in range(1, num_epochs + 1):
            self.train_model_help(epoch_itr)
            if (epoch_itr >= test_from and epoch_itr % test_step == 0) or epoch_itr == num_epochs:
                self.eval()
                # evaluate model
                # epoch_eval_start = time()

                test_score = evaluator.evaluate(self)
                test_score_str = ['%s=%.4f' % (k, test_score[k]) for k in test_score]

                print(test_score_str)
                updated, should_stop = early_stop.step(test_score, epoch_itr)

                if should_stop:
                    logger.info('Early stop triggered.')
                    break
                else:
                    # save best parameters
                    if updated:
                        torch.save(self.state_dict(), os.path.join(log_dir, 'best_model.p'))
                        # if self.anneal_cap == 1: print(self.anneal)

            total_train_time = time() - start
        #     if epoch_itr % self.display_step == 0:
        #         score_cumulator, cur_metric = self.test_model(epoch_itr)
        #         if cur_metric > best_metric:
        #             best_metric = cur_metric
        #             best_itr = epoch_itr
        #             # self.make_records(epoch_itr)
        #         elif epoch_itr - best_itr >= 30:
        #             break
        # total_train_time = time() - start
        # for metric in score_cumulator:
        #     score_by_ks = score_cumulator[metric]
        #     for k in score_by_ks:
        #         # if mean:
        #         #     scores['%s@%d' % (metric, k)] = score_by_ks[k].mean
        #         # else:
        #         scores['%s@%d' % (metric, k)] = score_by_ks[k].mean

        return early_stop.best_score, total_train_time

    def train_model_help(self, itr):
        epoch_cost = 0.
        epoch_cost1 = 0.
        epoch_cost2 = 0.
        self.user_list, self.item_list, self.label_list = MP_Utility.negative_sampling(self.num_users, self.num_items,
                                                                                    self.train_df[0],
                                                                                    self.train_df[1],
                                                                                    self.neg_sample_rate)
        start_time = time() * 1000.0
        num_batch = int(len(self.user_list) / float(self.batch_size)) + 1
        random_idx = np.random.permutation(len(self.user_list))

        for i in tqdm(range(num_batch)):
            #             print(i)
            batch_idx = None
            if i == num_batch - 1:
                batch_idx = random_idx[i * self.batch_size:]
            elif i < num_batch - 1:
                batch_idx = random_idx[(i * self.batch_size):((i + 1) * self.batch_size)]

            tmp_cost, tmp_cost1, tmp_cost2 = self.train_batch(self.user_list[batch_idx], self.item_list[batch_idx],
                                                              self.label_list[batch_idx])

            epoch_cost += tmp_cost
            epoch_cost1 += tmp_cost1
            epoch_cost2 += tmp_cost2

        print("Training //", "Epoch %d //" % itr, " Total cost = {:.5f}".format(epoch_cost),
              " Total cost1 = {:.5f}".format(epoch_cost1), " Total cost2 = {:.5f}".format(epoch_cost2),
              "Training time : %d ms" % (time() * 1000.0 - start_time))

    def train_batch(self, user_input, item_input, label_input):
        users = torch.Tensor(user_input).int()
        items = torch.Tensor(item_input).int()
        labels = torch.Tensor(label_input).float()
        total_loss = 0
        total_loss1 = 0
        total_loss2 = 0

        self.train()

        y_hat = self.forward(users, items)
        loss = F.mse_loss(y_hat, labels)
        self.regularization_term = self.regularization * (LA.norm(self.user_factors.weight.data, 'fro').item() +
                                                          LA.norm(self.item_factors.weight.data, 'fro').item())

        added_loss = loss.item() + self.regularization_term

        total_loss += added_loss
        total_loss1 += loss.item()
        total_loss2 += self.regularization_term

        # reset gradients
        self.optimizer.zero_grad()
        # backpropagate
        loss.backward()
        # update
        self.optimizer.step()

        self.eval()

        return (total_loss, total_loss1, total_loss2)

    # def test_model(self, itr):  # calculate the cost and rmse of testing set in each epoch
    #     start_time = time() * 1000.0
    #     P, Q = self.user_factors.weight, self.item_factors.weight
    #     P = P.detach().numpy()
    #     Q = Q.detach().numpy()
    #     Rec = np.matmul(P, Q.T)
    #
    #     score, precision, recall, ndcg = MF_Utility.MP_test_model_all(Rec, self.test_like, self.train_like, n_workers=10)
    #
    #     # print("Testing //", "Epoch %d //" % itr,
    #     #       "Accuracy Testing time : %d ms" % (time() * 1000.0 - start_time))
    #     # print("=" * 100)
    #     return score, recall[3]

    def get_rec(self):
        P, Q = self.user_factors.weight, self.item_factors.weight
        P = P.detach().numpy()
        Q = Q.detach().numpy()
        Rec = np.matmul(P, Q.T)
        return Rec

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        self.eval()
        batch_eval_pos = eval_pos_matrix[user_ids]
        with torch.no_grad():
            # eval_input = torch.Tensor(batch_eval_pos.toarray()).to(self.device)
            # P = torch.Tensor(self.user_list[user_ids]).int()
            # Q = torch.Tensor(self.item_list[user_ids]).int()
            Rec = self.get_rec()
            eval_output = Rec[user_ids, :]
            if eval_items is not None:
                eval_output[np.logical_not(eval_items)] = float('-inf')
            else:
                eval_output[batch_eval_pos.nonzero()] = float('-inf')
        self.train()
        return eval_output