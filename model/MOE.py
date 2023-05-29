import collections
import os
import math
import pickle
from time import time
from time import strftime
from scipy.sparse.csr import csr_matrix

import numpy as np
# from numpy.random import randint, binomial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from concurrent import futures
from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher
# from utils import Logger, set_random_seed
from sklearn.cluster import KMeans
from collections import OrderedDict
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_distances
import warnings
from tqdm import tqdm
import copy
import torch.distributed as dist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import KernelPCA
from experiment import EarlyStop
import matplotlib.pyplot as plt
# from softadapt import SoftAdapt

warnings.filterwarnings("ignore")
# PCA sklearn
class MOE(BaseRecommender):
    # add entropy added to the loss (ce), smaller is better
    def __init__(self, dataset, model_conf, device):
        super(MOE, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        # config
        self.num_experts = model_conf['num_experts']
        self.num_features = self.num_items
        self.batch_size = model_conf['batch_size']
        self.test_batch_size = model_conf['test_batch_size']

        self.lr = model_conf['lr']
        self.dropout = model_conf['dropout']
        ### change weight decay
        self.reg = model_conf['reg'] / self.num_experts
        print("weight decay:", self.reg)
        self.anneal = 0.
        self.anneal_cap = model_conf['anneal_cap']
        self.total_anneal_steps = model_conf['total_anneal_steps']

        print("... Model - MOE")
        self.num_local_threads = model_conf['num_local_threads']
        self.update_count = 0
        self.device = device
        """ Initialize gate weights (number of input features * number of experts * number of tasks)"""

        # # feed forward neural network
        # self.gate_kernels = gate_network(self.num_items, self.num_experts)

        # load validation data
        self.vali_data = dataset.vali_dict
        self.vali_ = np.zeros((self.num_users, self.num_items))
        for u in self.vali_data:
            self.vali_[u][self.vali_data[u]] = 1

        self.expert_kernels = nn.ModuleList()

        for i in range(self.num_experts):
            expert = MultVAE_Expert(dataset, model_conf, device)
            self.expert_kernels.append(expert)

        """ Initialize gate bias (number of experts * number of tasks)"""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)
        self.to(self.device) ###

        self.time = strftime('%Y%m%d-%H%M')
        self.vali_batch = None

    def forward(self, x, batch_idx):
        n = x.shape[0]
        expert_output_list = []
        kl_loss_list = []
        dimension = 0
        loss = None
        # loss = []

        self.vali_batch = torch.FloatTensor(self.vali_).to(self.device)[batch_idx] ###
        mask = 1 - x

        for i in range(self.num_experts):
            if self.training:
                expert_kernel, kl_loss = self.expert_kernels[i](x)
                kl_loss_list.append(kl_loss)
            else:
                expert_kernel = self.expert_kernels[i](x)
            # weighted here
            vali_loss = -torch.log(torch.exp(expert_kernel) / torch.exp(expert_kernel * mask).sum(1).reshape(-1, 1))

            expert_vali_loss = 1 / (vali_loss * self.vali_batch).sum(1).reshape(-1, 1)

            if i == 0:
                loss = expert_vali_loss
            else:
                loss = torch.cat((loss, expert_vali_loss), 1)

            aux = expert_kernel
            dimension = expert_kernel.shape[1]
            aux = torch.reshape(aux, (n, dimension))
            expert_output_list.append(torch.reshape(aux, [-1, 1, dimension]))  # [b, 1, dimension]
        expert_outputs = torch.cat(expert_output_list, 1)  # size: batch_size X expert X self.dimension

        """ Multiplying gates and experts"""
        loss = F.softmax(loss, dim=1)
        res = torch.matmul(torch.reshape(loss, [-1, 1, self.num_experts]), expert_outputs)
        final_outputs = torch.reshape(res, [-1, dimension])


        if self.training:
            weighted_kl_loss = torch.matmul(loss, torch.stack(kl_loss_list)).sum()
            return final_outputs, weighted_kl_loss
        else:
            return final_outputs

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        # here should train each expert
        exp_config = config['Experiment']

        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        log_dir = logger.log_dir
        users = np.arange(self.num_users)
        similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name, 'mainstream_scores')
        if not os.path.exists(similarity_dir):
            os.mkdir(similarity_dir)

        similarity_file = os.path.join(similarity_dir, 'moe_folder_f')
        if not os.path.exists(similarity_file):
            os.mkdir(similarity_file)
        # path_ = os.path.join(similarity_file, 'MOE_scores_' + strftime('%Y%m%d-%H%M') + '_tau_' + str(self.tau))
        path_ = os.path.join(similarity_file, 'MOE_scores_' + strftime('%Y%m%d-%H%M'))
        if not os.path.exists(path_):
            os.mkdir(path_)

        train_matrix = dataset.train_matrix.toarray()
        train_matrix = torch.FloatTensor(train_matrix).to(self.device) ###
        best_result = None

        # for epoch
        start = time()

        # # global epoch weight
        self.e_loss = torch.ones((self.num_users, 1))
        bw = None

        prev_loss = None
        loss_change = None

        for epoch in range(1, num_epochs + 1):
            self.train()

            epoch_loss = 0.0
            ce_loss_sum = 0.0
            kl_loss_sum = 0.0
            batch_loader = DataBatcher(users, batch_size=self.batch_size, drop_remain=False, shuffle=True)
            num_batches = len(batch_loader)
            # ======================== Train
            epoch_train_start = time()

            # run 10, 20, 30, 40 epochs first
            if epoch <= 40:
                bw = torch.ones((self.num_users, 1))
            else:
                loss_change_avg = loss_change.mean(1).reshape(-1, 1)
                alpha = 0.08
                beta = torch.ones(loss_change_avg.shape) * (1e-8)
                bw = torch.max(((loss_change_avg - loss_change_avg.mean() + 2 * alpha) / (2 * alpha)).reshape(-1, 1), beta)

            for b, batch_idx in enumerate(batch_loader):
                batch_matrix = train_matrix[batch_idx]

                if self.total_anneal_steps > 0:
                    self.anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
                else:
                    self.anneal = self.anneal_cap

                # epoch weight
                batch_weight = bw[batch_idx].to(self.device) ###
                batch_loss, ce, kl = self.train_model_per_batch(batch_matrix, batch_idx, batch_weight)

                epoch_loss += batch_loss
                ce_loss_sum += ce
                kl_loss_sum += kl

                if verbose and (b + 1) % verbose == 0:
                    print('batch %d / %d loss = %.4f' % (b + 1, num_batches, batch_loss))
            epoch_train_time = time() - epoch_train_start

            # calculate loss change, or 4 epoch (3 changes), 3 epoch (2 changes), less or equal
            if epoch <= 9:
                if epoch == 1:
                    pass
                elif epoch == 2:
                    loss_change = prev_loss - self.e_loss
                else:
                    loss_change = torch.cat((loss_change, prev_loss - self.e_loss), 1)
            else:
                loss_change = torch.cat((loss_change[:, 1:], prev_loss - self.e_loss), 1)

            prev_loss = copy.deepcopy(self.e_loss)

            epoch_info = ['epoch=%3d' % epoch, 'epoch loss=%.3f' % epoch_loss, 'ce loss=%.3f' % ce_loss_sum, 'kl loss=%.3f' % kl_loss_sum, 'train time=%.2f' % epoch_train_time]

            # ======================== Evaluate
            if (epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs:
                self.eval()
                # evaluate model
                epoch_eval_start = time()

                # # normal
                test_score = evaluator.evaluate_vali(self)

                updated, should_stop = early_stop.step(test_score, epoch)
                test_score_output = evaluator.evaluate(self)
                test_score_str = ['%s=%.4f' % (k, test_score_output[k]) for k in test_score_output]

                if should_stop:
                    logger.info('Early stop triggered.')
                    break
                else:
                    # save best parameters
                    if updated:
                        best_result = test_score_output
                        ndcg_test_all = evaluator.evaluate(self, mean=False) ##

                        with open(os.path.join(path_, str(self.num_experts) + '_moe_test_scores.npy'), 'wb') as f:
                            np.save(f, ndcg_test_all)

                        if self.anneal_cap == 1: print(self.anneal)

                epoch_eval_time = time() - epoch_eval_start
                epoch_time = epoch_train_time + epoch_eval_time

                epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
                epoch_info += test_score_str
            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

            if epoch % print_step == 0:
                logger.info(', '.join(epoch_info))

        total_train_time = time() - start
        return best_result, total_train_time

    def train_model_per_batch(self, batch_matrix, batch_idx, batch_weight=None):
        # zero grad
        self.optimizer.zero_grad()
        mask = 1 - batch_matrix

        output, kl_loss = self.forward(batch_matrix, batch_idx)

        vali_loss = -torch.log(torch.exp(output) / torch.exp(output * mask).sum(1).reshape(-1, 1))
        expert_vali_loss = (vali_loss * self.vali_batch).sum(1)

        self.e_loss[batch_idx] = expert_vali_loss.reshape(-1, 1).detach().cpu() ### update batch weights

        if batch_weight is None:
            ce_loss = -(F.log_softmax(output, 1) * batch_matrix).sum(1).mean()
        else:
            ce_loss = -((F.log_softmax(output, 1) * batch_matrix) * batch_weight.view(output.shape[0], -1)).sum(1).sum() / batch_weight.sum()
        loss = ce_loss + kl_loss * self.anneal

        # backward
        loss.backward()

        # step
        self.optimizer.step()

        self.update_count += 1

        # return loss, ce_loss, kl_loss, entropy_norm
        return loss, ce_loss, kl_loss

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        # simply multiple here
        self.eval()
        batch_eval_pos = eval_pos_matrix[user_ids]
        with torch.no_grad():
            eval_input = torch.Tensor(batch_eval_pos.toarray()).to(self.device) ###
            # eval_input = torch.Tensor(batch_eval_pos.toarray()).cuda()
            eval_output = self.forward(eval_input, user_ids).detach().cpu().numpy()

            if eval_items is not None:
                eval_output[np.logical_not(eval_items)] = float('-inf')
            else:
                eval_output[batch_eval_pos.nonzero()] = float('-inf')
        self.train()
        return eval_output

# normal MoE gate
class gate_network(nn.Module):
    def __init__(self, num_items, num_experts):
        super(gate_network, self).__init__()
        self.num_items = num_items
        self.num_experts = num_experts

        ## two layers
        hidden_size = 2000
        self.fc1 = nn.Linear(self.num_items, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.num_experts)

        # self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # two layers
        hidden = self.fc1(x)
        # hidden = self.dropout(hidden)
        relu = self.relu(hidden)
        output = self.fc2(relu)

        # return
        output = self.softmax(output)
        return output


class MultVAE_Expert(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(MultVAE_Expert, self).__init__(dataset, model_conf)
        self.num_items = dataset.num_items

        self.enc_dims = [self.num_items] + model_conf['enc_dims']
        self.dec_dims = self.enc_dims[::-1]
        self.dims = self.enc_dims + self.dec_dims[1:]

        self.total_anneal_steps = model_conf['total_anneal_steps']
        self.anneal_cap = model_conf['anneal_cap']

        self.dropout = model_conf['dropout']
        self.reg = model_conf['reg']

        self.lr = model_conf['lr']

        self.device = device

        self.build_graph()

    def build_graph(self):
        self.encoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:])):
            if i == len(self.enc_dims[:-1]) - 1:
                d_out *= 2
            # linear_layer = nn.Linear(d_in, d_out)
            # nn.init.uniform_(linear_layer.weight, -0.1, 0.1)
            # self.encoder.append(linear_layer)
            self.encoder.append(nn.Linear(d_in, d_out))
            if i != len(self.enc_dims[:-1]) - 1:
                self.encoder.append(nn.Tanh())

        self.decoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.dec_dims[:-1], self.dec_dims[1:])):
            # linear_layer = nn.Linear(d_in, d_out)
            # nn.init.uniform_(linear_layer.weight, -0.1, 0.1)
            # self.decoder.append(linear_layer)
            self.decoder.append(nn.Linear(d_in, d_out))
            if i != len(self.dec_dims[:-1]) - 1:
                self.decoder.append(nn.Tanh())

        # optimizer
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)

        # Send model to device (cpu or gpu)
        self.to(self.device)

    def forward(self, x):
        # encoder
        h = F.dropout(F.normalize(x), p=self.dropout, training=self.training)
        for layer in self.encoder:
            h = layer(h)

        # sample
        mu_q = h[:, :self.enc_dims[-1]]
        logvar_q = h[:, self.enc_dims[-1]:]  # log sigmod^2  batch x 200
        std_q = torch.exp(0.5 * logvar_q)  # sigmod batch x 200

        # F.kl_div()

        epsilon = torch.zeros_like(std_q).normal_(mean=0, std=0.01)
        sampled_z = mu_q + self.training * epsilon * std_q

        output = sampled_z
        for layer in self.decoder:
            output = layer(output)

        if self.training:
            kl_loss = ((0.5 * (-logvar_q + torch.exp(logvar_q) + torch.pow(mu_q, 2) - 1)).sum(1)).mean()
            return output, kl_loss
        else:
            return output

    # def predict(self, user_ids, eval_pos_matrix, eval_items=None):
    #     self.eval()
    #     batch_eval_pos = eval_pos_matrix[user_ids]
    #     with torch.no_grad():
    #         eval_input = torcsh.Tensor(batch_eval_pos.toarray()).to(self.device)
    #         eval_output = self.forward(eval_input).detach().cpu().numpy()
    #
    #         if eval_items is not None:
    #             eval_output[np.logical_not(eval_items)] = float('-inf')
    #         else:
    #             eval_output[batch_eval_pos.nonzero()] = float('-inf')
    #     self.train()
    #     return eval_output
