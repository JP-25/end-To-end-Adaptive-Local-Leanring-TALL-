import os
import math
import copy
import pickle
from time import time

import numpy as np
from numpy.random import randint, binomial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from concurrent import futures
from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher
from utils import Logger, set_random_seed
from sklearn.cluster import KMeans
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_distances
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

class MOE(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(MOE, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        # config
        self.units = model_conf['num_units']
        self.num_experts = model_conf['num_experts']
        self.num_tasks = 1
        self.num_features = self.num_items
        self.use_expert_bias = True
        self.use_gate_bias = True
        # no selen this time for recsys
        self.seqlen = None
        # self.expert = 'MultVAE_Expert'
        self.n_layers = 1
        self.batch_size = model_conf['batch_size']
        self.test_batch_size = model_conf['test_batch_size']

        self.lr = model_conf['lr']
        self.dropout = model_conf['dropout']
        self.reg = model_conf['reg']
        self.anneal = 0.
        self.anneal_cap = model_conf['anneal_cap']
        self.total_anneal_steps = model_conf['total_anneal_steps']

        print("... Model - MMOE")
        self.update_count = 0

        # self.runits = model_conf['runits']
        self.model_config = model_conf
        self.device = device
        """ Initialize gate weights (number of input features * number of experts * number of tasks)"""
        if self.seqlen is None:
            gate_kernels = torch.rand(
                (self.num_tasks, self.num_features, self.num_experts)
            ).float()
        else:
            gate_kernels = torch.rand(
                (self.num_tasks, self.seqlen * self.num_features, self.num_experts)
            ).float()

        self.expert_kernels = nn.ModuleList(
            [
                MultVAE_Expert(
                    self.num_items, self.model_config
                )
                for i in range(self.num_experts)
            ]
        )

        # self.expert_output = nn.ModuleList(
        #     [
        #         nn.Linear(self.num_features, 1).float()
        #         for i in range(self.num_experts)
        #     ]
        # )

        """Initialize expert bias (number of units per expert * number of experts)# Bias parameter"""
        if self.use_expert_bias:
            if self.seqlen is None:
                self.expert_bias = nn.Parameter(
                    torch.zeros(self.num_experts), requires_grad=True
                )
            else:
                self.expert_bias = nn.Parameter(
                    torch.zeros(self.num_experts, self.seqlen), requires_grad=True
                )

        """ Initialize gate bias (number of experts * number of tasks)"""
        if self.use_gate_bias:
            self.gate_bias = nn.Parameter(
                torch.zeros(self.num_tasks, 1, self.num_experts), requires_grad=True
            )

        self.gate_kernels = nn.Parameter(gate_kernels, requires_grad=True)
        self.task_bias = nn.Parameter(torch.zeros(self.num_tasks), requires_grad=True)
        self.dropout = nn.Dropout(0.25)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # self.kl_loss_list = []

    def forward(self, x):
        self.to(self.device)
        n = x.shape[0]
        expert_output_list = []
        kl_loss_list = []
        for i in range(self.num_experts):
            # here !!!
            if self.training:
                expert_kernels, kl_loss = self.expert_kernels[i](x)
                kl_loss_list.append(kl_loss)
            else:
                expert_kernels = self.expert_kernels[i](x)
            # expert_kernels, kl_loss = self.expert_kernels[i](x)
            # kl_loss_list.append(kl_loss)
            # print("kl loss: ", kl_loss)
            # expert_kernels = self.expert_kernels[i]
            # change self.expert_kernels[i] to expert_kernels
            # aux = torch.mm(x, expert_kernels)
            aux = expert_kernels
            aux = torch.reshape(aux, (n, expert_kernels.shape[1]))
            # print("expert_kernel shape: ", expert_kernels.shape)
            # aux = torch.reshape(aux, (expert_kernels.shape[1], n)) ###
            # print("L126: ", aux.size()) # n = 512, 512*3706, 3706 * 512
            # temp = torch.reshape(aux, [-1, 1, expert_kernels.shape[1]])
            # print("temp:", temp.shape)
            expert_output_list.append(torch.reshape(aux, [-1, 1, expert_kernels.shape[1]]))  # [b, 1, h]
            # here!!!
            # if i == 0:
            #     expert_outputs = self.expert_output[i](aux)
            # else:
            #     expert_outputs = torch.cat(
            #         (expert_outputs, self.expert_output[i](aux)), dim=1
            #     )
            # print("L126: ", expert_outputs.size())
        expert_outputs = torch.cat(expert_output_list, 1)  # size: batch_size X expert X self.dimension
        # print("experts output size: ", expert_outputs.shape)
        # user expert bias
        for i in range(self.num_experts):
            expert_outputs[i] = expert_outputs[i].add(self.expert_bias[i])
        expert_outputs = F.relu(expert_outputs)

        """ Calculating the gates"""
        # g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the paper T x n x E
        for index in range(self.num_tasks):
            if index == 0:
                gate_outputs = torch.mm(x, self.gate_kernels[index]).reshape(
                    1, n, self.num_experts
                )

            else:
                gate_outputs = torch.cat(
                    (
                        gate_outputs,
                        torch.mm(x, self.gate_kernels[index]).reshape(
                            1, n, self.num_experts
                        ),
                    ),
                    dim=0,
                )
        # user gate bias
        gate_outputs = gate_outputs.add(self.gate_bias)
        # one layer softmax function based on num of tasks, features, and experts
        gate_outputs = F.softmax(gate_outputs, dim=2)

        """ Multiplying gates and experts"""
        for task in range(self.num_tasks):
            gate = gate_outputs[task]
            # here need to change as well, 512*3706 shape, here!!!
            # gate shape and expert_outputs shape should be the same
            # print("L1: ", gate.shape)
            # print("L2: ", expert_outputs.shape)
            # temp2 = torch.reshape(gate, [-1, 1, self.num_experts])
            # print("L3: ", temp2.shape)
            res = torch.matmul(torch.reshape(gate, [-1, 1, self.num_experts]), expert_outputs)
            # print("res shape: ", res.shape)
            embeddings = torch.reshape(res, [-1, expert_kernels.shape[1]])
            # print("embed: ", embeddings.shape)

            # print("L3: ", torch.mul(gate, expert_outputs)) # give me 512 * 12
            # final_outputs_t = torch.mul(gate, expert_outputs).reshape(
            #     1, gate.shape[0], gate.shape[1]
            # )
            # final_outputs_t = final_outputs_t.add(self.task_bias[task])
            final_outputs_t = embeddings.add(self.task_bias[task])
            if task == 0:
                final_outputs = final_outputs_t
            else:
                # not useful yet
                final_outputs = torch.cat((final_outputs, final_outputs_t), dim=0)

            if self.training:
                mean_kl_loss = torch.mean(torch.stack(kl_loss_list))  ### change later maybe
                return final_outputs, mean_kl_loss
            else:
                return final_outputs
        # return final_outputs, mean_kl_loss

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        # here should train each expert
        exp_config = config['Experiment']

        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        log_dir = logger.log_dir
        train_matrix = dataset.train_matrix.toarray()
        train_matrix = torch.FloatTensor(train_matrix)
        users = np.arange(self.num_users)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)
        # loss_ = []
        # task_losses = np.zeros([self.num_tasks], dtype=np.float32)
        # self.criterion = [nn.BCEWithLogitsLoss().to(self.device) for i in range(self.num_tasks)]
        # self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        # for epoch
        start = time()
        for epoch in range(1, num_epochs + 1):
            self.train()

            # self.optimizer.zero_grad()
            epoch_loss = 0.0
            batch_loader = DataBatcher(users, batch_size=self.batch_size, drop_remain=False, shuffle=False)
            num_batches = len(batch_loader)
            # ======================== Train
            epoch_train_start = time()
            for b, batch_idx in enumerate(batch_loader):
                batch_matrix = train_matrix[batch_idx].to(self.device)
                if self.total_anneal_steps > 0:
                    self.anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
                else:
                    self.anneal = self.anneal_cap
                batch_loss = self.train_model_per_batch(batch_matrix)
                epoch_loss += batch_loss
                if verbose and (b + 1) % verbose == 0:
                    print('batch %d / %d loss = %.4f' % (b + 1, num_batches, batch_loss))
                # if use model(x) should be working, right now still using the multvae code
                # for task in range(self.num_tasks):
                #     label = batch_idx[1][:, task].long().to(self.device).reshape(-1, 1)
                #     epoch_loss += criterion[task](
                #         batch_matrix[task], label.float()
                #     )
            # epoch_loss.backward()
            """ Saving losses per epoch"""
            # loss_.append(epoch_loss.cpu().detach().numpy())


            epoch_train_time = time() - epoch_train_start

            epoch_info = ['epoch=%3d' % epoch, 'loss=%.3f' % epoch_loss, 'train time=%.2f' % epoch_train_time]

            # ======================== Evaluate
            if (epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs:
                self.eval()
                # evaluate model
                epoch_eval_start = time()

                test_score = evaluator.evaluate(self)
                test_score_str = ['%s=%.4f' % (k, test_score[k]) for k in test_score]

                updated, should_stop = early_stop.step(test_score, epoch)

                if should_stop:
                    logger.info('Early stop triggered.')
                    break
                else:
                    # save best parameters
                    if updated:
                        torch.save(self.state_dict(), os.path.join(log_dir, 'best_model.p'))
                        if self.anneal_cap == 1:
                            print(self.anneal)

                epoch_eval_time = time() - epoch_eval_start
                epoch_time = epoch_train_time + epoch_eval_time

                epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
                epoch_info += test_score_str
            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

            if epoch % print_step == 0:
                logger.info(', '.join(epoch_info))

        total_train_time = time() - start

        return early_stop.best_score, total_train_time

    def train_model_per_batch(self, batch_matrix, batch_weight=None):
        # zero grad
        # criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer.zero_grad()

        # model forwrad
        output, kl_loss = self.forward(batch_matrix) ###
        # output = self.forward(batch_matrix)
        # BCEWithLogitsLoss, actually just ce loss
        # ce_loss = -(F.log_softmax(output, 1) * batch_matrix).mean()
        if batch_weight is None:
            ce_loss = -(F.log_softmax(output, 1) * batch_matrix).sum(1).mean()
        else:
            ce_loss = -((F.log_softmax(output, 1) * batch_matrix) * batch_weight.view(output.shape[0], -1)).sum(
                1).mean()
        loss = ce_loss + kl_loss * self.anneal

        # dimension probably, can implement the equation above, reshape(-1, 1) to make a matrix into one single column
        # loss = criterion(
        #     output.reshape(-1, 1), batch_matrix.reshape(-1, 1).float()
        # )

        # backward
        loss.backward()
        # loss.backward(retain_graph=True)

        # step
        self.optimizer.step()

        self.update_count += 1

        return loss

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        # simply multiple here
        self.eval()
        batch_eval_pos = eval_pos_matrix[user_ids]
        with torch.no_grad():
            eval_input = torch.Tensor(batch_eval_pos.toarray()).to(self.device)
            eval_output = self.forward(eval_input).detach().cpu().numpy()

            if eval_items is not None:
                eval_output[np.logical_not(eval_items)] = float('-inf')
            else:
                eval_output[batch_eval_pos.nonzero()] = float('-inf')
        self.train()
        return eval_output


class MultVAE_Expert(nn.Module):
    def __init__(self, num_items, model_conf):
        super(MultVAE_Expert, self).__init__()
        # self.dataset = dataset
        # self.num_users = dataset.num_users
        # self.num_items = dataset.num_items
        self.num_items = num_items
        self.enc_dims = [self.num_items] + model_conf['enc_dims']
        self.dec_dims = self.enc_dims[::-1]
        self.dims = self.enc_dims + self.dec_dims[1:]

        # self.total_anneal_steps = model_conf['total_anneal_steps']
        # self.anneal_cap = model_conf['anneal_cap']

        self.dropout = model_conf['dropout']
        # self.reg = model_conf['reg']

        # self.batch_size = model_conf['batch_size']
        # self.test_batch_size = model_conf['test_batch_size']

        # self.lr = model_conf['lr']
        # self.eps = 1e-6
        # self.anneal = 0.
        # self.update_count = 0

        # self.device = device
        self.best_params = None

        # self.build_graph()
        self.encoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:])):
            if i == len(self.enc_dims[:-1]) - 1:
                d_out *= 2
            self.encoder.append(nn.Linear(d_in, d_out))
            if i != len(self.enc_dims[:-1]) - 1:
                self.encoder.append(nn.Tanh())

        self.decoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.dec_dims[:-1], self.dec_dims[1:])):
            self.decoder.append(nn.Linear(d_in, d_out))
            if i != len(self.dec_dims[:-1]) - 1:
                self.decoder.append(nn.Tanh())

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
        # return output