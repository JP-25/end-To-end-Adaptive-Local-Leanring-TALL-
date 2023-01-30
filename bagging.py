import collections
import os
import math
from time import time
from time import strftime
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from base.BaseRecommender import BaseRecommender
from dataloader import UIRTDatset
from dataloader.DataBatcher import DataBatcher
from utils import Tool
from utils import MP_Utility
from experiment import EarlyStop
from evaluation import Evaluator
from utils import Config, Logger, ResultTable, make_log_dir
import copy
from scipy.sparse.csr import csr_matrix


class MultVAE(nn.Module):
    def __init__(self, args, train_mat, device):
        super(MultVAE, self).__init__()
        self.args = args
        self.dataset = dataset
        self.num_users, self.num_items = train_mat.shape
        self.train_matrix = train_mat

        self.enc_dims = [self.num_items] + [args.hidden]
        self.dec_dims = self.enc_dims[::-1]
        self.dims = self.enc_dims + self.dec_dims[1:]

        self.batch_size = args.bs
        self.epoch = args.epoch
        self.lr = args.lr  # learning rate
        self.reg = args.reg  # regularization term trade-off
        self.dropout = args.dropout
        self.anneal_cap = args.anneal

        self.total_anneal_steps = 200000

        self.eps = 1e-6
        self.anneal = 0.
        self.update_count = 0

        self.device = device
        self.best_params = None
        self.es = EarlyStop(10, 'mean')

        # similarity_dir = os.path.join(dataset.data_dir, dataset.data_name, 'mainstream_scores')
        # similarity_file = os.path.join(similarity_dir, 'MS_similarity.npy')
        # self.ms = np.load(similarity_file)
        # weight_temp = self.ms / np.max(self.ms)
        # self.weight = (1 / weight_temp)

        self.build_graph()

    def build_graph(self):
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

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)

        # Send model to device (cpu or gpu)
        self.to(self.device)
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(param)

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

    def train_model(self, evaluator, early_stop):
        # exp_config = config['Experiment']
        #
        # num_epochs = exp_config['num_epochs']
        # print_step = exp_config['print_step']
        # test_step = exp_config['test_step']
        # test_from = exp_config['test_from']
        # verbose = exp_config['verbose']
        num_epochs = 10000
        verbose = 0
        print_step = 1
        test_step = 1
        test_from = 0

        # prepare dataset
        # dataset.set_eval_data('valid')
        users = np.arange(self.num_users)

        train_matrix = self.train_matrix.toarray()
        train_matrix = torch.FloatTensor(train_matrix)
        # best_result = 0.0
        # best_epoch = -1

        # for epoch
        start = time()
        for epoch in range(1, num_epochs + 1):
            # if epoch - best_epoch > 10:
            #     break
            self.train()

            epoch_loss = 0.0

            batch_loader = DataBatcher(users, batch_size=self.batch_size, drop_remain=False, shuffle=True)
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
                # weighted loss
                # batch_weight = torch.FloatTensor(self.weight[batch_idx]).to(self.device)
                # batch_loss = self.train_model_per_batch(batch_matrix, batch_weight)
                epoch_loss += batch_loss

                if verbose and (b + 1) % verbose == 0:
                    print('batch %d / %d loss = %.4f' % (b + 1, num_batches, batch_loss))
            epoch_train_time = time() - epoch_train_start

            epoch_info = ['epoch=%3d' % epoch, 'loss=%.3f' % epoch_loss, 'train time=%.2f' % epoch_train_time]

            # ======================== Evaluate
            if (epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs:
                self.eval()
                # evaluate model
                epoch_eval_start = time()

                test_score = evaluator.evaluate_vali(self)
                # test_score_str = ['%s=%.4f' % (k, test_score[k]) for k in test_score]
                # test_score_str = ['%s=[%s]' % (k, test_score[k]) for k in test_score]
                updated, should_stop = early_stop.step(test_score, epoch)

                test_score_output = evaluator.evaluate(self)
                test_score_str = ['%s=%.4f' % (k, test_score_output[k]) for k in test_score_output]
                _, _ = self.es.step(test_score_output, epoch)
                if should_stop:
                    print('Early stop triggered.')
                    break
                else:
                    # save best parameters
                    if updated:
                        # torch.save(self.state_dict(), os.path.join(log_dir, 'best_model.p'))

                        # save scores for all users
                        # rec = self.predict_all()
                        # ndcg_test_all = evaluator.evaluate(self, mean=False)
                        # similarity_dir = os.path.join(self.dataset.data_dir, self.dataset.data_name,
                        #                               'mainstream_scores')
                        # similarity_file = os.path.join(similarity_dir, 'MultVAE_scores')
                        # if not os.path.exists(similarity_file):
                        #     os.mkdir(similarity_file)
                        # with open(os.path.join(similarity_file, 'vae_scores.npy'), 'wb') as f:
                        #     np.save(f, ndcg_test_all)

                        if self.anneal_cap == 1: print(self.anneal)

                epoch_eval_time = time() - epoch_eval_start
                epoch_time = epoch_train_time + epoch_eval_time

                epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
                epoch_info += test_score_str
                # epoch_info += ['ndcg@20= ' + str(ndcg[3])]
            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

            if epoch % print_step == 0:
                print(epoch_info)
                # logger.info(', '.join(epoch_info))

        total_train_time = time() - start

        # return early_stop.best_score, total_train_time
        return self.es.best_score, total_train_time
        # return {'NDCG@20': ndcg[3]}, total_train_time

    def train_model_per_batch(self, batch_matrix, batch_weight=None):
        # zero grad
        self.optimizer.zero_grad()

        # model forwrad
        output, kl_loss = self.forward(batch_matrix)

        # loss
        # ce_loss = -(F.log_softmax(output, 1) * batch_matrix).mean()
        if batch_weight is None:
            ce_loss = -(F.log_softmax(output, 1) * batch_matrix).sum(1).mean()
        else:
            ce_loss = -((F.log_softmax(output, 1) * batch_matrix) * batch_weight.view(output.shape[0], -1)).sum(
                1).mean()
            # ce_loss = -((F.log_softmax(output, 1) * batch_matrix).sum(1) * batch_weight).mean()

        loss = ce_loss + kl_loss * self.anneal

        # backward
        loss.backward()

        # step
        self.optimizer.step()

        self.update_count += 1

        return loss

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
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

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--epoch', type=int, default=10000, help='number of epochs to train')
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

    # evaluator
    num_users, num_items = dataset.num_users, dataset.num_items

    # eval_mat = np.zeros((num_users, num_items)) # for calculate avg
    test_eval_pos, test_eval_target, vali_target, eval_neg_candidates = dataset.test_data()
    train_ratio_list = [0.7, 0.5, 0.3, 0]
    # train_ratio_list = [0.9] # for testing debugging
    local_num = 100

    for train_ratio in train_ratio_list:
        eval_mat = []
        weighted = None
        # log_dir = make_log_dir(os.path.join('bagging_results', 'avg'))
        # log_dir = make_log_dir(os.path.join('bagging_results', 'weighted'))
        log_dir = make_log_dir(os.path.join('bagging_results', 'weighted_softmax'))
        logger = Logger(log_dir)
        logger.info("training ratio is %s" % str(1 - train_ratio))
        for i in range(local_num):
            print("Local training " + str(i))
            print("start!!!")
            train_sample = test_eval_pos.toarray()
            train_size = int(train_ratio * len(train_sample))
            idx = np.random.choice(np.arange(len(train_sample)), train_size, replace=False).tolist()
            # train_sample.drop(idx, axis=0, inplace=True)
            train_sample_matrix = np.delete(train_sample, idx, axis=0)
            # train_sample.reset_index(drop=True, inplace=True)
            train_sample_matrix = csr_matrix(train_sample_matrix)
            test_dict = collections.defaultdict(list)
            vali_dict = collections.defaultdict(list)
            vali_user_idx = 0
            for user_id, item_ids in vali_target.items():
                if user_id in idx:
                    continue
                else:
                    vali_dict[vali_user_idx] = item_ids
                    vali_user_idx += 1
            test_id_idx = 0
            for user_id, item_ids in test_eval_target.items():
                if user_id in idx:
                    continue
                else:
                    test_dict[test_id_idx] = item_ids
                    test_id_idx += 1

            # test_evaluator = Evaluator(test_eval_pos, test_eval_target, eval_neg_candidates, **config['Evaluator'], num_users=num_users, num_items=num_items, item_id=dataset.item_id_dict)
            test_evaluator = Evaluator(train_sample_matrix, test_dict, vali_dict, eval_neg_candidates,
                                       **config['Evaluator'], num_users=None, num_items=None, item_id=None)
            evaluator = Evaluator(test_eval_pos, test_eval_target, vali_target, eval_neg_candidates, **config['Evaluator'],
                                  num_users=num_users, num_items=num_items, item_id=None)
            early_stop = EarlyStop(**config['EarlyStop'])
            model = MultVAE(args, train_sample_matrix, device)
            model.train_model(test_evaluator, early_stop)

            rec = model.predict(np.arange(len(train_sample)).tolist(), test_eval_pos)

            # eval_mat += rec
            # similar idea in mutiplication in MOE part
            eval_mat.append(np.reshape(rec, (-1, 1, num_items)))
            ndcg_all = evaluator.evaluate_all(rec, mean=False)
            ndcg20 = np.reshape(ndcg_all['NDCG@20'], (-1, 1))
            if i == 0:
                weighted = ndcg20
            else:
                weighted = np.append(weighted, ndcg20, axis=1)
            print("finsihed one local model training")

        # evaluation here
        eval_mat = np.concatenate(eval_mat, 1)
        # weighted_norm = weighted / np.sum(weighted, axis=1, keepdims=True)
        weighted_softmax = np.exp(weighted) / np.sum(np.exp(weighted), axis=1, keepdims=True)
        # output = np.matmul(np.reshape(weighted_norm, (-1, 1, local_num)), eval_mat)
        output = np.matmul(np.reshape(weighted_softmax, (-1, 1, local_num)), eval_mat)
        final_output = np.reshape(output, (-1, num_items))

        # final_output = np.mean(eval_mat, axis=1)  # average achieved here

        test_score = evaluator.evaluate_all(final_output)
        test_score_str = ['%s=%.4f' % (k, test_score[k]) for k in test_score]
        logger.info(test_score_str)
        # print(test_score_str)

        similarity_dir = os.path.join(dataset.data_dir, dataset.data_name, 'mainstream_scores')
        similarity_file = os.path.join(similarity_dir, strftime('%Y%m%d-%H%M') + '_bagging_scores_' + str(1 - train_ratio))
        # similarity_file = os.path.join(similarity_dir, strftime('%Y%m%d-%H%M') + '_bagging_scores_weighted_normalize' + str(1 - train_ratio))
        # similarity_file = os.path.join(similarity_dir, strftime('%Y%m%d-%H%M') + '_bagging_scores_weighted_softmax' + str(1 - train_ratio))
        if not os.path.exists(similarity_file):
            os.mkdir(similarity_file)
        ndcg_test = evaluator.evaluate_all(final_output, mean=False)
        with open(os.path.join(similarity_file, 'bagging_scores.npy'), 'wb') as f:
            np.save(f, ndcg_test)

        print("=" * 100)
