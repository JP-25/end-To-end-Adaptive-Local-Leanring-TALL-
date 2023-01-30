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
from multiprocessing import Process, Queue, Pool, Manager

import utils.Constant as CONSTANT
from dataloader import UIRTDatset
from evaluation import Evaluator

from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher
# from utils import Logger, set_random_seed
from sklearn.cluster import KMeans
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_distances
import warnings
from utils import Config, Logger, ResultTable, make_log_dir
import copy
from time import strftime

import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# np.random.seed(1)
# torch.random.manual_seed(1)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(1)

class LFT(nn.Module):
    def __init__(self, args, dataset, state_dict, device):
        super(LFT, self).__init__()
        self.num_user = dataset.num_users
        self.num_item = dataset.num_items

        self.t = args.t
        self.args = args
        self.anneal = args.anneal
        self.n_workers = args.n_workers
        self.local_bs = args.local_bs
        self.local_ep = args.local_ep
        self.lr = args.lr
        self.reg = args.reg

        self.train_mat = dataset.train_matrix.tocoo().toarray()

        self.device = device

        self.to(self.device)

        self.model = MultVAE(args, dataset, self.device)

        self.model.load_state_dict(state_dict)

        self.neighbor_users_list, self.neighbor_size_list = self.neighbor_users()

    def jaccard(self):
        num_rating_per_user = np.sum(self.train_mat, axis=1, keepdims=True)
        numerator = np.matmul(self.train_mat, self.train_mat.T)
        denominator = num_rating_per_user + num_rating_per_user.T - numerator
        denominator[denominator == 0] = 1
        Jaccard_mat = numerator / denominator
        Jaccard_mat *= (1 - np.eye(self.train_mat.shape[0]))
        return Jaccard_mat

    def cosine(self, x):
        numerator = np.matmul(x, x.T)
        denominator = np.sum(x ** 2, axis=1, keepdims=True) ** 0.5
        cosine_mat = numerator / denominator / denominator.T
        return cosine_mat

    def neighbor_users(self):
        Jaccard_mat = self.jaccard()
        distribution_mat = np.zeros_like(self.train_mat)
        for u in range(self.num_user):
            sim = Jaccard_mat[u]

            sim_threshold = self.args.sim_threshold
            sim_users = np.where(sim > sim_threshold)[0]
            k = 10
            if len(sim_users) < k:
                sim_users = np.argpartition(sim, -k)[-k:]

            alpha = self.args.alpha
            dist = alpha * self.train_mat[u, :] + (1 - alpha) * np.mean(self.train_mat[sim_users], axis=0, keepdims=True)
            distribution_mat[u, :] = dist

        cosine_mat = self.cosine(distribution_mat)
        neighbor_users_list = []
        neighbor_size_list = []
        s = 0
        for u in range(self.num_user):
            sim = copy.copy(cosine_mat[u])

            user_idx = np.where(sim > self.t)[0]

            k = 10
            if len(user_idx) < k:
                user_idx = np.argpartition(sim, -k)[-k:]
            k = 2000
            if len(user_idx) > k:
                user_idx = np.argpartition(sim, -k)[-k:]
            sim_users = user_idx

            s += len(sim_users)
            neighbor_users_list.append(sim_users)
            neighbor_size_list.append(len(sim_users))
        return neighbor_users_list, neighbor_size_list

    def local_update(self, user_id):

        local_train_mat = self.train_mat[self.neighbor_users_list[user_id], :]
        num_user = local_train_mat.shape[0]

        local_model = copy.deepcopy(self.model)
        local_model.to(self.device)
        local_model.train()

        optimizer = torch.optim.Adam(local_model.parameters(), lr=self.lr, weight_decay=self.reg)

        epoch_loss = []
        epochs = self.local_ep
        for iter in range(epochs):
            num_batch = int((num_user - 1) / self.local_bs) + 1
            random_idx = np.random.permutation(num_user)
            batch_loss = []
            for i in range(num_batch):
                batch_idx = random_idx[(i * self.local_bs):min(num_user, (i + 1) * self.local_bs)]
                batch_matrix = local_train_mat[batch_idx]
                batch_matrix = torch.FloatTensor(batch_matrix).to(self.device)
                optimizer.zero_grad()
                output, kl_loss = local_model.forward(batch_matrix)
                ce_loss = -(F.log_softmax(output, 1) * batch_matrix).sum(1).mean()
                loss = ce_loss + kl_loss * self.anneal
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(np.mean(batch_loss))

        local_model.eval()
        user_input = torch.FloatTensor(self.train_mat[user_id, :].reshape((1, -1))).to(self.device)
        user_Rec = local_model(user_input).detach().cpu().numpy()
        return user_Rec

    def local_test(self, user_list, user_Rec_queue, verbose=False):
        start_time = time()
        user_Rec = []
        if verbose:
            iterations = tqdm(user_list)
        else:
            iterations = user_list
        for uid in iterations:
            u_Rec = self.local_update(uid)
            user_Rec.append(u_Rec.reshape((1, -1)))
        user_Rec_queue.put((np.concatenate(user_Rec, axis=0), user_list))
        print('Evaluation process done, used %.2fs' % (time() - start_time))

    def run(self, evaluator):
        test_worker_size = int(self.num_user / self.n_workers + 0.5)
        # ======================== Evaluate ========================
        print('Meta-test')
        epoch_eval_start = time()
        Rec = np.zeros_like(self.train_mat)
        manager = Manager()
        user_Rec_queue = manager.Queue(maxsize=self.n_workers)
        processors = []

        for r in range(self.n_workers):
            p = Process(target=self.local_test,
                        args=(range(r * test_worker_size, min((r + 1) * test_worker_size, self.num_user)),
                              user_Rec_queue,
                              r == 0))
            processors.append(p)
            p.start()

        for p in processors:
            p.join()

        while not user_Rec_queue.empty():
            u_Rec, u_list = user_Rec_queue.get()
            Rec[u_list, :] = u_Rec
        test_score = evaluator.evaluate_all(Rec)
        test_score_str = ['%s=%.4f' % (k, test_score[k]) for k in test_score]
        print(test_score_str)

        similarity_dir = os.path.join(dataset.data_dir, dataset.data_name, 'mainstream_scores')
        similarity_file = os.path.join(similarity_dir, 'LFT_scores_wl')
        if not os.path.exists(similarity_file):
            os.mkdir(similarity_file)
        ndcg_all = evaluator.evaluate_all(Rec, mean=False)
        with open(os.path.join(similarity_file, strftime('%Y%m%d-%H%M') + '_LFT_scores_new_preprocess.npy'), 'wb') as f:
            np.save(f, ndcg_all)

        print("Save the best model")
        print("=" * 100)

        print("Testing time : %.2fs" % (time() - epoch_eval_start))

class MultVAE(nn.Module):
    def __init__(self, args, dataset, device):
        super(MultVAE, self).__init__()
        self.num_items = dataset.num_items

        self.enc_dims = [self.num_items] + [args.hidden]
        self.dec_dims = self.enc_dims[::-1]
        self.dims = self.enc_dims + self.dec_dims[1:]

        self.lr = args.lr  # learning rate
        self.reg = args.reg  # regularization term trade-off
        self.dropout = args.dropout

        self.anneal = args.anneal

        self.device = device

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
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)

        # Send model to device (cpu or gpu)
        # self.to(self.device)

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


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='LFT')
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--reg', type=float, default=0.01, help='regularization')
    parser.add_argument('--dropout', type=float, default=0, help='dropout')
    parser.add_argument('--anneal', type=float, default=0.2, help='anneal')
    parser.add_argument('--hidden', type=int, default=100, help='latent dimension')
    parser.add_argument('--t', type=float, default=0.2, help='t')
    parser.add_argument('--local_ep', type=int, default=30, help='training epochs during testing')
    parser.add_argument('--n_workers', type=int, default=5, help='n_workers')
    parser.add_argument('--alpha', type=float, default=0.7, help='alpha')
    parser.add_argument('--sim_threshold', type=float, default=0.1, help='sim_threshold')

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

    state_dict = torch.load(os.path.join('saves', 'WL', '7_20221230-0116', 'best_model.p'), map_location=device)

    # evaluator
    num_users, num_items = dataset.num_users, dataset.num_items
    test_eval_pos, test_eval_target, vali_target, eval_neg_candidates = dataset.test_data()
    # test_evaluator = Evaluator(test_eval_pos, test_eval_target, eval_neg_candidates, **config['Evaluator'],
    #                            num_users=num_users, num_items=num_items, item_id=dataset.item_id_dict)
    test_evaluator = Evaluator(test_eval_pos, test_eval_target, vali_target, eval_neg_candidates, **config['Evaluator'],
                               num_users=num_users, num_items=num_items, item_id=None)

    model = LFT(args, dataset, state_dict, device)
    model.run(test_evaluator)