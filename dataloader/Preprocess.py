import os
import math
from scipy.sparse.csr import csr_matrix
from tqdm import tqdm
import pickle
from datetime import datetime
import json
from scipy.io import loadmat

import numpy as np
import scipy.sparse as sp
import json
import pandas as pd
import copy
from collections import defaultdict

def read_raw_random(datapath, file_prefix, separator, prefix):
    # here is training and testing split preprocess
    if datapath[-10:] == "rating.mat":
        rating = loadmat(datapath)
        rating = rating['rating']
        rating_df = pd.DataFrame({'userId': rating[:, 0], 'itemId': rating[:, 1], 'rating': rating[:, 3]})
        rating_df.drop(columns=['rating'], inplace=True)
    else:
        rating_df = pd.read_csv(datapath, sep=separator, names=["userId", "itemId", "rating", "timestamp"], engine='python')
        rating_df.drop(columns=['timestamp'], inplace=True)
        rating_df.drop(columns=['rating'], inplace=True)
        print(len(rating_df))
    rating_df.drop_duplicates(subset=['itemId', 'userId'], keep='first', inplace=True)
    item_set = set(rating_df['itemId'].unique())
    user_set = set(rating_df['userId'].unique())
    rating_df.reset_index(drop=True, inplace=True)
    rdf_backup = copy.copy(rating_df)
    rdf = copy.copy(rdf_backup)

    print("initial: ", len(user_set), len(item_set))

    rdf['user_freq'] = rdf.groupby('userId')['userId'].transform('count')
    rdf['item_freq'] = rdf.groupby('itemId')['itemId'].transform('count')

    # ml-1m (9,9,9), yelp(8,8,19),  epinion(11, 11, 11), kindle_store(16, 16, 23), CDs(9, 9, 15)
    while np.min(rdf['user_freq']) <= 9:
        # iteratively remove items and users with less than 10 reviews (ml-1m),
        rdf.drop(rdf.index[rdf['user_freq'] <= 9], inplace=True)
        rdf.reset_index(drop=True, inplace=True)

        rdf['item_freq'] = rdf.groupby('itemId')['itemId'].transform('count')
        rdf.drop(rdf.index[rdf['item_freq'] <= 9], inplace=True)
        rdf.reset_index(drop=True, inplace=True)

        rdf['user_freq'] = rdf.groupby('userId')['userId'].transform('count')
        rdf.reset_index(drop=True, inplace=True)

    item_list = rdf['itemId'].unique()
    user_list = rdf['userId'].unique()
    # print("here")
    print(len(user_list), len(item_list))
    print('sparsity: ' + str(len(rdf) * 1.0 / (len(user_list) * len(item_list))))

    # get the user and item str id->int id dict
    i = 0
    user_old2new_id_dict = dict()
    for u in user_list:
        if not u in user_old2new_id_dict:
            user_old2new_id_dict[u] = i
            i += 1
    j = 0
    item_old2new_id_dict = dict()
    for i in item_list:
        if not i in item_old2new_id_dict:
            item_old2new_id_dict[i] = j
            j += 1

    for i in range(len(rdf)):
        rdf.at[i, 'userId'] = user_old2new_id_dict[rdf.at[i, 'userId']]
        rdf.at[i, 'itemId'] = item_old2new_id_dict[rdf.at[i, 'itemId']]
    item_list = rdf['itemId'].unique()
    user_list = rdf['userId'].unique()
    # get the df of train, vali, and test df
    rdf.reset_index(inplace=True, drop=True)
    train_df = rdf.copy()

    train_ratio = 0.7
    vali_ratio = 0.1
    test_ratio = 0.2

    vali_size = int(vali_ratio * len(rdf))
    test_size = int(test_ratio * len(rdf))

    vali_idx = np.random.choice(np.arange(len(train_df)),
                                vali_size,
                                replace=False).tolist()
    vali_df = train_df.copy()
    vali_df = vali_df.loc[vali_idx]
    train_df.drop(vali_idx, axis=0, inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    test_idx = np.random.choice(np.arange(len(train_df)),
                                test_size,
                                replace=False).tolist()
    test_df = train_df.copy()
    test_df = test_df.loc[test_idx]
    train_df.drop(test_idx, axis=0, inplace=True)

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    vali_df.reset_index(drop=True, inplace=True)
    train_df.head()
    train_df.drop(columns=['user_freq', 'item_freq'], inplace=True)
    test_df.drop(columns=['user_freq', 'item_freq'], inplace=True)
    vali_df.drop(columns=['user_freq', 'item_freq'], inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    vali_df.reset_index(drop=True, inplace=True)
    num_items = len(item_list)
    num_users = len(user_list)
    user_train_like = []
    user_test_like = []
    user_vali_like = []

    train_array = train_df[['userId', 'itemId']].values
    vali_array = vali_df[['userId', 'itemId']].values
    test_array = test_df[['userId', 'itemId']].values

    for u in user_list:
        train_like = (train_array[list(np.where(train_array[:, 0] == u)[0]), 1]).astype(int)
        vali_like = (vali_array[list(np.where(vali_array[:, 0] == u)[0]), 1]).astype(int)
        test_like = (test_array[list(np.where(test_array[:, 0] == u)[0]), 1]).astype(int)
        if len(vali_like) == 0:
            new_vali_idx = np.random.choice(np.arange(len(train_like)), size=1)
            new_vali = train_like[new_vali_idx]
            vali_like = np.array([new_vali])
            train_like = np.delete(train_like, new_vali_idx)
            train_array = np.delete(train_array,
                                    np.where((train_array[:, 0] == u) & (train_array[:, 1] == new_vali))[0], axis=0)
            vali_array = np.append(vali_array, [[u, new_vali]], axis=0)
        if len(test_like) == 0:
            new_test_idx = np.random.choice(np.arange(len(train_like)), size=1)
            new_test = train_like[new_test_idx]
            test_like = np.array([new_test])
            train_like = np.delete(train_like, new_test_idx)
            train_array = np.delete(train_array,
                                    np.where((train_array[:, 0] == u) & (train_array[:, 1] == new_test))[0], axis=0)
            test_array = np.append(test_array, [[u, new_test]], axis=0)

        user_train_like.append(train_like)
        user_vali_like.append(vali_like)
        user_test_like.append(test_like)

    # np.save(prefix + '/user_train_like.npy', np.array(user_train_like))
    # # np.save(prefix + './user_vali_like.npy', np.array(user_vali_like))
    # np.save(prefix + './user_test_like.npy', np.array(user_test_like))

    # train_like = np.array(user_train_like)
    # vali_like = np.array(user_vali_like)
    # test_like = np.array(user_test_like)

    user_array = train_array[:, 0]
    item_array = train_array[:, 1]
    train_sp_matrix = csr_matrix((np.ones(len(train_array)), (user_array, item_array)), shape=(num_users, num_items))
    test_dict = {u: [] for u in range(num_users)}
    vali_dict = {u: [] for u in range(num_users)}
    for user, item in test_array:
        if isinstance(item, int):
            test_dict[user].append(item)
        else:
            test_dict[user].extend(item)

    for user, item in vali_array:
        if isinstance(item, int):
            vali_dict[user].append(item)
        else:
            vali_dict[user].extend(item)


    # save
    # data_to_save = {
    #     'train': train_sp_matrix,
    #     'train_like': train_like,
    #     'test_like': test_like,
    #     'vali_like': vali_like,
    #     'num_users': num_users,
    #     'num_items': num_items
    # }
    data_to_save = {
        'train': train_sp_matrix,
        'test': test_dict,
        'vali': vali_dict,
        'num_users': num_users,
        'num_items': num_items
    }

    info_lines = []
    info_lines.append('# users: %d, # items: %d' % (num_users, num_items))

    info_lines.append("Sparsity : %.2f%%" % (len(rdf) * 100. / (len(user_list) * len(item_list))))

    with open(file_prefix + '.info', 'wt') as f:
        f.write('\n'.join(info_lines))

    with open(file_prefix + '.data', 'wb') as f:
        pickle.dump(data_to_save, f)

    print('Preprocess 10% validation, 20% testing, 70% training')


def read_raw_UIRT(datapath, separator, order_by_popularity=True):
    """
    read raw data (ex. ml-100k.rating)

    return U2IRT, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users, num_users, num_items, num_ratings

    """

    print("Loading the dataset from \"%s\"" % datapath)
    with open(datapath, "r") as f:
        lines = f.readlines()

    num_users, num_items = 0, 0
    user_to_num_items, item_to_num_users = {}, {}
    user_id_dict, item_id_dict = {}, {}

    for line in lines:
        user_id, item_id, _, _ = line.strip().split(separator)
        try:
            user_id = int(user_id)
            item_id = int(item_id)
        except:
            continue

        if user_id not in user_id_dict:
            user_id_dict[user_id] = num_users
            new_user_id = user_id_dict[user_id]

            user_to_num_items[new_user_id] = 1
            num_users += 1
        else:
            new_user_id = user_id_dict[user_id]
            user_to_num_items[new_user_id] += 1

        # Update the number of ratings per item
        if item_id not in item_id_dict:
            item_id_dict[item_id] = num_items

            new_item_id = item_id_dict[item_id]
            item_to_num_users[new_item_id] = 1
            num_items += 1
        else:
            new_item_id = item_id_dict[item_id]
            item_to_num_users[new_item_id] += 1

    if order_by_popularity:
        user_id_dict, user_to_num_items = order_id_by_popularity(user_id_dict, user_to_num_items)
        item_id_dict, item_to_num_users = order_id_by_popularity(item_id_dict, item_to_num_users)

    # BUILD U2IRTs
    U2IRT = {u: [] for u in user_to_num_items}
    for line in lines:
        user_id, item_id, rating, time = line.strip().split(separator)
        try:
            user_id = int(user_id)
            item_id = int(item_id)
            rating = float(rating)
            time = int(time)
        except:
            continue
        U2IRT[user_id_dict[user_id]].append((item_id_dict[item_id], rating, time))

    return U2IRT, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users


def order_id_by_popularity(object_id_dict, object_to_num):
    old_to_pop_dict = {}
    new_to_pop_dict = {}
    new_object_to_num = {}
    object_id_dict_sorted = sorted(object_to_num.items(), key=lambda x: x[-1], reverse=True)
    for pop, new_pop_tuple in enumerate(object_id_dict_sorted):
        new = new_pop_tuple[0]
        new_to_pop_dict[new] = pop
        new_object_to_num[pop] = object_to_num[new]
    for old, new in object_id_dict.items():
        old_to_pop_dict[old] = new_to_pop_dict[new]

    return old_to_pop_dict, new_object_to_num


def filter_min_item_cnt(U2IRT, min_item_cnt, user_id_dict):
    modifier = 0
    user_id_dict_sorted = sorted(user_id_dict.items(), key=lambda x: x[-1])
    for old_user_id, _ in user_id_dict_sorted:
        new_user_id = user_id_dict[old_user_id]
        IRTs = U2IRT[new_user_id]
        num_items = len(IRTs)

        if num_items < min_item_cnt:
            U2IRT.pop(new_user_id)
            user_id_dict.pop(old_user_id)
            modifier += 1
        elif modifier > 0:
            U2IRT[new_user_id - modifier] = IRTs
            user_id_dict[old_user_id] = new_user_id - modifier
    return U2IRT, user_id_dict


def preprocess(raw_file, file_prefix, prefix, leave_k, min_item_per_user=0, min_user_per_item=0, separator=',', order_by_popularity=True):
    """
    read raw data and preprocess

    """
    # # read raw data
    # U2IRT, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users = read_raw_UIRT(raw_file, separator, order_by_popularity)
    #
    # # filter out (min item cnt)
    # if min_item_per_user > 0:
    #     U2IRT, user_id_dict = filter_min_item_cnt(U2IRT, min_item_per_user, user_id_dict)
    #
    # # preprocess
    # # preprocess_lko(U2IRT, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users, file_prefix, leave_k)

    # here instead of leave k, use traditional methods.
    read_raw_random(raw_file, file_prefix, separator, prefix)

def preprocess_lko(U2IRT, user_id_dict, user_to_num_items, item_id_dict, item_to_num_users,  file_prefix, leave_k):
    num_users = len(user_id_dict)
    num_items = len(item_id_dict)
    num_ratings = 0

    train_data = np.zeros((num_users, num_items))
    test_dict = {u: [] for u in range(num_users)}


    for user in tqdm(U2IRT):
        # Sort the UIRTs by the ascending order of the timestamp.
        IRTs = sorted(U2IRT[user], key=lambda x: x[-1])
        num_ratings += len(IRTs)

        # test data
        for i in range(leave_k):
            IRT = IRTs.pop()
            test_dict[user].append(IRT[0])

        # train data
        for IRT in IRTs:
            train_data[user, IRT[0]] = 1

    train_sp_matrix = csr_matrix(train_data, shape=(num_users, num_items))

    # save
    data_to_save = {
        'train': train_sp_matrix,
        'test': test_dict,
        'num_users': num_users,
        'num_items': num_items
    }

    info_to_save = {
        'user_id_dict': user_id_dict,
        'user_to_num_items': user_to_num_items,
        'item_id_dict': item_id_dict,
        'item_to_num_users': item_to_num_users
    }

    ratings_per_user = [len(U2IRT[u]) for u in U2IRT]

    info_lines = []
    info_lines.append('# users: %d, # items: %d, # ratings: %d' % (num_users, num_items, num_ratings))
    info_lines.append("Sparsity : %.2f%%" % ((1 - (num_ratings / (num_users * num_items))) * 100))
    info_lines.append("Min/Max/Avg. ratings per users (full data): %d %d %.2f" % (
        min(ratings_per_user), max(ratings_per_user), np.mean(ratings_per_user)))

    with open(file_prefix + '.stat', 'wt') as f:
        f.write('\n'.join(info_lines))

    with open(file_prefix + '.data', 'wb') as f:
        pickle.dump(data_to_save, f)

    with open(file_prefix + '.info', 'wb') as f:
        pickle.dump(info_to_save, f)

    print('Preprocess Leave-%d-out finished.' % (leave_k))
