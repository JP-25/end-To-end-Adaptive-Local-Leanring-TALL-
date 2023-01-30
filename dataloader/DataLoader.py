import math
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp

def load_data_and_info(data_file, info_file):
    with open(data_file, 'rb') as f:
        data_dict = pickle.load(f)

    # with open(info_file, 'rb') as f:
    #     info_dict = pickle.load(f)
        
    # user_id_dict = info_dict['user_id_dict']
    # user_to_num_items = info_dict['user_to_num_items']
    # item_id_dict = info_dict['item_id_dict']
    # item_to_num_users = info_dict['item_to_num_users']

    num_users = data_dict['num_users']
    num_items = data_dict['num_items']

    train_sp_matrix = data_dict['train']
    ###
    test_sp_matrix = data_dict['test']
    vali_sp_matrix = data_dict['vali']


    # csr matrix to numpy
    train_df = train_sp_matrix.toarray()
    # train_df_two_columns = np.where(train_sp_matrix.toarray() == 1)
    # train_matrix = csr_matrix((np.ones(len(train_df)), (user_array, item_array)),
    #                                shape=(self.num_user, self.num_item))

    # print(train_sp_matrix.toarray())

    return train_sp_matrix, test_sp_matrix, vali_sp_matrix, num_users, num_items, train_df
