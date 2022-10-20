import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import multiprocessing
from datetime import datetime
from datetime import timedelta
from causalml.dataset import make_uplift_classification
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from causalml.inference.tree import uplift_tree_string, uplift_tree_plot
from causalml.inference.meta import LRSRegressor
from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor
import pickle
import numpy as np
from tqdm import tqdm

import os
import warnings

warnings.filterwarnings('ignore')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # for lightgbm to work

from odps import ODPS
from public_input import fg_json
from odps.udf import annotate


@annotate("string->bigint")
class ipint(object):
    def evaluate(self, access_key, access_passwd, project_name, project_url,
                 table_name, fg_json, label_name, start_day, end_day):
        odps = ODPS(access_key, access_passwd, project_name, endpoint=project_url)
        # project = odps.get_project()

        if odps.exist_table(table_name):
            # 获得表的字段/特征名称 变量类型 具体值
            json_table = fg_json
            propoty_c = []
            propoty_type_c = []
            propoty_type_accpedted_c = []
            for idx, feature_c in enumerate(json_table['features']):
                propoty_c.append(feature_c['feature_name'])
                propoty_type_c.append(feature_c['value_type'])
                if feature_c['value_type'] == 'Double':
                    propoty_type_accpedted_c.append(idx)

            table_c = odps.get_table(table_name)
            partition_c = [i.name for i in table_c.schema.partitions]
            partition_order = partition_c[0] + '=' + start_day

            # 使用哪些特征作为输入变量
            # 只使用Double
            propoty_c_use0 = [propoty_c[idx_cc] for idx_cc in propoty_type_accpedted_c]
            propoty_c_use = ['features'] + [label_name]
            input_variable = []
            output_variable = []
            for idx, record in tqdm(
                    enumerate(
                        odps.read_table(name=table_name, partition=partition_order, columns=propoty_c_use, start=0,
                                        limit=100000))):
                value_c = record.values
                input_variable_c = value_c[0].split('\02')
                input_variable.append([input_variable_c[idx_c] for idx_c in propoty_type_accpedted_c])
                output_variable.append(value_c[-1])

            input_variable = np.array(input_variable)
            # input_variable1 = np.array(input_variable)[:,propoty_type_accpedted_c]
            output_variable = np.array(output_variable)

            input_variable[input_variable == np.array(None)] = 0
            input_variable = np.array(input_variable, dtype=np.float)

            # 循环计算每个变量的因果影响
            ATE_list = []
            for i in tqdm(range(input_variable.shape[1])):
                treatment_c = input_variable[:, i]
                input_variable_new = np.delete(input_variable, i, axis=1)

                # propoty_c_use_now = np.delete(np.array(propoty_c_use[:-1]), i, axis=0)

                thresheld_c = np.median(treatment_c)
                treatment_c0 = copy.deepcopy(treatment_c)
                treatment_c0 = np.array(treatment_c0, dtype=np.str)
                treatment_c0[np.where(treatment_c <= thresheld_c)[0]] = 'control'
                treatment_c0[np.where(treatment_c > thresheld_c)[0]] = 'treatment'
                if len(set(treatment_c0.tolist())) <= 1:
                    re = 0
                else:
                    # # Train uplift tree
                    # lr = LRSRegressor()
                    # re, lb, ub = lr.estimate_ate(input_variable_new, treatment_c0, output_variable)
                    # re = np.abs(re)
                    tlearner = BaseTRegressor(LGBMRegressor(), control_name='control')
                    re, lb, ub = tlearner.estimate_ate(input_variable_new, treatment_c0, output_variable)
                    re = np.abs(re)

                ATE_list.append(re)
                print(ATE_list[-1])
            ATE_list = np.array(ATE_list)
            idx = np.argsort(ATE_list)[::-1].tolist()

            max_rank = 20
            idx_part = [i for i in idx[:max_rank]]

            rank_list = [propoty_c_use0[j] for j in idx_part]
            rank_score = ATE_list[idx_part].tolist()

            rank_dict = dict(zip(rank_list, rank_score))
            print(rank_dict)

        return rank_dict


def cal_treatment(tree):
    global num_leaf
    trueBranch = tree.trueBranch
    falseBranch = tree.falseBranch
    if not trueBranch:
        num_leaf += 1
        return tree.upliftScore[0]
    return cal_treatment(trueBranch) + cal_treatment(falseBranch)


def get_ATE(input):
    access_key, access_passwd, project_name, project_url, table_name, fg_json, \
    label_name, current_day, current_number, batchsize = \
        input[0], input[1], input[2], input[3], input[4], input[5], \
        input[6], input[7], input[8], input[9]

    global end_day

    odps = ODPS(access_key, access_passwd, project_name, endpoint=project_url)

    if odps.exist_table(table_name):
        # 获得表的字段/特征名称 变量类型 具体值
        json_table = fg_json
        propoty_c = []
        propoty_type_c = []
        propoty_type_accpedted_c = []
        for idx, feature_c in enumerate(json_table['features']):
            propoty_c.append(feature_c['feature_name'])
            propoty_type_c.append(feature_c['value_type'])
            if feature_c['value_type'] == 'Double' or feature_c['value_type'] == 'Integer':
                propoty_type_accpedted_c.append(idx)

        table_c = odps.get_table(table_name)
        partition_c = [i.name for i in table_c.schema.partitions]

        # 使用哪些特征作为输入变量
        # 只使用Double
        propoty_c_use0 = [propoty_c[idx_cc] for idx_cc in propoty_type_accpedted_c]
        propoty_c_use = ['features'] + [label_name]
        input_variable = []
        output_variable = []

        partition_order = partition_c[0] + '=' + current_day.strftime('%Y%m%d')
        for idx, record in tqdm(
                enumerate(odps.read_table
                              (name=table_name, partition=partition_order, columns=propoty_c_use,
                               start=current_number, limit=batchsize))):
            value_c = record.values
            input_variable_c = value_c[0].split('\02')
            input_variable.append([input_variable_c[idx_c] for idx_c in propoty_type_accpedted_c])
            output_variable.append(value_c[-1])
        current_number += idx + 1
        # 更新current_number & 跨日期补全
        while len(input_variable) < batchsize and current_day < end_day:
            batchsize_c = batchsize - len(input_variable)

            current_day = current_day + timedelta(days=1)
            partition_order = partition_c[0] + '=' + current_day.strftime('%Y%m%d')
            current_number = 0

            for idx, record in tqdm(
                    enumerate(odps.read_table
                                  (name=table_name, partition=partition_order, columns=propoty_c_use,
                                   start=current_number, limit=batchsize_c))):
                value_c = record.values
                input_variable_c = value_c[0].split('\02')
                input_variable.append([input_variable_c[idx_c] for idx_c in propoty_type_accpedted_c])
                output_variable.append(value_c[-1])

            current_number += idx + 1

        # 如果到达最后一天，需要截断
        if len(input_variable) < batchsize:
            current_day = current_day + timedelta(days=1)

        if len(input_variable) == 0:
            return {}, current_day, current_number

        input_variable = np.array(input_variable)
        output_variable = np.array(output_variable)

        input_variable[input_variable == np.array(None)] = 0
        input_variable = np.array(input_variable, dtype=np.float)

        # 循环计算每个变量的因果影响
        ATE_list = []
        for i in tqdm(range(input_variable.shape[1])):
            treatment_c = input_variable[:, i]
            input_variable_new = np.delete(input_variable, i, axis=1)

            thresheld_c = np.median(treatment_c)
            treatment_c0 = copy.deepcopy(treatment_c)
            treatment_c0 = np.array(treatment_c0, dtype=np.str)
            treatment_c0[np.where(treatment_c <= thresheld_c)[0]] = 'control'
            treatment_c0[np.where(treatment_c > thresheld_c)[0]] = 'treatment'
            if len(set(treatment_c0.tolist())) <= 1:
                re = 0
            else:
                # 采用LightGBM决策树：效率高 精度高 内存小
                try:
                    tlearner = BaseTRegressor(LGBMRegressor(), control_name='control')
                    re, lb, ub = tlearner.estimate_ate(input_variable_new, treatment_c0, output_variable)
                    re = np.abs(re)
                    # 调用PAI的一个决策树的接口，相当于大数据的计算放在PAI上
                except:
                    re = 0

            ATE_list.append(re)
            # print(ATE_list[-1])
        ATE_list = np.array(ATE_list)
        idx = np.argsort(ATE_list)[::-1].tolist()

        # max_rank = len(idx)
        # idx_part = [i for i in idx[:max_rank]]

        rank_list = [propoty_c_use0[j] for j in idx]
        rank_score = ATE_list[idx].tolist()

        rank_dict = dict(zip(rank_list, rank_score))

    return rank_dict, current_day, current_number

"--------------------input-----------------"
# 以下为阿里云授权账号的access key及密码（由于合同中离职保密条例都删掉了，用xxx代替）
fuliao_accesskey_id = 'xxx'
fuliao_accesskey_secret = 'xxx'
# 以下为datawalks空间的项目名称的endpoint地址
fuliao_project = 'pai_test_xjp'
fuliao_endpoint = 'xxx'
# 以下为datawalks空间中mc表的地址
table_name = 'xxx'
# 以下为datawalks空间中mc表中目标变量的列名称
label_feature_name = 'is_click'
# 以下为datawalks空间中mc表中使用数据的时间分区，一般是一个月，最后一天作为测试数据，之前的作为训练数据
start_day, end_day = '20220901', '20220929'
# 以下为一次处理的数据量
batchsize = 300000
# 以下为多线程的个数，根据服务器的CPU核心数决定，一般4核对应一个
num_pool = 3
"--------------------input-----------------"

start_day = datetime.strptime(start_day, '%Y%m%d')
end_day = datetime.strptime(end_day, '%Y%m%d')
assert end_day >= start_day

access_key, access_passwd, project_name, project_url, \
table_name, label_name, start_day, end_day \
    = fuliao_accesskey_id, fuliao_accesskey_secret, fuliao_project, fuliao_endpoint, \
      table_name, label_feature_name, start_day, end_day

# mini-batch遍历分区，考虑跨日期补全
# batchsize = 300000
current_day = start_day
current_number = 0
current_number1 = 0
re_rank_dict_all = {}

while current_day <= end_day:
    # num_pool = 3
    pool = multiprocessing.Pool(num_pool)
    pool_input = []
    current_number1 = copy.deepcopy(current_number)
    for _ in range(num_pool):
        print(current_day, current_number)
        pool_input.append([access_key, access_passwd, project_name, project_url,
                           table_name, fg_json, label_name, current_day, current_number1, batchsize])
        current_number1 = current_number1 + batchsize
    batch_result = pool.map(get_ATE, pool_input)

    for re in batch_result:
        re_rank_dict, current_day, current_number = re[0], re[1], re[2]
        print(current_day, current_number)

        for key_c, value_c in re_rank_dict.items():
            if key_c not in re_rank_dict_all:
                re_rank_dict_all[key_c] = []
            re_rank_dict_all[key_c].append(value_c)

outpath = './feature_dict.pkl'
with open(outpath, 'wb') as f:
    pickle.dump(re_rank_dict_all, f, pickle.HIGHEST_PROTOCOL)
print('save success result')

# re_rank_dict_all_mean = copy.deepcopy(re_rank_dict_all)
# feature_list = []
# score_list = []
# for key_c, value_c in re_rank_dict_all_mean.items():
#     feature_list.append(key_c)
#     value_c_c = np.mean(np.array(re_rank_dict_all_mean[key_c])).tolist()
#     if type(value_c_c) == list:
#         score_list.append(value_c_c[0])
#     else:
#         score_list.append(value_c_c)
#
# # 排序输出最重要的top个特征
# idx = np.argsort(np.array(score_list))[::-1].tolist()
#
# rank_list = [feature_list[j] for j in idx]
# rank_score = [score_list[j] for j in idx]
#
# feature_rank = dict(zip(rank_list, rank_score))
#
# print(feature_rank)
# print(rank_list[:50])