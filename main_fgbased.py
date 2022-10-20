import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from causalml.dataset import make_uplift_classification
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from causalml.inference.tree import uplift_tree_string, uplift_tree_plot
from causalml.inference.meta import LRSRegressor
from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor

import numpy as np
import pandas as pd
from IPython.display import Image
from sklearn.model_selection import train_test_split

import shap
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import time
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

import os
import warnings, pickle

warnings.filterwarnings('ignore')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # for lightgbm to work

outpath = './feature_dict.pkl'
with open(outpath, 'rb') as f:
    re_rank_dict_all = pickle.load(f)
print('save success result')

re_rank_dict_all_mean = copy.deepcopy(re_rank_dict_all)
feature_list = []
score_list = []
for key_c, value_c in re_rank_dict_all_mean.items():
    feature_list.append(key_c)
    value_c_c = np.mean(np.array(re_rank_dict_all_mean[key_c])).tolist()
    if type(value_c_c) == list:
        score_list.append(value_c_c[0])
    else:
        score_list.append(value_c_c)

# 排序输出最重要的top个特征
idx = np.argsort(np.array(score_list))[::-1].tolist()

rank_list = [feature_list[j] for j in idx]
rank_score = [score_list[j] for j in idx]

feature_rank = dict(zip(rank_list, rank_score))

print(feature_rank)
print(rank_list[:50])
