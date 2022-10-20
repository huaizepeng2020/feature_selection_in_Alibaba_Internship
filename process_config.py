import pickle
import numpy as np
import copy
from public_input import fg_json

"""---------------读取重要特征子集--------------------"""
# 打开特征子集
outpath = './feature_dict.pkl'
with open(outpath, 'rb') as feature_subset:
    re_rank_dict_all = pickle.load(feature_subset)
print('load success feature_subset')

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
min_causal_effect = 0.05
min_causal_effect = 0.1
min_causal_effect = 0.2
rank_list = [feature_list[j] for j in idx if score_list[j] >= min_causal_effect]
rank_score = [score_list[j] for j in idx if score_list[j] >= min_causal_effect]
feature_rank = dict(zip(rank_list, rank_score))
"""---------------读取重要特征子集--------------------"""

"""----------------读取字符串类型-----------------------"""
# 获得表的字段/特征名称 变量类型 具体值
json_table = fg_json
propoty_c = []
propoty_type_c = []
propoty_type_accpedted_c = []
all_string_feature = []
for idx, feature_c in enumerate(json_table['features']):
    propoty_c.append(feature_c['feature_name'])
    propoty_type_c.append(feature_c['value_type'])
    if feature_c['value_type'] == 'Double' or feature_c['value_type'] == 'Integer':
        propoty_type_accpedted_c.append(idx)
    if feature_c['value_type'] == 'String':
        all_string_feature.append(feature_c['feature_name'])
"""----------------读取字符串类型-----------------------"""

"""----------------所有特征子集-----------------------"""
all_used_feature_list = rank_list + all_string_feature
# 打印筛选之后的特征信息
print('筛选的因果影响阈值为：',min_causal_effect)
print('筛选之后的特征子集数量/全量特征：',len(all_used_feature_list), '/', idx)
print('字符串类型特征(全部保留)：',len(all_string_feature))
print('数值型特征(根据因果算法筛选)：',len(rank_list))
"""----------------所有特征子集-----------------------"""

# 打开 hello.txt 文件，指定「只读模式」
all_feature_config = open('moment_rec_dbmtl_v1_copy.config', 'r')
part_feature_config = open('moment_rec_dbmtl_v1_part_0_2.config', 'w')

# 使用 for 循环，将读到的内容，打印出来
num = 1
flag = False
for con in all_feature_config:
    if 'feature_names' in con:
        for feature_c in all_used_feature_list:
            if feature_c in con:
                part_feature_config.writelines(con)
                break
    else:
        part_feature_config.writelines(con)

# 最后需要将文件关闭
all_feature_config.close()
part_feature_config.close()

a = 1
