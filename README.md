# feature selection

##必须文件
### 1 public_input.py
包含fg所需的fg.json

###2 main_udf.py （生成文件6）
以maxcomputer表为输入，基于多线程分批次处理训练数据，
采用因果推断中的meta-learning算法，
得到fg.json中所有非字符串类型特征的重要性分数和排序

输入：
line242-259
包含以下信息
1. 阿里云授权账号的access key及密码
2. datawalks空间的项目名称的endpoint地址
3. datawalks空间中mc表的地址
4. datawalks空间中mc表中目标变量的列名称
5. datawalks空间中mc表中使用数据的时间分区，一般是一个月，最后一天作为测试数据，之前的作为训练数据
6. 一次处理的数据量
7. 多线程的个数，根据服务器的CPU核心数决定，一般4核对应一个

###3 main_fgbased.py
这是将文件2得到的所有非字符串类型特征的重要性分数的排序进行打印

###4 moment_rec_dbmtl_v1_copy.config
这是pai -name easy_rec_ext命令中 
以全量特征参与训练时的-Dconfig的参数

###5 process_config.py（生成文件7）
这是以moment_rec_dbmtl_v1_copy.config为输入，
根据因果推断阈值（改变min_causal_effect变量即可），
生成只包含重要特征的-Dconfig的参数

##非必须文件
###6 re.pkl
这是文件2生成中间文件，存储了所有非字符串类型特征的重要性分数

###7 moment_rec_dbmtl_v1_part.config
这是pai -name easy_rec_ext命令中 
以部分重要特征参与训练时的-Dconfig的参数

## 运行顺序
1. 拷贝文件1和4至此
2. 在文件2中根据所需的输入信息运行文件2
3. 运行文件3查看结果
4. 运行文件5
5. 之后训练模型在datawalks中进行

# 说明1
moment_rec_dbmtl_v1_copy.config是全量特征的config文件
moment_rec_dbmtl_v1_part.config是阈值为0.05的config文件
moment_rec_dbmtl_v1_part_0_1.config是阈值为0.1的config文件
moment_rec_dbmtl_v1_part_0_2.config是阈值为0.2的config文件

# 说明2
以上如果想变成pipeline形式，可以把以上命令写到一个sh文件里，可单独配一个config文件来单独给出所有输入信息，如accesskey和项目名称等。

# 说明3
以上文件全部删去了阿里云内部账号，url等涉密信息，本人已从阿里结束实习。

# 说明4
如果有任何问题，请联系zepenghuai6@gmail.com