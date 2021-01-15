import pandas as pd
import numpy as np
import random
from evaluation import *

# run this code to test evaluate

# data of celltype
celltype_data = pd.read_csv('./data/celltype.csv')
cols = celltype_data.columns
col1 = cols[0:-2]
data = celltype_data[col1]
data = np.array(data)
real_labels = np.array(celltype_data[cols[-1]])  # tissue
k = 10  # change k smaller if the code throws error

# data of covtype
# covtype_data = pd.read_csv('./data/covtype.csv')
# cols = covtype_data.columns
# col1 = cols[0:-1]
# data = covtype_data[col1]
# data = np.array(data)
# real_labels = np.array(covtype_data[cols[-1]])  # type
# k = 7  # change k smaller if the code throws error

# uncomment one of above paragraphs to get data

samples = random.sample(range(data.shape[0]), 10000)
data = data[samples, :]
real_labels = real_labels[samples]

evaluate = evaluation(data, real_labels, k, DEL_NUM=100)

# 初始的训练
evaluate.run()

# 测试三种算法基础效果
print("--------------------------------")
evaluate.test_basic()
print("--------------------------------")

# 测试删除效果
evaluate.test_deletion()
print("--------------------------------")

# 测试删除时间
evaluate.test_deletion_time()
print("--------------------------------")
