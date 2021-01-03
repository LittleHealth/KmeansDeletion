import pandas as pd
import numpy as np
import random
from evaluation import *

celltype_data = pd.read_csv('./data/celltype.csv')
# covtype_data = pd.read_csv('./data/covtype.csv')
#
cols = celltype_data.columns
# cols = covtype_data.columns
col1 = cols[0:-2]
# col1 = cols[0:-1]
data = celltype_data[col1]
# data = covtype_data[col1]
data = np.array(data)
real_labels = np.array(celltype_data[cols[-1]])  # tissue
# real_labels = np.array(covtype_data[cols[-1]])  # type

# print(data)

# %%

from QKmeans import QKmeans
from DCKmeans import DCKmeans, Kmeans
from utils import *

# %%

k = 10
# k = 7

samples = random.sample(range(data.shape[0]), 10000)
data = data[samples, :]
real_labels = real_labels[samples]

kmeans = Kmeans(k, 20)
centers, assignments, loss = kmeans.run(data)

qkmeans = QKmeans(k, 0.5)
qcenters, qassignments, qloss = qkmeans.run(data)
# qkmeans.run(data)

dckmeans = DCKmeans([k, k], [1, 16])
dccenters, dcassignments, dcloss = dckmeans.run(data, True)
# dckmeans.run(data, True)

DEL_NUM = 100

print('Simulation deletion stream for kmeans')
online_deletion_stream(DEL_NUM, kmeans)

print('Simulation deletion stream for qkmeans')
online_deletion_stream(DEL_NUM, qkmeans)

print('Simulation deletion stream for dckmeans')
online_deletion_stream(DEL_NUM, dckmeans)


# loss_ratio_q = loss_ratio(qloss, loss)
# print("qkmeans loss ratio: ", loss_ratio_q)
# loss_ratio_dc = loss_ratio(dcloss, loss)
# print("dckmeans loss ratio: ", loss_ratio_dc)
#
# silh = silhouette(data, assignments)
# print("kmeans silhouette score: ", silh)
#
# qsilhouette = silhouette(data, qassignments)
# print("qkmeans silhouette score: ", qsilhouette)
#
# dcsilhouette = silhouette(data, dcassignments)
# print("dckmeans silhouette score: ", dcsilhouette)
#
# mutual = mutual_info(real_labels, assignments)
# print("kmeans mutual info score: ", mutual)
#
# qmutual = mutual_info(real_labels, qassignments)
# print("qkmeans mutual info score: ", qmutual)
#
# dcmutual = mutual_info(real_labels, dcassignments)
# print("dckmeans mutual info score: ", dcmutual)

#  删除后qkmeans和dckmeans的三个指标
deleted_loss_q = 0
deleted_loss_dc = 0

deleted_sil_q = 0
deleted_sil_dc = 0

deleted_mutual_q = 0
deleted_mutual_dc = 0

#  运行删除后数据集的qkmeans和dckmeans的三个指标
rerun_loss_q = 0
rerun_loss_dc = 0

rerun_sil_q = 0
rerun_sil_dc = 0

rerun_mutual_q = 0
rerun_mutual_dc = 0

#  qkmeans删除
DEL_NUM = 100

# online_deletion_stream(DEL_NUM, qkmeans)
# deleted_data_q = qkmeans.data
#
# qcenters = qkmeans.centroids_res
# qassignments = qkmeans.clusters_res
# qloss = qkmeans.min_loss
#
# deleted_loss_q = qloss / loss
# deleted_sil_q = silhouette(deleted_data_q, qassignments)
#
# qcenters, qassignments, qloss = qkmeans.run(deleted_data_q)
# rerun_loss_q = qloss / loss
# rerun_sil_q = silhouette(deleted_data_q, qassignments)
#
# print(deleted_loss_q, rerun_loss_q)
# print(deleted_sil_q, rerun_sil_q)

#  dckmeans删除
# DEL_NUM = 100
#
# online_deletion_stream(DEL_NUM, dckmeans)
# deleted_data_dc = dckmeans.data
#
# dccenters = dckmeans.centers
# dcassignments = dckmeans.assignments
# dcloss = dckmeans.loss
# _, _, loss = kmeans.run(deleted_data_dc)
#
# deleted_loss_dc = dcloss / loss
# deleted_sil_dc = silhouette(deleted_data_dc, dcassignments)
#
#
# dccenters, dcassignments, dcloss = dckmeans.run(deleted_data_dc, True)
# rerun_loss_dc = dcloss / loss
# rerun_sil_dc = silhouette(deleted_data_dc, dcassignments)
#
# print(deleted_loss_dc, rerun_loss_dc)
# print(deleted_sil_dc, rerun_sil_dc)
