import numpy as np

class QKmeans():
    '''
    QKmenas类,根据论文实现Quantized K-Means算法
    主要变量:
        1.主要参数
        k: 聚类簇数
        iters: 算法循环次数
        epsilon: 网格粒度参数epsilon > 0
        gamma: 参数gamma,范围(0,1),用于平衡聚类大小,默认0.2
        2.和聚类有关的参数
        data: 训练集,n*d数组表示,n个数据,每个数据d维
        centroids: k个聚类的中心点集合
        loss: 损失函数值
        min_loss: 最小损失函数值
        clusters: n维数组，把每个数据点分到一个类中,初始化为0
        centroids_res,clusters_res: 分别存储一次训练完成后的中心点和数据分类
        empty_clusters: 空的聚类，在处理时会选择新的中心点
        pahse: 随机偏移,偏差值取(-1/2,1/2)
        r_phase: 记录每次迭代过程中的phase值
        terminate_i: 提前终止时记录停止的循环次数，否则为-1
        r_init_centroids: 记录kmeans++聚类中心的初始点,若删除需重新聚类分析
        momentum: 根据gamma算出的值，用于修正聚类大小
    '''

    def __init__(self,k,epsilon,iters=10,gamma=0.2):
        # 设置参数
        self.k = k
        self.epsilon = epsilon
        self.iters = iters
        self.gamma = gamma

    def run(self, X):
        # 设置数据
        self.data=X
        self.n=X.shape[0]
        self.d=X.shape[1]
        # 初始化
        self.min_loss = np.Infinity
        self.terminate_i = -1
        self.clusters_res = None
        self.centroids_res = None
        self.r_analog_c = np.zeros([self.iters + 1, self.k, self.d])
        self.r_quantized_c = np.zeros([self.iters + 1, self.k, self.d])
        self.r_phase = np.zeros([self.iters, self.d])
        self.r_clustersize = np.zeros([self.iters, self.k])
        self.momentum = self.gamma * self.n / self.k
        self.empty_clusters = []
        self.r_init_centroids = set()
        # 使用原有的kmeans算法计算簇中心点
        c_index = np.random.choice(self.n)
        self.centroids = np.zeros([self.k, self.d])
        self.centroids[0,:] = self.data[c_index,:]
        # 进行第一次聚类
        for i in range(1,self.k):
            centroids = self.centroids[0:i,:]
            if len(centroids.shape)==1:
                centroids = np.expand_dims(centroids,axis=0)

            D = np.zeros([self.n])
            for j in range(self.n):
                d = np.linalg.norm(self.data[j, :] - centroids, axis=1)
                D[i] = np.min(d)
            P = [dist ** 2 for dist in D]
            P = P / sum(P)
            c_index = np.random.choice(self.n,p=P)
            self.centroids[i,:] = self.data[c_index,:]
            self.r_init_centroids.add(c_index)
        self.r_analog_c[0] = self.centroids
        self.r_quantized_c[0] = self.centroids
        self._clustering()
        self.empty_clusters = []
        cluster_size = np.bincount(self.clusters)
        self.empty_clusters=np.where(cluster_size==0)
        self.phase = np.random.random([self.d]) - 0.5
        # 进行后面几次聚类
        for i in range(self.iters):
            self._choose_centroids()
            self._quantize(i)
            self._clustering()
            if self.min_loss <= self.loss:
                self.terminate_i = i
                break
            else:
                self.min_loss = self.loss
                self.centroids_res = self.centroids
                self.clusters_res = self.clusters

        return self.centroids_res, self.clusters_res, self.min_loss

    # 删除数据
    def delete(self, del_idx):
        if not self._invariance(del_idx):
            self.data = np.delete(self.data, del_idx, 0)
            self.n = self.n - 1
            self.loss = np.Infinity
            self.r_init_centroids = set()
            self.centroids_res = None
            return self.run(self.data)
        else:
            return self.centroids_res, self.clusters_res, self.min_loss

    # 聚类
    def _clustering(self):
        self.clusters = np.zeros([self.n]).astype(int)
        self.loss = 0
        for i in range(self.n):
            d = np.linalg.norm(self.data[i, :] - self.centroids, axis=1)
            self.clusters[i] = int(np.argmin(d))
            self.loss += np.min(d) ** 2
        self.loss = self.loss / self.n

    # 选择簇中心点
    def _choose_centroids(self):
        self.centroids = np.zeros([self.k, self.d])
        cluster_size = np.bincount(self.clusters)
        for j in range(self.k):
            data_indexs=np.where(self.clusters==j)
            data_slice=self.data[data_indexs,:]
            if not j in self.empty_clusters:
                self.centroids[j, :]=np.sum(data_slice,axis=1)/cluster_size[j]
        for j in self.empty_clusters:
            self._reinit_cluster(j)
        self.empty_clusters = []

    # 对空聚类重新聚类分析
    def _reinit_cluster(self, j):
        P = self.get_probality()
        j_prime = np.random.choice(self.n, p=P)
        self.r_init_centroids.add(j_prime)
        self.centroids[j, :] = self.data[j_prime, :]
        return j_prime

    # 计算随机选取的概率
    def get_probality(self):
        if len(self.centroids.shape) == 1:
            self.centroids = np.expand_dims(self.centroids, axis=0)
        D = np.zeros([self.n])
        for i in range(self.n):
            d = np.linalg.norm(self.data[i, :] - self.centroids, axis=1)
            D[i] = np.min(d)
        P = [dist ** 2 for dist in D]
        P = P / sum(P)
        return P

    # 量化
    def _quantize(self, i):
        self.r_analog_c[i + 1, :, :] = self.centroids
        cluster_size = np.bincount(self.clusters)

        for j in range(self.k):
            self.r_clustersize[i, j] = cluster_size[j]
            if (cluster_size[j] < self.momentum):
                self.centroids[j] = self._momentum_correction(
                    self.centroids[j], self.r_quantized_c[i, j], cluster_size[j])

        self.phase = np.random.random([self.d]) - 0.5
        self.centroids = self._quantize_c(self.centroids, self.epsilon, self.phase)

        self.r_phase[i, :] = self.phase
        self.r_quantized_c[i + 1] = self.centroids

    def _quantize_c(self, centroids, epsilon, phase):
        return (np.round(centroids / epsilon + phase) - phase) * epsilon

    # 对小聚类进行补偿
    def _momentum_correction(self, centroid_cur, centroid_prev, clustersize):
        return (clustersize / self.momentum) * centroid_cur + \
               ((self.momentum - clustersize) / self.momentum * centroid_prev )

    # 判断中心点是否不变
    def _invariance(self, del_idx):
        delete_data = self.data[del_idx, :]
        if del_idx in self.r_init_centroids:
            return False

        for i in range(self.iters):
            if i >= self.terminate_i and self.terminate_i >= 0:
                break

            analog_centroids = self.r_analog_c[i + 1, :, :]
            phase = self.r_phase[i, :]
            cluster_size = self.r_clustersize[i, :]
            d = np.linalg.norm(delete_data - analog_centroids, axis=1)
            assignment_idx = int(np.argmin(d))
            centroid = analog_centroids[assignment_idx, :]
            centroid_prev = self.r_quantized_c[i, assignment_idx, :]
            clustersize = cluster_size[assignment_idx]

            if clustersize < self.momentum:
                centroid = self._momentum_correction(
                    centroid, centroid_prev, clustersize)
                clustersize = self.momentum

            perturbed_centroid = centroid - delete_data / clustersize
            quant =self._quantize_c(centroid, self.epsilon, phase)
            quant_perturbed =self._quantize_c(perturbed_centroid, self.epsilon, phase)

            if not all(quant == quant_perturbed):
                return False

            self.r_clustersize[i - 1, assignment_idx] -= 1
            self.r_analog_c[i - 1, assignment_idx] = perturbed_centroid

        self.data[del_idx, :] = np.zeros(self.d)
        self.n -= 1
        self.momentum = self.gamma * self.n / self.k
        return True
