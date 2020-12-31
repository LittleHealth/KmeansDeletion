import numpy as np


class Kmeans():
    def __init__(self, k, iters=10):
        self.k = k
        self.iters = iters
        self.n, self.d, self.data, self.centers, self.assignments, self.loss, self.centers_idx \
            = 1, 0, None, None, None, [], []

    def delete(self, del_idx):
        self.data = np.delete(self.data, del_idx, axis=0)
        return self.run(self.data)

    def run(self, data):
        self.n, self.d = data.shape
        self.data, self.centers, self.assignments, self.loss, self.centers_idx \
            = data, None, None, [], []
        # 先找中心点
        self.init_centers()
        # 再根据中心点进行聚类
        for _ in range(self.iters):
            self.assign_cluster()
            # 对每个聚类中心更新，使用聚类内所有点的平均值
            for i in range(self.k):
                cluster = data[self.assignments == i]
                self.centers[i] = np.sum(cluster, axis=0) / len(cluster)
                # print("center:",self.centers[i])
        return self.centers, self.assignments, self.loss

    def init_centers(self):
        # 随机找到一个中心点
        idx = np.random.choice(self.n)
        self.centers_idx.append(idx)
        # print(type(self.data))
        # print(self.data[idx, :])
        centers = [self.data[idx, :]]
        for i in range(1, self.k):
            D = []
            for center in centers:
                # 计算每个点到这个聚类中心的距离
                d = np.linalg.norm(self.data - center, axis=1, ord=2)
                # d = np.sum((data - center)**2, axis = 1)
                D.append(d)
            # 一个很严重的问题，numpy的数组操作好多都没有，我该怎么用呢？
            # D储存的就是每个点到每个聚类中心的距离
            D = np.array(D)
            # P是每个点到最近的聚类中心的距离长度
            P = np.min(D, axis=0)
            # print("P:", P.shape, "  D:", D.shape)
            # 转化成概率
            # P = P / np.sum(P)
            # next_idx = np.random.choice(self.n, p=P)
            # 直接把距离最大的给筛选出来，是否需要在距离前几名中进行随机选取？，不然会导致噪声影响太大
            next_idx = np.argmax(P)
            self.centers_idx.append(next_idx)
            centers.append(self.data[next_idx, :])
            # print(P.shape, centers)
        self.centers = np.array(centers)

    def assign_cluster(self):
        self.assignments = np.zeros(self.n).astype(int)
        loss = 0
        D = []
        # 计算所有点到每个中心的距离
        for center in self.centers:
            d = np.linalg.norm(self.data - center, axis=1, ord=2)
            D.append(d)
        # 到哪个中心距离最短就属于哪个cluster
        self.assignments = np.argmin(D, axis=0)
        loss = np.sum(np.min(D, axis=0) ** 2) / self.n
        self.loss.append(loss)

    # def show(self, title):
    #     show_cluster(self.centers, self.assignments, self.data, title)


class DCNode(Kmeans):
    def __init__(self, k, iters=10):
        Kmeans.__init__(self, k=k, iters=iters)
        # 继承自Kmeans，添加children和parent，用于构造树
        self.children = []
        self.parent = None
        # 当前节点存储的数据
        self.node_data = []
        self.data_idx = []

    def run_node(self, d=None):
        if d is not None:
            self.d = d
        self.run(np.array(self.node_data).reshape(-1, self.d))


class DCKmeans():
    def __init__(self, ks, widths, iters=10):

        self.ks = ks
        self.widths = widths
        self.dc_tree = self.init_tree(ks, widths, iters)

        self.data_partion_table = dict()
        self.data = dict()
        self.dels = set()
        self.centers = None
        self.assignments = None
        self.loss = None
        self.n, self.d = 0, 0
        self.height = len(self.dc_tree)
        for i in range(self.height):
            self.data[i] = None

    def run(self, data, assignment=False):
        self.data = np.array(data)
        self.n, self.d = self.data.shape
        # data_layer_size = self.n
        # 从height-1减到0，这是为了将储存数据的self.data[i]设置成相应的0矩阵
        # for i in range(self.height - 1, -1, -1):
        #     self.data[i] = np.zeros((data_layer_size, self.d))
        #     print(f"self.data[{i}].shape:{self.data[i].shape}; data.shape")
        #     data_layer_size = self.ks[i] * self.widths[i]
        #
        num_leaves = len(self.dc_tree[-1])
        # 对每个数据点，随机抽取一个叶子节点，然后塞到里面去
        # TODO 考虑不用for循环
        # 所有数据点的表示都使用下标来表示
        for i in range(self.n):
            leaf_id = np.random.choice(num_leaves)
            # 维护划分表，记录每个数据点属于哪个叶子节点
            self.data_partion_table[i] = leaf_id
            # leaf维护自己当前存储的数据点
            leaf = self.dc_tree[-1][leaf_id]
            leaf.node_data.append(self.data[i])
            leaf.data_idx.append(i)
            # 前面初始化过data[k]，此时将data[k]的第i行设置为第i个数据
            # self.data[self.height - 1][i] = data[i]
        for h in range(self.height - 1, -1, -1):
            # print("h:", h)
            for w in range(self.widths[h]):
                # 对每层的数据节点进行训练聚类，然后依次让上层节点再进行训练
                subproblem = self.dc_tree[h][w]
                # 将当前高度height/层数level 进行聚类
                # subdata = np.array(subproblem.node_data)
                # subproblem.run(subdata)
                subproblem.run_node(self.d)
                # if w == 1 or w % 8 == 0:
                #     subproblem.show(f"{h}th level, {w}th node")
                if subproblem.parent is None:
                    # 到根节点了
                    self.centers = subproblem.centers
                else:
                    # 将子节点的中心点加入到母节点的数据集中
                    subproblem.parent.node_data.append(subproblem.centers)
        # 如果要求assignments，退化到原始的Kmeans算法；所以每次都只是按照不返回assignments实验
        if assignment is True:
            assignment_solver = Kmeans(self.ks[0])
            assignment_solver.data = self.data
            assignment_solver.centers = self.centers
            assignment_solver.assign_cluster()
            self.assignments = assignment_solver.assignments
            self.loss = assignment_solver.loss
        if self.assignments is None:
            self.assignments = np.zeros(self.n)
        return self.centers, self.assignments, self.loss

    def init_tree(self, ks, widths, iters):
        # print("init_tree:", ks, widths, iters)
        tree = [[DCNode(ks[0], iters)]]  # 根节点
        for i in range(1, len(widths)):
            k = ks[i]
            assert widths[i] % widths[i - 1] == 0, "Inconsistent widths in tree"
            # 下一层节点的数量的整数倍
            merge_factor = int(widths[i] / widths[i - 1])
            level = []
            for j in range(widths[i - 1]):
                parent = tree[i - 1][j]
                for _ in range(merge_factor):
                    child = DCNode(k, iters=10)
                    child.parent = parent
                    parent.children.append(child)
                    level.append(child)
            tree.append(level)
        return tree

    def delete(self, del_idx):
        leaf_idx = self.data_partion_table[del_idx]
        node = self.dc_tree[-1][leaf_idx]
        # 如果之前被删过就直接退出
        if node.data_idx.count(del_idx) == 0:
            return
        node.data_idx.remove(del_idx)
        node.node_data = self.data[list(node.data_idx)]
        while True:
            node.run_node()
            if node.parent is None:
                self.centers = node.centers
                break
            parent = node.parent
            child_idx = parent.children.index(node)
            parent.node_data[child_idx] = node.centers
            node = parent

