import inspect
import pickle

import numpy as np
import scipy.sparse as sp


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])  # A* = A+I，即对邻接矩阵加入自连接

    rowsum = np.array(adj_.sum(1))  # 对行求和，即得到节点的度
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())  # 得到D的-1/2次方矩阵d
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()  # 这一步的实质是做归一化，即A* × d转置 × d
    return sparse_to_tuple(adj_normalized)


def load_data(dataset):
    adj = np.loadtxt('./data/partitioned_data/{0}/orig/{0}_adj_orig.txt'.format(dataset), dtype=int)
    adj = sp.csr_matrix(adj)
    features = pickle.load(open("./data/partitioned_data/{0}/feature/{0}_feature.pkl".format(dataset), 'rb'))
    y_test = 0
    tx = 0
    ty = 0
    test_mask = 0
    labels = 0
    return adj, features, y_test, tx, ty, test_mask, labels


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # sp.matrix(data,offsets)是将data的元素每列的元素，按offset里的顺序在列上进行重新排列，offset里的值是偏移量
    # 具体可以参考https://blog.csdn.net/ChenglinBen/article/details/84424379
    # .diagonal()就是提取对角线元素
    # Remove diagonal elements删除对角线元素
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    # 把零元素都消除掉
    adj.eliminate_zeros()
    # Check that diag is zero:
    # np.diag(matrix)即提取matrix的对角线元素，todense() like toarray(),区别是一个是将存储方式由稀疏矩阵转成正常矩阵，另一个是转成array
    # assert检查是否对角线元素是否都被清空了
    assert np.diag(adj.todense()).sum() == 0

    # sp.triu(matrix)获取matrix的上三角矩阵，相应的，tril()是获取下三角矩阵
    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    # edges相当于组合，因为是上三角矩阵的edge，所以减少了一半的重复量，(4.6)与(6,4)不会同时存在，而只会保留(4,6)
    # edges_all相当于排列，就都包含了
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]

    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    # 随机选取一部分作为test与val
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    # 该函数请参考github中gae的写法，应该是更新了，这种方法应该是错的，或者说与python3不兼容
    # 其中，return部分或许应该改成np.any(rows_close)
    def ismember(a, b, tol=5):
        # 该函数的作用就是判断a元素是否存在于b集合中
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)
        # return (np.all(np.any(rows_close, axis=-1), axis=-1) and
        # np.all(np.any(rows_close, axis=0), axis=0))

    # test_edges_false是去生成一些本来就不存在的edges
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    # val_edges_false生成一些不存在于train与val的edges
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    #    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    # 如英文注释所说，这里将处理好的train_edges再重建出adj_train
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    print([var_name for var_name, var_val in callers_local_vars if var_val is var])
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]

def get_data(dataset):
    # Load data
    # adj, features, y_test, tx, ty, test_maks, true_labels = load_data(data_name)
    adj, features, y_test, tx, ty, test_maks, true_labels = load_data(dataset)  # e  ic gpcr nr luo

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    # 删除对角线元素
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train
    adj_dense = adj.toarray()

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    num_nodes = adj.shape[0]
    features_dense = features.tocoo().toarray()

    features = sparse_to_tuple(features.tocoo())
    # num_features是feature的维度
    num_features = features[2][1]
    # features_nonzero就是非零feature的个数
    features_nonzero = features[1].shape[0]

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    items = [
        adj, num_features, num_nodes, features_nonzero,
        pos_weight, norm, adj_norm, adj_label,
        features, true_labels, train_edges, val_edges,
        val_edges_false, test_edges, test_edges_false, adj_orig, features_dense, adj_dense, features_dense
    ]

    feas = {}

    print('num_features is:', num_features)
    print('num_nodes is:', num_nodes)
    print('features_nonzero is:', features_nonzero)
    print('pos_weight is:', pos_weight)
    print('norm is:', norm)

    for item in items:
        # item_name = [ k for k,v in locals().iteritems() if v == item][0]
        feas[retrieve_name(item)] = item

    feas['num_features'] = num_features
    feas['num_nodes'] = num_nodes
    return feas

def get_data_by_fold(dataset):
    # Load data
    # adj, features, y_test, tx, ty, test_maks, true_labels = load_data(data_name)
    adj, features, y_test, tx, ty, test_maks, true_labels = load_data(dataset)  # e  ic gpcr nr luo

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    # 删除对角线元素
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train
    adj_dense = adj.toarray()

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    num_nodes = adj.shape[0]
    features_dense = features.tocoo().toarray()

    features = sparse_to_tuple(features.tocoo())
    # num_features是feature的维度
    num_features = features[2][1]
    # features_nonzero就是非零feature的个数
    features_nonzero = features[1].shape[0]

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    items = [
        adj, num_features, num_nodes, features_nonzero,
        pos_weight, norm, adj_norm, adj_label,
        features, true_labels, train_edges, val_edges,
        val_edges_false, test_edges, test_edges_false, adj_orig, features_dense, adj_dense, features_dense
    ]

    feas = {}

    print('num_features is:', num_features)
    print('num_nodes is:', num_nodes)
    print('features_nonzero is:', features_nonzero)
    print('pos_weight is:', pos_weight)
    print('norm is:', norm)

    for item in items:
        # item_name = [ k for k,v in locals().iteritems() if v == item][0]
        feas[retrieve_name(item)] = item

    feas['num_features'] = num_features
    feas['num_nodes'] = num_nodes
    return feas

