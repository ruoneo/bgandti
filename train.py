import argparse

import numpy as np
import tensorflow as tf
from dppy.finite_dpps import FiniteDPP
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

from input import get_data
from evaluation import Evaluator
from model import BGAN
from optimizer import Optimizer, update


# 解析参数
def parse_args():
    parser = argparse.ArgumentParser(description='BGANDTI')
    parser.add_argument('--hidden1', type=int, default=32, help='隐藏层1神经元数量.')
    parser.add_argument('--hidden2', type=int, default=32, help='隐藏层2神经元数量.')
    parser.add_argument('--hidden3', type=int, default=64, help='隐藏层3神经元数量.')
    parser.add_argument('--learning_rate', type=float, default=.6 * 0.001, help='学习率')
    parser.add_argument('--discriminator_learning_rate', type=float, default=0.0001, help='判别器学习率')  # luo 判别器学习率0.0001， 其它数据集0.001
    parser.add_argument('--epoch', type=int, default=250, help='迭代次数')
    parser.add_argument('--seed', type=int, default=50, help='用来打乱数据集')
    parser.add_argument('--features', type=int, default=1, help='是(1)否(0)使用特征')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, default=0., help='Weight for L2 loss on embedding matrix.')
    parser.add_argument('--dataset', type=str, default='luo', help='使用的数据集')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    settings = parse_args()

    while True:
        try:
            # 读数据
            feas = get_data(settings.dataset)
            test_edges = np.column_stack((feas['test_edges'], np.ones((feas['test_edges'].shape[0], 1), dtype=np.int32)))
            test_edges_false = np.column_stack((np.array(feas['test_edges_false']), np.zeros((np.array(feas['test_edges_false']).shape[0], 1), dtype=np.int32)))
            test_dataset = np.vstack((test_edges, test_edges_false))

            # DPP采样和PCA降维
            DPP = FiniteDPP('correlation', **{'K': feas['adj'].toarray()})
            pca = PCA(n_components=settings.hidden2)
            DPP.sample_exact_k_dpp(size=20)  # e 21 ic 6 gpcr 3 luo 20
        except ValueError:
            print("A中的rank小于设定值，不能以设定值的大小进行采样!")
            continue
        else:
            break

    np.savetxt('results/emb/{0}/emb_{1}.csv'.format(settings.dataset, 0), feas['features_dense'], delimiter=',')
    np.savetxt('results/emb/{0}/y_test.csv'.format(settings.dataset), test_dataset, delimiter=',', fmt='%d')
    index = DPP.list_of_samples[0]
    feature_sample = feas['features_dense']
    feature_sample = pca.fit_transform(feature_sample)
    kde = KernelDensity(bandwidth=0.7).fit(np.array([feature_sample[i] for i in index]))

    # 预输入的数据
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'features_dense': tf.placeholder(tf.float32, shape=[feas['adj'].shape[0], feas['num_features']], name='real_distribution'),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'real_distribution': tf.placeholder(dtype=tf.float32, shape=[feas['adj'].shape[0], settings.hidden2], name='real_distribution')
    }

    # 构造模型
    model = BGAN(placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'], settings)

    # 定义优化器
    optimizer = Optimizer(model.ae_model, model.model_z2g, model.D_Graph, model.discriminator, placeholders, feas['pos_weight'], feas['norm'], model.d_real, feas['num_nodes'], model.GD_real,
                          settings)

    # 初始化会话和权重
    # 配置显存自动增长
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 存储不同阶段结果
    val_roc_score = []
    record = []
    record_emb = []

    # Train model
    for epoch in range(settings.epoch):

        emb, avg_cost = update(model.ae_model, optimizer.opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'], kde, feas['features_dense'], settings)

        lm_train = Evaluator(feas['val_edges'], feas['val_edges_false'])
        roc_curr, ap_curr, _, aupr_score = lm_train.get_roc_score(emb, feas)
        val_roc_score.append(roc_curr)
        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss={:.5f}, d_loss={:.5f}, g_loss={:.5f}, GD_loss={:.5f}, GG_loss={:.5f}".format(avg_cost[0], avg_cost[1], avg_cost[2], avg_cost[3], avg_cost[4]),
              "val_roc={:.5f}".format(val_roc_score[-1]), "val_ap=", "{:.5f}".format(ap_curr), "val_aupr=", "{:.5f}".format(aupr_score))

        if (epoch + 1) % 10 == 0:
            lm_test = Evaluator(feas['test_edges'], feas['test_edges_false'])
            roc_score, ap_score, _, aupr_score = lm_test.get_roc_score(emb, feas)
            print('Test ROC score: ' + str(roc_score), 'Test AUPR score: ' + str(aupr_score), 'Test AP score: ' + str(ap_score))
            record.append([roc_score, aupr_score, ap_score])
            record_emb.append(emb)
            np.savetxt('results/emb/{0}/emb_{1}.csv'.format(settings.dataset, epoch + 1), emb, delimiter=',')
    rec = np.array(record)
    emb = record_emb[rec[:, 0].tolist().index(max(rec[:, 0].tolist()))]
    ana = record[rec[:, 0].tolist().index(max(rec[:, 0].tolist()))]
    # ana_pr = record[rec[:, 1].tolist().index(max(rec[:, 1].tolist()))]
    print('The peak [auc] test_roc={:.7f}, aupr={:.7f}, ap={:.7f}'.format(ana[0], ana[1], ana[2]))
    # print('The peak [aupr] test_roc={:.7f}, aupr={:.7f}, ap={:.7f}'.format(ana_pr[0], ana_pr[1], ana_pr[2]))
