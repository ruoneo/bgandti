import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

epochs = [0, 10, 250]
marks = ["(A)", "(B)", "(C)"] # , "()", "e"]
datasets = ['e'] # , 'ic', 'gpcr', 'nr', 'luo']

for dataset in datasets:
    for epoch, mark in zip(epochs, marks):
        coords = np.loadtxt("../results/emb/{}/y_test.csv".format(dataset), dtype=int, delimiter=',')

        A = np.loadtxt('../results/emb/{}/emb_{}.csv'.format(dataset, epoch), delimiter=',')

        drug_features = A[coords[:, 0], :]

        target_features = A[coords[:, 1], :]

        edges_features = drug_features * target_features

        t_sne_features = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(edges_features)
        plt.scatter(x=t_sne_features[:, 0], y=t_sne_features[:, 1], c=coords[:, 2], cmap='jet')
        plt.title("{} epoch {}".format(mark,epoch))
        plt.savefig("../results/emb/{}/epoch_{}.svg".format(dataset,epoch))
        plt.show()
