import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

data = np.random.rand(64, 10)  # 64个样本，每个样本维度为10
target = np.arange(2).repeat(32)  # 生成64个标签，用于区分样本目标
t_sne_features = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(data)
plt.scatter(x=t_sne_features[:, 0], y=t_sne_features[:, 1], c=target, cmap='jet')
plt.show()

