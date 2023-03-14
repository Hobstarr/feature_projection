import numpy as np
from keras.datasets import mnist, fashion_mnist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import time
import umap

sns.set_style('whitegrid')
# import dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reduce the testing set to 10000 to speed up, and normalise and reshape.
x_train_small = np.reshape(x_train[:10000]/255, (10000, 784))
y_train_small = y_train[:10000]
x_test_reshape = np.reshape(x_test/255, (10000, 784))

# reducing the dimensions and then building a model
# using the reduced dimensions.
n_list = []
score_list = []
for n in range(2,100):
    n_list.append(n)
    pca = PCA(n_components = n)
    pca_embedding = pca.fit_transform(x_train_small)
    knn_small = KNeighborsClassifier(n_neighbors = 5)
    knn_small.fit(pca_embedding, y_train_small)

    pca_test_embedding = pca.transform(x_test_reshape)
    score_list.append(knn_small.score(pca_test_embedding, y_test))

# using a model with all the dimensions
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train_small, y_train_small)
knn.score(x_train_small, y_train_small)
full_score = knn.score(x_test_reshape, y_test)

plt.plot(n_list, score_list, label = 'feature projected model')
plt.axhline(full_score, label = 'full model', 
            color = 'green', linestyle = 'dashed')
plt.legend()
plt.title('feature projected model vs full model \n' +
          'for PCA reduced dimension 5-Nearest Neighbours model')
plt.ylabel('score')
plt.xlabel('dimensions')
plt.show()

# PCA embedding
pca = PCA(n_components = 2)
pca_embedding = pca.fit_transform(x_train_small)

plt.scatter(pca_embedding[:,0], pca_embedding[:,1], s = 0.3,
            c = y_train_small, cmap = 'tab10')
plt.title('PCA embedding of mnist digist')
plt.tick_params(bottom=False, left=False, 
                labelbottom = False, labelleft = False)
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.show()

# UMAP embedding
umapp = umap.UMAP()
umap_embedding = umapp.fit_transform(x_train_small)

plt.scatter(umap_embedding[:,0], umap_embedding[:,1], s = 0.1,
            c = y_train_small, cmap = 'tab10')
plt.title('umap embedding of mnist digist')
plt.tick_params(bottom=False, left=False, 
                labelbottom = False, labelleft = False)
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.show()