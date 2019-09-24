import chars2vec as c2v
import sklearn.decomposition
import matplotlib.pyplot as plt
import tensorflow as tf

""" Mute tensorflow warning """
tf.logging.set_verbosity(tf.logging.ERROR)

""" 2D Visualization script by using PCA on the vectorization of a list of words"""

c2v_model = c2v.load_model("train_fr_150")

""" Words to visualize """
words = ['est', 'ezt', 'zest',
         'carotte', 'carote', 'carottte',
         'langage', 'language', 'langqge',
         'francais', 'franssais', 'francqis',
         'bread', 'brad', 'breod', 'broad']

word_embeddings = c2v_model.vectorize(words)

""" Optional print of euclidean distances between vectors """
def print_distance(words):
    import numpy as np

    print("\t", end = '')
    for word in words:
        print("%-10.6s" % word, end = '\t')
    print("")
    for i, vec1 in enumerate(word_embeddings):
        print(words[i], end = ' ')
        for vec2 in word_embeddings:
            print("%10.4f" % np.linalg.norm(vec1 - vec2), end = '\t')
        print("")

"""Embedding 2D projection with PCA"""
projection_2d = sklearn.decomposition.PCA(n_components=2).fit_transform(word_embeddings)

f = plt.figure(figsize=(8, 6))

for j in range(len(projection_2d)):
    plt.scatter(projection_2d[j, 0], projection_2d[j, 1],
                marker=('$' + words[j] + '$'),
                s=400 * len(words[j]), label=j,
                facecolors='red' if words[j]
                            in ['est', 'langage', 'francais', 'carotte', 'bread'] else 'black')
plt.show()


"""Embedding 3D projection with PCA"""
projection_3d = sklearn.decomposition.PCA(n_components=3).fit_transform(word_embeddings)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for j in range(len(projection_3d)):
    ax.scatter(projection_3d[j,0], projection_3d[j,1], projection_3d[j,2],
               marker=('$' + words[j] + '$'), s=200 * len(words[j]), label=j,
               facecolors='red' if words[j]
                           in ['est', 'langage', 'francais', 'carotte', 'bread'] else 'black')

plt.show()
