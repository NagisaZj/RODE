import torch as th
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import copy
import numpy as np
import matplotlib.pyplot as plt
import os


n_clusters = 5
path = '/data/RODE-v1/storage/RODE/3s5z_vs_3s6z/action-20-500k/models/'
sub_ds = next(os.walk(path))[1]
sub_ds.sort()

files = []

for sub_d in sub_ds:
    d_path = os.path.join(path, sub_d)
    sub_sub_ds = next(os.walk(d_path))[1]
    sub_sub_ds = [('0' + sub_sub_d if int(sub_sub_d) < 1000 else sub_sub_d) for sub_sub_d in sub_sub_ds]
    sub_sub_ds.sort()

    files.append(os.path.join(d_path, sub_sub_ds[-1]))

for file_i, file in enumerate(files):
    action_repr_array = th.load("{}/action_repr.pt".format(file), map_location=lambda storage, loc: storage).cpu().numpy()

    k_means = KMeans(n_clusters=n_clusters, random_state=0).fit(action_repr_array)

    spaces = []
    for cluster_i in range(n_clusters):
        spaces.append((k_means.labels_ == cluster_i).astype(np.float))

    print(file_i + 1, 'After Clustering', spaces)

    o_spaces = copy.deepcopy(spaces)
    spaces = []

    for space_i, space in enumerate(o_spaces):
        _space = copy.deepcopy(space)
        _space[0] = 0.
        _space[1] = 0.

        if _space.sum() == 2.:
            spaces.append(o_spaces[space_i])
        if _space.sum() >= 3:
            _space[:6] = 1.
            spaces.append(_space)

    for space in spaces:
        space[0] = 1.

    print(file_i + 1, 'After Processing', spaces)

    if len(spaces) < 3:
        spaces.append(spaces[0])
        spaces.append(spaces[1])

    print(file_i + 1, 'After Adding', spaces)

    pca = PCA(n_components=2, svd_solver='full')
    x = pca.fit_transform(action_repr_array)

    n_actions = 15
    color = ['y', 'y', 'gold', 'gold', 'b', 'r'] + ['g' for _ in range(6)] + ['coral' for _ in range(3)]

    font_size = 18  # 22
    legend_font_size = 28  # 24

    figure = plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=color, s=120)
    for j in range(n_actions):
        plt.annotate(j, (x[j, 0], x[j, 1]))
    # plt.show()
    plt.tick_params('x', labelsize=font_size)
    plt.tick_params('y', labelsize=font_size)
    plt.tight_layout()
    grey = 248.
    plt.gca().set_facecolor([grey / 255, grey / 255, grey / 255])
    plt.grid()
    # plt.show()
    plt.savefig("3s5z_vs_3s6z-ar/" + str(file_i + 1) + "-ar.png".format(0))
    plt.close()

    print()
    print()
    print()
