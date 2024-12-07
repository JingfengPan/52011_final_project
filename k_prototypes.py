import os
import time
import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def k_prototypes(dataset, n_clus, nums_len, name):
    attr_len = dataset.shape[1]
    num_indices = np.arange(nums_len)
    cat_indices = np.arange(nums_len, attr_len).tolist()

    scaler = StandardScaler()
    scaled_numeric_columns = scaler.fit_transform(dataset[:, num_indices])
    combined_selection = np.hstack((scaled_numeric_columns, dataset[:, cat_indices]))

    st = time.time()
    if name == 'partsupp':
        km = KMeans(n_clusters=n_clus, random_state=0)
        print(f'{name} K-Means {n_clus} Clustering...')
        clusters = km.fit_predict(combined_selection)
    else:
        kp = KPrototypes(n_clusters=n_clus, n_jobs=-1)
        print(f'{name} K-Prototypes {n_clus} Clustering...')
        clusters = kp.fit_predict(combined_selection, categorical=cat_indices)
    et = time.time()
    t_clus = et - st

    output = pd.DataFrame(clusters)
    if not os.path.exists(f'./labels/{name}'):
        os.makedirs(f'./labels/{name}')
    if os.path.exists(f'./labels/{name}/{name}_{n_clus}_clus.csv'):
        os.remove(f'./labels/{name}/{name}_{n_clus}_clus.csv')
    output.to_csv(f'./labels/{name}/{name}_{n_clus}_clus.csv', index=False, header=False)

    return clusters, t_clus
