import os
import time
import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from preprocess_data import preprocess_data
from preprocess_df_electronics import preprocess_df_electronics
from preprocess_flight import preprocess_flight
from preprocess_nypd import preprocess_nypd
from preprocess_submissions import preprocess_submissions


def k_prototypes(dataset, n_clus, nums_len, name):
    attr_len = dataset.shape[1]
    num_indices = np.arange(nums_len)
    cat_indices = np.arange(nums_len, attr_len).tolist()

    scaler = StandardScaler()
    scaled_numeric_columns = scaler.fit_transform(dataset[:, num_indices])
    combined_selection = np.hstack((scaled_numeric_columns, dataset[:, cat_indices]))

    st = time.time()
    if name == 'partsupp' or name == '100k_a':
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

    return t_clus, clusters


names = ['Submissions']
nums_lens = [7]
n_clus_list = [12]
for name, nums_len in zip(names, nums_lens):
    for n_clus in n_clus_list:
        file_path = f'./datasets/{name}.csv'
        if name in ('orders', 'partsupp', '100k_a'):
            with open(file_path) as f:
                raw_dataset = f.readlines()
        else:
            if name == 'flight':
                raw_dataset = pd.read_csv(file_path, delimiter='|')
            else:
                raw_dataset = pd.read_csv(file_path, delimiter=',')
        if name in ('orders', 'partsupp', '100k_a'):
            dataset = preprocess_data(raw_dataset, name)
        elif name == 'df_electronics':
            dataset = preprocess_df_electronics(raw_dataset)
        elif name == 'flight':
            dataset = preprocess_flight(raw_dataset)
        elif name == 'nypd':
            dataset = preprocess_nypd(raw_dataset)
        elif name == 'Submissions':
            dataset = preprocess_submissions(raw_dataset)
        _, clusters = k_prototypes(np.array(dataset), n_clus, nums_len, name)
