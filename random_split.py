import os
import random
import time
import pandas as pd
from parallel_column_wise_compression import parallel_column_wise_compression


def random_split_one_by_one(dataset, n_clus):
    clusters = []
    for i in range(len(dataset)):
        clusters.append(i % n_clus)

    return clusters


def random_split_chunk_by_chunk(dataset, n_clus, buffer_size=10000):
    clusters = []

    total_chunks = len(dataset) // buffer_size + (1 if len(dataset) % buffer_size > 0 else 0)

    for chunk_index in range(total_chunks):
        cluster_label = chunk_index % n_clus
        for _ in range(buffer_size):
            if len(clusters) >= len(dataset):
                break
            clusters.append(cluster_label)

    return clusters


names = ['Submissions']
comp_methods = ['gzip', 'lz4', 'zstd']
n_clus = 10

print('Random Split One by One')
for name in names:
    for comp_method in comp_methods:
        data_path = f'./datasets/{name}.csv'
        with open(data_path, encoding='utf-8') as dp:
            data = dp.readlines()

        labels = random_split_one_by_one(data, n_clus)
        _, comp_time = parallel_column_wise_compression(10, data_path, labels, comp_method, delimiter=',', random_state=1)
