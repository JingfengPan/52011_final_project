import gzip
import json
import random
import re
import time
import lz4.frame
import zstandard as zstd
from joblib import Parallel, delayed


def cluster_wise_split(n_clus, data, labels, delimiter):
    clusters_data = [[] for _ in range(n_clus)]

    for i, line in enumerate(data):
        regex_row = re.sub(r'"[^"]*"', lambda match: match.group().replace(',', ''), line).split(delimiter)
        clusters_data[int(labels[i])].append(regex_row)

    return clusters_data


def compress(data, comp_method):
    # Convert each row back to a string
    string_data = ["".join(row) if isinstance(row, list) else row for row in data]

    # Combine all rows into a single byte sequence
    bytes_data = "\n".join(string_data).encode()

    if comp_method == 'gzip':
        compressed = gzip.compress(bytes_data)
    elif comp_method == 'lz4':
        compressed = lz4.frame.compress(bytes_data)
    elif comp_method == 'zstd':
        compressed = zstd.compress(bytes_data)
    else:
        raise ValueError("Unsupported compression format")
    compressed_size = len(compressed)

    return compressed_size


def cluster_wise_compress(n_clus, data_path, labels, comp_method, delimiter):
    compressed_size = 0
    with open(data_path, encoding='utf-8') as dp:
        raw_dataset = dp.readlines()

    clusters_data = cluster_wise_split(n_clus, raw_dataset, labels, delimiter)
    tasks = [(cluster, comp_method) for cluster in clusters_data]
    start_time = time.time()
    compressed_data = Parallel(n_jobs=n_clus)(delayed(compress)(*task) for task in tasks)
    end_time = time.time()

    compressed_size += sum(compressed_data)
    compression_time = end_time - start_time

    return compressed_size, compression_time
