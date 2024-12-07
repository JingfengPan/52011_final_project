import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from k_prototypes import k_prototypes
from compress import cluster_wise_compress, compress
from classifiers import train_and_save_model, load_and_test_model
from preprocess import preprocess_data, preprocess_flight, preprocess_nypd


# Use K-Means or K-Prototypes to cluster data
def cluster(name, nums_len, n_clus):
    file_path = f'./datasets/{name}.csv'

    if name in ('orders', 'partsupp'):
        with open(file_path) as f:
            raw_dataset = f.readlines()
    else:
        if name == 'flight':
            raw_dataset = pd.read_csv(file_path, delimiter='|')
        else:
            raw_dataset = pd.read_csv(file_path, delimiter=',')

    if name in ('orders', 'partsupp'):
        dataset = preprocess_data(raw_dataset, name)
    elif name == 'flight':
        dataset = preprocess_flight(raw_dataset)
    elif name == 'nypd':
        dataset = preprocess_nypd(raw_dataset)

    clusters, cluster_time = k_prototypes(np.array(dataset), n_clus, nums_len, name)

    return clusters, cluster_time


def compare_clusters(names, nums_lens, n_clus_list, comp_methods):
    for name, nums_len in zip(names, nums_lens):
        print('--------------------------------------------------------------------------------------------\n')
        print(f'Dataset: {name}\n')
        file_path = f'./datasets/{name}.csv'
        with open(file_path) as f:
            raw_dataset = f.readlines()

        original_size = os.stat(file_path).st_size
        for comp_method in comp_methods:
            start_time = time.time()
            comp_size = compress(raw_dataset, comp_method)
            end_time = time.time()
            comp_time = end_time - start_time
            comp_ratio = original_size / comp_size
            print('Compression method:', comp_method)
            print(f'Comp Ratio: {comp_ratio}')
            print(f'Comp Time: {comp_time}s\n')

        if name in ('orders', 'partsupp', 'flight'):
            delimiter = '|'
        else:
            delimiter = ','
        print('--------------------------------------------------------------------------------------------\n')
        print('Clustering:\n')

        for n_clus in n_clus_list:
            # labels, cluster_time = cluster(name, nums_len, n_clus)
            # print(f'Clustering time for {name} with {n_clus} clusters: {cluster_time}s\n')
            labels = pd.read_csv(f'./labels/{name}/{name}_{n_clus}_clus.csv', header=None).squeeze().tolist()
            if name in ('flight', 'nypd'):
                labels = np.concatenate(([0], labels))
            print(f'{n_clus} clusters:\n')
            for comp_method in comp_methods:
                comp_size, comp_time = cluster_wise_compress(n_clus, file_path, labels, comp_method, delimiter)
                comp_ratio = original_size / comp_size
                print('Compression method:', comp_method)
                print(f'Comp Ratio: {comp_ratio}')
                print(f'Comp Time: {comp_time}s\n')


# Train classifier models using 5 different classifiers
def train_model(raw_dataset, name, nums_len, n_clus, class_name, train_size):
    clus_labels = pd.read_csv(f'./labels/{name}/{name}_{n_clus}_clus.csv', header=None)
    raw_train_data, _, train_labels, _ = train_test_split(raw_dataset, clus_labels, train_size=train_size, random_state=0)

    if name in ('orders', 'partsupp'):
        train_data = preprocess_data(raw_train_data, name)
    elif name == 'flight':
        train_data = preprocess_flight(raw_train_data)
    elif name == 'nypd':
        train_data = preprocess_nypd(raw_train_data)

    print(f'Training models for {name} dataset with {n_clus} clusters...')
    model_path, train_time = train_and_save_model(np.array(train_data), train_labels, name, class_name, train_size, nums_len, n_clus)
    print(f'{class_name} model training time: {train_time}\n')


# Test classifier models (using trained models to label data)
def test_model(raw_dataset, name, nums_len, n_clus, class_name, comp_methods, train_size):
    file_path = f'./datasets/{name}.csv'

    if name in ('orders', 'partsupp'):
        test_data = preprocess_data(raw_dataset, name)
    elif name == 'flight':
        test_data = preprocess_flight(raw_dataset)
    elif name == 'nypd':
        test_data = preprocess_nypd(raw_dataset)

    if name in ('orders', 'partsupp', 'flight'):
        delimiter = '|'
    else:
        delimiter = ','

    print(f'Testing models for {name} dataset with {n_clus} clusters...')
    model_path = f'./models/{name}_{n_clus}/{class_name}_{train_size}.joblib'
    scaler_path = f'./models/{name}_{n_clus}/{train_size}_scaler.joblib'
    predictions, test_time = load_and_test_model(np.array(test_data), model_path, scaler_path, class_name, nums_len)
    print(f'{class_name} model testing time: {test_time}\n')
    if name in ('flight', 'nypd'):
        predictions = np.concatenate(([0], predictions))
    original_size = os.stat(file_path).st_size

    for comp_method in comp_methods:
        comp_size, comp_time = cluster_wise_compress(n_clus, file_path, predictions, comp_method, delimiter)
        comp_ratio = original_size / comp_size
        print('Compression method:', comp_method)
        print(f'Comp Ratio: {comp_ratio}')
        print(f'Comp Time: {comp_time}s\n')


def compare_classifiers(names, nums_lens, comp_methods, class_names):
    n_clus = 10
    train_size = 0.1

    for name, nums_len in zip(names, nums_lens):
        print('--------------------------------------------------------------------------------------------\n')
        print(f'Dataset: {name}\n')
        file_path = f'./datasets/{name}.csv'
        if name in ('orders', 'partsupp'):
            with open(file_path) as f:
                raw_dataset = f.readlines()
        else:
            if name == 'flight':
                raw_dataset = pd.read_csv(file_path, delimiter='|')
            else:
                raw_dataset = pd.read_csv(file_path, delimiter=',')
        for class_name in class_names:
            print(f"Classifier: {class_name} with train size: {train_size}")
            train_model(raw_dataset, name, nums_len, n_clus, class_name, train_size)
            test_model(raw_dataset, name, nums_len, n_clus, class_name, comp_methods, train_size)


def compare_train_sizes(names, nums_lens, comp_methods, train_sizes):
    class_name = 'QDA'
    n_clus = 10

    for name, nums_len in zip(names, nums_lens):
        print('--------------------------------------------------------------------------------------------\n')
        print(f'Dataset: {name}\n')
        file_path = f'./datasets/{name}.csv'
        if name in ('orders', 'partsupp'):
            with open(file_path) as f:
                raw_dataset = f.readlines()
        else:
            if name == 'flight':
                raw_dataset = pd.read_csv(file_path, delimiter='|')
            else:
                raw_dataset = pd.read_csv(file_path, delimiter=',')
        for train_size in train_sizes:
            print(f"Classifier: {class_name} with train size: {train_size}")
            train_model(raw_dataset, name, nums_len, n_clus, class_name, train_size)
            test_model(raw_dataset, name, nums_len, n_clus, class_name, comp_methods, train_size)


def main():
    names = ['orders', 'partsupp', 'flight', 'nypd']
    nums_lens = [4, 4, 4, 9]
    n_clus_list = [4, 6, 8, 10, 12]
    comp_methods = ['gzip', 'lz4', 'zstd']
    class_names = ['DecisionTree', 'QDA', 'MLP', 'GaussianNB', 'LogisticRegression']
    train_sizes = [0.05, 0.1, 0.15, 0.2, 0.25]

    compare_clusters(names, nums_lens, n_clus_list, comp_methods)
    compare_classifiers(names, nums_lens, comp_methods, class_names)
    compare_train_sizes(names, nums_lens, comp_methods, train_sizes)


if __name__ == '__main__':
    main()
