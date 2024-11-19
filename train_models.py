import numpy as np
import pandas as pd
from preprocess_data import preprocess_data
from preprocess_df_electronics import preprocess_df_electronics
from preprocess_flight import preprocess_flight
from preprocess_nypd import preprocess_nypd
from preprocess_submissions import preprocess_submissions
from classifiers import split_train_test_set, train_and_save_model


def main():
    names = ['orders', 'partsupp', '100k_a', 'df_electronics', 'flight', 'nypd', 'Submissions']
    nums_lens = [4, 4, 4, 3, 4, 9, 7]
    n_clus_list = [10]
    class_names = ['GaussianNB']
    train_sizes = [0.8]
    for name, nums_len in zip(names, nums_lens):
        file_path = f'./datasets/{name}.csv'
        for n_clus in n_clus_list:
            print('Processing {} dataset with {} clusters...'.format(name, n_clus))
            if name in ('orders', 'partsupp', '100k_a'):
                with open(file_path) as f:
                    raw_dataset = f.readlines()
            else:
                if name == 'flight':
                    raw_dataset = pd.read_csv(file_path, delimiter='|')
                else:
                    raw_dataset = pd.read_csv(file_path, delimiter=',')
            clus_labels = pd.read_csv(f'./labels/{name}/{name}_{n_clus}_clus.csv', header=None)
            for train_size in train_sizes:
                if train_size == 1.0:
                    raw_train_data = raw_dataset
                    train_labels = clus_labels
                else:
                    raw_train_data, train_labels = split_train_test_set(raw_dataset, clus_labels, train_size, 0, 'insert', name)
                if name in ('orders', 'partsupp', '100k_a'):
                    train_data = preprocess_data(raw_train_data, name)
                elif name == 'df_electronics':
                    train_data = preprocess_df_electronics(raw_train_data)
                elif name == 'flight':
                    train_data = preprocess_flight(raw_train_data)
                elif name == 'nypd':
                    train_data = preprocess_nypd(raw_train_data)
                elif name == 'Submissions':
                    train_data = preprocess_submissions(raw_train_data)
                for class_name in class_names:
                    model_path, _ = train_and_save_model(np.array(train_data), train_labels, name, class_name, 'insert', train_size, nums_len, n_clus)

            # raw_train_data, train_labels = split_train_test_set(raw_dataset, clus_labels, 0.2, 0, 'no_overlap', name)
            # train_data = preprocess_submissions(raw_train_data)
            # for class_name in class_names:
            #     model_path, _ = train_and_save_model(np.array(train_data), train_labels, name, class_name, 'no_overlap', 0.2, nums_len)

            # raw_train_data, train_labels = split_train_test_set(raw_dataset, clus_labels, 0, 0.1, 'update', name)
            # train_data = preprocess_submissions(raw_train_data)
            # for class_name in class_names:
            #     model_path, _ = train_and_save_model(np.array(train_data), train_labels, name, class_name, 'update', 0.1, nums_len)


if __name__ == '__main__':
    main()
