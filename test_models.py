import numpy as np
import pandas as pd
from incremental_k_prototypes import incremental_k_prototypes
from preprocess_data import preprocess_data
from preprocess_flight import preprocess_flight
from preprocess_nypd import preprocess_nypd
from preprocess_submissions import preprocess_submissions
from preprocess_arrests import preprocess_arrests
from classifiers import load_and_test_model
from parallel_column_wise_compression import parallel_column_wise_compression


def main():
    names = ['Submissions']
    nums_lens = [7]
    class_names = ['DecisionTree', 'QDA', 'MLP', 'GaussianNB', 'LogisticRegression']
    comp_methods = ['gzip', 'lz4', 'zstd']
    n_clus = 10
    for name, nums_len in zip(names, nums_lens):
        data_path = f'./datasets/{name}.csv'
        comp_ratios_list = []
        raw_dataset = pd.read_csv(data_path, delimiter=',')
        test_data = preprocess_submissions(raw_dataset)
        for class_name in class_names:
            model_path = f'./models/{name}_{n_clus}/{class_name}_insert_0.8.joblib'
            scaler_path = f'./models/{name}_{n_clus}/insert_0.8_scaler.joblib'
            predictions = load_and_test_model(np.array(test_data), model_path, scaler_path, class_name, nums_len)
            predictions = np.concatenate(([0], predictions))
            for comp_method in comp_methods:
                comp_ratio, comp_time = parallel_column_wise_compression(n_clus, data_path, predictions, comp_method, delimiter=',', random_state=0)
                # comp_ratios_list.append(comp_ratio)
        # gzip_list = comp_ratios_list[::3]
        # lz4_list = comp_ratios_list[1::3]
        # zstd_list = comp_ratios_list[2::3]
        # with open('Comp_Ratios_Lists.txt', 'a') as f:
        #     f.write(f'{name}_gzip_comp_ratios_list = {gzip_list}\n')
        #     f.write(f'{name}_lz4_comp_ratios_list = {lz4_list}\n')
        #     f.write(f'{name}_zstd_comp_ratios_list = {zstd_list}\n\n')


if __name__ == '__main__':
    main()
