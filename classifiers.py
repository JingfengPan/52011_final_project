import os
import time
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def split_train_test_set(features, labels, train_size, test_size, case, name, n_clus=10):
    # features = np.array(dataset)
    if case == 'no_overlap':
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, train_size=train_size, random_state=0)
    elif case == 'insert':
        features_train, _, labels_train, _ = train_test_split(features, labels, train_size=train_size, random_state=0)
        features_test = features
        # labels_test = labels
    elif case == 'update':
        features_same, features_diff, labels_same, labels_diff = train_test_split(features, labels, test_size=test_size*2, random_state=0)
        features_before, features_after, labels_before, labels_after = train_test_split(features_diff, labels_diff, test_size=0.5, random_state=0)
        features_train = pd.concat([features_same, features_before])
        features_test = pd.concat([features_same, features_after])
        labels_train = pd.concat([labels_same, labels_before])
        # labels_test = pd.concat([labels_same, labels_after])

        # features_train = np.concatenate((features_same, features_before))
        # features_test = np.concatenate((features_same, features_after))
        # labels_train = np.concatenate((labels_same, labels_before))
        # labels_test = np.concatenate((labels_same, labels_after))

    # if case == 'update':
    #     percentage = test_size
    # else:
    #     percentage = train_size

    # if not os.path.exists(f'./data/{name}_{n_clus}'):
    #     os.makedirs(f'./data/{name}_{n_clus}')
    # features_test_output = pd.DataFrame(features_test)
    # features_test_output_path = f'./data/{name}_{n_clus}/{name}_{case}_{percentage}.csv'
    # if os.path.exists(features_test_output_path):
    #     os.remove(features_test_output_path)
    # features_test_output.to_csv(features_test_output_path, index=False)

    return features_train, labels_train   # features_test, labels_test


def train_and_save_model(train_data, train_labels, name, class_name, case, percentage, nums_len, n_clus):
    classifiers = {
        'DecisionTree': DecisionTreeClassifier(),
        'QDA': QuadraticDiscriminantAnalysis(),
        'MLP': MLPClassifier(),  # hidden_layer_sizes=(10,), max_iter=100
        'GaussianNB': GaussianNB(),
        'LogisticRegression': LogisticRegression()  # max_iter=50
    }

    clf = classifiers[class_name]
    if class_name in ('QDA', 'MLP', 'LogisticRegression'):
        # Initialize the StandardScaler
        scaler = StandardScaler()

        # Scale the numeric data only
        num_scaled = scaler.fit_transform(train_data[:, :nums_len])

        # Concatenate the scaled numeric data with the categorical data
        cat_data = train_data[:, nums_len:]
        train_data = np.hstack((num_scaled, cat_data))

        if not os.path.exists(f'./models/{name}_{n_clus}'):
            os.makedirs(f'./models/{name}_{n_clus}')
        scaler_path = f'./models/{name}_{n_clus}/{case}_{percentage}_scaler.joblib'
        if not os.path.exists(scaler_path):
            joblib.dump(scaler, scaler_path)


    # Training
    print(f'Training {class_name} classifier...')
    train_start = time.time()
    clf.fit(np.array(train_data, dtype=float), train_labels)
    train_end = time.time()
    train_time = train_end - train_start

    # Saving the trained model
    if not os.path.exists(f'./models/{name}_{n_clus}'):
        os.makedirs(f'./models/{name}_{n_clus}')
    model_path = f'./models/{name}_{n_clus}/{class_name}_{case}_{percentage}.joblib'
    if os.path.exists(model_path):
        os.remove(model_path)
    joblib.dump(clf, model_path)
    print(f"Saved {class_name} classifier to {model_path}")

    return model_path, train_time


def load_and_test_model(test_data, model_path, scaler_path, class_name, nums_len):
    # Load the model
    clf = joblib.load(model_path)
    if class_name in ('QDA', 'MLP', 'LogisticRegression'):
        scaler = joblib.load(scaler_path)

        # Scale the numeric data only
        num_scaled = scaler.transform(test_data[:, :nums_len])

        # Concatenate the scaled numeric data with the categorical data
        cat_data = test_data[:, nums_len:]
        test_data = np.hstack((num_scaled, cat_data))

    # Testing
    print(f'Predicting {model_path}...')
    # test_start = time.time()
    predictions = clf.predict(test_data.astype(float))
    # test_end = time.time()
    # test_time = test_end - test_start

    # accuracy = np.mean(predictions == np.array(test_labels))

    return predictions
