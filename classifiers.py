import os
import time
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_and_save_model(train_data, train_labels, name, class_name, train_size, nums_len, n_clus):
    classifiers = {
        'DecisionTree': DecisionTreeClassifier(),
        'QDA': QuadraticDiscriminantAnalysis(),
        'MLP': MLPClassifier(hidden_layer_sizes=(10,), max_iter=100),
        'GaussianNB': GaussianNB(),
        'LogisticRegression': LogisticRegression(max_iter=50)
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
        scaler_path = f'./models/{name}_{n_clus}/{train_size}_scaler.joblib'
        if not os.path.exists(scaler_path):
            joblib.dump(scaler, scaler_path)

    # Training
    # print(f'Training {class_name} classifier...')
    train_start = time.time()
    clf.fit(np.array(train_data, dtype=float), train_labels)
    train_end = time.time()
    train_time = train_end - train_start

    # Saving the trained model
    if not os.path.exists(f'./models/{name}_{n_clus}'):
        os.makedirs(f'./models/{name}_{n_clus}')
    model_path = f'./models/{name}_{n_clus}/{class_name}_{train_size}.joblib'
    if os.path.exists(model_path):
        os.remove(model_path)
    joblib.dump(clf, model_path)
    # print(f"Saved {class_name} classifier to {model_path}")

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
    # print(f'Predicting {model_path}...')
    test_start = time.time()
    predictions = clf.predict(test_data.astype(float))
    test_end = time.time()
    test_time = test_end - test_start

    # accuracy = np.mean(predictions == np.array(test_labels))

    return predictions, test_time
