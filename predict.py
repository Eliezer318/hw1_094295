from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import sys
import os

from data import get_imputer, trend_and_features, data_preprocessing
from train import train_model


def get_classifier(clf_path):
    if os.path.exists(clf_path):
        print(f'Used cached classifier {clf_path}')
        return pickle.load(open(clf_path, 'rb'))
    print('Training new classifier')
    all_data = data_preprocessing(amount_of_rows=40)
    clf = train_model(all_data, verbose=True, t=0.47)
    pickle.dump(clf, open(clf_path, 'wb'))
    return clf


def get_test_data(folder_path):
    imputer = get_imputer()
    features_list, ids_list = [], []
    for file_name in tqdm(sorted(os.listdir(folder_path)), desc='Gathering the data from the test folder'):
        df = pd.read_csv(f'{folder_path}/{file_name}', sep='|')
        # if there is no SepsisLabel we will put -1
        sepsis = -np.ones(df.shape[0]) if 'SepsisLabel' not in df.columns else df['SepsisLabel'].values
        df['SepsisLabel'] = sepsis
        df = df.drop(['SepsisLabel', 'Unit1', 'Unit2', 'ICULOS'], axis=1)
        values = imputer.transform(df.interpolate(limit_direction='both', axis=0).values)
        df = pd.DataFrame(values.round(2), columns=df.columns)
        df['SepsisLabel'] = sepsis
        features, label = trend_and_features(df, amount_of_rows=40, is_test=True)
        patient_id = file_name.split('.')[0].split('_')[-1]
        features_list.append(features)
        ids_list.append(patient_id)
    return np.array(features_list), np.array(ids_list)

os.makedirs('cache', exist_ok=True)
all_features, ids = get_test_data(folder_path=sys.argv[1])
clf: RandomForestClassifier = get_classifier(clf_path='cache/random_forest.pkl')
answers = {'Id': ids.reshape(-1), 'SepsisLabel': clf.predict(all_features).astype(int).reshape(-1)}
answers = {'Id': ids.reshape(-1), 'SepsisLabel': (clf.predict_proba(all_features)[:, 1] >= 0.47).astype(int).reshape(-1)}
pd.DataFrame(answers)[['Id', 'SepsisLabel']].to_csv('prediction.csv', index=False)