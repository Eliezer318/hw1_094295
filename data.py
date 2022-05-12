from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import f1_score
import statsmodels.api as sm


import pandas as pd
import numpy as np

from tqdm import tqdm
import pickle
import os


def cut_first_rows(patient_df: pd.DataFrame, amount_of_rows=10, end=None) -> pd.DataFrame:
    """
    :param end:
    :param patient_df: patient dataframe
    :param amount_of_rows: amount of rows to keep before the first true sepsis label
    :return: remove rows with 0 that are not 10 before last row or first row with label sepsis=1.
    """
    idx = patient_df.shape[0] if end is None else end
    return patient_df.iloc[max(idx - amount_of_rows, 0):idx]


#  take statistics per feature and concatenate into vector
def trend_and_features(patient_df: pd.DataFrame, amount_of_rows=10, is_test=False) -> (np.ndarray, int):
    assert patient_df.shape[0] > 0
    if is_test:
        try:
            label = patient_df.SepsisLabel.max()  # todo change to -1
            end_idx = patient_df.SepsisLabel.argmax() + 1 if (label == 1) else patient_df.shape[0]  # find last relevant
        except Exception as e:
            label, end_idx = -1, patient_df.shape[0]
            print(e)
    else:
        label = patient_df.SepsisLabel.max()
        end_idx = patient_df.SepsisLabel.argmax() + 1 if (label == 1) else patient_df.shape[0]  # find last relevant row
    fixed_values = np.array([patient_df.loc[0, 'Age'], patient_df.loc[0, 'Gender']])
    patient_df = patient_df.drop(['Age', 'Gender', 'SepsisLabel'], axis=1)  # remove fixed values and label
    trend = np.vstack([sm.tsa.seasonal_decompose(patient_df.loc[:, [col]], model='additive', period=1, extrapolate_trend=1).trend for col in patient_df.columns]).T
    patient_values = cut_first_rows(patient_df, amount_of_rows, end=end_idx).values
    trend = cut_first_rows(pd.DataFrame(trend), amount_of_rows, end=end_idx).values
    assert trend.shape[0] > 0
    mean, std = patient_values.mean(axis=0), patient_values.std(axis=0)
    t_min, t_max = trend.min(axis=0), trend.max(axis=0)
    first, last, diff, last_diff = trend[0], trend[-1], trend[-1] - trend[0], trend[-1] - trend[-2 if len(trend) >= 2 else -1]
    return np.concatenate([np.vstack((first, last, diff, mean, t_max, t_min, std, last_diff)).flatten('F'), fixed_values]), label


def get_imputer():
    train_tables = sorted([f'/home/student/data/train/{path}' for path in os.listdir(f'/home/student/data/train')])
    path_imputer = f'cache/imputer.pkl'
    if os.path.isfile(path_imputer):
        print('Using cached Imputer')
        imputer = pickle.load(open(path_imputer, 'rb'))
    else:
        print('Training a new Imputer')
        full_data = pd.concat([pd.read_csv(path, sep='|').interpolate(limit_direction='both', axis=0) for path in train_tables])
        imputer = IterativeImputer(random_state=0)
        imputer.fit(full_data.drop(['SepsisLabel', 'Unit1', 'Unit2', 'ICULOS'], axis=1).values)
        pickle.dump(imputer, open(path_imputer, 'wb'))
    return imputer


def data_preprocessing(amount_of_rows=10) -> dict:
    # for on train, test, for on each psv file and after interpolation.
    path_all_data = f'cache/data_{amount_of_rows=}.pkl'
    if os.path.isfile(path_all_data):  # if given configuration of data exist - return it.
        return pickle.load(open(path_all_data, 'rb'))

    train_tables = sorted([f'/home/student/data/train/{path}' for path in os.listdir(f'/home/student/data/train')])
    test_tables = sorted([f'/home/student/data/test/{path}' for path in os.listdir(f'/home/student/data/test')])

    # train imputer on all the data.
    imputer = get_imputer()

    all_data = {}
    for state, tables_path in zip(['train', 'test'], [train_tables, test_tables]):
        features_list, labels_list, ids_list = [], [], []
        for path in tqdm(tables_path, desc=f'Create features for {state} set'):
            df = pd.read_csv(path, sep='|')
            # if there is no SepsisLabel we will put -1
            sepsis = -np.ones(df.shape[0]) if 'SepsisLabel' not in df.columns else df['SepsisLabel'].values
            df = df.drop(['SepsisLabel', 'Unit1', 'Unit2', 'ICULOS'], axis=1)
            values = imputer.transform(df.interpolate(limit_direction='both', axis=0).values)
            df = pd.DataFrame(values.round(2), columns=df.columns)
            df['SepsisLabel'] = sepsis
            assert not df.isnull().values.any(), 'There is null value!'

            # extract features
            features, label = trend_and_features(df, amount_of_rows=amount_of_rows, is_test=(state == 'test'))
            patient_id = path.split("/")[-1].split(".")[-2].split('_')[-1]
            features_list.append(features)
            labels_list.append(label)
            ids_list.append(patient_id)

        all_data[state] = (np.array(features_list), np.array(labels_list), np.array(ids_list))
    pickle.dump(all_data, open(path_all_data, 'wb'))
    return all_data


# check_f1_score('prediction.csv')
