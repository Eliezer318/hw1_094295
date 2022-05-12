from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from data import data_preprocessing
from train import train_model


def get_graphs():
    print('Training new classifier')
    all_data = data_preprocessing(amount_of_rows=40)
    depths = list(range(1, 15))
    f1_scores = []
    auc_scores = []
    for depth in range(1, 15):
        f1, roc = train_model(all_data, verbose=False, t=0.47, **{'tree_depth': depth, 'clf_or_f1': 'f1'})
        f1_scores.append(f1)
        auc_scores.append(roc)
        print(f'{depth=}, f1={f1_scores[-1]: .4f}, auc={auc_scores[-1]: .4f}')

    plt.plot(depths, f1_scores)
    plt.xlabel('Tree Depth')
    plt.ylabel('F1 Score')
    plt.title('F1 Score as function of tree depth')
    plt.show()

    plt.figure()
    plt.plot(depths, auc_scores)
    plt.xlabel('Tree Depth')
    plt.ylabel('Auc Score')
    plt.title('Auc Score as function of tree depth')
    plt.show()


def get_graphs_mlp():
    print('Training new classifier')
    all_data = data_preprocessing(amount_of_rows=40)
    depths = range(100, 501, 100)
    f1_scores = []
    auc_scores = []
    for hid in [500]:
        f1, auc = train_model(all_data, verbose=False, t=0.47, **{"model_type": 'mlp', "clf_or_f1": "f1", "hidden_dim": hid})
        f1_scores.append(f1)
        auc_scores.append(auc)
        print(f'{hid=}, f1={f1_scores[-1]: .4f}, auc={auc_scores[-1]: .4f}')

    plt.plot(depths, f1_scores)
    plt.xlabel('Hidden Layers Dimension')
    plt.ylabel('F1 Score')
    plt.title('F1 Score as function of Hidden Dim')
    plt.show()

    plt.figure()
    plt.plot(depths, auc_scores)
    plt.xlabel('Hidden Layers Dimension')
    plt.ylabel('Auc Score')
    plt.title('Auc Score as function of Hidden Dim')
    plt.show()


def check_f1_score(prediction_path: str):
    df = pd.read_csv(prediction_path)
    test_tables = sorted([f'/home/student/data/test/{path}' for path in os.listdir(f'/home/student/data/test')])
    path2id = lambda path: int(path.split("/")[-1].split(".")[-2].split('_')[-1])
    ss = {path2id(path): pd.read_csv(path, sep='|').SepsisLabel.max() for path in tqdm(test_tables)}
    pred = df.SepsisLabel.tolist()
    true = [ss[pid] for pid in df.Id]
    print(f'f1 score = {f1_score(true, pred): .4f} auc score is {roc_auc_score(true, pred): .4f}')
    ConfusionMatrixDisplay.from_predictions(true, pred)
    plt.show()


# check_f1_score('prediction.csv')
