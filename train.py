from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def print_metrics(state, clf, features, labels, t=0.5, verbose=True):
    pred = (clf.predict_proba(features)[:, 1] >= t).astype(int)
    f1 = f1_score(labels, pred)
    acc = accuracy_score(labels, pred)
    precision = precision_score(labels, pred)
    recall = recall_score(labels, pred)
    auc = roc_auc_score(labels, pred)
    if verbose:
        print(f'{state} f1 score {f1: .4f}')
        print(f'{state} precision {precision: .4f}')
        print(f'{state} recall {recall: .4f}')
        print(f'{state} accuracy {acc: .4f}')
        print()
    return f1, auc, acc, precision, recall


def train_model(data: dict, t=0.5, verbose=True, **kwargs):
    model_type = kwargs.get('model_type', 'rf')
    train_features, train_labels, _ = data['train']
    test_features, test_labels, _ = data['test']
    if model_type == 'rf':
        clf = RandomForestClassifier(n_estimators=500, max_depth=kwargs.get('tree_depth', 8), class_weight='balanced')
    else:
        hid = kwargs.get('hidden_dim', 100)
        clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(hid, hid), max_iter=1000, tol=1e-4))
    clf.fit(train_features, train_labels)
    print_metrics('train', clf, train_features, train_labels, t, verbose)
    f1, roc = print_metrics('test', clf, test_features, test_labels, t, verbose)[:2]
    if kwargs.get('clf_or_f1', 'clf') == 'clf':
        return clf
    else:
        return f1, roc
