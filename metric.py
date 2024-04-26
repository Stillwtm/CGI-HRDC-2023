import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score


def classification_metrics(y_true, y_pred):
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    f1 = f1_score(y_true, y_pred, average='macro')
    spe = np.mean(specificity(y_true, y_pred))
    return dict(qwk=qwk, f1=f1, spe=spe)


def specificity(y_true: np.array, y_pred: np.array, classes: set = None):
    if classes is None:
        classes = set(np.concatenate((np.unique(y_true), np.unique(y_pred))))
    specs = []
    for cls in classes:
        y_true_cls = np.array((y_true == cls), np.uint8)
        y_pred_cls = np.array((y_pred == cls), np.uint8)
        specs.append(recall_score(y_true_cls, y_pred_cls, pos_label=0))
    return specs
