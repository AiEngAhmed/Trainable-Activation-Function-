from sklearn.metrics import roc_auc_score
import numpy as np
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import *

def evaluation(true_Y, pred_Y):
    brier_score = brier_score_loss(true_Y, pred_Y)
    auc_score = roc_auc_score(true_Y, pred_Y)
    pred_Y = np.where(np.asarray(pred_Y) >= 0.5, 1, 0)

    f_score_weighted = f1_score(true_Y, pred_Y, average='weighted')
    f_score_micro = f1_score(true_Y, pred_Y, average='micro')
    f_score_macro = f1_score(true_Y, pred_Y, average='macro')
    acc = accuracy_score(true_Y, pred_Y)
    gmean = geometric_mean_score(true_Y, pred_Y, average='weighted')
    return auc_score, acc, gmean, brier_score #,f_score_micro, f_score_macro, f_score_weighted,