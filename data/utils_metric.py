from cProfile import label
from unittest import result
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, average_precision_score
from scipy.stats import norm

def get_confusion_matrix():
    label_path = r''
    pred_path = r''
    label_all = np.load(label_path)
    pred_all = np.load(pred_path)
    for i in range(4):
        label = np.array(label_all[:, i], dtype=np.uint8).tolist()
        pred = np.array(pred_all[:, i] > 0.5, dtype=np.uint8).tolist()
        tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
        auprc = average_precision_score(label, pred)
        print(auprc)

def delong_test(auc_1, auc_2, m):
    v = 1/(2*m) * (auc_1 * (1-auc_1) + auc_2 * (1-auc_2) + (auc_1 - auc_2) ** 2)
    z = (auc_1 - auc_2) / np.sqrt(v)
    p_value = 2 * (1 - norm.cdf(abs(z)))
    print(z)
    print(p_value)



if __name__ == '__main__':
    pass
