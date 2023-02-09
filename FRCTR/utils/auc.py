# -*- coding: UTF-8 -*-
"""
@project: RefineCTR
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def get_auc(y_labels, y_scores):
    auc = roc_auc_score(y_labels, y_scores)
    print('AUC calculated by sklearn tool is {}'.format(auc))
    return auc


def calculate_auc_func1(y_labels, y_scores):
    pos_sample_ids = [i for i in range(len(y_labels)) if y_labels[i] == 1]
    neg_sample_ids = [i for i in range(len(y_labels)) if y_labels[i] == 0]

    sum_indicator_value = 0
    for i in pos_sample_ids:
        for j in neg_sample_ids:
            if y_scores[i] > y_scores[j]:
                sum_indicator_value += 1
            elif y_scores[i] == y_scores[j]:
                sum_indicator_value += 0.5

    auc = sum_indicator_value / (len(pos_sample_ids) * len(neg_sample_ids))
    print('AUC calculated by function1 is {:.2f}'.format(auc))
    return auc


def calculate_auc_func2(y_labels, y_scores):
    samples = list(zip(y_scores, y_labels))
    print(samples)
    rank = [(values2, values1) for values1, values2 in sorted(samples, key=lambda x: x[0])]
    print(rank)
    pos_rank = [i + 1 for i in range(len(rank)) if rank[i][0] == 1]
    print(pos_rank)
    pos_cnt = np.sum(y_labels == 1)
    neg_cnt = np.sum(y_labels == 0)
    auc = (np.sum(pos_rank) - pos_cnt * (pos_cnt + 1) / 2) / (pos_cnt * neg_cnt)
    print('AUC calculated by function2 is {:.2f}'.format(auc))
    return auc


if __name__ == '__main__':
    y_labels = np.array([1, 1, 0, 0, 0])
    y_scores = np.array([1, 0.8, 0.2, 0.4, 0.5])
    calculate_auc_func2(y_labels, y_scores)
    print(roc_auc_score(y_labels, y_scores))
