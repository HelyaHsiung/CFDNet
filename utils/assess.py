import numpy as np


def calculate_kappa(pred_mask, gt_mask):
    """
    计算Kappa系数
    :param pred_mask: predict mask
    :param gt_mask: gt mask
    :return: Kappa Coefficient
    """
    """
    True Positive           False Positive
    False Negative          True Positive
    """
    confuse_matrix = np.zeros((2, 2))
    confuse_matrix[0, 0] = (np.logical_and(pred_mask, gt_mask)).sum()  # TP
    confuse_matrix[0, 1] = (np.logical_and(pred_mask, np.logical_not(gt_mask))).sum()  # FP
    confuse_matrix[1, 0] = (np.logical_and(np.logical_not(pred_mask), gt_mask)).sum()  # FN
    confuse_matrix[1, 1] = (np.logical_and(np.logical_not(pred_mask), np.logical_not(gt_mask))).sum()  # TN

    cross_item = confuse_matrix.sum(axis=0) @ confuse_matrix.sum(axis=1)
    KaPpa = (confuse_matrix.sum()*confuse_matrix.trace() - cross_item + 1e-6)/((confuse_matrix.sum())**2 - cross_item + 1e-6)
    return KaPpa


def calculate_iou(pred_mask, gt_mask):
    """
    计算IoU交叠率
    :param pred_mask: predict mask
    :param gt_mask: gt mask
    :return: IoU Coefficient
    """
    numerator = np.logical_and(pred_mask, gt_mask).sum()
    denominator = np.logical_or(pred_mask, gt_mask).sum()
    IoU = (numerator + 1e-6) / (denominator + 1e-6)
    return IoU


def calculate_accuracy(pred_mask, gt_mask):
    pred_mask = pred_mask.flatten()
    gt_mask = gt_mask.flatten()

    accuracy = np.sum(pred_mask == gt_mask) / len(pred_mask)
    return accuracy


def calculate_precision(pred_mask, gt_mask):
    pred_mask = pred_mask.flatten()
    gt_mask = gt_mask.flatten()

    true_positive = np.sum((pred_mask == 1) & (gt_mask == 1))
    false_positive = np.sum((pred_mask == 1) & (gt_mask == 0))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    return precision


def calculate_recall(pred_mask, gt_mask):
    pred_mask = pred_mask.flatten()
    gt_mask = gt_mask.flatten()

    true_positive = np.sum((pred_mask == 1) & (gt_mask == 1))
    false_negative = np.sum((pred_mask == 0) & (gt_mask == 1))

    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    return recall


def calculate_f1_score(pred_mask, gt_mask):
    precision = calculate_precision(pred_mask, gt_mask)
    recall = calculate_recall(pred_mask, gt_mask)

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return f1_score


def calculate_false_alarm(pred_mask, gt_mask):
    """
    False alarm rate = FP / (FP + TN)
    """
    # FP: 预测为目标但实际是背景
    fp = np.logical_and(pred_mask == 1, gt_mask == 0).sum()
    # TN: 预测为背景且实际是背景
    tn = np.logical_and(pred_mask == 0, gt_mask == 0).sum()

    denominator = fp + tn
    if denominator == 0:
        return 0.0  # 避免除以零
    return fp / denominator


def calculate_miss_rate(pred_mask, gt_mask):
    """
    Miss rate = FN / (TP + FN)
    """
    # FN: 实际是目标但预测为背景
    fn = np.logical_and(pred_mask == 0, gt_mask == 1).sum()
    # TP: 实际是目标且预测为目标
    tp = np.logical_and(pred_mask == 1, gt_mask == 1).sum()

    denominator = tp + fn
    if denominator == 0:
        return 0.0  # 避免除以零
    return fn / denominator
