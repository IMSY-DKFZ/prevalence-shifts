from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef, f1_score, balanced_accuracy_score

from src.prev.utils import Metric
from expected_cost.ec import average_cost


def find_sensitivity_threshold(labels: torch.Tensor, logits: torch.Tensor,
                               min_class: int, step: float = 0.01) -> float:
    """
    Finds threshold for .95 sensitivity
    """
    threshold = 0
    soft_scores = F.softmax(logits, dim=1)
    sensitivity = 1
    while sensitivity > 0.95:
        threshold_preds = soft_scores[:, min_class] > threshold
        P = torch.sum(labels == min_class)
        TP = torch.sum((labels == min_class) & (threshold_preds == 1))
        sensitivity = TP / P
        threshold += step
    return threshold - 2 * step


def accuracy(labels: torch.Tensor, preds: torch.Tensor) -> float:
    return torch.sum(labels == preds) / len(labels)


def find_best_threshold(labels: torch.Tensor, logits: torch.Tensor, min_class: int = 0,
                        metric: Metric = Metric.ACCURACY, step: float = 0.01,
                        priors: Optional[np.ndarray] = None) -> float:
    """Compute threshold for a given metric

    :param labels: true labels
    :type labels: torch.Tensor
    :param logits: logit values
    :type logits: torch.Tensor
    :param min_class: index of smallest class used to compute F1 score, defaults to 0
    :type min_class: int, optional
    :param metric: metric to be computed, defaults to Metric.ACCURACY
    :type metric: Metric, optional
    :param step: step size in threshold sweep, defaults to 0.01
    :type step: float, optional
    :param priors: priors to be used in EC computation, defaults to None
    :type priors: Optional[np.ndarray], optional
    :raises RuntimeError: raised when non binary task provided
    :raises ValueError: priors provided for metric which does not require them
    :return: threshold value
    :rtype: float
    """
    if logits.size(1) > 2:
        raise RuntimeError(f'Was given a non-binary task (C={logits.size(1)} for threshold search!')
    if metric != Metric.EC_ADJUSTED and priors:
        raise ValueError('Priors given for metric without weighing!')
    threshold = 0
    if (metric == Metric.ACCURACY) | (metric == Metric.MCC) | (metric == Metric.F1):
        best_value = -1
    else:
        best_value = 1000
    logits = F.softmax(logits, dim=1)[:, 0]
    for t in torch.arange(0, 1 + step, step):
        preds = (logits < t)
        if metric == Metric.EC or metric == Metric.EC_ADJUSTED or metric == Metric.EC_NORM:
            value = average_cost(labels, preds, priors=priors)
            if best_value > value:
                threshold = t
                best_value = value
        else:
            if metric == Metric.ACCURACY:
                value = accuracy(labels, preds)
            elif metric == Metric.MCC:
                value = matthews_corrcoef(labels, preds)
            elif metric == Metric.F1:
                value = f1_score(labels, preds, pos_label=min_class)
            if best_value < value:
                threshold = t
                best_value = value
    return threshold.item()


def find_best_thresholds(labels: torch.Tensor, logits: torch.Tensor, min_class: int = 0,
                         step: float = 0.01, priors: Optional[np.ndarray] = None,
                         est_priors: Optional[np.array] = None) -> Dict[Metric, float]:
    """Compute thresholds for all metrics in a single pass

    :param labels: label values
    :type labels: torch.Tensor
    :param logits: logits used to compute thresholds
    :type logits: torch.Tensor
    :param min_class: smallest class used for computing F1 score, defaults to 0
    :type min_class: int, optional
    :param step: step size in threshold sweep, defaults to 0.01
    :type step: float, optional
    :param priors: priors for computing EC_ADJUSTED, defaults to None
    :type priors: Optional[np.ndarray], optional
    :param est_priors: priors for computing EC_EST, defaults to None
    :type est_priors: Optional[np.array], optional
    :raises RuntimeError: error raised if non binary task given
    :return: dictionary of threshold values per metric
    :rtype: Dict[Metric, float]
    """
    if logits.size(1) > 2:
        raise RuntimeError(f'Was given a non-binary task (C={logits.size(1)} for threshold search!')
    # initialize optimal metric values
    acc_threshold = 0.
    f1_threshold = 0.
    mcc_threshold = 0.
    ec_threshold = 0.
    ec_adjusted_threshold = 0.
    ec_est_threshold = 0.
    balanced_acc_threshold = 0.
    best_mcc_value = -1
    best_acc_value = 0
    best_f1_value = 0
    best_ec_value = 1
    best_ec_adjusted_value = 1
    best_ec_est_value = 1
    best_balanced_acc_value = 0

    logits = F.softmax(logits, dim=1)[:, 0]
    # sweep threshold value to find optimum for each metric
    for t in torch.arange(0, 1 + step, step):
        preds = (logits < t)
        value = average_cost(labels, preds)
        if best_ec_value > value:
            ec_threshold = t.item()
            best_ec_value = value
        value = average_cost(labels, preds, priors=priors)
        if best_ec_adjusted_value > value:
            ec_adjusted_threshold = t.item()
            best_ec_adjusted_value = value
        value = average_cost(labels, preds, priors=est_priors)
        if best_ec_est_value > value:
            ec_est_threshold = t.item()
            best_ec_est_value = value
        value = accuracy(labels, preds)
        if best_acc_value < value:
            acc_threshold = t.item()
            best_acc_value = value
        value = matthews_corrcoef(labels, preds)
        if best_mcc_value < value:
            mcc_threshold = t.item()
            best_mcc_value = value
        value = f1_score(labels, preds, pos_label=min_class)
        if best_f1_value < value:
            f1_threshold = t.item()
            best_f1_value = value
        value = balanced_accuracy_score(labels, preds)
        if best_balanced_acc_value < value:
            balanced_acc_threshold = t.item()
            best_balanced_acc_value = value
    return {Metric.EC: ec_threshold, Metric.EC_ADJUSTED: ec_adjusted_threshold, Metric.ACCURACY: acc_threshold,
            Metric.MCC: mcc_threshold, Metric.F1: f1_threshold,
            Metric.EC_NORM: ec_threshold, Metric.EC_NORM_ADJUSTED: ec_adjusted_threshold,
            Metric.EC_EST: ec_est_threshold, Metric.EC_NORM_EST: ec_est_threshold,
            Metric.BALANCED_ACC: balanced_acc_threshold}
