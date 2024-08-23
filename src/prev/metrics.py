import enum
from typing import Optional

import torch
from expected_cost.ec import average_cost
from sklearn.metrics import f1_score, precision_recall_fscore_support, matthews_corrcoef, balanced_accuracy_score, \
    roc_auc_score


class Metric(enum.Enum):
    """Performance metrics."""
    MCC = 'MCC'
    ACCURACY = 'Accuracy'
    F1 = 'F1 Score'
    EC = "EC"
    EC_ADJUSTED = "EC (dep. prev.)"
    EC_NORM = "NEC"
    EC_NORM_ADJUSTED = "NEC (dep. prev.)"
    EC_NORM_EST = "NEC (est. prev.)"
    EC_EST = "EC (est. prev.)"
    AUROC = "AUROC"
    BALANCED_ACC = "Bal. Accuracy"


def accuracy(labels: torch.Tensor, preds: torch.Tensor) -> float:
    """Accuracy metric."""
    return (torch.sum(labels == preds) / len(labels)).item()


def compute_metric(metric: Metric,
                   labels: torch.Tensor,
                   preds: torch.Tensor,
                   min_class: int,
                   exact_priors: Optional[torch.Tensor] = None,
                   estimated_priors: Optional[torch.Tensor] = None) -> float:
    """
    Perform inference on prediction with a given metric.

    :param metric: the metric to calculate
    :param labels: the reference labels
    :param preds: predictions by a model (potentially after re-calibration)
    :param min_class: index of the minority class, necessary for binary tasks and F1-Score
    :param exact_priors: priors for expected cost (EC_ADJUSTED and EC_NORM_ADJUSTED)
    :param estimated_priors: priors for expected cost (EC_EST and EC_NORM_EST)
    :return: metric value
    """
    if metric == Metric.ACCURACY:
        return accuracy(labels, preds)
    elif metric == Metric.EC:
        return average_cost(labels, preds)
    elif metric == Metric.EC_NORM:
        return average_cost(labels, preds, adjusted=True)
    elif metric == Metric.EC_ADJUSTED:
        if exact_priors is None:
            raise ValueError('Priors not given for EC adjusted')
        return average_cost(labels, preds, priors=exact_priors.numpy())
    elif metric == Metric.EC_EST:
        if estimated_priors is None:
            raise ValueError('Priors not given for EC estimated')
        return average_cost(labels, preds, priors=estimated_priors.numpy())
    elif metric == Metric.EC_NORM_ADJUSTED:
        if exact_priors is None:
            raise ValueError('Priors not given for NEC adjusted')
        return average_cost(labels, preds, priors=exact_priors.numpy(), adjusted=True)
    elif metric == Metric.EC_NORM_EST:
        if estimated_priors is None:
            raise ValueError('Priors not given for NEC estimated')
        return average_cost(labels, preds, priors=estimated_priors.numpy(), adjusted=True)
    elif metric == Metric.F1:
        if len(torch.bincount(labels)) == 2:
            return f1_score(labels, preds, pos_label=min_class, zero_division=0)
        else:
            return precision_recall_fscore_support(labels, preds, average=None, zero_division=0)[2][min_class]
    elif metric == Metric.MCC:
        return matthews_corrcoef(labels, preds)
    elif metric == Metric.BALANCED_ACC:
        return balanced_accuracy_score(labels, preds)
    elif metric == Metric.AUROC:
        if len(torch.bincount(labels)) == 2:
            return roc_auc_score(labels, torch.nn.Softmax(dim=1)(preds)[:, -1])
        else:
            return roc_auc_score(labels, torch.nn.Softmax(dim=1)(preds), multi_class='ovr')
    else:
        raise ValueError('Metric value not implemented')


def compute_all_metrics(labels: torch.Tensor, logits: torch.Tensor, preds: torch.Tensor,
                        min_class: int, exact_priors: torch.Tensor, estimated_priors: torch.Tensor, metrics=Metric):
    """
    Computes values of the provided metrics.

    :param labels: tensor of true labels
    :param logits: tensor of logits (needed to compute AUROC)
    :param preds: tensor of predictions
    :param min_class: smallest class with respect to which F1 score is computed
    :param exact_priors: priors used to compute EC_ADJUSTED and NEC_ADJUSTED
    :parem estimated_priors: priors used to compute EC_EST and NEC_EST
    :param metrics: metrics to be computed
    :return results: dictionary of metric values
    """
    results = {}

    if not isinstance(preds, dict):
        preds = {m: preds for m in metrics}

    for metric in metrics:
        if metric == Metric.AUROC:
            results[metric] = compute_metric(metric=metric, labels=labels, preds=logits,
                                             min_class=min_class, exact_priors=exact_priors,
                                             estimated_priors=estimated_priors)
        else:
            results[metric] = compute_metric(metric=metric, labels=labels, preds=preds[metric],
                                             min_class=min_class, exact_priors=exact_priors,
                                             estimated_priors=estimated_priors)
    return results
