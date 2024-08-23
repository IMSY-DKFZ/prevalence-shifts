import enum
from typing import Type

import numpy as np
import quapy as qp
import torch
from quapy.method.aggregative import ACC
from quapy.method.base import BaseQuantifier
from sklearn.base import BaseEstimator, ClassifierMixin


class QuantificationMethod(enum.Enum):
    """Quantification is the process of estimating prevalences in unlabeled data."""
    CC = "CC"
    EMQ = "EMQ"
    BBSE = "BBSE"
    ACC = "ACC"
    PCC = "PCC"
    EM = "EM"
    DMx = 'DMx'
    PACC = "PACC"
    KDEyCS = 'KDEyCS'
    KDEyHD = 'KDEyHD'
    KDEyML = 'KDEyML'
    DMy = 'HDy'


def absolute_error(true_prev, estimated_prev):
    """Absolute error (L1 norm of the difference between true and estimated) - a common quantification metric."""
    return np.abs(true_prev - estimated_prev).sum()


def compute_w_hat_and_mu_hat(train_labels, train_predictions, test_predictions, epsilon=1e-8):
    """
    Compute ŵ (w hat) and μ̂ (mu hat) using the BBSE method.

    :param train_labels: (array-like) True labels of the training set
    :param train_predictions: (array-like) Model predictions on the training set
    :param test_predictions: (array-like) Model predictions on the test set
    :return estimated ŵ and estimated μ̂
    """
    # Get the number of classes
    num_classes = len(np.unique(train_labels))

    # Compute the confusion matrix C_y,y_hat
    C_y_yhat = np.zeros((num_classes, num_classes))
    for true_label, pred_label in zip(train_labels, train_predictions):
        C_y_yhat[true_label, pred_label] += 1

    # Normalize the confusion matrix
    C_y_yhat = C_y_yhat / len(train_labels)

    # Compute μ̂_y_hat (the average predictions on the test set)
    mu_y_hat = np.bincount(test_predictions, minlength=num_classes) / len(test_predictions)

    # Compute ŵ by solving the linear system
    w_hat = np.linalg.solve(C_y_yhat.T, mu_y_hat)

    # Compute the empirical distribution of labels in the training set
    p_y = np.bincount(train_labels, minlength=num_classes) / len(train_labels)

    # Compute μ̂
    mu_hat = p_y * np.maximum(w_hat, epsilon)

    return w_hat, mu_hat


def adjust_priors_qp(train_probabilities: torch.Tensor, train_labels: torch.Tensor, test_probabilities: torch.Tensor,
                     test_labels: torch.Tensor, method: Type[BaseQuantifier] = ACC, epsilon: float = 1e-8):
    """
    Perform quantification using quapy library.

    :param train_probabilities: training / validation probabilities (softmax logits), will be used by some quantifiers
    :param train_labels: training / validation labels, will be used by some quantifiers
    :param test_probabilities: probabilities on the split to predict prevalences upon
    :param test_labels: (unused)
    :param method: quapy quantifier class to use
    :param epsilon: minimum estimated prevalence of a class, avoids predicting absolute zero, which might break code
    :return: estimated prevalences using provided quapy quantifier
    """
    # convert data to qp format
    dev_data = qp.data.LabelledCollection(train_probabilities,
                                          train_labels)
    app_data = qp.data.LabelledCollection(test_probabilities,
                                          test_labels)
    dset = qp.data.base.Dataset(training=dev_data, test=app_data)

    identity_class = IdentityClassifier(train_probabilities.shape[1])
    model = method(identity_class)
    model.fit(dset.training)
    priors = model.quantify(dset.test.instances)
    priors[priors == 0] += epsilon
    return priors


class IdentityClassifier(BaseEstimator, ClassifierMixin):
    """Helping dummy wrapper to use quapy."""

    def __init__(self, n_classes=2):
        self.n_classes = n_classes
        self.classes_ = np.arange(n_classes)

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        return X

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
