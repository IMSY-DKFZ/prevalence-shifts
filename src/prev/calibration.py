import enum
from typing import Dict, Union, Tuple

import torch
from MetricsReloaded.metrics.calibration_measures import CalibrationMeasures
from psrcal.calibration import calibrate, AffineCalLogLoss
from psrcal.losses import Brier

from src.prev.data_loading import Kind, Split


class CalibrationMethod(enum.Enum):
    """Variants for handling re-calibration."""
    NONE = 'No re-calibration'
    TEMPERATURE_SCALING = 'TempScal'
    AFFINE = 'Affine'
    ADAPTED_TRAIN_WEIGHTS = 'Retraining (dep. prev.)'
    TEMPERATURE_SCALING_REWEIGHTED = 'TempScal (dep. prev.)'
    ADAPTED_TRAIN_WEIGHTS_AND_TEMPERATURE_SCALING_REWEIGHTED = 'Retraining<br>+ TempScal (dep. prev.)'
    ADAPTED_TRAIN_WEIGHTS_AND_AFFINE_SCALING_REWEIGHTED = 'Retraining<br>+ Affine (dep. prev.)'
    ADAPTED_TRAIN_WEIGHTS_ACC = 'Retraining (est. prev.)'
    ADAPTED_TRAIN_WEIGHTS_AND_TEMPERATURE_SCALING_REWEIGHTED_ACC = 'Retraining<br>+ TempScal (est. prev.)'
    ADAPTED_TRAIN_WEIGHTS_AND_AFFINE_SCALING_REWEIGHETD_ACC = 'Retraining<br>+ Affine (est. prev.)'
    AFFINE_REWEIGHTED = 'Affine (dep. prev.)'
    AFFINE_ACC = 'Affine (est. prev.)'


def calibrate_logits_fast(data: Dict[Kind, Dict[Split, torch.Tensor]],
                          calibration: CalibrationMethod = CalibrationMethod.NONE,
                          prior=None) -> Union[
    Tuple[torch.Tensor, Dict[Split, torch.Tensor]], Dict[Split, torch.Tensor]]:
    """
    Calibrates logits according to strategy. Max return prior estimations if requested. Faster than previous
    implementation.

    :param data: labels and logits for various splits, see data_loading.get_values for details
    :param calibration: CalibrationMethod
    :param prior: Instead of weights please provide estimated priors for TEMPERATURE_SCALING_REWEIGHTED and AFFINE_REWEIGHTED
    :return:
    """
    # check kwargs
    if calibration not in CalibrationMethod:
        raise ValueError(f'provided incorrect calibration method {calibration}')
    if calibration == CalibrationMethod.ADAPTED_TRAIN_WEIGHTS:
        raise ValueError(f'adapted train weights do not need re-calibration')
    if calibration in [CalibrationMethod.NONE, CalibrationMethod.TEMPERATURE_SCALING, CalibrationMethod.AFFINE]:
        if prior is not None:
            raise ValueError(f'no prior needed for {calibration}')
    # simple case
    if calibration == CalibrationMethod.NONE:
        return {Split.DEV_TEST: data[Kind.LOGITS][Split.DEV_TEST], Split.APP_TEST: data[Kind.LOGITS][Split.APP_TEST]}
    # calibrate parameters
    use_bias = 'AFFINE' in calibration.name
    stacked_test_logits = torch.cat([data[Kind.LOGITS][Split.DEV_TEST], data[Kind.LOGITS][Split.APP_TEST]])
    # do calibration optimization
    cal_test_logits, _ = calibrate(trnscores=data[Kind.LOGITS][Split.DEV_CAL],
                                   trnlabels=data[Kind.LABELS][Split.DEV_CAL],
                                   tstscores=stacked_test_logits,
                                   calclass=AffineCalLogLoss,
                                   bias=use_bias,
                                   priors=prior, quiet=True)
    # return format
    dev_test_calibrated, app_test_calibrated = torch.split(cal_test_logits.detach(),
                                                           split_size_or_sections=[
                                                               data[Kind.LOGITS][Split.DEV_TEST].size(0),
                                                               data[Kind.LOGITS][Split.APP_TEST].size(0)])
    calibrated_logits = {Split.DEV_TEST: dev_test_calibrated, Split.APP_TEST: app_test_calibrated}
    return calibrated_logits


def calc_calibration_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Calculates calibration metrics.

    :param logits: model logits
    :param labels: sample labels
    :return: dict with multiple calibration metrics and their values
    """
    pred_probabilities = torch.nn.functional.softmax(logits, dim=1).numpy()
    measures = CalibrationMeasures(pred_proba=pred_probabilities, ref=labels.numpy(), measures=None)
    cwce = measures.class_wise_expectation_calibration_error()
    rbs = measures.root_brier_score()
    log_probs = torch.log(torch.nn.functional.softmax(logits, dim=1))
    brier = Brier(log_probs, labels, norm=False).item()
    brier_norm = Brier(log_probs, labels, norm=True).item()
    return {'cwce': cwce, 'rbs': rbs, 'bs': brier, 'nbs': brier_norm}

