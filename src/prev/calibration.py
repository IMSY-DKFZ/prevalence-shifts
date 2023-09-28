import enum
from typing import Dict, Union, Optional, Tuple

import torch
from MetricsReloaded.metrics.calibration_measures import CalibrationMeasures

from src.prev.data_loading import Kind, Split
from psrcal.calibration import calibrate, AffineCalLogLoss
from psrcal.losses import Brier

VERBOSE_CALIBRATION = False


class CalibrationMethod(enum.Enum):
    NONE = 'None'
    TEMPERATURE_SCALING = 'TempScal'
    AFFINE = 'Affine'
    TEMPERATURE_SCALING_ESTIMATED = 'TempScal (est. prev.)'
    TEMPERATURE_SCALING_REWEIGHTED = 'TempScal (dep. prev.)'
    AFFINE_ESTIMATED = 'Affine (est. prev.)'
    AFFINE_ESTIMATED_005 = 'Affine (est. prev.) std=0.05'
    AFFINE_ESTIMATED_0075 = 'Affine (est. prev.) std=0.075'
    AFFINE_ESTIMATED_01 = 'Affine (est. prev.) std=0.1'
    AFFINE_REWEIGHTED = 'Affine (dep. prev.)'


def missmatch_priors(priors: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    """
    Method to perturb priors, mimics uncertainty in application.

    :param priors: exact priors
    :param std: degree of perturbation
    :return:
    """
    guess = torch.normal(priors, std=std)
    # make sure at least one percent for each class
    guess = torch.clamp(guess, min=0.01)
    # normalize
    return guess / guess.sum()


def calibrate_logits_fast(data: Dict[Kind, Dict[Split, torch.Tensor]],
                          calibration: CalibrationMethod = CalibrationMethod.NONE,
                          prior=None, return_estimate=False, std: Optional[int] = None) -> Union[
    Tuple[torch.Tensor, Dict[Split, torch.Tensor]], Dict[Split, torch.Tensor]]:
    """
    Calibrates logits according to strategy. Max return prior estimations if requested. Faster than previous
    implementation.

    :param data: labels and logits for various splits, see data_loading.get_values for details
    :param calibration: CalibrationMethod
    :param prior: Instead of weights please provide estimated priors for TEMPERATURE_SCALING_REWEIGHTED and AFFINE_REWEIGHTED
    :param return_estimate: only for TEMPERATURE_SCALING_ESTIMATED and AFFINE_ESTIMATED, returns estimated priors as well
    :param std: only for TEMPERATURE_SCALING_ESTIMATED and AFFINE_ESTIMATED, determines search depth, defaults to 2
    :return:
    """
    # check kwargs
    if calibration not in CalibrationMethod:
        raise ValueError(f'provided incorrect calibration method {calibration}')
    if 'ESTIMATED' not in calibration.name:
        if std is not None:
            raise ValueError(f'no std needed for {calibration}')
        if return_estimate:
            raise ValueError(f'no return estimation available for {calibration}')
    if calibration in [CalibrationMethod.NONE, CalibrationMethod.TEMPERATURE_SCALING, CalibrationMethod.AFFINE,
                       CalibrationMethod.TEMPERATURE_SCALING_ESTIMATED, CalibrationMethod.AFFINE_ESTIMATED]:
        if prior is not None:
            raise ValueError(f'no prior needed for {calibration}')
    # simple case
    if calibration == CalibrationMethod.NONE:
        return {Split.DEV_TEST: data[Kind.LOGITS][Split.DEV_TEST], Split.APP_TEST: data[Kind.LOGITS][Split.APP_TEST]}
    # check if prior estimation necessary
    if 'ESTIMATED' in calibration.name:
        if not isinstance(std, float) or std <= 0 or std >= 1:
            raise ValueError('std needs to be a float within (0, 1)')
        real_prior = torch.bincount(data[Kind.LABELS][Split.APP_TEST]) / data[Kind.LABELS][Split.APP_TEST].size(0)
        prior = missmatch_priors(priors=real_prior, std=std)
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
    if return_estimate:
        return prior, calibrated_logits
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
    # they are just to slow...
    # metrics['ece_kde'][identifier] = measures.kernel_based_ece()
    # metrics['kce'][identifier] = measures.kernel_calibration_error()
    rbs = measures.root_brier_score()
    log_probs = torch.log(torch.nn.functional.softmax(logits, dim=1))
    brier = Brier(log_probs, labels, norm=False).item()
    brier_norm = Brier(log_probs, labels, norm=True).item()
    return {'cwce': cwce, 'rbs': rbs, 'bs': brier, 'nbs': brier_norm}
