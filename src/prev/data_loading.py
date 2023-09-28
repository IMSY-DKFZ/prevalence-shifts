import enum
from pathlib import Path
from typing import Dict

import torch


class Split(enum.Enum):
    DEV_TEST = 'dev_test'
    DEV_CAL = 'dev_cal'
    APP_TEST = 'app_test'


class Kind(enum.Enum):
    LOGITS = 'logits'
    LABELS = 'labels'


def get_values(task: str, data_root: Path, proj: str = 'mic23_predictions_original_0') -> Dict[Kind, Dict[Split, torch.Tensor]]:
    """
    Loads all logits for a task from the mml project inside the data folder.

    :param proj_wo_seed: mml project name, seed will be appended as with an additional underscore before
    :param task: task name
    :param data_root: root path for all data
    :param seed: the seed of the experiment
    :return: two level dict, first determining Kind, then determining Split
    """
    folder = next((data_root / f"{proj}" / "PREDICTIONS").glob(f'*TASK_{task}*miccai*'))
    val_set = torch.load(folder / "preds-val-fold-0_0001.pt")
    val_logits = torch.stack([item['logits'] for item in val_set]).float()
    val_labels = torch.tensor([item['target'] for item in val_set])
    test_set = torch.load(folder / "preds-test-fold-0_0001.pt")
    app_test_logits = torch.stack(
        [item['logits'] for item in test_set if item['sample_id'].startswith('app_test')]).float()
    app_test_labels = torch.stack([item['target'] for item in test_set if item['sample_id'].startswith('app_test')])
    dev_test_logits = torch.stack(
        [item['logits'] for item in test_set if item['sample_id'].startswith('dev_test')]).float()
    dev_test_labels = torch.stack([item['target'] for item in test_set if item['sample_id'].startswith('dev_test')])
    return {Kind.LOGITS: {Split.DEV_CAL: val_logits, Split.DEV_TEST: dev_test_logits, Split.APP_TEST: app_test_logits},
            Kind.LABELS: {Split.DEV_CAL: val_labels, Split.DEV_TEST: dev_test_labels, Split.APP_TEST: app_test_labels}}
