import enum
from pathlib import Path
from typing import Dict

import torch

# identifiers for the 30 tasks we used
all_tasks = ['lapgyn4_surgical_actions', 'lapgyn4_instrument_count', 'lapgyn4_anatomical_actions',
             'nerthus_bowel_cleansing_quality', 'hyperkvasir_therapeutic-interventions', 'cholec80_grasper_presence',
             'cholec80_hook_presence', 'idle_action_recognition', 'brain_tumor_classification',
             'brain_tumor_type_classification', 'chexpert_enlarged_cardiomediastinum', 'chexpert_cardiomegaly',
             'chexpert_edema', 'chexpert_consolidation', 'chexpert_pneumonia', 'chexpert_pneumothorax',
             'chexpert_pleural_effusion', 'chexpert_fracture', 'pneumonia_classification', 'covid_xray_classification',
             'deep_drid_dr_level', 'deep_drid_quality', 'kvasir_capsule_anatomy', 'mura_xr_wrist', 'mura_xr_shoulder',
             'mura_xr_humerus', 'mura_xr_hand', 'mura_xr_forearm', 'mura_xr_finger', 'mura_xr_elbow']

# identifiers of the binary tasks
binary_tasks = ['hyperkvasir_therapeutic-interventions', 'cholec80_grasper_presence', 'cholec80_hook_presence',
                'idle_action_recognition', 'brain_tumor_classification', 'chexpert_enlarged_cardiomediastinum',
                'chexpert_cardiomegaly', 'chexpert_edema', 'chexpert_consolidation', 'chexpert_pneumonia',
                'chexpert_pneumothorax', 'chexpert_pleural_effusion', 'chexpert_fracture',
                'pneumonia_classification', 'covid_xray_classification', 'deep_drid_quality',
                'kvasir_capsule_anatomy', 'mura_xr_wrist', 'mura_xr_shoulder', 'mura_xr_humerus',
                'mura_xr_hand', 'mura_xr_forearm', 'mura_xr_finger', 'mura_xr_elbow']

# identifiers for the example tasks
example_tasks = ['lapgyn4_anatomical_actions', 'lapgyn4_surgical_actions', 'hyperkvasir_therapeutic-interventions']

# identifiers for the example binary tasks
example_binary_tasks = ['hyperkvasir_therapeutic-interventions']

task_name_map = {'lapgyn4_surgical_actions': 'LapGyn4<br>(surg. ac.)',
                 'lapgyn4_instrument_count': 'LapGyn4<br>(inst. cnt.)',
                 'lapgyn4_anatomical_actions': 'LapGyn4<br>(ana. ac.)',
                 'nerthus_bowel_cleansing_quality': 'Nerthus',
                 'hyperkvasir_therapeutic-interventions': 'HyperKvasir<br>(thera. int.)',
                 'cholec80_grasper_presence': 'Cholec80<br>(grasper)',
                 'cholec80_hook_presence': 'Cholec80<br>(hook)',
                 'idle_action_recognition': 'CatRel<br>Comp',
                 'brain_tumor_classification': 'Brain<br>Tumor',
                 'brain_tumor_type_classification': 'Cheng<br>Dataset',
                 'chexpert_enlarged_cardiomediastinum': 'CheXpert<br>(enl. card.)',
                 'chexpert_cardiomegaly': 'CheXpert<br>(cardiom.)',
                 'chexpert_edema': 'CheXpert<br>(edema)',
                 'chexpert_consolidation': 'CheXpert<br>(consol.)',
                 'chexpert_pneumonia': 'CheXpert<br>(pneumonia)',
                 'chexpert_pneumothorax': 'CheXpert<br>(pneumot.)',
                 'chexpert_pleural_effusion': 'CheXpert<br>(pl. eff.)',
                 'chexpert_fracture': 'CheXpert<br>(fracture)',
                 'pneumonia_classification': 'Chest X-Ray<br>ZhangLab',
                 'covid_xray_classification': 'Covid XRay<br>Dataset',
                 'deep_drid_dr_level': 'DeepDRiD<br>(dr level)',
                 'deep_drid_quality': 'DeepDRiD<br>(quality)',
                 'kvasir_capsule_anatomy': 'Kvasir-Cap.<br>(anatomy)',
                 'mura_xr_wrist': 'MURA<br>(wrist)',
                 'mura_xr_shoulder': 'MURA<br>(shoulder)',
                 'mura_xr_humerus': 'MURA<br>(humerus)',
                 'mura_xr_hand': 'MURA<br>(hand)',
                 'mura_xr_forearm': 'MURA<br>(forearm)',
                 'mura_xr_finger': 'MURA<br>(finger)',
                 'mura_xr_elbow': 'MURA<br>(elbow)'}


class Split(enum.Enum):
    """Describes the three splits we use logits from in post-precessing and performance estimation."""
    DEV_TEST = 'dev_test'
    DEV_CAL = 'dev_cal'
    APP_TEST = 'app_test'


class Kind(enum.Enum):
    """Distinguishes two kinds of data."""
    LOGITS = 'logits'
    LABELS = 'labels'


def get_values(task: str, data_root: Path, proj: str = 'mic23_predictions_original_0') -> Dict[
    Kind, Dict[Split, torch.Tensor]]:
    """
    Loads all logits for a task from the mml project inside the data folder. Is capable to handle both the old MML
    file structure and the new one (between the 2023 experiments and the 2024 experiments some internals have changed).

    :param proj: mml project name
    :param task: task name
    :param data_root: root path for all data
    :return: two level dict, first determining Kind, then determining Split
    """
    prediction_root = data_root / f"{proj}" / "PREDICTIONS"
    sub_folders = list(prediction_root.iterdir())
    if len(sub_folders) == 0:
        raise RuntimeError(f"No predictions found for {proj}.")
    if any(folder.name.startswith('DSET') for folder in sub_folders):
        # old format detected - use glob to catch correct folder
        folder = next((data_root / f"{proj}" / "PREDICTIONS").glob(f'*TASK_{task}*miccai*'))
        val_set = torch.load(folder / "preds-val-fold-0_0001.pt")
        test_set = torch.load(folder / "preds-test-fold-0_0001.pt")
    else:
        # new format detected - deduce seed by any element
        seed = next(prediction_root.iterdir()).name.split('+')[1].split('?')[1]
        val_set = torch.load(prediction_root / f'{task}+miccai?{seed}+nested?0' / 'preds-test-fold-0_0001.pt')
        test_set = torch.load(prediction_root / f'{task}+miccai?{seed}' / 'preds-test-fold-0_0001.pt')
    # load predictions and create tensors
    val_logits = torch.stack([item['logits'] for item in val_set]).float()
    val_labels = torch.tensor([item['target'] for item in val_set])
    app_test_logits = torch.stack(
        [item['logits'] for item in test_set if item['sample_id'].startswith('app_test')]).float()
    app_test_labels = torch.stack([item['target'] for item in test_set if item['sample_id'].startswith('app_test')])
    dev_test_logits = torch.stack(
        [item['logits'] for item in test_set if item['sample_id'].startswith('dev_test')]).float()
    dev_test_labels = torch.stack([item['target'] for item in test_set if item['sample_id'].startswith('dev_test')])
    # return as clearly descriptive dict
    return {Kind.LOGITS: {Split.DEV_CAL: val_logits, Split.DEV_TEST: dev_test_logits, Split.APP_TEST: app_test_logits},
            Kind.LABELS: {Split.DEV_CAL: val_labels, Split.DEV_TEST: dev_test_labels, Split.APP_TEST: app_test_labels}}
