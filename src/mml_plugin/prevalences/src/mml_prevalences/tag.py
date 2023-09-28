import logging
import random
from collections import Counter

from mml.core.data_loading.task_attributes import TaskType
from mml.core.data_preparation.task_creator import TaskCreator, implements_action
from mml.core.data_preparation.utils import TaskCreatorActions

logger = logging.getLogger(__name__)


@implements_action(TaskCreatorActions.MODIFY)
def miccai_transform(self: TaskCreator, seed: str = '42') -> None:
    """
    Special version of task transformation we need for the MICCAI 2023 submission. Has
    * 10% of data with equal class fractions left aside as development test data
    the remaining 80% of data are split with identical class distribution to
    * 50% development training data
    * 10% development validation data
    * 30% application testing data

    this manifests as follows in the meta information
    first train fold is development validation split
    other train folds are development training data
    test data is stacked with first half being dev_test, then app_test, sample IDs are prefixed accordingly

    :return:
    """
    # log
    logger.info(f"MICCAI redistribution of task {self.current_meta['alias']}.")
    self.protocol('MICCAI redistribution.')
    # checks
    if self.current_meta['task_type'] != TaskType.CLASSIFICATION:
        raise RuntimeError('Since class occurrences are important for sample selection, MICCAI transform is '
                           'currently only available for classification tasks.')
    if self.fm.preprocessed_data in self.dset_path.parents:
        raise RuntimeError('Since MICCAI tag modifies the test data, this can only be run on a raw (none '
                           'preprocessed) version of a task. If this task already has been preprocessed with '
                           'some preprocessing id X, consider creating the task with some dummy preprocessing id '
                           'e.g. mml mode=info (task settings as before) preprocessing=example and run your '
                           'original command afterwards.')
    # set self.data['train'] and self.data['test'], recalculate class_occ if necessary
    old_folds_n = len(self.current_meta['train_folds'])
    all_samples = self.current_meta['train_tuples']
    n_classes = len(self.current_meta['class_occ'])
    logger.debug(f'Previous occurences {self.current_meta["class_occ"]}, size {len(all_samples)}')
    # remove the 10% dev-test
    dev_test_class_size = int(0.1 * len(all_samples) / n_classes)
    logger.debug(f'require {dev_test_class_size} samples per class in dev test split')
    if any([dev_test_class_size > 0.35 * occ for cls, occ in self.current_meta['class_occ'].items()]):
        raise RuntimeError(f'Impossible splitting with occ {self.current_meta["class_occ"]}')
    dev_test_ids = set()
    for enum_idx, class_idx in enumerate(self.current_meta['idx_to_class']):
        random.seed(int(seed) + 10 * enum_idx)
        all_class_indices = [k for k, elem in all_samples.items() if elem['class'] == class_idx]
        random.shuffle(all_class_indices)
        dev_test_ids.update(all_class_indices[:dev_test_class_size])
    logger.debug(f'size dev test {len(dev_test_ids)}')
    reduced_samples = {k: v for k, v in all_samples.items() if k not in dev_test_ids}
    # use split folds mechanism to extract the application test split balanced
    self.data = {'train': reduced_samples}
    self.split_folds(n_folds=2, ensure_balancing=True, fold_0_fraction=0.3 / 0.9, seed=3*int(seed),
                     ignore_state=True)
    training_ids = self.current_meta['train_folds'][1]
    app_test_ids = self.current_meta['train_folds'][0]
    logger.debug(f'training size {len(training_ids)}, app test size {len(app_test_ids)}')
    self.data = {'train': {s_id: all_samples[s_id] for s_id in training_ids},
                 'test': {f'dev_test_{s_id}': all_samples[s_id] for s_id in dev_test_ids}}
    self.data['test'].update({f'app_test_{s_id}': all_samples[s_id] for s_id in app_test_ids})
    # update class occurrences
    self.current_meta['class_occ'] = Counter([self.current_meta['idx_to_class'][all_samples[s_id]['class']]
                                              for s_id in training_ids])
    # calculate validation split and run self.split_folds
    fraction = 0.1 / 0.6
    self.split_folds(n_folds=old_folds_n, ensure_balancing=True, fold_0_fraction=fraction, seed=5*int(seed),
                     ignore_state=True)
    logger.debug(
        f'Final occurences {self.current_meta["class_occ"]}, size {sum(self.current_meta["class_occ"].values())}')
