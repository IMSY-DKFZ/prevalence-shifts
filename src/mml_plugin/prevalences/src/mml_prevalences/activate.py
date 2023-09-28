from mml_prevalences.tag import miccai_transform

from mml.core.data_preparation.task_creator import TASK_CREATOR_TAG_MAP, TaskCreator

TaskCreator.miccai_transform = miccai_transform

TASK_CREATOR_TAG_MAP['miccai'] = 'miccai_transform'
