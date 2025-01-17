# MML prevalences plugin

This is the MML prevalences plugin, providing a task tag to reproduce the data splits used in the papers:
  - "Deployment of Image Analysis Algorithms under Prevalence Shifts", see [Springer](https://doi.org/10.1007/978-3-031-43898-1_38) and [ArXiV](https://arxiv.org/abs/2303.12540). 
  - "Navigating Prevalence Shifts in Image Analysis Algorithm Deployment" (unpublished)
  
This specific version has been updated to be used in conjunction with the latest version of MML.

> NOTE: This package is a plugin for a larger framework named mml. It is only necessary to install mml-core and this 
> plugin if you intend to train your own models aka. generate logit predictions.

# Install

You need to install `mml-core` first, see [documentation](https://mml.readthedocs.io/en/latest/index.html). 
Detailed instructions are given at the top level README of this repository. After following these steps you can 

```commandline
pip install .
```

from this directory to install the plugin in the same virtual environment. 

# Usage

To split a task according to our strategy add a tag to its alias containing the key `miccai` and a value that determines 
the random seed e.g. `42`. You need to use the proper separators `+` and `?`. So for example the task alias `test_task_one` 
becomes `test_task_one+miccai?13`. The modified task alias can either be used in a `cfg/tasks` file or as a `task_list` 
command line value.
