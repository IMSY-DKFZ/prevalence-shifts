# MML prevalences plugin

This is the MML prevalences plugin, providing a task tag to reproduce the data splits used in paper 
"Deployment of Image Analysis Algorithms under Prevalence Shifts", see [ArXiV](https://arxiv.org/abs/2303.12540). To 
be published at the [MICCAI 2023](https://conferences.miccai.org/2023/en/) conference.

> DISCLAIMER: This package is a plugin for a larger library named mml. The mml dependency has not been published yet - 
> this means so far the plugin cannot be installed. We are working on it and release this dependency later. 

# Install

You need to install `mml-core` first, see [documentation](https://imsy.pages.dkfz.de/ise/mml/). Afterward you can 

```commandline
pip install .
```

from this directory to install the plugin in the same virtual environment. 

# Usage

To split a task according to our strategy add a tag to its alias containing the key `miccai` and a value that determines 
the random seed e.g. `42`. You need to use the proper separators `+` and `?`. So for example the task alias `test_task_one` 
becomes `test_task_one+miccai?13`. The modified task alias can either be used in a `cfg/tasks` file or as a `task_list` 
command line value.
