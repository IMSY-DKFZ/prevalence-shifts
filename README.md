# Navigating Prevalence Shifts in Image Analysis Algorithm Deployment

This repo serves as reproduction code for the following papers 
  * "Deployment of Image Analysis Algorithms under Prevalence Shifts" see [Springer](https://doi.org/10.1007/978-3-031-43898-1_38) and [ArXiV](https://arxiv.org/abs/2303.12540). It was published as conference paper at [MICCAI 2023](https://conferences.miccai.org/2023/en/).
  * "Navigating Prevalence Shifts in Image Analysis Algorithm Deployment" an extended version of the former, published in the [Medical Image Analysis](https://www.sciencedirect.com/science/article/pii/S1361841525000520) journal.

In the following, the reproduction descriptions only focus on the latter, extended version. To reproduce our original 
results please use the respective git tag and check out "miccai23".

## Overview

- [code structure](#code-structure)
- [installation](#installation)
- [image data preparation](#image-data-preparation)
- [model training and prediction generation](#training-and-predictions)
- [experiments and figures](#experiments-and-figures)


## code structure

The code of this repository is structured as follows in `src`:

 - `mml-plugin` implements a [`mml`](https://github.com/IMSY-DKFZ/mml) plugin to re-distribute samples within a task according to our needs
 - `prev` contains definitions and routines that are shared through our experiments (note that `__init__.py` modifies `psrcal` behaviour)
 - `training_scripts` contains all commands with respect to task data (prepare, preprocess) and neural networks (train and predict) 
 - the notebooks `1_...` to `8_...` contain the steps to reproduce our experiments 

## pre-requisites

To run the notebooks and reproduce the plots exactly you might need to install the font we used. 
 - [download](http://mirrors.ctan.org/fonts/newcomputermodern/otf/NewCM10-Regular.otf) `NewCM10-Regular.otf`
 - place it in the `/data` folder at project root (necessary for notebook 5)
 - install the font (the details of this step depend on your OS)

## results for example tsks

`mml` was primarily used to handle the imaging datasets, run the model training and produce the prediction logits for 
this project. Here we show how to obtain results for 3 example tasks without the use of `mml`:

 - create a virtualenv with conda and install python 3.10, install the requirements

```commandline
conda create --yes --name prev python=3.10
conda activate prev
git clone https://github.com/IMSY-DKFZ/prevalence-shifts.git
cd prevalences
pip install -r requirements.txt
```

 - Run the following notebooks to follow our experiments: 
    - `3_prevalence_estimation.ipynb` - Research Question 1, creates figures 5 and C.11
    - `4_calibration.ipynb` - Research Question 2a, creates figure 6
    - `6_decision_rule.ipynb` - Research Question 2b, creates figure 9
    - `7_validation_metrics.ipynb` - Research Question 2c, creates figure 10
  - In each of the notebooks set `EXAMPLE_TASKS_ONLY = False` in the first cell to limit the analysys to tasks available without access to mml.

## installation

These instructions handle the **full** reproduction including the installation of `mml` (see above for a light setup).

 - create a virtualenv with conda and install python 3.10

```commandline
conda create --yes --name prev python=3.10
conda activate prev
```

 - install [mml-core](https://mml.readthedocs.io/en/latest/install.html) and [mml-tasks plugin](https://mml.readthedocs.io/en/latest/api/plugins/tasks.html)

```commandline
pip install mml-core
pip install mml-tasks
```

 - install local prevalence plugin and other requirements

```commandline
git clone https://github.com/IMSY-DKFZ/prevalence-shifts.git
cd prevalences
pip install -r requirements.txt
cd src/mml_plugin/prevalences
pip install .
```

- setup [environment variables](https://mml.readthedocs.io/en/latest/install.html#local-variables) for `mml`

```commandline
cd ../../..
mml-env-setup
nano mml.env  # modify at least MML_DATA_PATH, MML_RESULTS_PATH and MML_LOCAL_WORKERS accordingly 
pwd | conda env config vars set MML_ENV_PATH=$(</dev/stdin)/mml.env
conda activate prev
```

## image data preparation

> Note: We do not own the datasets used for this study and many datasets have to be downloaded manually. Instructions 
> will be given along the "01_create_cmds" process. 

- the data and predictions generation process in handled with the `mml` framework
- the commands to leverage `mml` are generated in `1_generate_predictions.ipynb` and stored in `training_scripts`
  - `01_create_cmds.txt` for data download / task generation
  - `02_pp_cmds.txt` for data preprocessing
  - `03_tag_cmds.txt` for the splitting of tasks according to our experimental setup (train-validation-development test and deployment test)
  - `05_dataseed_cmds.txt` for creating 5 additional splittings with different splitting seeds
- if the commands shall be run on some external infrastructure (like a GPU cluster) the `1_generate_predictions.ipynb` contains configuration possibilities to adapt the txt files
- the commands can be run locally by `bash 0X_XXX_cmds.txt` (stick to the order indicated by numbering)

## training and predictions

> Note: Model training requires appropriate hardware (e.g. a GPU), ideally on a distributed cluster. 

- once more `mml` is leveraged for this step and the commands are generated in `1_generate_predictions.ipynb` and stored in `training_scripts`
  - for uncertainty assessment (`8_uncertainty.ipynb`):
    - `04_reproduce_cmds.txt` for re-training the original experiments with 6 additional seeds on the original splits
    - `06_dataseed_predict_cmds.txt` for re-training each task once per additional splittings
  - for re-calibration assessment (`4_calibration.ipynb`):
    - `07_retraining_cmds.txt` for re-training on the original splits with adapted loss weights according to exact deployment prevalences (for each IR 1.0, 1.5, ..., 10.0)
    - `08_retraining_estimated_cmds.txt` for re-training on the original splits with adapted loss weights according to estimated deployment prevalences - using ACC (for each IR 1.0, 1.5, ..., 10.0)
- the files can be run or adapted as mentioned before
- keep in mind that they incorporate (6+5+19+19) * 30 = 1470 training+prediction pipelines and take some time to complete
- in our analysis we also used the original training and previous 3 additional seeds next to `06_dataseed_predict_cmds.txt` (in sum 10 seeded repetitions on the original split)


## experiments and figures

> DISCLAIMER: For licensing reasons we may not provide all predictions, but attach some sample predictions in /data.
> We also provide intermediate results:
>  - `24_prev_estimation_df.pkl` - holds all quantification results (RQ1)
>  - `24_recalibration_results.csv` - holds all re-calibration results (RQ2a)
>  - `24_decision_rule_results_....pkl` - holds results on applying various decision rules (RQ2b)
>  - `24_metric_performance_....pkl` - holds all metric evaluations (RQ2c)

- navigate to the top level folder named `data` and store the project folders generated by the previous commands in there, more precisely
  - locate your `MML_RESULTS_PATH` as provided in the [installation](#installation)
  - within search the project folders 
    - original publication (n=4): 
      - `mic23_predictions_original_0`
      - `mic23_predictions_reproduce_0`, ..., `.._2`
    - generated by `04_reproduce_cmds.txt` (n=6):
      - `mic23_predictions_original_10`, ..., `.._15`
    - generated by `06_dataseed_predict_cmds.txt` (n=5):
      - `mic23_predictions_datasplit_seed_3`, `..._31`, `..._314`, `..._3141`, `..._31415`
    - generated by `07_retraining_cmds.txt` (n=19):
      - `mic23_predictions_extension_balanced_0_1.0`, `..._0_1.5`, `..._0_2.0`, ..., `..._0_10.0`
    - generated by `08_retraining_estimated_cmds.txt` (n=19):
      - `mic23_predictions_extension_balanced_estimated_0_1.0`, `..._0_1.5`, `..._0_2.0`, ..., `..._0_10.0`
  - copy those to the `data` folder next to `src`
- for reproducing full results set `EXAMPLE_TASKS_ONLY = False` in the first cell of notebooks `3_prevalence_estimation.ipynb`, `4_calibration.ipynb`, `6_decision_rule.ipynb`, and `7_validation_metrics.ipynb`.
- now you can load the predictions inside the jupyter notebooks, they should run straight forward and produce the figures in `results`, next to `src` and `data`

Notebooks:

- `2_plot_examples.ipynb` - Task overview, Figure 3
- `3_prevalence_estimation.ipynb` - Research Question 1, creates Figures 4 and D.10
- `4_calibration.ipynb` - Research Question 2a, creates Figure 5
- `5_threshold_visualization.ipynb` - Research Question 2b, creates Figures 2, 6, and 7
- `6_decision_rule.ipynb` - Research Question 2b, creates Figure 8
- `7_validation_metrics.ipynb` - Research Question 2c, creates Figure 9
- `8_uncertainty.ipynb` - Creates uncertainty tables, and Figure E.11
