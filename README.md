# Navigating Prevalence Shifts in Image Analysis Algorithm Deployment

This repo serves as reproduction code for the following papers 
  * "Deployment of Image Analysis Algorithms under Prevalence Shifts" see [Springer](https://doi.org/10.1007/978-3-031-43898-1_38) and [ArXiV](https://arxiv.org/abs/2303.12540). It was published as conference paper at [MICCAI 2023](https://conferences.miccai.org/2023/en/).
  * "Navigating Prevalence Shifts in Image Analysis Algorithm Deployment" (unpublished) an extended version of the former, submitted to the Medical Image Analysis journal.

In the following, the reproduction descriptions only focus on the latter, extended version. To reproduce our original 
results please use the respective git tag and check out "miccai23".

## Overview

> DISCLAIMER: The mml dependency has not been published yet - this means so far the training part cannot be 
> reproduced publicly. We are working on it and release this dependency later. All evaluation and plotting scripts are 
> available. We provide three sample tasks with the produced predictions in /data/. [mml-free reproducibility](#mml-free-reproducibility)
> provides instructions to produce plots for the sample tasks.

- [code structure](#code-structure)
- [installation](#installation)
- [image data preparation](#image-data-preparation)
- [model training and prediction generation](#training-and-predictions)
- [experiments and figures](#experiments-and-figures)


## code structure

The code of this repository is structured as follows in `src`:

 - `mml-plugin` implements a [`mml`](https://git.dkfz.de/imsy/ise/mml) plugin to re-distribute samples within a task according to our needs
 - `prev` contains definitions and routines that are shared through our experiments (note that `__init__.py` modifies `psrcal` behaviour)
 - `training_scripts` contains all commands with respect to task data (prepare, preprocess) and neural networks (train and predict) 
 - the notebooks `1_...` to `8_...` contain the steps to reproduce our experiments 

## pre-requisites

To run the notebooks and reproduce the plots exactly you might need to install the font we used. 
 - [download](http://mirrors.ctan.org/fonts/newcomputermodern/otf/NewCM10-Regular.otf) `NewCM10-Regular.otf`
 - place it in the `/data` folder at project root (necessary for notebook 5)
 - install the font (the details of this step depend on your OS)

## mml-free reproducibility

 - create a virtualenv with conda and install python 3.10, install the requirements

```commandline
conda update -n base -c defaults conda
conda create --yes --name prev python=3.10
conda activate prev
git clone https://github.com/IMSY-DKFZ/prevalence-shifts.git
cd prevalences
pip install -r requirements.txt
```

 - Run the following notebooks to follow our experiments: 
    - `3_prevalence_estimation.ipynb` - Research Question 1, creates figures 5 and C.11
    - `4_calibration.ipynb` - Research Question 2a, creates figure 6
    - `5_threshold_visualization.ipynb` - Research Question 2b, creates figures 3, 7, and 8
    - `6_decision_rule.ipynb` - Research Question 2b, creates figure 9
    - `7_validation_metrics.ipynb` - Research Question 2c, creates figure 10
    - `8_uncertainty.ipynb` - Creates uncertainty tables
## installation

 > DISCLAIMER: mml is not yet public. Please follow the installation instructions from [mml-free reproducibility](#mml-free-reproducibility).

 - create a virtualenv with conda and install python 3.10

```commandline
conda update -n base -c defaults conda
conda create --yes --name prev python=3.10
conda activate prev
```

 - install [mml-core](https://imsy.pages.dkfz.de/ise/mml/install.html#virtual-environment) and [mml-data plugin](https://imsy.pages.dkfz.de/ise/mml/api/plugins/data.html)

```commandline
pip install --index-url https://mmlToken:<personal_access_token>@git.dkfz.de/api/v4/projects/89/packages/pypi/simple mml-core==0.13.3
pip install mml-data==0.4.1 --index-url https://__token__:<your_personal_token>@git.dkfz.de/api/v4/projects/89/packages/pypi/simple
```

 - install local prevalence plugin and other requirements

```commandline
git clone https://github.com/IMSY-DKFZ/prevalence-shifts.git
cd prevalences
pip install -r requirements.txt
cd src/mml_plugin/prevalences
pip install .
```

 - install the fonts (http://mirrors.ctan.org/fonts/newcomputermodern/otf/NewCM10-Regular.otf)
 - setup environment variables for `mml`

```commandline
cd ../../..
mml-env-setup
nano mml.env  # modify at least MML_DATA_PATH, MML_RESULTS_PATH and MML_LOCAL_WORKERS accordingly 
pwd | conda env config vars set MML_ENV_PATH=$(</dev/stdin)/mml.env
conda activate prev
```

## image data preparation

> DISCLAIMER: This section requires mml which is not yet public.

- the data and predictions generation process in handled with the `mml` framework
- the commands to leverage `mml` are generated in `1_generate_predictions.ipynb` and stored in `training_scripts`
  - `01_create_cmds.txt` for data download / task generation
  - `02_pp_cmds.txt` for data preprocessing
  - `03_tag_cmds.txt` for the splitting of tasks according to our experimental setup (train-validation-development test and deployment test)
  - `05_dataseed_cmds.txt` for creating 5 additional splittings with different splitting seeds
- if the commands shall be run on some external infrastructure (like a GPU cluster) the `1_generate_predictions.ipynb` contains configuration possibilities to adapt the txt files
- the commands can be run locally by `bash 0X_XXX_cmds.txt` (stick to the order indicated by numbering)

## training and predictions

> DISCLAIMER: This section requires mml which is not yet public.

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
- now you can load the predictions inside the jupyter notebooks, they should run straight forward and produce the figures in `results`, next to `src` and `data`

Notebooks:

- `2_plot_examples.ipynb` - Task overview, Figure 4
- `3_prevalence_estimation.ipynb` - Research Question 1, creates Figures 5 and C.11
- `4_calibration.ipynb` - Research Question 2a, creates Figure 6
- `5_threshold_visualization.ipynb` - Research Question 2b, creates Figures 3, 7, and 8
- `6_decision_rule.ipynb` - Research Question 2b, creates Figure 9
- `7_validation_metrics.ipynb` - Research Question 2c, creates Figure 10
- `8_uncertainty.ipynb` - Creates uncertainty tables
