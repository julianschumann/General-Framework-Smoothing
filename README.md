# Randomized Smoothing as a defense against adversarial attacks.
This repository contains the code necessary to reproduce the results reported in the underlying paper. It uses the [STEP framework](https://github.com/DAI-Lab-HERALD/General-Framework/tree/main) for the evaluation of trajectory prediction models.

For this, the following steps need to be executed.

## Get the data
Clone the repository, and then follow the instructions in the linked readmes to get the raw data for [L-GAP](Framework/Data_sets/CoR_left_turns), [NuScenes](Framework/Data_sets/NuScenes), and [rounD](Framework/Data_sets/RounD_round_about).

## Train the models
Open a terminal in the [Framework folder](Framework), and there run the following two files:
- [simulations_train.py](Framework/simulations_train.py)
- [simulations_train_smooth.py](Framework/simulations_train_smooth.py)

Once this is complete, run the following files to make the trained models available for evaluation
- [duplicate_model_LGAP.py](Framework/duplicate_model_LGAP.py)
- [duplicate_model_RounD.py](Framework/duplicate_model_RounD.py)

## Evaluate the models
In the final step, one can then run the evaluation of those models, which includes the application of the adversarial attacks.
- [simulations_LGAP_Tpp_base.py](Framework/simulations_LGAP_Tpp_base.py)
- [simulations_LGAP_Tpp.py](Framework/simulations_LGAP_Tpp.py)
- [simulations_LGAP_ADAPT_base.py](Framework/simulations_LGAP_ADAPT_base.py)
- [simulations_LGAP_ADAPT.py](Framework/simulations_LGAP_ADAPT.py)
- [simulations_RounD_Tpp_base.py](Framework/simulations_RounD_Tpp_base.py)
- [simulations_RounD_Tpp.py](Framework/simulations_RounD_Tpp.py)
- [simulations_RounD_ADAPT_base.py](Framework/simulations_RounD_ADAPT_base.py)
- [simulations_RounD_ADAPT.py](Framework/simulations_RounD_ADAPT.py)

## Visualization
To generate the *.tex* files underlying the tables presented in the paper, as well as plots of the results, run the file [analysis.py](Framework/analysis.py).
