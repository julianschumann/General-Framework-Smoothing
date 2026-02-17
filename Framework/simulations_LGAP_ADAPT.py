#%%
import copy
from experiment import Experiment

# Draw latex figure
Experiment_name = 'Perturbations'
new_experiment = Experiment(Experiment_name)


#%% Select modules
# Select the models to be trained
Models = ['adapt_aydemir']

# Select the params for the datasets to be considered
Data_params = [{'dt': 0.1, 'num_timesteps_in': (15,15), 'num_timesteps_out': (20, 20)}]

# Select the spitting methods to be considered
Splitters = [{'Type': 'no_split', 'repetition': 0, 'train_pert': False, 'test_pert': True}]

# Select the datasets
Data_sets = []
perturbation = {'attack': 'Adversarial_Control_Action',
                'data_set_dict': {'scenario': 'CoR_left_turns', 'max_num_agents': None, 't0_type': 'col_set', 'conforming_t0_types': []},
                'data_param': Data_params[0],
                'splitter_dict': Splitters[0],
                'model_dict': Models[0],
                'num_samples_perturb': 20,
                'max_number_iterations': 100,
                'alpha': 0.01,
                'gamma': 0.99,
                'loss_function_1': None,
                'loss_function_2': None,
                'barrier_function_past': 'Time_Trajectory_specific',
                'barrier_function_future': 'Trajectory_specific',
                'distance_threshold_past': 0.9,
                'distance_threshold_future': 0.9,
                'log_value_past': 2.5,
                'log_value_future': 2.5,
                'GT_data': 'no'}
# Vary loss functions and distance thresholds
for scenario in ['CoR_left_turns']:
    for model_dict in Models:
        for loss_function_1, loss_function_2 in [
            ('ADE_Y_GT_Y_Pred_Max', None),
            ]:
            for distance_threshold in [0.25, 0.5, 1.0]:
                # Define specific perturbation
                perturbation_i = copy.deepcopy(perturbation)
                perturbation_i['splitter_dict']['test_pert'] = False
                dataset_unperturbed = {'scenario': scenario,  'max_num_agents': None, 't0_type': 'crit', 'conforming_t0_types': []}
                perturbation_i['data_set_dict'] = dataset_unperturbed
                perturbation_i['model_dict'] = model_dict
                perturbation_i['loss_function_1'] = loss_function_1
                perturbation_i['loss_function_2'] = loss_function_2
                perturbation_i['distance_threshold_past'] = distance_threshold
                perturbation_i['distance_threshold_future'] = distance_threshold
                dataset = {'scenario': scenario,  'max_num_agents': None, 't0_type': 'crit', 'conforming_t0_types': [], 'perturbation': perturbation_i}
                Data_sets.append(dataset)

# Add models using smoothing during evaluation
for model_name in ['adapt_aydemir']:
    for addition in ['_smoooth_eval']:
        model_name_smooth = model_name + addition
        for smoothing_method in ['positions', 'control_matched']:
            for sigma in [0.25, 0.5, 1.0]:
                model_dict = {'model': model_name_smooth, 'kwargs': {'smoothing_sigma': sigma, 'smoothing_method': smoothing_method}}
                Models.append(model_dict)
            
# Select the metrics to be used
Metrics = [
    {'metric': 'ADE_indep', 'kwargs': {'include_pov': False}}
]

new_experiment.set_modules(Data_sets, Data_params, Splitters, Models, Metrics)

#%% Other settings
# Set the number of different trajectories to be predicted by trajectory prediction models.
num_samples_path_pred = 6

# Deciding wether to enforce start times (or allow later predictions if not enough input data is available)
enforce_prediction_times = True

# determine if the upper bound for n_O should be enforced, or if prediction can be made without
# underlying output data (might cause training problems)
enforce_num_timesteps_out = True

# Determine if the useless prediction (i.e, prediction you cannot act anymore)
# should be exclude from the dataset
exclude_post_crit = True

# Decide wether missing position in trajectory data can be extrapolated
allow_extrapolation = True

# Use all available agents for predictions
agents_to_predict = 'predefined'

# Determine if allready existing results shoul dbe overwritten, or if not, be used instead
overwrite_results = 'no'

# Determine if the model should be evaluated on the training set as well
evaluate_on_train_set = False

# Determine if predictions should be saved
save_predictions = True

# Select method used for transformation function to path predictions
model_for_path_transform = 'trajectron_salzmann_old'

new_experiment.set_parameters(model_for_path_transform, num_samples_path_pred, 
                              enforce_num_timesteps_out, enforce_prediction_times, 
                              exclude_post_crit, allow_extrapolation, 
                              agents_to_predict, overwrite_results, 
                              save_predictions, evaluate_on_train_set)

#%% Run experiment
new_experiment.run()

# Load results
Results = new_experiment.load_results()

import numpy as np
np.set_printoptions(precision=3, suppress=True)
print(np.squeeze(Results).T)
np.save('Results_LGAP_ADAPT.npy', np.squeeze(Results).T)
