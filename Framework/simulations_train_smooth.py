from experiment import Experiment

# Draw latex figure
Experiment_name = 'Train_nuscenes_cor_left_turns_trajectron++'
new_experiment = Experiment(Experiment_name)

#%% Select modules
# # Select the datasets
Data_sets = [     
    {'scenario': 'RounD_round_about',  'max_num_agents': None, 't0_type': 'all', 'conforming_t0_types': []},
]

# Select the params for the datasets to be considered
Data_params = [{'dt': 0.1, 'num_timesteps_in': (15,15), 'num_timesteps_out': (20, 20)}] 

# Select the spitting methods to be considered
Splitters = [{'Type': 'no_split', 'repetition': 0}]

# Select the models to be trained
Models = []
for model_name in ['trajectron_salzmann', 'adapt_aydemir']:
    model_name_smooth = model_name + '_smooth'
    for sigma in [0.25, 0.5, 1.0]:
        for smoothing_method in ['positions', 'position_matched', 'all', 'control', 'control_matched']:
            # Adapt only uses positions, so skip some methods
            if (model_name == 'adapt_aydemir') and (smoothing_method not in ['positions', 'control_matched']):
                continue
            model_dict = {'model': model_name_smooth, 'kwargs': {'smoothing_sigma': sigma, 'smoothing_method': smoothing_method}}
            Models.append(model_dict)

# Select the metrics to be used
Metrics = ['ADE_indep']

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
agents_to_predict = 'V'

# Determine if allready existing results shoul dbe overwritten, or if not, be used instead
overwrite_results = 'no'

# Determine if the model should be evaluated on the training set as well
evaluate_on_train_set = True

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
Results, Train_results, Loss = new_experiment.load_results(plot_if_possible = True,
                                                           return_train_results = True,
                                                           return_train_loss = True)

