import os
import shutil

# Get the current path
path = os.sep.join(os.path.dirname(os.path.realpath(__file__)).split(os.sep)) + os.sep + 'Results' + os.sep 


# Define replacements to look for
old = 'Comb_N_L__--mixed_ss--dt=0.10_nI=15m15_nO=20m20_EC--max_000--agents_V--identi_split_0_pert=0'
new = 'Lgap_lturn--t0=crit__ss--dt=0.10_nI=15m15_nO=20m20_EC--max_000--agents_0--identi_split_0_pert=0'

data_set_old = 'Combined dataset (NuS & L-GAP)' # NONE
data_set_new = 'L-GAP (left turns)'

# Add certain parts (to prevent accidental doubbleings)
prefix = ''
suffix = ''

old_term = prefix + old + suffix
new_term = prefix + new + suffix

if data_set_old is None:
    data_set_folders = os.listdir(path)
else:
    data_set_folders = [data_set_old]
for data_set_folder in data_set_folders: 
    folder_path = path + data_set_folder + os.sep + 'Models' + os.sep
    if not os.path.exists(folder_path):
        continue
    if data_set_new is not None and data_set_new != data_set_old:
        folder_path_new = path + data_set_new + os.sep + 'Models' + os.sep
    else:
        folder_path_new = folder_path
    model_files = os.listdir(folder_path)
    for model_file in model_files:
        if old_term not in model_file:
            continue
        model_file_split = model_file.split(old_term)
        if len(model_file_split) > 1:
            model_file_new = new_term.join(model_file_split)
            old_file = folder_path + model_file
            new_file = folder_path_new + model_file_new
            if os.path.exists(new_file):
                print('Model already exists.')
            else:
                if os.path.isfile(old_file):
                    shutil.copyfile(old_file, new_file)
                else:
                    # Copy the whole folder
                    shutil.copytree(old_file, new_file)
