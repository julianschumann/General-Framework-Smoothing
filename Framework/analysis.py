import os
import pandas as pd
import numpy as np
np.set_printoptions(precision=3, suppress=True)


# Write Latex files for tables
sigma = [0.25, 0.5, 1.0]
dmax = [0.0, 0.25, 0.5, 1.0]

smoothing_methods = ['pos~\\eqref{eq:smooth_pos}', 'ctrl~\\eqref{eq:smooth_ctrl}']
smoothing_applied = ['eval', '\\makecell[l]{train \& \\\\eval}']
for model_name in ['Tpp', 'ADAPT']:
    for data_set_name in ['LGAP', 'RounD']:
        # Load data
        file_name = f'Results_{data_set_name}_{model_name}.npy'
        base_file_name = f'Results_{data_set_name}_{model_name}_base.npy'

        if not (os.path.exists(file_name) and os.path.exists(base_file_name)):
            continue
        Results = np.load(file_name)  # shape: 7/13 (num_models) x 3 (num_datasets)
        Base_Results = np.load(base_file_name)  # shape: 7/13 (num_models)

        Results = np.concatenate((Base_Results[:, None], Results), axis=1) # shape: 7/13 x 4
        Results_no = Results[0]
        Results_smooth = Results[1:] # shape: 6/12 x 4
        Results_smooth = Results_smooth.reshape(-1, 2, 3, Results.shape[-1])  # shape: 1/2 (smoothing types) x 2 (smoothing methods) x 3 (sigmas) x 4 (perturbations) 

        import matplotlib.pyplot as plt
        # Show dependence of results between dmax and results
        plt.figure(figsize=(12,12))
        plt.title(f'{data_set_name} - {model_name}')
        plt.plot(dmax, Results_no, 'x-', c='k', label='No Smoothing')
        linestyles = ['--', ':']
        for i_type, smoothing_type in enumerate(smoothing_applied):
            # Switch between dashed and dotted lines
            linstyle = linestyles[i_type % len(linestyles)]
            if i_type >= Results_smooth.shape[0]:
                continue
            colors = [(1,0,0), (0,0,1)]
            for i_approach in range(2):
                # Switch between red and blue colors
                color = colors[i_approach]
                for i_sigma, sigma_value in enumerate(sigma):
                    # With increasing sigma, make color lighter
                    factor = 1 - 0.6 * sigma_value
                    color_sigma = (1 - (1 - color[0]) * factor, 1 - (1 - color[1]) * factor, 1 - (1 - color[2]) * factor)
                    result = Results_smooth[i_type,i_approach,i_sigma]
                    plt.plot(dmax, result, marker='x', linestyle=linstyle, color=color_sigma, label=f'{smoothing_type}, {smoothing_methods[i_approach]}, $\\sigma$={sigma_value}')
        plt.xlabel('$d_{\\max}$ [$m$]')
        plt.ylabel('ADE [m]')
        plt.legend()
        plt.savefig(f'plot_{data_set_name}_{model_name}.pdf')
        plt.close()
        # Get the best model for each datasset
        model_best_indices = np.argmin(Results, axis=0)  # shape: (4,)

        string = ''

        # Write header for tabularx table with 7 columns: 3L, 4C
        string += '\\begin{tabularx}{\\linewidth}{|X|X|X|YYYY|}\n'
        string += '\\hline\n'
        string += ' \\multicolumn{3}{|c|}{Smoothing approach} & \\multicolumn{4}{c|}{$d_{\\max}$ [$m$]} \\\\ \n'
        string += ' \\multicolumn{1}{|c}{Applied} & \\multicolumn{1}{c}{Goal} & $\\sigma$ [$m$] &  (0.0) & 0.25 & 0.5 & 1.0 \\\\ \n'
        string += '\\hline \n'


        # First row: Normal model without smoothing
        string += '\\multicolumn{2}{|c|}{No Smoothing} & (0.0) & '
        for i_pert in range(4):
            string += f'{Results_no[i_pert]:.3f} '
            if i_pert < 3:
                string += '& '
            else:
                string += ' \\\\ \n'



        # Then first SE, then ST if available
        for i_type, smoothing_type in enumerate(smoothing_applied):
            if i_type >= Results_smooth.shape[0]:
                continue

            for i_approach in range(2):
                # Add midrule for each smoothing type
                if i_approach == 0:
                    string += '\\hline\n'
                else:
                    string += '\\cline{2-7}\n'
                for i_sigma, sigma_value in enumerate(sigma):
                    if i_approach == 0 and i_sigma == 0:
                        # Use multirow with 6 rows to indicate smoothing type
                        string += f'\\multirow{{6}}{{*}}{{{smoothing_type}}} & '
                    else:
                        string += ' & '
                    
                    # If i_approach , use multirow with 3 rows to indicate smoothing method
                    if i_sigma == 0:
                        string += f'\\multirow{{3}}{{*}}{{{smoothing_methods[i_approach]}}} & '
                    else:
                        string += ' & '
                    
                    # Add sigma value
                    string += f'{sigma_value} & '
                    # Add perturbation results
                    for i_pert in range(4):
                        string += f'{Results_smooth[i_type,i_approach,i_sigma,i_pert]:.3f} '
                        if i_pert < 3:
                            string += '& '
                        else:
                            string += ' \\\\ \n'
        string += '\\hline\n'
        string += '\\end{tabularx}\n'

        # Go throughg each column, and make the lowest values bold
        lines = string.split('\n')
        data_line_idx = np.arange(5, len(lines)-2)
        # Remove lines with 'line' inside them
        data_line_idx = [i for i in data_line_idx if 'line' not in lines[i]]
        data_lines = [lines[i] for i in data_line_idx]
        # For each column, get the index of the minimum value
        line_min = [data_line_idx[i] for i in model_best_indices]

        # In the specific line in lines, make value bold
        for col_idx, line_idx in enumerate(line_min):
            line = lines[line_idx]
            parts = line.split('&')
            # Find the position of the value to bold
            i_part = len(parts) - 4 + col_idx
            # Overwrite that part with bold
            parts_i_split = parts[i_part].split('\\\\')
            parts_i_split[0] = ' \\textbf{' + parts_i_split[0].strip() + '} '
            parts[i_part] = '\\\\'.join(parts_i_split)
            # Reconstruct the line
            lines[line_idx] = '&'.join(parts)
        # Reconstruct the string
        string = '\n'.join(lines)
        # Write to file
        with open(f'table_{data_set_name}_{model_name}.tex', 'w') as f:
            f.write(string)
    




