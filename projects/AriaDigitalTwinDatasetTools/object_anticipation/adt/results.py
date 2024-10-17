import numpy as np
import os                                   
import json
import matplotlib.pyplot as plt
from utils.evaluation import LLMEvaluation
from itertools import product               # added by Petros ()
import pandas as pd

from plots import (
    plot_results,
    plot_bar_metrics,
    plot_precision_vs_recall,
    plot_tp_fp_pie,
    plot_combined_metrics,
)


# Custom writing function to match the desired output format
def write_custom_json(data, file_path):
    
    with open(file_path, 'w') as f:
        
        # Start the JSON array
        f.write('[\n')
        
        for i, item in enumerate(data):
            
            # Dump each item and remove the surrounding list brackets
            json_item = json.dumps(item)
            
            # Write each item followed by a comma, except the last item
            if i < len(data) - 1:
                f.write(f'    {json_item},\n')
            else:
                f.write(f'    {json_item}\n')
        
        # End the JSON array
        f.write(']\n')

# Project path
project_path = "Documents/projectaria_sandbox/projectaria_tools/projects/AriaDigitalTwinDatasetTools/object_anticipation/adt/"
sequences = ['Apartment_release_clean_seq150_M1292'] # , 'Apartment_release_work_seq107_M1292']
# sequences = ['Apartment_release_work_seq107_M1292'] # , 'Apartment_release_clean_seq150_M1292']

# ==============================================
# Parameters Settting
# ==============================================

# Parameters for the language model module
time_thresholds = [1]                            # Time (in seconds) before interaction to activate the LLM

avg_dot_threshold_highs = [0.7]                  # Filter objects: keep only those with an average dot product above this value
avg_dot_threshold_lows = [0.2]                   # Filter objects: keep only those with an average dot product above this minimum value
avg_distance_threshold_highs = [3]               # Filter objects: keep only those with an average distance below this value
avg_distance_threshold_lows = [1]                # Filter objects: keep only those with an average distance above this minimum value

high_dot_thresholds = [0.9]                      # Count objects with dot product values exceeding this threshold
distance_thresholds = [2]                        # Count objects with distance values below this threshold
high_dot_counters_threshold = [45]               # Keep objects that exceed this count for dot product values above the threshold
low_distance_counters_threshold = [30]           # Keep objects that exceed this count for distance values below the threshold

variables_window_times = [3.0]                   # Sliding time window (in seconds) for tracking these parameters

# Parameters for the LLM reactivation module
minimum_time_deactivated = [2.0]                 # Minimum time (in seconds) the LLM remains deactivated after querying
maximum_time_deactivated = [5.0]                 # Maximum time (in seconds) before the LLM is activated again after querying
user_relative_movement = [1.5]                   # Threshold for user's relative movement (distance) after LLM is queried
object_percentage_overlap = [0.7]                # Percentage of overlap required for objects near the user to trigger reactivation

# Generate all combinations of the parameters
param_combinations = [
    {
        "time_threshold": t,
        "avg_dot_high": adh,
        "avg_dot_low": adl,
        "avg_distance_high": adhg,
        "avg_distance_low": adlg,
        "high_dot_threshold": hdt,
        "distance_threshold": dt,
        "high_dot_counters_threshold": hdct,
        "low_distance_counters_threshold": ldct,
        "window_time": w, 
        "minimum_time_deactivated": mintd,                
        "maximum_time_deactivated": maxtd,             
        "user_relative_movement": urm,                 
        "object_percentage_overlap" : obo,   
    }
    for t, adh, adl, adhg, adlg, hdt, dt, hdct, ldct, w, mintd, maxtd, urm, obo in product(
        time_thresholds, avg_dot_threshold_highs, avg_dot_threshold_lows, 
        avg_distance_threshold_highs, avg_distance_threshold_lows,
        high_dot_thresholds, distance_thresholds,
        high_dot_counters_threshold, low_distance_counters_threshold, variables_window_times, 
        minimum_time_deactivated, maximum_time_deactivated, user_relative_movement,object_percentage_overlap   
    )
]
        
def main():
    
    # Initialize storage for metrics
    results = []

    for sequence in sequences:

        # ==============================================
        # Define the filenames / folder names 
        # ==============================================
        
        # Folder to load the parameters compinations 
        parameters_comb = os.path.join(project_path, 'utils', 'json', 'param_combinations.json')

        # Folder to write the ground truth data - Where the filtered ground t
        ground_truth_folder = os.path.join(project_path, 'data', 'gt', sequence)
        os.makedirs(ground_truth_folder, exist_ok = True)

        # ==============================================
        # Load --> Parameters / Ground Truth 
        # ==============================================

        # Load the Parameters
        with open(parameters_comb, 'r') as file:
            loaded_param_combinations = json.load(file)

        # Load the Ground Truth
        with open(os.path.join(project_path, 'data', 'gt', sequence, 'objects_that_moved.json'), 'r') as json_file:
            original_objects_that_moved_dict = json.load(json_file)

        with open(os.path.join(project_path, 'data', 'gt', sequence, 'user_object_movement.json'), 'r') as json_file:
            user_object_movement = json.load(json_file)

        # Filter the Ground Truth Data based on the user's motion
        filtered_objects = {k: float(v) for k, v in user_object_movement.items() if v > 1.0}
        ground_truth = {float(k): v for k, v in original_objects_that_moved_dict.items() if v in filtered_objects}
        
        # Write the filtered ground truth interactions (filtered by user's motion) in a json  
        with open(os.path.join(ground_truth_folder, 'filtered_ground_truth.json'), 'w') as file:
            json.dump (ground_truth, file, indent =4)
        
        # ==============================================
        # Run through all the predictions 
        # ==============================================
        
        # Predictions
        for parameters in loaded_param_combinations:
            
            # Parameters folder name
            parameter_folder_name = (
                f"time_{parameters['time_threshold']}_"
                f"highdot_{parameters['high_dot_threshold']}_"
                f"highdotcount_{parameters['high_dot_counters_threshold']}_"
                f"dist_{parameters['distance_threshold']}_"
                f"distcount_{parameters['low_distance_counters_threshold']}"
            )
                
            # ==============================================
            # Load the predictions
            # ==============================================
            
            # Load the predictions from each corresponding folder 
            with open(os.path.join(project_path, 'data', 'predictions', sequence, parameter_folder_name, 'large_language_model_prediction.json')) as json_file:
                predictions_dict = json.load(json_file)

            # LLM predictions
            LLM_predictions = {float(k): v for k, v in predictions_dict.items()}

            # ==============================================
            # Calculate the metrics 
            # ==============================================
            
            # Initialize the LLMEvaluation class
            evaluation = LLMEvaluation(LLM_predictions, ground_truth)

            # Adjust the predictions with ground truth items and calculate FP and FN
            evaluation.calculate_FP_FN_GT_correspondances()
            
            # after getting the correspondances calculate the final FP, TP 
            evaluation.calculate_final_TP_FP()
            
            # Calculate accuracy, precision, recall
            (model_overall_accuracy, 
            precision, 
            recall, 
            llm_activation_sensitivity, 
            llm_interaction_accuracy, 
            Tp, Fp, Fn,
            total_ground_truth, total_llm_predictions, total_llm_activations, total_actual_correspondences, 
            correspondances) = evaluation.calculate_metrics()
            
            # result
            result = {
                'sequence': sequence,
                'parameters': parameter_folder_name,
                'total_ground_truth':  total_ground_truth, 
                'total_llm_predictions': total_llm_predictions,
                'total_llm_activations': total_llm_activations,
                'total_actual_correspondences': total_actual_correspondences, 
                'Tp': Tp,
                'Fp': Fp,
                'Fn': Fn,
                'model_overall_accuracy': model_overall_accuracy,
                'precision': precision,
                'recall': recall,
                'llm_activation_sensitivity': llm_activation_sensitivity,
                'llm_interaction_accuracy': llm_interaction_accuracy
            }

            results.append(result)

            print(correspondances)
            print(result)   

            # Define the folders that we will write the results       
            result_folder = os.path.join(project_path, 'data', 'results', sequence, parameter_folder_name)
            os.makedirs(result_folder, exist_ok=True)  

            # Write the correspondances
            write_custom_json(correspondances, os.path.join(result_folder, 'correspondances.json'))
            
            # Write the results
            with open(os.path.join(result_folder,'results.json'), 'w') as file:
                json.dump (result, file, indent=4)

    # ==============================================
    # Write the results 
    # ==============================================
    
    # Define the folders that we will write the results       
    results_folder = os.path.join(project_path, 'data', 'results')
    os.makedirs(results_folder, exist_ok=True)  

    # Write the results
    with open(os.path.join(results_folder, 'results.json')):
        json.dump (results, file, )

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(results)

    # Export the DataFrame to a CSV file
    df.to_csv(os.path.join(results_folder, 'llm_predictions_results.csv'), index=False)

    # ==============================================
    # Plot the results 
    # ==============================================
    
    # plot_results(results)
    # plot_bar_metrics(results)
    # plot_precision_vs_recall(results)
    # plot_tp_fp_pie(results)
    # plot_combined_metrics(results)

if __name__ == "__main__":
    main()