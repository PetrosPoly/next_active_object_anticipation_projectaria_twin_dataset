import os
import json
import time
import multiprocessing as mp
from itertools import product
from utils.evaluation import LLMEvaluation
import pandas as pd

# Project path
project_path = "Documents/projectaria_sandbox/projectaria_tools/projects/AriaDigitalTwinDatasetTools/object_anticipation/adt/"
sequences = ['Apartment_release_clean_seq150_M1292'] #, 'Apartment_release_work_seq107_M1292']

# Parameters for the language model module (unchanged)
time_thresholds = [2] # [1, 2, 3, 4, 5]
avg_dot_threshold_highs = [0.7]
avg_dot_threshold_lows = [0.2]
avg_distance_threshold_highs = [3]
avg_distance_threshold_lows = [1]
high_dot_thresholds = [0.9]
distance_thresholds = [2]
high_dot_counters_threshold = [15, 30, 45, 60]
distance_counters_threshold = [15, 30, 45, 60]
variables_window_times = [3.0]

# Parameters for the LLM reactivation module (unchanged)
minimum_time_deactivated = [2.0]
maximum_time_deactivated = [5.0]
user_relative_movement = [1.5]
object_percentage_overlap = [0.7]

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
        "distance_counters_threshold": dct,
        "window_time": w, 
        "minimum_time_deactivated": mintd,                
        "maximum_time_deactivated": maxtd,             
        "user_relative_movement": urm,                 
        "object_percentage_overlap" : obo
    }
    for t, adh, adl, adhg, adlg, hdt, dt, hdct, dct, w, mintd, maxtd, urm, obo in product(
        time_thresholds, avg_dot_threshold_highs, avg_dot_threshold_lows, 
        avg_distance_threshold_highs, avg_distance_threshold_lows,
        high_dot_thresholds, distance_thresholds,
        high_dot_counters_threshold, distance_counters_threshold, variables_window_times, 
        minimum_time_deactivated, maximum_time_deactivated, user_relative_movement, object_percentage_overlap   
    )
]

# Function to filter ground truth data
def filter_ground_truth(sequence):
    # Load ground truth data
    with open(os.path.join(project_path, 'data', 'gt', sequence, 'objects_that_moved.json'), 'r') as json_file:
        original_objects_that_moved_dict = json.load(json_file)

    with open(os.path.join(project_path, 'data', 'gt', sequence, 'user_object_movement.json'), 'r') as json_file:
        user_object_movement = json.load(json_file)

    # Filter the ground truth data based on the user's motion
    filtered_objects = {k: float(v) for k, v in user_object_movement.items() if v > 1.0}
    ground_truth = {float(k): v for k, v in original_objects_that_moved_dict.items() if v in filtered_objects}
    
    # Save filtered ground truth (optional, if needed later)
    ground_truth_folder = os.path.join(project_path, 'data', 'gt', sequence)
    os.makedirs(ground_truth_folder, exist_ok=True)
    with open(os.path.join(ground_truth_folder, 'filtered_ground_truth.json'), 'w') as file:
        json.dump(ground_truth, file, indent=4)

    return ground_truth

def run_simulation(parameters, ground_truth, sequence):  
    results = []

    # Parameters folder name
    parameter_folder_name = (
        f"time_{parameters['time_threshold']}_"
        f"highdot_{parameters['high_dot_threshold']}_"
        f"highdotcount_{parameters['high_dot_counters_threshold']}_"
        f"dist_{parameters['distance_threshold']}_"
        f"distcount_{parameters['distance_counters_threshold']}"
    )

    # Load predictions
    with open(os.path.join(project_path, 'data', 'predictions', sequence, parameter_folder_name, 'large_language_model_prediction.json')) as json_file:
        predictions_dict = json.load(json_file)

    # LLM predictions and evaluation logic
    LLM_predictions = {float(k): v for k, v in predictions_dict.items()}

    evaluation = LLMEvaluation(LLM_predictions, ground_truth)
    evaluation.calculate_FP_FN_GT_correspondances()
    evaluation.calculate_final_TP_FP()
    metrics = evaluation.calculate_metrics()

    # Store results
    result = {
        'sequence': sequence,
        'parameters': parameter_folder_name,
        'model_overall_accuracy': metrics[0],
        'precision': metrics[1],
        'recall': metrics[2],
        'llm_activation_sensitivity': metrics[3],
        'llm_interaction_accuracy': metrics[4],
        'Tp': metrics[5],
        'Fp': metrics[6],
        'Fn': metrics[7]
    }
    results.append(result)

    return results

if __name__ == "__main__":
    # Filter ground truth once per sequence
    filtered_ground_truths = {sequence: filter_ground_truth(sequence) for sequence in sequences}

    # Run simulations in parallel and pass the filtered ground truth to each process
    with mp.Pool(mp.cpu_count()) as pool:
        all_results = []
        for sequence in sequences:
            sequence_results = pool.starmap(run_simulation, [(params, filtered_ground_truths[sequence], sequence) for params in param_combinations])
            all_results.extend(sequence_results)

    # Flatten the list of lists
    flattened_results = [item for sublist in all_results for item in sublist]

    # Save final results to JSON and CSV
    results_folder = os.path.join(project_path, 'data', 'results')
    os.makedirs(results_folder, exist_ok=True)

    with open(os.path.join(results_folder, 'final_results.json'), 'w') as file:
        json.dump(flattened_results, file, indent=4)

    df = pd.DataFrame(flattened_results)
    df.to_csv(os.path.join(results_folder, 'llm_predictions_results.csv'), index=False)

if __name__ == "__main__":
    
    # Filter ground truth once per sequence
    filtered_ground_truths = {sequence: filter_ground_truth(sequence) for sequence in sequences}

    # Get available CPUs
    available_cpus = mp.cpu_count()

    # Record the start time
    start_time = time.time()

    # Create a multiprocessing pool
    with mp.Pool(available_cpus) as pool:
        all_results = []
        
        # Run the parameter combinations in parallel for each sequence
        for sequence in sequences:
            # Queue up all parameter combinations for this sequence
            sequence_results = pool.starmap(run_simulation, [(params, filtered_ground_truths[sequence], sequence) for params in param_combinations])
            all_results.extend(sequence_results)

    # Record the end time
    end_time = time.time()

    # Calculate the total time taken
    total_time = end_time - start_time

    # Print the total runtime
    print(f"Total runtime: {total_time:.2f} seconds")

    # Flatten the list of lists
    flattened_results = [item for sublist in all_results for item in sublist]

    # Save final results to JSON and CSV
    results_folder = os.path.join(project_path, 'data', 'results')
    os.makedirs(results_folder, exist_ok=True)

    with open(os.path.join(results_folder, 'final_results.json'), 'w') as file:
        json.dump(flattened_results, file, indent=4)

    df = pd.DataFrame(flattened_results)
    df.to_csv(os.path.join(results_folder, 'llm_predictions_results.csv'), index=False)
