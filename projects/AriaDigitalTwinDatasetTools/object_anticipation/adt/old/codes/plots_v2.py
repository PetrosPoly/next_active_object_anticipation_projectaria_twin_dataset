import numpy as np
import matplotlib.pyplot as plt
import os

# Lists to store the metrics
accuracies = []
precisions = []
recalls = []
true_positives = []
false_positives = []
parameter_sets = []

def plot_results(results, project_path):
    sequences = list(set([result['sequence'] for result in results]))

    # Create plots for each sequence
    for sequence in sequences:
        sequence_results = [r for r in results if r['sequence'] == sequence]

        plot_folder = os.path.join(project_path, 'plots', sequence)
        os.makedirs(plot_folder, exist_ok=True)

        # Extract the metric values
        overall_model_accuracies = [r['model_overall_accuracy'] for r in sequence_results]
        precisions = [r['precision'] for r in sequence_results]
        recalls = [r['recall'] for r in sequence_results]
        llm_activation_sensitivities = [r['llm_activation_sensitivity'] for r in sequence_results]
        llm_interaction_accuracies = [r['llm_interaction_accuracy'] for r in sequence_results]
        parameter_names = [r['parameters'] for r in sequence_results]

        # Refine parameter names (shorten if needed)
        parameter_names_short = [param.replace("time_", "t_").replace("highdot_", "hd_").replace("dist_", "d_") for param in parameter_names]

        # Plotting
        plt.figure(figsize=(12, 8))

        plt.plot(parameter_names_short, overall_model_accuracies, label='Overall Model Accuracy', marker='o', color='blue')
        plt.plot(parameter_names_short, precisions, label='Precision', marker='x', color='orange')
        plt.plot(parameter_names_short, recalls, label='Recall', marker='s', color='green')
        plt.plot(parameter_names_short, llm_activation_sensitivities, label='LLM Activation Sensitivity', marker='^', color='red')
        plt.plot(parameter_names_short, llm_interaction_accuracies, label='LLM Interaction Accuracy', marker='d', color='purple')

        plt.title(f'Metrics for Sequence: {sequence}', fontsize=16)
        plt.xlabel('Parameter Combination', fontsize=14)
        plt.ylabel('Metric Values', fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.grid(True)
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()

        # Save the plot with a descriptive name
        plt.savefig(os.path.join(plot_folder, 'metrics_plot.png'), format='png')
        plt.close()

def plot_bar_metrics(results, project_path):
    sequences = list(set([result['sequence'] for result in results]))

    for sequence in sequences:
        sequence_results = [r for r in results if r['sequence'] == sequence]

        plot_folder = os.path.join(project_path, 'plots', sequence)
        os.makedirs(plot_folder, exist_ok=True)

        # Extract the metric values
        overall_model_accuracies = [r['model_overall_accuracy'] for r in sequence_results]
        precisions = [r['precision'] for r in sequence_results]
        recalls = [r['recall'] for r in sequence_results]
        parameter_names = [r['parameters'] for r in sequence_results]

        # Create a bar plot
        index = np.arange(len(parameter_names))
        bar_width = 0.2

        plt.figure(figsize=(12, 8))
        plt.bar(index, overall_model_accuracies, bar_width, label='Accuracy', color='blue')
        plt.bar(index + bar_width, precisions, bar_width, label='Precision', color='orange')
        plt.bar(index + 2 * bar_width, recalls, bar_width, label='Recall', color='green')

        plt.xlabel('Parameter Combination', fontsize=14)
        plt.ylabel('Metric Values', fontsize=14)
        plt.xticks(index + bar_width, parameter_names, rotation=45, ha="right")
        plt.title(f'Comparison of Metrics for Sequence: {sequence}', fontsize=16)
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(plot_folder, 'bar_metrics_plot.png'), format='png')
        plt.close()
    
def plot_precision_vs_recall(results, project_path):
    sequences = list(set([result['sequence'] for result in results]))

    for sequence in sequences:
        sequence_results = [r for r in results if r['sequence'] == sequence]

        plot_folder = os.path.join(project_path, 'plots', sequence)
        os.makedirs(plot_folder, exist_ok=True)

        precisions = [r['precision'] for r in sequence_results]
        recalls = [r['recall'] for r in sequence_results]
        accuracies = [r['model_overall_accuracy'] for r in sequence_results]
        parameter_names = [r['parameters'] for r in sequence_results]

        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(recalls, precisions, s=[a * 100 for a in accuracies], c=accuracies, cmap='viridis', alpha=0.7)

        plt.title(f'Precision vs Recall for Sequence: {sequence}', fontsize=16)
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.colorbar(scatter, label='Overall Accuracy')

        for i, param in enumerate(parameter_names):
            plt.annotate(param, (recalls[i], precisions[i]), fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, 'precision_vs_recall.png'), format='png')
        plt.close()

def plot_tp_fp_pie(results):
    sequences = list(set([result['sequence'] for result in results]))

    for sequence in sequences:
        sequence_results = [r for r in results if r['sequence'] == sequence]

        plot_folder = os.path.join(project_path, 'plots', sequence)
        os.makedirs(plot_folder, exist_ok=True)

        for result in sequence_results:
            labels = ['True Positives', 'False Positives']
            sizes = [result['Tp'], result['Fp']]
            colors = ['green', 'red']
            
            # Check if sizes contain NaN or invalid values
            if any(np.isnan(sizes)) or any(np.isinf(sizes)):
                print(f"Skipping pie chart in {sequence} and for {result['parameters']} due to NaN or Inf values in Tp/Fp")
                continue

            if sum(sizes) == 0:
                print(f"Skipping pie chart in {sequence} and for {result['parameters']} as both Tp and Fp are zero")
                continue

            plt.figure(figsize=(6, 6))
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
            plt.title(f'TP vs FP for {result["parameters"]}', fontsize=16)
            plt.tight_layout()

            plt.savefig(os.path.join(plot_folder, f'tp_fp_pie_{result["parameters"]}.png'), format='png')
            plt.close()

def plot_combined_metrics(results):
    sequences = list(set([result['sequence'] for result in results]))

    for sequence in sequences:
        sequence_results = [r for r in results if r['sequence'] == sequence]

        plot_folder = os.path.join(project_path, 'plots', sequence)
        os.makedirs(plot_folder, exist_ok=True)

        overall_model_accuracies = [r['model_overall_accuracy'] for r in sequence_results]
        recalls = [r['recall'] for r in sequence_results]
        parameter_names = [r['parameters'] for r in sequence_results]

        plt.figure(figsize=(10, 6))
        plt.plot(parameter_names, overall_model_accuracies, label='Accuracy', marker='o', color='blue')
        plt.plot(parameter_names, recalls, label='Recall', marker='x', color='green')

        plt.title(f'Accuracy and Recall for Sequence: {sequence}', fontsize=16)
        plt.xlabel('Parameter Combination', fontsize=14)
        plt.ylabel('Metric Values', fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(os.path.join(plot_folder, 'combined_metrics.png'), format='png')
        plt.close()

if __name__ == "__main__":
    main()


