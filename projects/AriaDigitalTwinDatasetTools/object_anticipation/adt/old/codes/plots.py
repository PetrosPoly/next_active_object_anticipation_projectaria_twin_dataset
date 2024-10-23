import pandas as pd
import os

# import the plot functions
from plots_functions import (
    plot_activation_sensitivity,
    plot_llm_prediction_accuracy,
    plot_activation_sensitivity_heatmap,
    plot_activation_sensitivity_3d,
)

project_path = "Documents/projectaria_sandbox/projectaria_tools/projects/AriaDigitalTwinDatasetTools/object_anticipation/adt/"
file_path = os.path.join(project_path, 'data', 'results', 'llm_predictions_results.csv')

def main(): 

    # data 
    df = pd.read_csv(file_path)

    # Display the first few rows of the dataset
    print(df.head(50))

    # plots 
    plot_activation_sensitivity(df)
    plot_llm_prediction_accuracy(df)

    # plots that don't work
    # plot_activation_sensitivity_3d(df)
    # plot_activation_sensitivity_heatmap(df)

if __name__ == "__main__":
    main()