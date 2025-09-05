import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, mannwhitneyu
from statannotations.Annotator import Annotator

def compare_multiple_models(model_results, pairs_to_compare, output_filename):
    """
    Calculates pairwise Wasserstein distances for specified model pairs and
    creates a statistically annotated multiple box plot.
    """
    all_distances_data = []

    # --- 1. Calculate the distances for each specified pair ---
    print("Calculating pairwise Wasserstein distances...")
    for model_a_name, model_b_name in pairs_to_compare:
        model_a_betas = model_results[model_a_name]
        model_b_betas = model_results[model_b_name]
        
        comparison_label = f'{model_a_name} vs.\n{model_b_name}'
        
        for r1 in model_a_betas:
            for r2 in model_b_betas:
                dist = wasserstein_distance(r1, r2)
                all_distances_data.append({
                    'Comparison': comparison_label,
                    'Wasserstein Distance': dist
                })

    # --- 2. Prepare data for plotting ---
    plot_data = pd.DataFrame(all_distances_data)

    # --- 3. Create the annotated multiple box plot ---
    print(f"Generating annotated box plot to {output_filename}...")
    x = "Comparison"
    y = "Wasserstein Distance"
    
    # Get the order for the x-axis from the pairs we defined
    order = [f'{p[0]} vs.\n{p[1]}' for p in pairs_to_compare]
    
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(data=plot_data, x=x, y=y, order=order, hue=x, hue_order=order)
    
    # Define the pairs for the statistical annotation
    # This will draw a bracket comparing the boxes on the plot: list all the pairs
    pairs_for_annotation = [(order[i], order[j]) for i in range(len(order)) for j in range(i+1, len(order))]

    
    # Initialize the annotator
    annotator = Annotator(ax, pairs_for_annotation, data=plot_data, x=x, y=y, order=order)
    
    # Specify the test and add the annotations
    annotator.configure(test='t-test_paired', text_format='star', loc='inside', verbose=2)
    annotator.apply_and_annotate()
    
    plt.title('Discrepancy Between Pairs of Spatial Models')
    plt.ylabel('Wasserstein Distance')
    plt.xlabel('Model Pair Comparison')
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close()
    print("Plot saved successfully.")

# --- Main Execution ---
if __name__ == "__main__":
    
    # 1. Define the models and their corresponding data files
    model_files = {
        "Gland Fission": "gland_fission_50_runs.pkl",
        "Boundary Growth": "Boundary_50_runs.pkl",
        'Non Spatial' : "Nonspatial_50_runs.pkl",
        'Subclonal Gland Fission' : "Subclonal_gland_fission_50_runs.pkl",
        'Subclonal Boundary Growth' : "Subclonal_Boundary_50_runs.pkl",
        "Sobclonal Non Spatial": "Subclonal_Non_Spatial_50_runs.pkl"
    }

    # 2. Load all results into a dictionary
    all_model_results = {}
    for model_name, file_path in model_files.items():
        try:
            with open(file_path, 'rb') as f:
                all_model_results[model_name] = pickle.load(f)
            print(f"Loaded {len(all_model_results[model_name])} results for {model_name}.")
        except FileNotFoundError:
            print(f"Error: Could not find file {file_path}. Please generate it first.")
            exit()

    # 3. Specify which pairs of models you want to compare
    pairs = [
        # ("Gland Fission", "Non Spatial"),
        # ("Gland Fission", "Boundary Growth"),
        # ("Gland Fission", "Subclonal Gland Fission"),
        # ("Subclonal Gland Fission", "Non Spatial"),
        # ("Subclonal Gland Fission", "Boundary Growth"),
        # ("Boundary Growth", "Non Spatial"),
        # ("Subclonal Boundary Growth", "Non Spatial"),
        # ("Subclonal Boundary Growth", "Boundary Growth"),
        # ("Subclonal Boundary Growth", "Gland Fission"),
        ("Subclonal Gland Fission", "Sobclonal Non Spatial"),
        ("Subclonal Boundary Growth", "Subclonal Gland Fission"),
        ("Subclonal Boundary Growth", "Sobclonal Non Spatial"),
        ("Sobclonal Non Spatial", "Non Spatial")

        
    ]

    # 4. Generate the plot
    compare_multiple_models(
        model_results=all_model_results,
        pairs_to_compare=pairs,
        output_filename="multi_model_comparison.png"
    )
