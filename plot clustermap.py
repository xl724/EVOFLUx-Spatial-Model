import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from spatial_mode import Simulation, Deme, Grid, plot_longitudinal, scatter_with_color, \
    convert_to_continuous_colour, plot_demes_from_data, plot_demes_from_data_subclone, multivariate_hypergeometric

def plot_beta_clustermap_pdf(simulation_history, output_filename):
    """
    Generates a multi-page PDF of clustermaps showing the total beta value
    at each timepoint after the deme count exceeds 100.
    """
    print(f"Generating beta clustermap PDF to {output_filename}...")
    
    # 1. Find the first timepoint with > 100 demes
    start_timepoint = -1
    for t, snapshot in enumerate(simulation):
        if len(snapshot) > 100:
            start_timepoint = t
            break
    
    if start_timepoint == -1:
        print("Simulation did not reach 100 demes. No plot generated.")
        return

    
    cmap = sns.diverging_palette(253, 11, s=60, l=40, sep=80, as_cmap=True)

    for t in range(start_timepoint, len(simulation_history)):
        if t % 20 == 0:
            print(f"Processing timepoint {t}...")
            snapshot = simulation_history[t]
            
            # Prepare beta data DataFrame (same as before)
            df_beta = pd.DataFrame({
                deme_id: data['beta'] for deme_id, data in snapshot.items()
            }).dropna(axis=0, how='all')
            df_beta = df_beta.loc[df_beta.var(axis=1) > 0]
            if df_beta.empty: continue

            grid_center = 75
            
            # Calculate radius and angle for each deme
            radius_data = {}
            angle_data = {}
            for deme_id, data in snapshot.items():
                x, y = data['coords']
                dx, dy = x - grid_center, y - grid_center
                radius_data[deme_id] = np.sqrt(dx**2 + dy**2)/(np.sqrt(2)*grid_center)  # Normalize radius to [0, 1]
                # Use arctan2 for a full -pi to +pi range, then normalize to 0-1
                angle_data[deme_id] = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)

            # Create the DataFrame with all three annotations
            sample_info = pd.DataFrame({
                'Radius': radius_data,
                'Angle': angle_data
            })
            
            # Create the color mapping for each annotation track
            col_colors = pd.DataFrame({
                'Radius': convert_to_continuous_colour(sample_info['Radius'], cmap="Oranges"),
                'Angle': convert_to_continuous_colour(sample_info['Angle'], cmap="twilight")
            }, index=sample_info.index)

            # Create the clustermap with the multi-layered col_colors
            g = sns.clustermap(df_beta, cmap=cmap,
                               col_colors=col_colors, 
                               cbar_kws={'label':'Fraction\nmethylated'},
                               vmin=0, vmax=1,
                               yticklabels=False, xticklabels=False)
            
            g.fig.suptitle(f'Beta Clustering at Timepoint {t}')
            # Adjust the font size of the main color bar label
            g.ax_cbar.set_label('Fraction\nmethylated')
            g.ax_cbar.yaxis.label.set_fontsize(10)
            
            # Radius legend
            radius_norm = plt.Normalize(sample_info['Radius'].min(), sample_info['Radius'].max())
            radius_sm = plt.cm.ScalarMappable(cmap="Oranges", norm=radius_norm)
            radius_cbar_ax = g.fig.add_axes([0.85, 0.6, 0.2, 0.02])
            radius_cbar = g.fig.colorbar(radius_sm, cax=radius_cbar_ax, orientation='horizontal')
            radius_cbar.set_label('Radius from Center', fontsize=10)

            # Angle legend
            angle_norm = plt.Normalize(0, 360) # Angle in degrees
            angle_sm = plt.cm.ScalarMappable(cmap="twilight", norm=angle_norm)
            angle_cbar_ax = g.fig.add_axes([0.85, 0.5, 0.2, 0.02])
            angle_cbar = g.fig.colorbar(angle_sm, cax=angle_cbar_ax, orientation='horizontal')
            angle_cbar.set_ticks([0, 90, 180, 270, 360])
            angle_cbar.set_label('Angle (Degrees)', fontsize=10)
            # Save the figure
            output_dir = "Multiple_Cluster_Subclonal_gland_fission"
            g.fig.savefig(f"{(os.path.join(output_dir, output_filename))}_t{t}.png", dpi=150, bbox_inches='tight')
            plt.close(g.fig)
            
    print("Finished generating clustermap series.")

import os
#For visualizing the subclone:
def plot_beta_clustermap_pdf_subclone(simulation_history, output_filename):
    """
    Generates a multi-page PDF of clustermaps showing total beta value,
    annotated by radius, angle, and subclonal fraction.
    """
    print(f"Generating annotated clustermap series to {output_filename}...")
    # ... (code to create the output directory is the same) ...

    start_timepoint = -1
    for t, snapshot in enumerate(simulation_history):
        if len(snapshot) > 100:
            start_timepoint = t
            break
    
    if start_timepoint == -1:
        print("Simulation did not reach 100 demes. No plot generated.")
        return

    cmap = sns.diverging_palette(253, 11, s=60, l=40, sep=80, as_cmap=True)

    for t in range(start_timepoint, len(simulation_history)):
        if t % 20 == 0:
            print(f"Processing timepoint {t}...")
            snapshot = simulation_history[t]
            
            # Prepare beta data DataFrame (same as before)
            df_beta = pd.DataFrame({
                deme_id: data['beta'] for deme_id, data in snapshot.items()
            }).dropna(axis=0, how='all')
            df_beta = df_beta.loc[df_beta.var(axis=1) > 0]
            if df_beta.empty: continue

            grid_center = 75
            
            # Calculate radius and angle for each deme
            radius_data = {}
            angle_data = {}
            for deme_id, data in snapshot.items():
                x, y = data['coords']
                dx, dy = x - grid_center, y - grid_center
                radius_data[deme_id] = np.sqrt(dx**2 + dy**2)/(np.sqrt(2)*grid_center)  # Normalize radius to [0, 1]
                # Use arctan2 for a full -pi to +pi range, then normalize to 0-1
                angle_data[deme_id] = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)

            # Create the DataFrame with all three annotations
            sample_info = pd.DataFrame({
                'Radius': radius_data,
                'Angle': angle_data,
                'Subclonal Fraction': {deme_id: data['subclonal_fraction'] for deme_id, data in snapshot.items()}
            })
            
            # Create the color mapping for each annotation track
            col_colors = pd.DataFrame({
                'Radius': convert_to_continuous_colour(sample_info['Radius'], cmap="Oranges"),
                'Angle': convert_to_continuous_colour(sample_info['Angle'], cmap="twilight"),
                'Subclonal Fraction': convert_to_continuous_colour(sample_info['Subclonal Fraction'], cmap="Greens")
            }, index=sample_info.index)

            # Create the clustermap with the multi-layered col_colors
            g = sns.clustermap(df_beta, cmap=cmap,
                               col_colors=col_colors, 
                               cbar_kws={'label':'Fraction\nmethylated'},
                               vmin=0, vmax=1,
                               yticklabels=False, xticklabels=False)
            
            g.fig.suptitle(f'Beta Clustering at Timepoint {t}')
            # Adjust the font size of the main color bar label
            g.ax_cbar.set_label('Fraction\nmethylated')
            g.ax_cbar.yaxis.label.set_fontsize(10)

            # Manually create and position the SUBCLONE color bar
            subclone_norm = plt.Normalize(0, 1)
            subclone_sm = plt.cm.ScalarMappable(cmap="Greens", norm=subclone_norm)
            subclone_sm.set_array([])
            subclone_cbar_ax = g.fig.add_axes([0.85, 0.4, 0.2, 0.02]) # Positioned below the main cbar
            subclone_cbar = g.fig.colorbar(subclone_sm, cax=subclone_cbar_ax, orientation='horizontal')
            subclone_cbar.set_label('Subclonal Fraction', fontsize=10)
            
            # Radius legend
            radius_norm = plt.Normalize(sample_info['Radius'].min(), sample_info['Radius'].max())
            radius_sm = plt.cm.ScalarMappable(cmap="Oranges", norm=radius_norm)
            radius_cbar_ax = g.fig.add_axes([0.85, 0.6, 0.2, 0.02])
            radius_cbar = g.fig.colorbar(radius_sm, cax=radius_cbar_ax, orientation='horizontal')
            radius_cbar.set_label('Radius from Center', fontsize=10)

            # Angle legend
            angle_norm = plt.Normalize(0, 360) # Angle in degrees
            angle_sm = plt.cm.ScalarMappable(cmap="twilight", norm=angle_norm)
            angle_cbar_ax = g.fig.add_axes([0.85, 0.5, 0.2, 0.02])
            angle_cbar = g.fig.colorbar(angle_sm, cax=angle_cbar_ax, orientation='horizontal')
            angle_cbar.set_ticks([0, 90, 180, 270, 360])
            angle_cbar.set_label('Angle (Degrees)', fontsize=10)
            # Save the figure
            output_dir = "Multiple_Cluster_Subclonal_gland_fission"
            g.fig.savefig(f"{(os.path.join(output_dir, output_filename))}_t{t}.png", dpi=150, bbox_inches='tight')
            plt.close(g.fig)
            
    print("Finished generating clustermap series.")

# def plot_beta_clustermap_pdf_subclone(self, output_filename):
#     """
#     Generates a multi-page PDF of clustermaps showing total beta value,
#     annotated by the subclonal fraction in each deme.
#     """
#     print(f"Generating subclonal clustermap PDF to {output_filename}...")
    
#     start_timepoint = -1
#     for t, snapshot in enumerate(simulation):
#         if len(snapshot) > 100:
#             start_timepoint = t
#             break
    
#     if start_timepoint == -1:
#         print("Simulation did not reach 100 demes. No plot generated.")
#         return

#     cmap = sns.diverging_palette(253, 11, s=60, l=40, sep=80, as_cmap=True)

#     # iter = 0
#     for t in range(start_timepoint, len(simulation)):
#         if t%10 == 0 :
#             # iter += 1
#             print(f"Processing the page {t}")
#             snapshot = simulation[t]
            
#             # Prepare beta data DataFrame (Rows: CpGs, Columns: Demes)
#             beta_data = {deme_id: data['beta'] for deme_id, data in snapshot.items()}
#             df_beta = pd.DataFrame(beta_data)
#             df_beta = df_beta.dropna(axis=0, how='all').loc[df_beta.var(axis=1) > 0]
#             if df_beta.empty: continue

#             # Prepare annotation DataFrame (Rows: Demes, Columns: Annotations)
#             sample_info = pd.DataFrame({
#                 'Subclonal Fraction': {deme_id: data['subclonal_fraction'] for deme_id, data in snapshot.items()}
#             })
            
#             # Create the color mapping for the annotation track
#             col_colors = sample_info.apply(lambda series: convert_to_continuous_colour(series, cmap="Greens", vmin=0, vmax=1))
            
#             # Create the clustermap
#             g = sns.clustermap(df_beta, cmap=cmap,
#                             col_colors=col_colors, 
#                             cbar_kws={'label':'Fraction\nmethylated'}, 
#                             vmin=0, vmax=1,
#                             yticklabels=False, xticklabels=False)
            
#             g.fig.suptitle(f'Beta Clustering with Subclone Fraction at Timepoint {t}')
#             # Adjust the font size of the main color bar label
#             g.ax_cbar.set_label('Fraction\nmethylated')
#             g.ax_cbar.yaxis.label.set_fontsize(10)

#             # # Create a manual legend for the subclonal fraction
#             # norm = plt.Normalize(0, 1)
#             # sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
#             # sm.set_array([])
#             # cbar_ax = g.fig.add_axes([0.05, 0.8, 0.2, 0.02])
#             # cbar = g.fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
#             # cbar.set_label('Subclonal Fraction')

#             # 2. Manually create and position the MAIN color bar (for the heatmap)
#             # main_cbar_ax = g.fig.add_axes([0.05, 0.8, 0.2, 0.02]) # Position: [left, bottom, width, height]
#             # main_cbar = g.fig.colorbar(g.ax_heatmap.get_children()[0], 
#             #                         cax=main_cbar_ax, 
#             #                         orientation='horizontal')
#             # main_cbar.set_label('Fraction Methylated')

#             # 3. Manually create and position the SUBCLONE color bar
#             subclone_norm = plt.Normalize(0, 1)
#             subclone_sm = plt.cm.ScalarMappable(cmap="Greens", norm=subclone_norm)
#             subclone_sm.set_array([])
#             subclone_cbar_ax = g.fig.add_axes([0.85, 0.85, 0.2, 0.02]) # Positioned below the main cbar
#             subclone_cbar = g.fig.colorbar(subclone_sm, cax=subclone_cbar_ax, orientation='horizontal')
#             subclone_cbar.set_label('Subclonal Fraction', fontsize=10)
#             output_dir = "Cluster_Subclonal_gland_fission"
#             # Save the figure to PNG
#             g.fig.savefig(f"{(os.path.join(output_dir,output_filename))}_t{t}.png", dpi=150, bbox_inches='tight')
#             # Close the figure to free memory
#             plt.close(g.fig)
        
#     print("Finished generating PDF.")


from matplotlib.collections import LineCollection
import colorsys
import numpy as np
import random

def plot_budging_trajectories(simulation, output_filename, num_to_plot=100, sampling_interval=10):
    """
    Visualizes the movement (budging) trajectory of a random subset of demes with
    an independent, full-range color gradient for each path.
    The timepoint is start from when the deme count exceeds 100 and ends when it stop expanding:
    in order to plot the dynamic trajectory of the budging process.

    Args:
        output_filename (str): The path to save the plot image.
        sampling_interval (int): Plot a point every N timepoints.
    """
    print(f"Generating budging trajectory plot to {output_filename}...")
    if not simulation:
        print("Warning: No simulation history to plot.")
        return
    # Find the Start and End Timepoints for Plotting
    start_t = -1
    end_t = -1
    last_deme_count = 0

    for t, snapshot in enumerate(simulation):
        current_deme_count = len(snapshot)
        
        # Find the first timepoint where deme count exceeds 100
        if start_t == -1 and current_deme_count > 100:
            start_t = t
            
        # Find the first timepoint where the deme count stabilizes
        if start_t != -1 and current_deme_count == last_deme_count:
            end_t = t
            break # Stop searching once stabilization is found
            
        last_deme_count = current_deme_count

    # If the conditions are not met, exit or plot the whole history
    if start_t == -1:
        print("Simulation did not reach 100 demes. No trajectory plot generated.")
        return
    if end_t == -1: # If it never stabilized, plot to the end
        end_t = len(simulation)

    print(f"Plotting trajectories from timepoint {start_t} to {end_t}...")
    
    # Slice the history to the desired time window
    history_slice = simulation[start_t:end_t]

    # Extract Trajectories within the Time interval, for each individual deme
    deme_trajectories = {}
    deme_start_times = {}
    for t_offset, snapshot in enumerate(history_slice):
        t_actual = start_t + t_offset
        for deme_id, data in snapshot.items():
            if deme_id not in deme_trajectories:
                deme_trajectories[deme_id] = []
                deme_start_times[deme_id] = t_actual
            deme_trajectories[deme_id].append(data['coords'])
    

    all_deme_ids = list(deme_trajectories.keys())
    if len(all_deme_ids) > num_to_plot:
        demes_to_plot = random.sample(all_deme_ids, num_to_plot)
    else:
        demes_to_plot = all_deme_ids

    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Use the logic from plot_grid to draw the final state
    dim = 150
    background = np.full((dim, dim, 3), 1.0)  # White background
    
    ax.imshow(background, origin='lower', extent=[0, dim, 0, dim])

    # Draw the trajectory for each deme
    cmap = plt.get_cmap('viridis_r') 
    for deme_id in demes_to_plot:
        path = deme_trajectories[deme_id]
        if len(path) < 2: 
            continue

        start_time = deme_start_times[deme_id]
        
        # Normalize the colors for this deme's path
        points = np.array(path).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors = np.linspace(0, 1, len(segments))  # Ensure the colors range from 0 to 1

        # Create the LineCollection for this deme
        norm = plt.Normalize(0, 1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=3, alpha=0.8)
        lc.set_array(colors)  # Set the normalized colors

        # Add the colored line to the plot
        ax.add_collection(lc)

        # Get and plot the sampled points on top
        sampled_points = np.array([
            coords for i, coords in enumerate(path) 
            if (start_time + i) % sampling_interval == 0
        ])
        
        if len(sampled_points) > 0:
            ax.scatter(sampled_points[:, 0], sampled_points[:, 1], c='white', s=10, zorder=3, ec='black', lw=0.5)

    # Finalize and save the plot
    ax.set_title('Deme Budging Trajectories')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_xlim(0, dim)
    ax.set_ylim(0, dim)
    ax.set_aspect('equal')
    
    # Add a colorbar for the time gradient
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Time (Simulation Step)')
    output_dir = "Subclonal_simulation_plots_gland_fission"
    plt.savefig((os.path.join(output_dir,output_filename)), dpi=150, bbox_inches='tight')
    plt.close(fig)
    

# --- Main Analysis Script ---
if __name__ == "__main__":
    
    # 1. Load the saved simulation object from the file
    saved_filepath = "Subclonal_gland_fission_simulation_final_state.pkl"
    
    print(f"Loading saved simulation state from {saved_filepath}...")
    
    try:
        with open(saved_filepath, 'rb') as f:
            # The 'rb' means read in binary mode.
            simulation = pickle.load(f)
        print("Load successful.")
        print(f"Loaded simulation up to Time: {len(simulation):.2f}")
        

    except FileNotFoundError:
        print(f"Error: Save file not found at '{saved_filepath}'. Please run the main simulation first.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading: {e}")
        exit()

    # 2. Plot the beta clustermap PDF
    # output_filename = "test_clustermap"
    # plot_beta_clustermap_pdf(simulation, output_filename)
    # 3. Plot the subclonal beta clustermap PDF
    output_filename_subclone = "beta_evolution_summary_subclonal_clustermap_gland_fission"
    plot_beta_clustermap_pdf_subclone(simulation, output_filename_subclone)

    # 4. Plot the budging trajectories
    # output_filename_trajectories = "budging_trajectories_subclonal_gland_fission.png"
    # plot_budging_trajectories(simulation, output_filename_trajectories, sampling_interval=50)
    
    # Example: Perform biopsies and create a correlation plot
    # biopsy_locations = [
    #     (45, 45),
    #     (45, 105),
    #     (75, 75),
    #     (105, 45),
    #     (105, 105)  # Center of the grid
    # ]
    # biopsy_size = 15 # A deme square
    # biopsy_data = []
    # for x, y in biopsy_locations:
    #     result = simulation.perform_virtual_biopsy(center_x=x, center_y=y, size=biopsy_size)
    #     if result:
    #         biopsy_data.append(result)

    # if biopsy_data:
    #     simulation.plot_biopsy_correlation(biopsy_data, "re-plotted_correlation.png")

    # Example: Re-generate the multi-page PDF
    # simulation.plot_beta_heatmap_pdf("re-plotted_heatmap.pdf")

    # Test saving and loading a simple object
    # test_object = {"key": "value"}
    # test_filepath = "test.pkl"

    # # Save the object
    # with open(test_filepath, 'wb') as f:
    #     pickle.dump(test_object, f)

    # # Load the object
    # with open(test_filepath, 'rb') as f:
    #     loaded_object = pickle.load(f)
    # print("Loaded object:", loaded_object)

    