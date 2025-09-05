import numpy as np
import random
import math
from scipy import stats, linalg
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
from scipy.special import logsumexp, gammaln, logit, softmax
from time import time
import dynesty
from dynesty import NestedSampler
from dynesty.results import print_fn
from multiprocess import Pool, cpu_count
import joblib
from skimage.morphology import binary_dilation, square
from scipy.spatial import KDTree
import seaborn as sns
from scipy.stats import wasserstein_distance
from matplotlib.backends.backend_pdf import PdfPages
import colorsys
from matplotlib.collections import LineCollection
from statannotations.Annotator import Annotator
import pickle

import evoflux
from evoflux import generate_next_timepoint, multinomial_rvs, initialise_cancer, grow_cancer
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['axes.labelsize'] = 20

"""
We use params dictionary to manage simulation parameters, which includes:
- dim_grid: The dimension of the square grid (Nx = Ny).
- migration_type: The type of spatial model to use (0, 1, 2, or 3).
- migration_edge_only: Whether migration is restricted to the edges of the grid.
- G: The carrying capacity of each deme
- mu, gamma, nu, zeta: Transition rates for methylation states.
- init_migration_rate: Initial migration rate.
- init_pop: Initial population size in the center deme.

- baseline_death_rate: Baseline death rate for cells.
- density_dept_death_rate: Death rate dependent on cell density.
"""

baseline_death_rate = 0.1
density_dept_death_rate = 100
# Define a custom function for the lower triangle to show Wasserstein distance
def lower_triangle_wasserstein(x, y, **kwargs):
    ax = plt.gca()  # Get the current axis
    # Calculate the Wasserstein distance
    distance = wasserstein_distance(x, y)
    # Plot the scatter plot
    ax.scatter(x, y, alpha=0.5, **kwargs)
    # Add the Wasserstein distance as text in the center of the plot
    ax.text(0.5, 0.5, f"W={distance:.2f}", fontsize=10, ha='center', va='center', transform=ax.transAxes)

def plot_beta_comparison(betas_group1, betas_group2, group1_label, group2_label, title, output_filename=None):
    """
    Plots the histogram and calculates the Wasserstein distance between two groups of beta values.

    Args:
        betas_group1 (list or np.array): Beta values for the first group.
        betas_group2 (list or np.array): Beta values for the second group.
        group1_label (str): Label for the first group (e.g., "Gland Fission").
        group2_label (str): Label for the second group (e.g., "Boundary Growth").
        title (str): Title for the plot.
        output_filename (str, optional): If provided, saves the plot to this file.
    """
    if len(betas_group1) > 0 and len(betas_group2) > 0:
        # Calculate Wasserstein distance
        w_distance = wasserstein_distance(betas_group1, betas_group2)
        print(f"\n--- Comparison Results ---")
        print(f"Wasserstein Distance between distributions of {group1_label} and {group2_label}: {w_distance:.4f}")
    else:
        print("\nCould not calculate Wasserstein distance because one of the groups has no results.")
        return

    # Plot the histograms
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(12, 7))
    
    plt.hist(betas_group1, bins=100, alpha=0.7, density=True, label=f'{group1_label} (N={len(betas_group1)})')
    plt.hist(betas_group2, bins=100, alpha=0.7, density=True, label=f'{group2_label} (N={len(betas_group2)})')
    
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel('Beta Value (Methylation Fraction)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=12)
    plt.xlim(0, 1)
    
    # Add text with the Wasserstein distance on the plot
    plt.text(0.05, 0.9, f'Wasserstein Distance: {w_distance:.4f}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
             
    # Show or save the plot
    if output_filename:
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_filename}")
    else:
        plt.show()

def plot_beta_histogram_comparison(betas_gland, betas_boundary, betas_nonspatial, output_filename=None):
    """
    Plots the histogram of beta values for gland fission, boundary growth, and non-spatial models on the same plot.

    Args:
        betas_gland (list or np.array): Beta values for the gland fission model.
        betas_boundary (list or np.array): Beta values for the boundary growth model.
        betas_nonspatial (list or np.array): Beta values for the non-spatial model.
        output_filename (str, optional): If provided, saves the plot to this file.
    """
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(12, 7))
    
    # Plot histograms for each group
    plt.hist(betas_nonspatial, bins=100, alpha=0.45, density=True, label=f'Non-Spatial (N={len(betas_nonspatial)})', color='orange')
    plt.hist(betas_gland, bins=100, alpha=0.45, density=True, label=f'Gland Fission (N={len(betas_gland)})', color='blue')
    plt.hist(betas_boundary, bins=100, alpha=0.45, density=True, label=f'Boundary Growth (N={len(betas_boundary)})', color='green')

    
    # Add title and labels
    plt.title('Comparison of Final Beta Value Distributions', fontsize=16, weight='bold')
    plt.xlabel('Beta Value (Methylation Fraction)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=12)
    plt.xlim(0, 1)
    
    # Show or save the plot
    if output_filename:
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_filename}")
    else:
        plt.show()

# Define a function to calculate the absolute difference and use it to color 
# the scatter plots
# add the legend of wasserstein distance
def scatter_with_color(x, y, s=3, cmap='viridis_r', 
                       vmin = None, vmax = None, 
                       edgecolors = 'Face',
                       linewidths = 0.5,  **kwargs):
    abs_diff = np.abs(x - y)
    plt.scatter(x, y, s, c=abs_diff, cmap=cmap,
                vmin = vmin, vmax = vmax, 
                edgecolors=edgecolors, linewidths=linewidths)
    # Calculate the Wasserstein distance
    # The 'u_values' and 'v_values' are the data points themselves
    distance = wasserstein_distance(u_values=x, v_values=y)
    
    # 3. Add the distance as text to the plot
    # We place it in the top-left corner of the plot axes
    ax = plt.gca()
    ax.text(0.05, 0.9, f'W-dist: {distance:.3f}', 
            transform=ax.transAxes, 
            fontsize=9, 
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))

def plot_longitudinal(df, 
                      outpath = None, 
                      title = None,
                      corner = False,
                      kde = False,
                      color = False,
                      labels = None,
                      edgecolors = 'Face',
                      cmap='viridis_r',
                      vmin = 0, 
                      vmax = 0.25):

    grid = sns.PairGrid(data= df)
    if corner:
        for i in range(grid.axes.shape[0]):
            for j in range(grid.axes.shape[1]):
                if i < j:
                    grid.axes[i, j].set_axis_off()
    else:
        # Map a density plot to the lower triangle
        grid.map_upper(sns.kdeplot, shade=True, thresh=0.05, cmap = 'Blues',
                    cut = 0)

    # Map a histogram to the diagonal
    grid.map_diag(plt.hist, bins = np.linspace(0, 1, 101), alpha = 0.4, linewidth=None)
    if kde:
        grid.map_lower(sns.kdeplot, cut = 0, color = 'k', 
                       shade=False, thresh=0.05, levels = 4,
                       alpha=0.3)
    if color:
        grid.map_lower(scatter_with_color, s = 3, vmin = vmin, vmax = vmax,
                       cmap = cmap, edgecolors = edgecolors,
                       linewidths=0.3)
        # Call the new annotation function for the lower grid
        # grid.map_lower(scatter_and_annotate_distance, s=10, vmin=vmin, vmax=vmax,
        #                cmap=cmap, edgecolors=edgecolors,
        #                linewidths=0.3)
        # norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        # Map a scatter plot to the upper triangle
        grid.map_lower(plt.scatter, s = 3)

    for i in range(grid.axes.shape[0]):
        for j in range(grid.axes.shape[1]):
            if i > j:
                grid.axes[1, 0].set_ylim([0, 1])
    
    if title is not None:
        grid.fig.suptitle(title)

    if labels is not None:
        for i in range(grid.axes.shape[0]):
            for j in range(grid.axes.shape[1]):
                x_var = grid.axes[i, j].get_xlabel()
                y_var = grid.axes[i, j].get_ylabel()
                if x_var in labels.keys():
                    grid.axes[i, j].set_xlabel(labels[x_var])
                if y_var in labels.keys():  
                    grid.axes[i, j].set_ylabel(labels[y_var])

    plt.tight_layout()
    if outpath is not None:
        grid.fig.savefig(outpath, dpi = 600)
        plt.close()

# def convert_to_continuous_colour(series, cmap="Blues"):
#     """Maps a continuous series (like subclonal fraction) to a color palette."""
#     norm = plt.Normalize(series.min(), series.max())
#     return series.map(sns.color_palette(cmap, as_cmap=True)).map(norm)


def convert_to_continuous_colour(series, cmap="Blues", vmin=0, vmax=1):
    """
    Maps a continuous series (like subclonal fraction) to a color palette
    by first normalizing the data.
    """
    # 1. Create a normalizer to map the data from its range [vmin, vmax] to [0, 1]
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # 2. Get the colormap function
    cmap_func = sns.color_palette(cmap, as_cmap=True)
    
    # 3. Define a function that applies the normalization THEN the colormap
    def map_color(val):
        normalized_val = norm(val)
        return cmap_func(normalized_val)
        
    # 4. Apply this correct sequence to every value in the series
    return series.map(map_color)

def plot_demes_from_data(x_coords, y_coords, color_values, dim_grid, output_path):
    """
    Creates and saves a grid plot using a sequential colormap to show deme fullness:
    proportion: population / carrying capacity (N / G).
    """
    if not x_coords:
        print("Warning: No demes to plot.")
        return

    toPlot = np.full((dim_grid, dim_grid), np.nan)

    # Populate the array with the continuous color values for each deme
    for i in range(len(x_coords)):
        plot_x = int(x_coords[i])
        plot_y = int(y_coords[i])
        if 0 <= plot_x < dim_grid and 0 <= plot_y < dim_grid:
            toPlot[plot_y, plot_x] = color_values[i]
    
    # from colormap sequential
    # use heat colormap for continuous color representation
    # cmap = plt.get_cmap('YlOrRd')  # Sequential colormap from yellow to red
    # Use `set_bad` to make any `np.nan` values in our data appear white.
    cmap = mpl.cm.get_cmap("YlOrRd").copy()
    cmap.set_bad(color='white')

    # Normalize to the range [0,1]
    norm = mpl.colors.Normalize(vmin=0, vmax=1.0)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8)) # Adjusted figsize for color bar
    mesh = ax.pcolormesh(toPlot, cmap=cmap, norm=norm)
    
    # Add a color bar to serve as a legend for the continuous colors.
    # set the colorbar bigger?
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Deme Fullness (Population / Capacity)')
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"Saved plot to {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close(fig)


def plot_demes_from_data_subclone(x_coords, y_coords, rgb_colors, dim_grid, output_path):
    """
    Creates and saves a grid plot using direct RGB values for each deme.
    Includes a 2D color legend to explain the blending of two populations.
    """

    # Initiate the plotting array to hold RGB values. Shape: (height, width, 3 channels)
    # We use (1,1,1) for white, representing empty space.
    toPlot = np.ones((dim_grid, dim_grid, 3), dtype=float)

    # Populate the array with the (R, G, B) color tuple for each existing deme.
    for i in range(len(x_coords)):
        plot_x = int(x_coords[i])
        plot_y = int(y_coords[i])
        if 0 <= plot_x < dim_grid and 0 <= plot_y < dim_grid:
            toPlot[plot_y, plot_x, :] = rgb_colors[i] # Assign the RGB tuple
            
   

    fig, ax = plt.subplots(figsize=(12, 12))
    
    # imshow can directly render an (M, N, 3) array of RGB values.
    ax.imshow(toPlot, interpolation='none', origin='lower')
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    # --- Create a Custom 2D Color Legend ---
    # This creates a new small set of axes on the main figure for the legend
    legend_ax = fig.add_axes([0.8, 0.8, 0.15, 0.15])
    
    # Define the base colors again, normalized to 0-1
    color_subclone = np.array([0, 77/255, 64/255])    # Green
    color_origin = np.array([255/255, 193/255, 7/255]) # Yellow

    # Create the 2D gradient for the legend
    legend_grid = np.zeros((64, 64, 3))
    origin_vals = np.linspace(0, 1, 64)   # Corresponds to y-axis
    subclone_vals = np.linspace(0, 1, 64) # Corresponds to x-axis

    for i, o_frac in enumerate(origin_vals):
        for j, s_frac in enumerate(subclone_vals):
            # Apply the same blending logic as in the main function
            blended_color = (color_origin * o_frac) + (color_subclone * s_frac)
            legend_grid[i, j, :] = np.clip(blended_color, 0, 1)

    legend_ax.imshow(legend_grid, origin='lower')
    legend_ax.set_title('Clone Mix', fontsize=10)
    legend_ax.set_xlabel('Subclone (Green)', fontsize=8)
    legend_ax.set_ylabel('Original (Yellow)', fontsize=8)
    legend_ax.set_xticks([0, 63])
    legend_ax.set_xticklabels(['0%', '100%'])
    legend_ax.set_yticks([0, 63])
    legend_ax.set_yticklabels(['0%', '100%'])
    # # blue and red version:
    # # Create a 2D gradient from blue to red to purple
    # legend_grid = np.zeros((64, 64, 3))
    # r_vals = np.linspace(0, 1, 64)
    # b_vals = np.linspace(0, 1, 64)
    # legend_grid[:, :, 0] = r_vals[np.newaxis, :] # Red channel varies along x-axis
    # legend_grid[:, :, 2] = b_vals[:, np.newaxis] # Blue channel varies along y-axis
    
    # legend_ax.imshow(legend_grid, origin='lower')
    # legend_ax.set_title('Clone Mix', fontsize=10)
    # legend_ax.set_xlabel('Original', fontsize=8)
    # legend_ax.set_ylabel('Subclone', fontsize=8)
    # legend_ax.set_xticks([0, 63])
    # legend_ax.set_xticklabels(['0%', '100%'])
    # legend_ax.set_yticks([0, 63])
    # legend_ax.set_yticklabels(['0%', '100%'])

    # HSV version---
    # legend_ax = fig.add_axes([0.7, 0.7, 0.2, 0.2]) # Positioned the legend
    
    # # Create a 2D gradient that mimics the HSV mapping
    # legend_grid = np.zeros((64, 64, 3))
    # for i in range(64): # Y-axis: Total Fullness
    #     for j in range(64): # X-axis: Subclone Fraction
    #         subclone_frac = j / 63.0
    #         total_full = i / 63.0
            
    #         hue = subclone_frac * 0.66
    #         saturation = 1.0
    #         value = total_full**0.5 # Apply the same non-linear transform
            
    #         legend_grid[i, j, :] = colorsys.hsv_to_rgb(hue, saturation, value)
    
    #legend_ax.imshow(legend_grid, origin='lower', aspect='auto')
    # legend_ax.set_title('Legend', fontsize=10)
    # legend_ax.set_xlabel('Subclone Fraction', fontsize=8)
    # legend_ax.set_ylabel('Total Fullness', fontsize=8)
    # legend_ax.set_xticks([0, 63])
    # legend_ax.set_xticklabels(['0%', '100%'])
    # legend_ax.set_yticks([0, 63])
    # legend_ax.set_yticklabels(['0%', '100%'])


    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"Saved plot to {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close(fig)


def multivariate_hypergeometric(colors, nsample, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    colors = np.asarray(colors, dtype=int)
    variate = np.zeros(np.shape(colors), dtype=int)
    
    # `remaining` is the cumulative sum of `colors` from the last
    # element to the first; e.g. if `colors` is [3, 1, 5], then
    # `remaining` is [9, 6, 5].
    remaining = np.cumsum(colors[::-1, :], axis=0)[::-1, :].astype(int)
    nsamples = np.full(np.shape(colors)[1], nsample, dtype=int)
    for i in range(np.shape(colors)[0]-1):
        # print(f"Sampling from colors[{i}, :]: {colors[i, :]}, remaining[{i+1}, :]: {remaining[i+1, :]}, nsamples: {nsamples}")
        variate[i, :] = rng.hypergeometric(colors[i, :], remaining[i+1, :],
                                nsamples).astype(int)
        nsamples -= variate[i, :]

    variate[-1, :] = nsamples.astype(int)

    return variate

class Deme:
    """
    A single deme represent a local patch of cells (a gland) on the grid.
    Each deme has a carrying capacity G, a location coordinate (x, y), and a population of cells,
    and methylation states for each CpG site.
    """
    def __init__(self, x, y, G,  NSIM, rng, creation_time, unique_id):
        self.x = x
        self.y = y
        self.G = G #Cell carrying capacity of this deme
        self.NSIM = NSIM  # Number of simulations for this deme at the beginning of the simulation
        self.N = 0   # total number of cells in this deme
        self.creation_time = creation_time  # Time when this deme was created
        self.rng = rng
        self.unique_id = unique_id # an integer to identify the appearing order 

        # Methylation states for each CpG site
        self.m = np.zeros(NSIM, dtype=int) # Homozygous methylated cells per locus
        self.k = np.zeros(NSIM, dtype=int) # Heterozygous cells per locus
        self.w = np.zeros(NSIM, dtype=int) # Homozygous unmethylated cells per locus

        # Population and methylation states for SUBCLONE
        self.N2 = 0
        self.m2 = np.zeros(NSIM, dtype=int)
        self.k2 = np.zeros(NSIM, dtype=int)
        self.w2 = np.zeros(NSIM, dtype=int)
        
        self.beta_value = np.zeros(NSIM, dtype=float) # Methylation fraction per locus
        self.subclonal_fraction = 0
        self.empty_neighbor_count = 0 

    def calculate_beta_value(self):
        """
        Calculate the methylation fraction: betacancer.
        """
        if self.N + self.N2 > 0:
            if self.N2 >0: 
                assert np.all(self.m + self.k + self.w == self.N),\
                    f"Total cells in deme {self.unique_id} does not match: {self.m + self.k + self.w} != {self.N}"
                assert np.all(self.m2 + self.k2 + self.w2 == self.N2),\
                    f"Total subclone cells in deme {self.unique_id} does not match: {self.m2 + self.k2 + self.w2} != {self.N2}"
                total_N = self.N + self.N2
                total_m = self.m + self.m2
                total_k = self.k + self.k2
                self.beta_value = (total_k + 2 * total_m) / (2 * total_N)
            else:
                assert np.all(self.m + self.k + self.w == self.N),\
                    f"Total cells in deme {self.unique_id} does not match: {self.m + self.k + self.w} != {self.N}"
                # self.beta_value = (self.k + 2 * self.m) / (2 * self.N)
                total_N = self.N + self.N2
                total_m = self.m + self.m2
                total_k = self.k + self.k2
                self.beta_value = (total_k + 2 * total_m) / (2 * total_N)
        else:
            self.beta_value = np.zeros(self.NSIM, dtype=float)

    def calculate_subclonal_fraction(self):
        total_pop = self.N + self.N2
        self.subclonal_fraction = self.N2 / total_pop if total_pop > 0 else 0
    
    def update_methylation(self, dt, params, dt2=0):
        """
        Applies methylation transitions to the cells within this deme for a time dt.
        Arguments:
            dt: time step for the simulation - float
            params: dictionary containing the parameters for the simulation: mu, gamma, nu, zeta
            m,k,w: demes methylation states 
        """
        if self.N + self.N2 == 0: 
            print("Warning: Deme is empty, no methylation update will be performed.")
            return
        gamma = params['gamma']
        mu = params['mu']
        nu = params['nu']
        zeta = params['zeta']
        rng = self.rng
        NSIM = len(self.m)
        old_w = self.w
        # Use sequential rounds of binomial sampling to calculate how many cells transition between each state
        m_to_k, k_out, w_to_k = rng.binomial(
                                    n = (self.m, self.k, self.w), 
                                    p = np.tile([2*gamma*dt, 
                                        (nu + zeta)*dt, 2*mu*dt], [NSIM, 1]).T)

        k_to_m = rng.binomial(n=k_out, p = np.repeat(nu / (nu + zeta), NSIM))

        self.m = self.m - m_to_k + k_to_m
        self.k = self.k - k_out + m_to_k + w_to_k
        self.w = self.N - self.m - self.k
        k_to_w = k_out - k_to_m
        # check if w==w+ k_to_w-w_to_k
        new_w = old_w + k_to_w - w_to_k
        if np.any(new_w!=self.w):
            print("m:", self.m)
            print("k:", self.k)
            print("w:", self.w)
            print("m_to_k:", m_to_k)
            print("k_out:", k_out)
            print("k_to_m:", k_to_m)
            print("k_to_w:", k_to_w)
            print("w_to_k:", w_to_k)
        assert np.all(new_w==self.w),\
            f"w is not conserved: {old_w} + {k_to_w} - {w_to_k} != {self.w}"
        
        if self.N2 ==0:
            return (self.m, self.k, self.w)
        else:
            # Update the subclone methylation states
            old_w2 = self.w2
            m_to_k2, k_out2, w_to_k2 = rng.binomial(
                                    n = (self.m2, self.k2, self.w2), 
                                    p = np.tile([2*gamma*dt2, 
                                        (nu + zeta)*dt2, 2*mu*dt2], [NSIM, 1]).T)

            k_to_m2 = rng.binomial(n=k_out2, p = np.repeat(nu / (nu + zeta), NSIM))

            self.m2 = self.m2 - m_to_k2 + k_to_m2
            self.k2 = self.k2 - k_out2 + m_to_k2 + w_to_k2
            self.w2 = self.N2 - self.m2 - self.k2

            k_to_w2 = k_out2 - k_to_m2
            # check if w==w+ k_to_w-w_to_k
            new_w2 = old_w2 + k_to_w2 - w_to_k2
            if np.any(new_w2!=self.w2):
                print("m2:", self.m2)
                print("k2:", self.k2)
                print("w2:", self.w2)
                print("m_to_k2:", m_to_k2)
                print("k_out2:", k_out2)
                print("k_to_m2:", k_to_m2)
                print("k_to_w2:", k_to_w2)
                print("w_to_k2:", w_to_k2)
            return (self.m, self.k, self.w, self.m2, self.k2, self.w2)
       
#---- ---- ---- ---- ---- ---- -------------------------------------_-``
    #need to correct
    def spatial_structure(self, grid, migration_type):
        if migration_type == 0:
            self.random_cell_migrates(grid)
        elif migration_type == 1:
            self.invasive_cell_migrates(grid)
        elif migration_type == 2:
            if self.N >= self.K:
                self.gland_fission(grid)
        elif migration_type == 3:
            if self.N >= self.K and self.is_at_boundary(grid):
                self.boundary_fission(grid)

    # for cell_invasion model
    def choose_event_for(birth_rate, death_rate, migration_rate, rng):
        total = birth_rate + death_rate + migration_rate
        r = rng.uniform(0, 1)
        if r < birth_rate / total:
            return 'birth'
        elif r < (birth_rate + death_rate) / total:
            return 'death'
        else:
            return 'migration'

class Grid:
    """
    Grid represent the global spatial structure of the simulation, containing multiple demes.
    Manages the spatial arrangement of demes: 
    Uses a dictionary to map coordinate tuples (x, y) to Deme objects.
    """
    def __init__(self, dim_grid,migration_type, rng):
        """
        dim_grid: An integer for the width and height of the square simulation grid
        migration_type: The type of spatial model to use:
                        0: Random cell migration
                        1: Invasive cell migration
                        2: Gland fission
                        3: Boundary growth fission
        """
        self.dim_grid = dim_grid
        # The dictionary to store demes, keyed by (x, y) coordinates
        self.demes = {}
        self.rng=rng
        self.migration_type = migration_type
        # Initialize the grid with a single deme at the center is at the class Simulation initialization
        
    def add_deme(self, deme):
        """
        Adds a deme to the grid at its coordinates (x,y).
        """
        if (deme.x, deme.y) not in self.demes:
            self.demes[(deme.x, deme.y)] = deme

    def get_deme(self, x, y):
        """
        Retrieves a deme from a given coordinate (x,y), returns None if not found.
        """
        return self.demes.get((x, y))

    def get_random_neighbor_coords(self, x, y):
        """
        In cell migration:
        Selects a random neighboring coordinate of the destination deme:
        direction: up, down, left, right from the current deme (x,y).
        """
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        # if self.params['diagonal']:
        #     neighbors.extend([(x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)])
        neighbors.extend([(x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)])
        # choose random neighbor until it is empty
        neighbors = self.rng.permutation(neighbors)  # Shuffle neighbors to randomize the search order
        for neighbor in neighbors:
            if 0 <= neighbor[0] < self.dim_grid and 0 <= neighbor[1] < self.dim_grid:
                if self.get_deme(neighbor[0], neighbor[1]) is None:
                    return neighbor
        return None  # No empty neighbor found

    
class Simulation:
    """
    The main class that handles the simulation logic.
    Initializes the grid, demes, and manages the mythlation state transition within the deme, the cell growth,...
    and spatial model.
    """
    def __init__(self, params):
        self.params = params
        self.rng = np.random.default_rng()
        self.grid = Grid(params['dim_grid'], params['migration_type'], self.rng)
        self.tau = params['tau'] # Start clock at tau
        
        #self.migration_edge_only= params.get('migration_edge_only', True)  # Default to True if not specified

        self.num_all_cells = 0
        self.beta_history = []
        self.ordered_deme_ids = [] # This will now store integer IDs to track the beta values of each deme
        self.seen_deme_ids = set()   # store IDs
        self.next_deme_id = 0      # A counter to generate new unique IDs
        self.initialize_simulation()
        
    def get_new_deme_id(self):
        """Generates and returns a new unique integer ID for a deme."""
        new_id = self.next_deme_id
        self.next_deme_id += 1
        return new_id
    
    def initialize_simulation(self):
        """
        Initialize the simulation with a single deme at the center of the grid at time tau,
        initialize epigenetic states by initialise_cancer from Evoflux model,
        initialize the birth and migration rates (for cell invasion model).
        """

        m0, k0, w0 = initialise_cancer(
            self.params['tau'], self.params['mu'], self.params['gamma'],
            self.params['nu'], self.params['zeta'], self.params['NSIM'], self.rng
        )
        # Assign the first unique ID
        initial_id = self.get_new_deme_id()
        center_x = self.params['dim_grid'] // 2
        center_y = self.params['dim_grid'] // 2
        initial_deme = Deme(
            center_x, center_y, self.params['G'], 
            creation_time=self.params['tau'], NSIM=self.params['NSIM'], rng=self.rng,
            unique_id=initial_id
        )
        
        initial_deme.N = self.params['init_pop']
        initial_deme.m = m0 
        initial_deme.k = k0 
        initial_deme.w = w0
        print("initial deme N:", initial_deme.N)
        print("initial deme m:", initial_deme.m)
        print("initial deme k:", initial_deme.k)
        print("initial deme w:", initial_deme.w) 
        initial_deme.calculate_beta_value()
        initial_deme.calculate_subclonal_fraction()
        initial_deme.creation_time = self.tau # Set the creation time to tau
        self.grid.add_deme(initial_deme)

        # Initialize the tracking lists with the first deme's ID
        self.ordered_deme_ids.append(initial_id)
        self.seen_deme_ids.add(initial_id)
        # set initial birth, death, and migration rates
        # self.birth_rate = self.params['init_birth_rate']
        # self.death_rate = self.params['baseline_death_rate'] 
        # self.migration_rate = self.params['init_migration_rate']
    
    # def get_deme_coordinates(self, origin_deme):
    #     """
    #     In deme fission:
    #     Determines the coordinates (x, y) for a new deme based on the origin deme's coordinates.
    #     If migration_edge_only is True, it returns the coordinates adjacent to the origin deme.
    #     If migration_edge_only is False, it returns the starting point of the budging process.

    #     Args:
    #         origin_deme: The deme that is undergoing fission.
    #     Returns:
    #         A tuple (x, y) representing the coordinates for the new deme.
    #     """
    #     old_x, old_y = origin_deme.x, origin_deme.y
    #     params = self.params
    #     migration_edge_only=params['migration_edge_only']
    #     #  1. Fission is restricted to the tumour boundary (Boundary growth model)
    #     if migration_edge_only:  
    #         return self.grid.get_random_neighbor_coords(old_x, old_y)

    #     #   2. Fission can happen anywhere, cause budge demes (Gland fission model)
    #     else:
    #         # find the starting location for the `budge_demes` chain reaction: 
    #         # simulates a line of force radiating from the dividing deme in a random direction and finds where it intersects the grid boundary.

    #         theta = self.rng.uniform(0, 2 * math.pi)
            
    #         if abs(theta - math.pi/2) < 1e-5 or abs(theta - 3*math.pi/2) < 1e-5:
    #             tan_theta = 1e9 
    #         else:
    #             tan_theta = math.tan(theta)

    #         dim = self.grid.dim_grid
            
    #         # 2. Determine which quadrant the angle points to, relative to the origin deme and which edge of the grid the line will hit first.
    #         if theta >= 0 and theta < math.pi/2: # Quadrant I (points Down-Right)
    #             quadrant_boundary_slope = (dim - 1 - old_y) / (dim - 1 - old_x) if (dim - 1 - old_x) != 0 else float('inf')
    #             if tan_theta < quadrant_boundary_slope:
    #                 # Hits the RIGHT edge
    #                 x_to_fill = dim - 1
    #                 y_to_fill = round(old_y + (dim - 1 - old_x) * tan_theta)
    #             else:
    #                 # Hits the BOTTOM edge
    #                 x_to_fill = round(old_x + (dim - 1 - old_y) / tan_theta)
    #                 y_to_fill = dim - 1
    #         elif theta >= math.pi/2 and theta < math.pi: # Quadrant II (points Down-Left)
    #             quadrant_boundary_slope = (dim - 1 - old_y) / (0 - old_x) if old_x != 0 else float('inf')
    #             if tan_theta > quadrant_boundary_slope:
    #                 # Hits the LEFT edge
    #                 x_to_fill = 0
    #                 y_to_fill = round(old_y - old_x * tan_theta)
    #             else:
    #                 # Hits the BOTTOM edge
    #                 x_to_fill = round(old_x + (dim - 1 - old_y) / tan_theta)
    #                 y_to_fill = dim - 1
    #         elif theta >= math.pi and theta < 3*math.pi/2: # Quadrant III (points Up-Left)
    #             quadrant_boundary_slope = (0 - old_y) / (0 - old_x) if old_x != 0 else float('inf')
    #             if tan_theta < quadrant_boundary_slope:
    #                 # Hits the LEFT edge
    #                 x_to_fill = 0
    #                 y_to_fill = round(old_y - old_x * tan_theta)
    #             else:
    #                 # Hits the TOP edge
    #                 x_to_fill = round(old_x - old_y / tan_theta)
    #                 y_to_fill = 0
    #         else: # Quadrant IV (points Up-Right)
    #             quadrant_boundary_slope = (0 - old_y) / (dim - 1 - old_x) if (dim - 1 - old_x) != 0 else float('inf')
    #             if tan_theta > quadrant_boundary_slope:
    #                 # Hits the RIGHT edge
    #                 x_to_fill = dim - 1
    #                 y_to_fill = round(old_y + (dim - 1 - old_x) * tan_theta)
    #             else:
    #                 # Hits the TOP edge
    #                 x_to_fill = round(old_x - old_y / tan_theta)
    #                 y_to_fill = 0
           
    #         return (int(x_to_fill), int(y_to_fill))


    def get_nearest_empty_spot_in_direction(self, start_x, start_y, angle):
        """
        Walks from a starting point in a given direction until it finds the first
        empty deme. Returns the coordinates of that deme.
        """
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        current_x, current_y = float(start_x), float(start_y)

        # Walk step-by-step, up to the maximum possible grid dimension
        for _ in range(self.grid.dim_grid * 2):
            current_x += dx
            current_y += dy
            
            ix, iy = round(current_x), round(current_y)

            # Check if we've gone off the grid
            if not (0 <= ix < self.grid.dim_grid and 0 <= iy < self.grid.dim_grid):
                print("Reached the edge of the grid without finding an empty deme. Within the loop")
                print("Current coordinates:", ix, iy)
                return None # No empty spot found before hitting the boundary

            # If we find an empty spot, return its coordinates
            if self.grid.get_deme(ix, iy) is None:
                return (ix, iy)

        print("No empty deme found until", ix, iy)        
        return None # No empty spot found


    def get_deme_coordinates(self, origin_deme):
        """
        Determines the coordinates for a new deme.
        For the budging model, it now finds the nearest empty spot in a random direction.
        """
        old_x, old_y = origin_deme.x, origin_deme.y
        params = self.params
        migration_edge_only = params['migration_edge_only']

        # --- MODE 1: Boundary Growth Model (migration_edge_only = True) ---
        if migration_edge_only:
            # return a random adjacent neighbor.
            return self.grid.get_random_neighbor_coords(old_x, old_y)

        # --- MODE 2: Gland Fission with Cohesive Growth (migration_edge_only = False) ---
        else:
            # find the closest empty spot in a random direction.
            
            # 1. Pick a random angle.
            theta = self.rng.uniform(0, 2 * math.pi)
            
            # 2. "Walk" from the origin deme in that direction and find the first empty cell.
            new_coords = self.get_nearest_empty_spot_in_direction(old_x, old_y, theta)
            
            # 3. If a spot is found, return it.
            if new_coords:
                return new_coords
            else:
                # If no empty spot is found in that direction (i.e., the path is blocked
                # all the way to the edge), the fission attempt in this direction fails.
                # We return the origin's own coordinates as a signal of failure.
                print(f"fail to find an empty deme at the {old_x, old_y} in direction {theta}") 
                return (old_x, old_y)
    
    def calculate_empty_neighbors(self, deme):
        """Calculates and returns the number of empty neighboring demes for a given deme."""
        count = 0
        # Define potential neighbor coordinates
        neighbors = [(deme.x + 1, deme.y), (deme.x - 1, deme.y),
                     (deme.x, deme.y + 1), (deme.x, deme.y - 1)]
        
        # Include diagonal neighbors if the simulation parameter: diagonal migration is True
        migration_diagonal = self.params['migration_diagonal']
        if migration_diagonal:
            neighbors.extend([(deme.x + 1, deme.y + 1), (deme.x - 1, deme.y - 1),
                              (deme.x + 1, deme.y - 1), (deme.x - 1, deme.y + 1)])

        for nx, ny in neighbors:
            # Check if the neighbor is within the grid boundaries
            if 0 <= nx < self.grid.dim_grid and 0 <= ny < self.grid.dim_grid:
                # Check if the neighbor coordinate is empty (no deme object)
                if self.grid.get_deme(nx, ny) is None:
                    count += 1
        return count
    
    def budge_demes(self, origin_deme, x_to_fill, y_to_fill):
        """
        Shifts a line of demes to make space for a new deme next to the origin.
        x_to_fill, y_to_fill: Provide the direction of budging, which is the first empty spot
        """
        old_x, old_y = origin_deme.x, origin_deme.y
        # print(f"Budge demes: origin_deme at ({old_x}, {old_y}), target coordinates ({x_to_fill}, {y_to_fill})")
        # This loop continues until an empty spot is created next to the origin_deme.
        while True:
            xdiff = old_x - x_to_fill
            ydiff = old_y - y_to_fill
            
            # Determine the next deme in the line to be moved.
            # The logic with ROOT2-1 ensures diagonal-like pushing on a square grid.
            ROOT2_MINUS_1 = math.sqrt(2) - 1
            hypoten = math.sqrt(xdiff**2 + ydiff**2) 
            #print(f"Budge: xdiff={xdiff}, ydiff={ydiff}")
            x_to_move, y_to_move = x_to_fill, y_to_fill
            if abs(xdiff / hypoten) >= ROOT2_MINUS_1:
                x_to_move += np.sign(xdiff)
            if abs(ydiff / hypoten) >= ROOT2_MINUS_1:
                y_to_move += np.sign(ydiff)

            # If the next deme to move is the origin itself, no more budging is needed.
            if (x_to_move, y_to_move) == (old_x, old_y):
                break

            # Move the deme from (x_to_move, y_to_move) to (x_to_fill, y_to_fill)
            deme_to_move = self.grid.get_deme(x_to_move, y_to_move)
            if deme_to_move:
                # Update the grid dictionary if the deme is not empty
                del self.grid.demes[(x_to_move, y_to_move)]
                self.grid.demes[(x_to_fill, y_to_fill)] = deme_to_move
                # Update the deme's internal coordinates
                deme_to_move.x = x_to_fill
                deme_to_move.y = y_to_fill
            
            # The next empty spot we need to fill is where we just moved a deme from.
            x_to_fill, y_to_fill = x_to_move, y_to_move
            
        return (x_to_fill, y_to_fill) # Return the final empty coordinates next to the origin.

    def is_deme_at_boundary(self, deme):
        """Helper function to check if a deme has at least one empty neighbor."""
        neighbors = [(deme.x + 1, deme.y), (deme.x - 1, deme.y),
                    (deme.x, deme.y + 1), (deme.x, deme.y - 1)]
        # if diagonal migration is allowed           
        migration_diagonal = self.params['migration_diagonal']
        if migration_diagonal:
            neighbors += [(deme.x + 1, deme.y + 1), (deme.x - 1, deme.y - 1),
                            (deme.x + 1, deme.y - 1), (deme.x - 1, deme.y + 1)]

        neighbors = self.rng.permutation(neighbors)  # Shuffle neighbors to randomize the search order
        for nx, ny in neighbors:
            if (0 <= nx < self.grid.dim_grid and 0 <= ny < self.grid.dim_grid):
                if self.grid.get_deme(nx, ny) is None:
                    return nx, ny # Found an empty neighbor
        return None

    def move_cells(self, origin_deme, new_deme, dividing_beyond_the_edge):
        """
        Moves half the cells and their methylation states from an origin deme to a new deme.
        For odd counts, the extra cell/state is randomly assigned to each deme with 50/50 probability.
        The methylation states m,k,w always satisfy: m+k+w = N for both demes, in every site.

        Args:
            origin_deme (Deme): The deme that is being split.
            new_deme (Deme): The new deme that will receive half of the cells.
            dividing_beyond_the_edge (bool): If True, the origin deme is on the edge and simply halves its population.
        """
        
        if np.any(origin_deme.m + origin_deme.k + origin_deme.w != origin_deme.N):
            print("x and y:", origin_deme.x, origin_deme.y)
            print("origin N:", origin_deme.N)
            print("origin m:", origin_deme.m)
            print("origin k:", origin_deme.k)
            print("origin w:", origin_deme.w)
           
        assert np.all(origin_deme.m + origin_deme.k + origin_deme.w == origin_deme.N), \
            f"originorigin N mismatch mkw {origin_deme.N}"

        assert origin_deme.N+origin_deme.N2 > 0, "Origin deme must have a positive population size."

        #  fission between two demes
        n_for_new_deme = origin_deme.N // 2
        n_for_origin_deme = origin_deme.N // 2
        if origin_deme.N % 2 != 0:
            # a bool to randomly assign the extra cell to one of the demes
            new_extra = self.rng.binomial(n=1, p=0.5)  # 50/50 chance
            # Randomly assign the extra cell to one of the demes
            if new_extra:
                n_for_new_deme += 1
            else:
                n_for_origin_deme += 1

        # Assign final counts to the new deme and update origin deme
        origin_deme.N = n_for_origin_deme
        
        # Split the methylation state arrays (m, k, w)
        # Initialize the new deme's methylation arrays
        new_m = np.zeros_like(origin_deme.m)
        new_k = np.zeros_like(origin_deme.k)
        new_w = np.zeros_like(origin_deme.w)

        odd_mkw = np.stack([origin_deme.m%2, origin_deme.k%2, origin_deme.w%2], axis=0)
        
        odd_number = np.sum(odd_mkw, axis=0)

        if n_for_new_deme > n_for_origin_deme:
            odd_number_to_new = odd_number//2 + 1
        else:
            odd_number_to_new = odd_number//2
        
        # use Multivariate Hypergeometric distribution to sample the odd numbers
        new_odd = multivariate_hypergeometric(odd_mkw, odd_number_to_new, self.rng)
        even_mkw = np.stack([origin_deme.m//2, origin_deme.k//2, origin_deme.w//2], axis=0).astype(np.int64)
        
        new_mkw = even_mkw + new_odd
        new_m, new_k, new_w = new_mkw[0], new_mkw[1], new_mkw[2]
        # print("type of new_m:", new_m.dtype)
        om=origin_deme.m
        ow=origin_deme.w
        ok=origin_deme.k
        
        origin_deme.m -= new_m
        origin_deme.k -= new_k
        origin_deme.w -= new_w

        if not dividing_beyond_the_edge:
            #new deme is not None
            new_deme.N = n_for_new_deme
            new_deme.m = new_m
            new_deme.k = new_k
            new_deme.w = new_w

         
        if np.all(origin_deme.m + origin_deme.k + origin_deme.w!=origin_deme.N):
            print("origin m:", om)
            print("origin k:", ok)
            print("origin w:", ow)
            print("N", origin_deme.N)
            print("m", origin_deme.m)
            print("k", origin_deme.k)
            print("w", origin_deme.w)
            print("new N", new_deme.N)
            print("new m", new_deme.m)
            print("new k", new_deme.k)
            print("new w", new_deme.w)
        if np.any(new_deme.m)<0 or np.any(new_deme.k)<0 or np.any(new_deme.w)<0:
            print("Negative methylation states detected in new deme!")
            print(f"New deme m: {new_deme.m}, k: {new_deme.k}, w: {new_deme.w}")
            print(f"New deme N: {new_deme.N}")
        if np.any(origin_deme.m)<0 or np.any(origin_deme.k)<0 or np.any(origin_deme.w)<0:
            print("Negative methylation states detected in origin deme!")
            print(f"Origin deme m: {origin_deme.m}, k: {origin_deme.k}, w: {origin_deme.w}")
            print(f"Origin deme N: {origin_deme.N}")

        assert np.all(origin_deme.m + origin_deme.k + origin_deme.w==origin_deme.N), \
             f"Origin deme N mismatch m+k+w"
        assert np.all(new_deme.m + new_deme.k + new_deme.w == new_deme.N), \
            f"New deme N mismatch m+k+w, m={new_deme.m}, k={new_deme.k}, w={new_deme.w}, N={new_deme.N}"
        assert np.all(origin_deme.m>=0) and np.all(origin_deme.k>=0) and np.all(origin_deme.w >= 0), \
            f"Origin deme has negative methylation states: m={origin_deme.m}, k={origin_deme.k}, w={origin_deme.w}, N={origin_deme.N}"
        # Recalculate methylation fractions for both demes
        # origin_deme.calculate_beta_value()
        # new_deme.calculate_beta_value()

        if origin_deme.N2 >0:
            # move cells as well for the subclonal
            if np.any(origin_deme.m2 + origin_deme.k2 + origin_deme.w2 != origin_deme.N2):
                print("origin N:", origin_deme.N2)
                print("origin m:", origin_deme.m2)
                print("origin k:", origin_deme.k2)
                print("origin w:", origin_deme.w2)
            
            assert np.all(origin_deme.m2 + origin_deme.k2 + origin_deme.w2 == origin_deme.N2), \
                f"originorigin N mismatch mkw {origin_deme.N2}"

            assert origin_deme.N2 > 0, "Origin deme must have a positive population size."

            #  fission between two demes
            n_for_new_deme2 = origin_deme.N2// 2
            n_for_origin_deme2 = origin_deme.N2 // 2
            if origin_deme.N2 % 2 != 0:
                # a bool to randomly assign the extra cell to one of the demes
                new_extra = self.rng.binomial(n=1, p=0.5)  # 50/50 chance
                # Randomly assign the extra cell to one of the demes
                if new_extra:
                    n_for_new_deme2 += 1
                else:
                    n_for_origin_deme2 += 1

            # Assign final counts to the new deme and update origin deme
            origin_deme.N2 = n_for_origin_deme2
            
            # Split the methylation state arrays (m, k, w)
            # Initialize the new deme's methylation arrays
            new_m2 = np.zeros_like(origin_deme.m2)
            new_k2 = np.zeros_like(origin_deme.k2)
            new_w2 = np.zeros_like(origin_deme.w2)

            odd_mkw2 = np.stack([origin_deme.m2%2, origin_deme.k2%2, origin_deme.w2%2], axis=0)
            
            odd_number2 = np.sum(odd_mkw2, axis=0)

            if n_for_new_deme2 > n_for_origin_deme2:
                odd_number_to_new2 = odd_number2//2 + 1
            else:
                odd_number_to_new2 = odd_number2//2
            
            # use Multivariate Hypergeometric distribution to sample the odd numbers
            print(f"origin deme at ({origin_deme.x}, {origin_deme.y}) has N2= {origin_deme.N2}.")
            new_odd2 = multivariate_hypergeometric(odd_mkw2, odd_number_to_new2, self.rng)
            even_mkw2 = np.stack([origin_deme.m2/2, origin_deme.k2//2, origin_deme.w2//2], axis=0).astype(np.int64)
            new_mkw2 = even_mkw2 + new_odd2
            new_m2, new_k2, new_w2 = new_mkw2[0], new_mkw2[1], new_mkw2[2]

            new_m2 = new_m2.astype(int)
            new_k2 = new_k2.astype(int)
            new_w2 = new_w2.astype(int)

            om2=origin_deme.m2
            ow2=origin_deme.w2
            ok2=origin_deme.k2
            
            origin_deme.m2 -= new_m2
            origin_deme.k2 -= new_k2
            origin_deme.w2 -= new_w2

            if not dividing_beyond_the_edge:
                #new deme is not None
                new_deme.N2 = n_for_new_deme2
                new_deme.m2 = new_m2
                new_deme.k2 = new_k2
                new_deme.w2 = new_w2
        
            
            if np.all(origin_deme.m2 + origin_deme.k2 + origin_deme.w2!=origin_deme.N2):
                print("origin m:", om)
                print("origin k:", ok)
                print("origin w:", ow)
                print("N", origin_deme.N2)
                print("m", origin_deme.m2)
                print("k", origin_deme.k2)
                print("w", origin_deme.w2)
                print("new N", new_deme.N2)
                print("new m", new_deme.m2)
                print("new k", new_deme.k2)
                print("new w", new_deme.w2)
            if np.any(new_deme.m2)<0 or np.any(new_deme.k2)<0 or np.any(new_deme.w2)<0:
                print("Negative methylation states detected in new deme!")
                print(f"New deme m: {new_deme.m2}, k: {new_deme.k2}, w: {new_deme.w2}")
                print(f"New deme N: {new_deme.N2}")
            if np.any(origin_deme.m2)<0 or np.any(origin_deme.k2)<0 or np.any(origin_deme.w2)<0:
                print("Negative methylation states detected in origin deme!")
                print(f"Origin deme m: {origin_deme.m2}, k: {origin_deme.k2}, w: {origin_deme.w2}")
                print(f"Origin deme N: {origin_deme.N2}")

            assert np.all(origin_deme.m2 + origin_deme.k2 + origin_deme.w2==origin_deme.N2), \
                f"Origin deme N mismatch m+k+w"
            assert np.all(new_deme.m2 + new_deme.k2 + new_deme.w2 == new_deme.N2), \
                f"New deme N mismatch m+k+w, m={new_deme.m2}, k={new_deme.k2}, w={new_deme.w2}, N={new_deme.N2}"
            assert np.all(origin_deme.m2>=0) and np.all(origin_deme.k2>=0) and np.all(origin_deme.w2 >= 0), \
                f"Origin deme has negative methylation states: m={origin_deme.m2}, k={origin_deme.k2}, w={origin_deme.w2}, N={origin_deme.N2}"
            # Recalculate methylation fractions for both demes
            # origin_deme.calculate_beta_value()
            # new_deme.calculate_beta_value()

    def deme_fission(self, origin_deme):
        """
        Executes fission for a deme, WITH budging.
        """
        grid_length = self.grid.dim_grid
        for deme in self.grid.demes.values():
            if deme.x == 0 or deme.x == grid_length - 1 or deme.y == 0 or deme.y == grid_length - 1:
                print(f"Found deme ({deme.x}, {deme.y}) already reached the boundary. Stop the deme fission.")
                self.params['exit_code'] = 1
                return None
                    
        migration_edge_only = self.params['migration_edge_only']
        # Fission can only happen if the deme is full.
        assert origin_deme.N+origin_deme.N2 >= self.params['G'], \
            f"Fission failed: Origin deme at ({origin_deme.x}, {origin_deme.y}) has insufficient population size {origin_deme.N}+{origin_deme.N2} < {self.params['G']}."
        dividing_beyond_the_edge = False
        
        coords = self.get_deme_coordinates(origin_deme)
        if coords is not None:
            x_to_fill, y_to_fill = coords
        else:
            if migration_edge_only:
                print(f"Fission failed: No empty neighbor found for deme at ({origin_deme.x}, {origin_deme.y}).")
                # No empty neighbor found, cannot fission
                return

        
        #check if the target coordinates are off-grid or occupied.
        is_off_grid = not (0 <= x_to_fill < self.grid.dim_grid and 0 <= y_to_fill < self.grid.dim_grid)
        is_occupied = self.grid.get_deme(x_to_fill, y_to_fill) is not None if not is_off_grid else False
        
        if migration_edge_only:
            fission_coords = self.get_deme_coordinates(origin_deme)
            # fission fails if the target is off-grid or occupied, decided by is_deme_at_boundary
            if fission_coords is None:
                print(f"Fission failed: No empty neighbor found for deme at ({origin_deme.x}, {origin_deme.y}).")
                # No empty neighbor found, cannot fission
                return
            else:
                x_to_fill, y_to_fill = fission_coords
            
        else: # Budge demes in gland fission model
            if is_off_grid:
                # It means the fissioning deme is on an edge, trying to push outwards.
                dividing_beyond_the_edge = True
                print(f" Attempting to budge demes: Target coordinates ({x_to_fill}, {y_to_fill}) are off-grid.")
                self.params['exit_code']=1
                
                return 
            elif is_occupied:
                print(f" Attempting to budge demes: Target coordinates ({x_to_fill}, {y_to_fill}) are occupied.")
                print(f"the deme is about to fission is", origin_deme.x, origin_deme.y)
            
                self.params['exit_code'] = 1
                return
            else:
                final_coords = self.budge_demes(origin_deme, x_to_fill, y_to_fill)
                x_to_fill, y_to_fill = final_coords
        
        new_deme = None
        if not dividing_beyond_the_edge:
            if (x_to_fill, y_to_fill) in self.grid.demes:
                # If the target coordinates are occupied, we cannot create a new deme.
                print(f"Cannot create a new deme: Target coordinates ({x_to_fill}, {y_to_fill}) are occupied.")
            new_deme = Deme(x_to_fill, y_to_fill, self.params['G'],  self.params['NSIM'], self.rng, 
                            creation_time=self.tau, unique_id=self.get_new_deme_id())
            self.grid.add_deme(new_deme)
            new_deme.creation_time = self.tau  # Set the creation time to tau
            if new_deme.N!=0 or new_deme.m.any() or new_deme.k.any() or new_deme.w.any():
                print(f"New deme at ({x_to_fill}, {y_to_fill}) has non-zero population or methylation states.")
        self.move_cells(origin_deme, new_deme, dividing_beyond_the_edge)
        # return the new deme if it was created, otherwise None, for calculate deme population
        return new_deme

#Here we develop THREE method to decide the direction of budging in the BOUNDARY GROWTH MODEL,
# which is different to random direction in the gland fission model.

# 1. The shortest path to the edge of the grid: only 4 directions: up, down, left, right.
    def find_shortest_path_to_edge(self, deme):
        """
        Calculates the shortest distance from a deme's coordinates to one of the four grid edges
        and returns the direction vector for that path.

        Returns:
            A tuple (dx, dy) representing the direction vector.
        """
        dim = self.grid.dim_grid
        x = deme.x
        y = deme.y

        # Calculate distances to each edge
        distances = {
            (0, -1): y,              # To bottom edge
            (0, 1): dim - 1 - y,     # To top edge
            (-1, 0): x,              # To left edge
            (1, 0): dim - 1 - x      # To right edge
        }
        
        # Find the direction with the minimum distance
        direction = min(distances, key=distances.get)
        theta = math.atan2(direction[1], direction[0])

        return theta

#2. The outward budge direction based on the grid center.
    def find_outward_budge_direction(self, deme):
        """
        Finds an outward-pointing direction from the grid center to the deme,
        then snaps it to the nearest of 8 discrete grid directions (including diagonals).
     
        Returns:
            A tuple (dx, dy) representing one of 8 directions (e.g., (1,0), (1,-1), etc.).
        """
        # Calculate the center of the simulation grid
        grid_center_x = (self.grid.dim_grid ) //2
        grid_center_y = (self.grid.dim_grid ) //2

        # 1. Get the vector from the grid's center to the current deme.
        vector_x = deme.x - grid_center_x
        vector_y = deme.y - grid_center_y

        # 2. Calculate the continuous angle (0 to 2*pi) of this vector.
        angle = math.atan2(vector_y, vector_x)

        # 3. Snap the ideal angle to the nearest 45-degree increment (pi/4 radians).
        #    This selects the closest of the 8 available grid directions.
        snapped_angle = round(angle / (math.pi / 4)) * (math.pi / 4)

        # 4. Convert the snapped angle back into a discrete (dx, dy) grid vector.
        #    The round() function handles any minor floating point inaccuracies.
        dx = int(round(math.cos(snapped_angle)))
        dy = int(round(math.sin(snapped_angle)))

        return snapped_angle

# 3. The direction along with the newest neighbor according to its creation_time.
    def find_direction_away_from_newest_neighbor(self, deme):
        """
        Finds the newest neighboring deme and returns a direction vector pointing to it.
        This simulates budging away from the most recent source of pressure.

        Returns:
            A tuple (dx, dy) representing one of 8 directions.
        """
        newest_neighbor = None
        max_creation_time = -1

        # Define potential neighbor coordinates
        neighbor_coords = [
            (deme.x + 1, deme.y), (deme.x - 1, deme.y),
            (deme.x, deme.y + 1), (deme.x, deme.y - 1)
        ]
        diagonal = self.params['diagonal']
        if diagonal:
            neighbor_coords.extend([
                (deme.x + 1, deme.y + 1), (deme.x - 1, deme.y - 1),
                (deme.x + 1, deme.y - 1), (deme.x - 1, deme.y + 1)
            ])

        # 1. Find the neighbor with the largest creation_time
        for nx, ny in neighbor_coords:
            if 0 <= nx < self.grid.dim_grid and 0 <= ny < self.grid.dim_grid:
                neighbor = self.grid.get_deme(nx, ny)
                if neighbor and neighbor.creation_time > max_creation_time:
                    max_creation_time = neighbor.creation_time
                    newest_neighbor = neighbor

        assert newest_neighbor is not None, \
            f"No newest neighbors found for deme at ({deme.x}, {deme.y})."
        # 2. Determine the direction based on the newest neighbor
        if newest_neighbor:
            # The vector points to the newest neighbor, from the deme that needs to budge
            vector_x = newest_neighbor.x - deme.x
            vector_y = newest_neighbor.y - deme.y
            
            # Normalize the vector into a discrete (dx, dy) grid direction
            dx = int(np.sign(vector_x))
            dy = int(np.sign(vector_y))
            theta = math.atan2(vector_y, vector_x)
            return theta
        else:
            # Fallback: If the deme has no neighbors (unlikely for a trapped deme),
            # use the outward-from-center method as a default.
            return self.find_outward_budge_direction(deme)

    def find_direction_with_least_budge_number(self, origin_deme):
        # This method finds the direction that requires the least number of budges to reach the grid edge.
           
        min_budges = float('inf')
        best_angle = None
        start_x, start_y = origin_deme.x, origin_deme.y

        # Check 8 discrete directions (every 45 degrees)
        for angle_deg in range(0, 360, 45):
            angle_rad = math.radians(angle_deg)
            dx = math.cos(angle_rad)
            dy = math.sin(angle_rad)
            
            current_x, current_y = float(start_x), float(start_y)
            budge_count = 0

            # "Walk" in this direction until we hit the edge of the grid
            while True:
                current_x += dx
                current_y += dy
                ix, iy = round(current_x), round(current_y)

                # Stop if we've gone off the grid
                if not (0 <= ix < self.grid.dim_grid and 0 <= iy < self.grid.dim_grid):
                    break
                    
                # If we cross a space that contains a deme, increment the budge count
                if self.grid.get_deme(ix, iy) is not None:
                    budge_count += 1

            # If this direction is better than the best one found so far, update it
            if budge_count < min_budges:
                min_budges = budge_count
                best_angle = angle_rad

        # If all directions are blocked (unlikely), default to a random angle
        if best_angle is None:
            return self.rng.uniform(0, 2 * math.pi)

        return best_angle   

    def deme_fission_boundary_budge(self, origin_deme):
        """
        Executes a special fission for a trapped (internally surrounded) deme, in the BOUNDARY GROWTH model.
        It budges a line of demes in the direction of the nearest grid edge
        to make space for the new deme.
        The direction can be calculated by above THREE methods.
        """
        # Fission can only happen if the deme is full.
        assert origin_deme.N+origin_deme.N2 >= self.params['G'], \
            f"Fission failed: Origin deme at ({origin_deme.x}, {origin_deme.y}) has insufficient population size {origin_deme.N} < {self.params['G']}."
        dividing_beyond_the_edge = False
        theta = self.find_direction_with_least_budge_number(origin_deme)
        coords = self.get_nearest_empty_spot_in_direction(origin_deme.x, origin_deme.y, theta)
        if coords is not None:
            x_to_fill, y_to_fill = coords
            #check if the target coordinates are off-grid or occupied.
            is_off_grid = not (0 <= x_to_fill < self.grid.dim_grid and 0 <= y_to_fill < self.grid.dim_grid)
            is_occupied = self.grid.get_deme(x_to_fill, y_to_fill) is not None if not is_off_grid else False
        else:
            print(f"Buding Fission failed: No empty spot found for deme at ({origin_deme.x}, {origin_deme.y}) in direction {theta}.")
            # No empty neighbor found, cannot fission
            self.params['exit_code'] = 1
            return
        

        if is_off_grid:
            # It means the fissioning deme is on an edge, trying to push outwards.
            dividing_beyond_the_edge = True
            print(f" Attempting to budge demes: Target coordinates ({x_to_fill}, {y_to_fill}) are off-grid.")
            self.params['exit_code']=1
            return
        
        elif is_occupied:
            print(f" Attempting to budge demes: Target coordinates ({x_to_fill}, {y_to_fill}) are occupied.")
            self.params['exit_code'] = 1
            return
        else:
            final_coords = self.budge_demes(origin_deme, x_to_fill, y_to_fill)
            x_to_fill, y_to_fill = final_coords
        
        new_deme = None
        if not dividing_beyond_the_edge:
            if (x_to_fill, y_to_fill) in self.grid.demes:
                # If the target coordinates are occupied, we cannot create a new deme.
                print(f"Cannot create a new deme: Target coordinates ({x_to_fill}, {y_to_fill}) are occupied.")
            new_deme = Deme(x_to_fill, y_to_fill, self.params['G'], self.params['NSIM'], self.rng, 
                            creation_time=self.tau, unique_id=self.get_new_deme_id())
            self.grid.add_deme(new_deme)
            new_deme.creation_time = self.tau
            if new_deme.N!=0 or new_deme.m.any() or new_deme.k.any() or new_deme.w.any():
                print(f"New deme at ({x_to_fill}, {y_to_fill}) has non-zero population or methylation states.")
        self.move_cells(origin_deme, new_deme, dividing_beyond_the_edge)

        # return the new deme if it was created, otherwise None, for calculate deme population
        return new_deme
   
        
    def calculate_deme_pop_glandfission(self, dt, params):
        """
        Calculates the explicit population of each deme after a time interval dt,
        assuming total population grows as exp(theta*t) (multinomially distributed to each deme),
        and growth occurs via gland fission. 

        Args:
            dt (float): The time interval to simulate forward.
            params['theta']=:theta (float): The global growth rate parameter.
        """
        theta = params['theta']

        if dt <= 0:
            raise ValueError("Time interval dt must be positive.")
        
        eligible_demes = [d for d in self.grid.demes.values() if d.N < d.G]
        if not eligible_demes:
            print("No eligible demes to grow")
            return

        # Calculate the growth factor (gf) needed for each eligible deme to reach capacity K:
        # gf = K / current_population
        growth_factors = {
            d: self.G / (d.N + 1e-9) for d in eligible_demes
            }
        
        #The deme requiring the smallest growth factor will be the next one (maybe more than one deme) to become full.
        gf_min = min(growth_factors.values())
        
        # Calculate the time required for latest deme to reach capacity K and fission
        # target_t = ln(gf_min) / theta
        # We handle the case where gf_min might be < 1 if a deme is over capacity.
        if gf_min < 1:
            target_t = 0 # Fission can happen immediately.
        else:
            target_t = math.log(gf_min) / theta

        # CASE 1: The next fission happens after time interval dt.
        if target_t > dt:
            final_growth_factor = math.exp(theta * dt)
            # Update the population of every single deme  and methylation by this factor.
            for deme in self.grid.demes.values():
                old_N = deme.N
                deme.N = round(final_growth_factor* deme.N)
                m, k, w = grow_cancer(
                    deme.m, deme.k, deme.w, old_N, deme.N, deme.rng)
                m, k, w = generate_next_timepoint(
                    m, k, w, params['mu'], params['gamma'],params['nu'], params['zeta'], deme.N, dt, deme.rng)
                deme.m = m
                deme.k = k
                deme.w = w

            # Update
            self.tau+= dt
            # self.birth_rate = set_birth_rate()
            # self.death_rate = set_death_rate()
            # self.migration_rate = set_migration_rate()
            return

        # CASE 2: A fission event occurs WITHIN our time interval `dt`.
        else:
            self.tau+= target_t

            # Update all deme populations to their size at the moment of fission.
            for deme in self.grid.demes.values():
                old_N = deme.N
                deme.N = round(gf_min* deme.N)
                m, k, w = grow_cancer(
                    deme.m, deme.k, deme.w, old_N, deme.N, deme.rng)
                m, k, w = generate_next_timepoint(
                    m, k, w, params['mu'], params['gamma'],params['nu'], params['zeta'], deme.N, dt, deme.rng)
                deme.m = m
                deme.k = k
                deme.w = w
            # Identify all demes that have just reached capacity G,
            # check with a small tolerance for floating point inaccuracies.
            full_demes = [
                d for d in self.grid.demes.values() if d.N>= self.params['G'] - 1e-9
            ]
            for deme_to_split in full_demes:
                self.deme_fission(deme_to_split)

            # recursive: now calculate what happens in the remaining time.
            remaining_dt = dt - target_t
            self.calculate_deme_pop(remaining_dt, theta)

   
#we use multinomial_rvs to distribute growth, instead of assuming all demes grow at the same rate
    def calculate_deme_pop_glandfission_mno(self, dt, params, S_i, S_iplus1):
        """
        Calculates the explicit population of each deme after a time interval dt,
        using a time-stepped approach with multinomial growth distribution.

        Growth is distributed multinomially to non-full demes,
        and all demes that reach capacity fission simultaneously, 
        with excess population being distributed to new non-full demes.

        Args:
            dt (float): The time interval to simulate forward.
            params (dict): The simulation parameters dictionary: theta, G, mu, gamma, nu, zeta.
        """
        theta = params['theta']
        G = params['G']

        # Total population before and after the time step dt: S=exp(theta*(t+dt))
        S_old = S_i
        delta_S = S_iplus1 - S_old

        assert delta_S >= 0, "Population growth must be non-negative."

        # Distribute growth multinomially to elligible demes that are not yet full.
        eligible_demes = [d for d in self.grid.demes.values() if d.N < G]
        if not eligible_demes:
            print("No eligible demes to grow")
            return
        # # Check if there are full demes that need to be fissioned.
        # if any (d.N >= G for d in self.grid.demes.values()):
        #     print(f"Time: {self.tau:.2f}. Found full demes" )

        if eligible_demes:
            # Calculate the total population of only the eligible demes.
            S_eligible = sum(d.N for d in eligible_demes)
            
            if S_eligible > 0:
                # the probability for multinomial distribution: p_i = N_i / S_eligible
                p = np.array([d.N / S_eligible for d in eligible_demes])
                growth_per_deme = multinomial_rvs(delta_S, p.reshape(-1, 1), self.rng).flatten()
                
                # Update the populations of the eligible demes: because of the deme fission later,
                # the methyltion states will be updated for N<=G
                for i, deme in enumerate(eligible_demes):
                    growth = growth_per_deme[i]
                    if growth == 0: continue
                    
                    old_N = deme.N
                    new_N = old_N + growth
                    
                    # Grow the methylation states：
                    if old_N > 0:
                        if new_N >= G:
                            
                            m, k, w = grow_cancer(deme.m, deme.k, deme.w, old_N, G, deme.rng)
                            deme.m = m
                            deme.k = k
                            deme.w = w
                            deme.N = new_N
                            
                        else:
                            
                            m, k, w = grow_cancer(deme.m, deme.k, deme.w, old_N, new_N, deme.rng)
                            deme.m = m
                            deme.k = k
                            deme.w = w
                            deme.N = new_N
                    else:
                        print(f"Deme at ({deme.x}, {deme.y}) has no cells to grow from. Skipping growth.")
      

        # Identify full demes and deme fission
        # If migration_edge_only is True, only consider demes at the boundary (erosion: erosion mode from morphology model)
        # If migration_edge_only is False, consider all demes that are full: Gland Fission model.
        migration_edge_only = self.params['migration_edge_only']
        erosion = self.params['erosion']
        if migration_edge_only:
            if erosion:
                full_demes = [
                d for d in self.find_boundary_demes()
                if d.N >= G ]
            else:
                full_demes = [
                d for d in self.grid.demes.values() 
                if d.N >= G and self.is_deme_at_boundary(d)]
        else:
            full_demes = [d for d in self.grid.demes.values() if d.N >= G]

        
        for deme_to_split in full_demes:
            excess_population = deme_to_split.N - G
            
            # Set the deme to size G for the split
            deme_to_split.N = G
            
            # Execute the deme fission
            new_deme = self.deme_fission(deme_to_split)
            if new_deme is None:
                self.params['exit_code'] = 1
                print("The new deme is None because demes reach the boundary, exit the expansion.")
                for deme in self.grid.demes.values():
                    if deme.N>G:
                        deme.N=G
                break
            # add the excess population equally to the origin and new deme
            if new_deme and excess_population > 0:
                excess_per_deme = excess_population // 2
                excess_origin_deme = excess_per_deme
                excess_new_deme = excess_per_deme
                if excess_population % 2 != 0:
                    # Randomly assign the extra cell to one of the demes
                    if self.rng.random() < 0.5:
                        excess_origin_deme +=1
                    else:
                        excess_new_deme +=1
                deme_to_split.N += excess_origin_deme
                new_deme.N += excess_new_deme

                # update methylation state for the original deme
                m,k,w = grow_cancer(
                    deme_to_split.m, deme_to_split.k, deme_to_split.w,
                    deme_to_split.N - excess_origin_deme, deme_to_split.N, self.rng)
                deme_to_split.m = m
                deme_to_split.k = k
                deme_to_split.w = w
                
                # update methylation state for the new deme
                m, k, w = grow_cancer(
                    new_deme.m, new_deme.k, new_deme.w,
                    new_deme.N - excess_new_deme, new_deme.N, self.rng)
                new_deme.m = m
                new_deme.k = k
                new_deme.w = w

# For subclonal, we define the calculation of number of cells of origin clone and subclone,
# and the growth of methylation states of two clones, in the deme fission for once.
# Notice it is mainly for Gland fission model and the Boundary growth model when remaining full demes fission, with the SUBCLONE.
    def update_deme_fission_subclonal(self, deme_to_split, boundary=False):
        G = self.params['G']
        new_N = deme_to_split.N
        new_N2= deme_to_split.N2
        deme_to_split.N = deme_to_split.m[0]+ deme_to_split.k[0] + deme_to_split.w[0]
        deme_to_split.N2 = deme_to_split.m2[0]+ deme_to_split.k2[0] + deme_to_split.w2[0]
        excess_population_1 = new_N - deme_to_split.N
        excess_population_2 = new_N2 - deme_to_split.N2
        excess_population = excess_population_1 + excess_population_2
        
        assert deme_to_split.N + deme_to_split.N2== G, \
            f"when fission at ({deme_to_split.x}, {deme_to_split.y}), deme_to_split.N + deme_to_split.N2 should be G, but got {deme_to_split.N} + {deme_to_split.N2} = {deme_to_split.N + deme_to_split.N2}"
        # Execute the deme fission
        if boundary:
            # In the boundary growth model, we use the deme_fission_boundary_budge method
            new_deme = self.deme_fission_boundary_budge(deme_to_split)
        else:
            # In the gland fission model, we use the deme_fission method
            new_deme = self.deme_fission(deme_to_split)
        
        if new_deme is None:
            self.params['exit_code'] = 1
            print("The new deme is None because demes reach the boundary, exit the expansion.")  
            return
        
        if new_deme.N==0 or new_deme.N2==0 or deme_to_split.N==0 or deme_to_split.N2==0:
            print("deme to split N is", deme_to_split.N, ", deme to split N2 is", deme_to_split.N2,"at the: ", deme_to_split.x, deme_to_split.y)
            print("new deme N is", new_deme.N, ", new deme N2 is", new_deme.N2, "at the: ", new_deme.x, new_deme.y)

    
        # add the excess population equally to the origin and new deme
        if new_deme and excess_population > 0:
            excess_per_deme_1 = excess_population_1 // 2
            excess_origin_deme_1 = excess_per_deme_1
            excess_new_deme_1 = excess_per_deme_1
            if excess_population_1 % 2 != 0:
                # Randomly assign the extra cell to one of the demes
                if self.rng.random() < 0.5:
                    excess_origin_deme_1 +=1
                else:
                    excess_new_deme_1 +=1
            deme_to_split.N += excess_origin_deme_1
            new_deme.N += excess_new_deme_1

            # Now handle the second clone
            excess_per_deme_2 = excess_population_2 // 2
            excess_origin_deme_2 = excess_per_deme_2
            excess_new_deme_2 = excess_per_deme_2
            if excess_population_2 % 2 != 0:
                # Randomly assign the extra cell to one of the demes
                if self.rng.random() < 0.5:
                    excess_origin_deme_2 +=1
                else:
                    excess_new_deme_2 +=1
            deme_to_split.N2 += excess_origin_deme_2
            new_deme.N2 += excess_new_deme_2

            if deme_to_split.N + deme_to_split.N2 <= G:
                # update methylation state for the original deme
                m,k,w = grow_cancer(
                    deme_to_split.m, deme_to_split.k, deme_to_split.w,
                    deme_to_split.N - excess_origin_deme_1, deme_to_split.N, self.rng)
                deme_to_split.m = m
                deme_to_split.k = k
                deme_to_split.w = w
            
                # update methylation state for the original deme, subclone
                m2, k2, w2 = grow_cancer(
                    deme_to_split.m2, deme_to_split.k2, deme_to_split.w2,
                    deme_to_split.N2 - excess_origin_deme_2, deme_to_split.N2, self.rng)
                
                deme_to_split.m2 = m2
                deme_to_split.k2 = k2
                deme_to_split.w2 = w2

                assert np.all(deme_to_split.m + deme_to_split.k + deme_to_split.w == deme_to_split.N), \
                    f"Origin deme at ({deme_to_split.x}, {deme_to_split.y}) has N={deme_to_split.N} but m+k+w={deme_to_split.m + deme_to_split.k + deme_to_split.w}."
                assert np.all(deme_to_split.m2 + deme_to_split.k2 + deme_to_split.w2 == deme_to_split.N2), \
                    f"Origin deme at ({deme_to_split.x}, {deme_to_split.y}) has N2={deme_to_split.N2} but m2+k2+w2={deme_to_split.m2 + deme_to_split.k2 + deme_to_split.w2}."
            else:
                # control the deme size to be G
                growth = excess_origin_deme_1
                growth2 = excess_origin_deme_2
                old_N = deme_to_split.N - growth
                old_N2 = deme_to_split.N2 - growth2
                remain_part = G - old_N - old_N2
                methylation_N = old_N + round(remain_part* (growth / (growth + growth2)))
                methylation_N2 = G- methylation_N

                m, k, w = grow_cancer(
                    deme_to_split.m, deme_to_split.k, deme_to_split.w,
                    old_N, methylation_N, self.rng)

                deme_to_split.m = m
                deme_to_split.k = k
                deme_to_split.w = w

                m2, k2, w2 = grow_cancer(
                    deme_to_split.m2, deme_to_split.k2, deme_to_split.w2,
                    old_N2, methylation_N2, self.rng)
                deme_to_split.m2 = m2
                deme_to_split.k2 = k2
                deme_to_split.w2 = w2
                assert np.all(deme_to_split.m + deme_to_split.k + deme_to_split.w + deme_to_split.m2 + deme_to_split.k2 + deme_to_split.w2 == G), \
                    f"Origin deme at ({deme_to_split.x}, {deme_to_split.y}) has N={deme_to_split.N} + N2={deme_to_split.N2} but m+k+w+m2+k2+w2={deme_to_split.m + deme_to_split.k + deme_to_split.w + deme_to_split.m2 + deme_to_split.k2 + deme_to_split.w2}."
            
            if new_deme.N + new_deme.N2 <= G:
                # update methylation state for the new deme
                m, k, w = grow_cancer(
                    new_deme.m, new_deme.k, new_deme.w,
                    new_deme.N - excess_new_deme_1, new_deme.N, self.rng)
                new_deme.m = m
                new_deme.k = k
                new_deme.w = w
                
                # update methylation state for the new deme, subclone
                m2, k2, w2 = grow_cancer(
                    new_deme.m2, new_deme.k2, new_deme.w2,
                    new_deme.N2 - excess_new_deme_2, new_deme.N2, self.rng)
                new_deme.m2 = m2
                new_deme.k2 = k2
                new_deme.w2 = w2
                assert np.all(new_deme.m + new_deme.k + new_deme.w == new_deme.N), \
                    f"New deme at ({new_deme.x}, {new_deme.y}) has N={new_deme.N} but m+k+w={new_deme.m + new_deme.k + new_deme.w}."
                assert np.all(new_deme.m2 + new_deme.k2 + new_deme.w2 == new_deme.N2), \
                    f"New deme at ({new_deme.x}, {new_deme.y}) has N2={new_deme.N2} but m2+k2+w2={new_deme.m2 + new_deme.k2 + new_deme.w2}."

            else:
                # control the deme size to be G
                growth = excess_new_deme_1
                growth2 = excess_new_deme_2
                old_N = new_deme.N - growth
                old_N2 = new_deme.N2 - growth2
                remain_part = G - old_N - old_N2
                methylation_N = old_N + round(remain_part* (growth / (growth + growth2)))
                methylation_N2 = G- methylation_N

                m, k, w = grow_cancer(
                    new_deme.m, new_deme.k, new_deme.w,
                    old_N, methylation_N, self.rng)

                new_deme.m = m
                new_deme.k = k
                new_deme.w = w

                m2, k2, w2 = grow_cancer(
                    new_deme.m2, new_deme.k2, new_deme.w2,
                    old_N2, methylation_N2, self.rng)
                
                new_deme.m2 = m2
                new_deme.k2 = k2
                new_deme.w2 = w2

                assert np.all(new_deme.m + new_deme.k + new_deme.w + new_deme.m2 + new_deme.k2 + new_deme.w2 == G), \
                    f"New deme at ({new_deme.x}, {new_deme.y}) has N={new_deme.N} + N2={new_deme.N2} but m+k+w+m2+k2+w2={new_deme.m + new_deme.k + new_deme.w + new_deme.m2 + new_deme.k2 + new_deme.w2}."
        return new_deme
    
# For subclonal, we define the calculation of number of cells and methylation states similarly.
# Compared with the Gland fission model, the difference is to handle the methylation states, and number of cells
# of the origin clone and subclone, in the deme fission for once.
    def calculate_deme_pop_glandfission_mno_subclonal(self, dt, params, S_i, S_iplus1, S2_i, S2_iplus1):
        """
        Calculates the explicit population of each deme after a time interval dt,
        using a time-stepped approach with multinomial growth distribution.

        Growth is distributed multinomially to non-full demes,
        and all demes that reach capacity fission simultaneously, 
        with excess population being distributed to new non-full demes.

        Args:
            dt (float): The time interval to simulate forward.
            params (dict): The simulation parameters dictionary: theta, G, mu, gamma, nu, zeta.
        """
        theta = params['theta']
        G = params['G']

        # Total population before and after the time step dt: S=exp(theta*(t+dt))
        S_old = S_i
        delta_S = S_iplus1 - S_old
        delta_S2 = S2_iplus1 - S2_i
        assert delta_S >= 0, "Population growth must be non-negative."
        assert delta_S2 >= 0, "Subclonal population growth must be non-negative."

        # Distribute growth multinomially to elligible demes that are not yet full.
        eligible_demes = [d for d in self.grid.demes.values() if (d.N+d.N2) < G]
        if not eligible_demes:
            print("No eligible demes to grow")
            return
        # # Check if there are full demes that need to be fissioned.
        # if any (d.N >= G for d in self.grid.demes.values()):
        #     print(f"Time: {self.tau:.2f}. Found full demes")

        if eligible_demes:
            # Calculate the total population of only the eligible demes.
            S_eligible = sum((d.N) for d in eligible_demes)
            S2_eligible = sum((d.N2) for d in eligible_demes)

            if S_eligible+S2_eligible > 0:
                # the probability for multinomial distribution: p_i = N_i / S_eligible
                p = np.array([d.N / S_eligible for d in eligible_demes])
                p2 = np.array([d.N2 / S2_eligible for d in eligible_demes])
                growth_per_deme = multinomial_rvs(delta_S, p.reshape(-1, 1), self.rng).flatten()
                growth_per_deme2 = multinomial_rvs(delta_S2, p2.reshape(-1, 1), self.rng).flatten()
                
                # Update the populations of the eligible demes: because of the deme fission later,
                # the methyltion states will be updated for N<=G
                for i, deme in enumerate(eligible_demes):
                    growth = growth_per_deme[i]
                    growth2 = growth_per_deme2[i]
                    if growth == 0 and growth2 == 0: continue
                    
                    old_N = deme.N
                    new_N = old_N + growth
                    old_N2 = deme.N2
                    new_N2 = old_N2 + growth2
                    
                    # Grow the methylation states：
                    if old_N+old_N2 > 0:
                        if new_N+new_N2 >= G:
                            # grow the methylation states of two CLONES according to the proportions of the two clones.
                            remain_part = G - old_N - old_N2
                            methylation_N = old_N + round(remain_part* (growth / (growth + growth2)))
                            methylation_N2 = G- methylation_N
                            
                            if old_N==0:
                                print("old N is zero at the deme.x, deme.y:", deme.x, deme.y)
                                print("old N2 is", old_N2)
                                print("new N:", new_N, "new N2:", new_N2)
                                print("methylation N:", methylation_N, "methylation N2:", methylation_N2)
                            if old_N2==0:
                                print("old N2 is zero at the deme.x, deme.y:", deme.x, deme.y)
                            
                            if old_N>0:
                                m, k, w = grow_cancer(deme.m, deme.k, deme.w, old_N, methylation_N, deme.rng)
                                deme.m = m
                                deme.k = k
                                deme.w = w
                            if old_N2>0:
                                m2, k2, w2 = grow_cancer(deme.m2, deme.k2, deme.w2, old_N2, methylation_N2, deme.rng)
                                deme.m2 = m2
                                deme.k2 = k2
                                deme.w2 = w2

                            deme.N = new_N
                            deme.N2 = new_N2
                            
                        else:
                            if old_N==0:
                                print("old N is zero at the deme.x, deme.y:", deme.x, deme.y)
                                print("old N2 is", old_N2)
                                print("new N:", new_N, "new N2:", new_N2)
                    
                            if old_N2==0:
                                print("old N2 is zero at the deme.x, deme.y:", deme.x, deme.y)
                            

                            if old_N>0:
                                m, k, w = grow_cancer(deme.m, deme.k, deme.w, old_N, new_N, deme.rng)
                                deme.m = m
                                deme.k = k
                                deme.w = w
                                deme.N = new_N
                            if old_N2>0:
                                m2, k2, w2 = grow_cancer(deme.m2, deme.k2, deme.w2, old_N2, new_N2, deme.rng)
                                deme.m2 = m2
                                deme.k2 = k2
                                deme.w2 = w2
                                deme.N2 = new_N2
                    else:
                        print(f"Deme at ({deme.x}, {deme.y}) has no cells to grow from. Skipping growth.")
        # for d in self.grid.demes.values():
        #     print(f"Deme at ({d.x}, {d.y}) has N={d.N}, N2={d.N2}")  
    

        # Identify full demes and deme fission
        # If migration_edge_only is True, only consider demes at the boundary (erosion: erosion mode from morphology model)
        # If migration_edge_only is False, consider all demes that are full: Gland Fission model.
        migration_edge_only = self.params['migration_edge_only']
        erosion = self.params['erosion']
        full_demes = [d for d in self.grid.demes.values() if (d.N+d.N2) >= G]

        iter = 0
        while len(full_demes)>0:
            iter+=1
            assert iter<3, "many loops!"
            
            if self.params['exit_code'] == 1:
                print("Exit code is 1, stop the deme fission.")
                for deme in self.grid.demes.values():
                    if deme.N+deme.N2>G:
                        deme.N = deme.m[0]+ deme.k[0] + deme.w[0]
                        deme.N2 = deme.m2[0]+ deme.k2[0] + deme.w2[0]
                break

            for deme_to_split in full_demes:
                new_deme = self.update_deme_fission_subclonal(deme_to_split)
                if new_deme is None:
                    self.params['exit_code'] = 1
                    print("The new deme is None because demes reach the boundary, exit the loop of fission.")
                    break
            full_demes = [d for d in self.grid.demes.values() if (d.N+d.N2) >= G]

# Similarly, we can define the function to calculate deme population for subclonal model,
# And the spatial model is BOUNDARY GROWTH.
    def calculate_deme_pop_boundary_growth_subclonal(self, dt, params, S_i, S_iplus1, S2_i, S2_iplus1):
        
        print(f"Beginning:")
        for deme in self.grid.demes.values():
            print(f"At the beginning, Deme at ({deme.x}, {deme.y}) has N={deme.N}.")

        delta_S = S_iplus1 - S_i
        delta_S2 = S2_iplus1 - S2_i
        assert delta_S >= 0, "Population growth must be non-negative."
        assert delta_S2 >= 0, "Population growth must be non-negative for Subclone."

        G = params['G']
        erosion = params['erosion']

        # Distribute growth multinomially to elligible demes that are not yet full.
        if erosion:
            boundary_demes = self.find_boundary_demes()
        else:
            boundary_demes = [
                d for d in self.grid.demes.values() if self.is_deme_at_boundary(d)
            ]
        
        eligible_demes = [d for d in self.grid.demes.values() if d.N+d.N2 < G]
        if not eligible_demes:
            print("No eligible demes to grow")
            return
        old_N_eligible = np.zeros(len(eligible_demes))
        old_N2_eligible = np.zeros(len(eligible_demes))

        if eligible_demes:
            eligible_demes_1 = [d for d in eligible_demes if d.N > 0]
            eligible_demes_2 = [d for d in eligible_demes if d.N2 > 0]
            # Calculate the total population of only the eligible demes.
            S_eligible = sum((d.N) for d in eligible_demes)
            S2_eligible = sum((d.N2) for d in eligible_demes)

        for i, deme in enumerate(eligible_demes):
            old_N_eligible[i] = deme.N
            old_N2_eligible[i] = deme.N2
        
        
        # for deme in fission_candidates:
        #     print(f"Before distribute excess S, Fission candidate at ({deme.x}, {deme.y}) has N={deme.N} + N2={deme.N2} >= G={G}.")
        
        excess_S = delta_S
        excess_S2 = delta_S2
        new_eligible_demes = eligible_demes.copy()

        # Different to gland fission model: the demes that are not fission candidates but N+N2>G should only have N+N2=G, 
        # and the excess population will be sum up and multinomial distributed to those eligible demes again.
        # big_iter=0
        iter = 0
        while excess_S > 0 or excess_S2 > 0:
            iter+=1
            assert iter<100, "many loops!"
            new_eligible_demes = [d for d in self.grid.demes.values() if d.N+d.N2 < G]+boundary_demes
            if len(new_eligible_demes)>0:
                S_eligible = sum((d.N) for d in new_eligible_demes)
                S2_eligible = sum((d.N2) for d in new_eligible_demes)
                # for i, deme in enumerate(new_eligible_demes):
                #     old_N_eligible[i] = deme.N
                #     old_N2_eligible[i] = deme.N2
                iter += 1
                assert iter<100,\
                    f"Too many iterations in boundary growth subclonal model when distributing excess S and S2. Excess S={excess_S}, Excess S2={excess_S2}."
                assert len(new_eligible_demes) > 0, \
                    f"No eligible demes to distribute excess S={excess_S} and excess S2={excess_S2}. Iteration {iter}."

                if excess_S == 0 and excess_S2 > 0:
                    new_eligible_demes_2 = [d for d in new_eligible_demes if d.N2 > 0]
                elif excess_S2 == 0 and excess_S > 0:
                    new_eligible_demes_1 = [d for d in new_eligible_demes if d.N > 0]
                else:
                    # excess_S>0 and excess_S2>0
                    assert excess_S > 0 and excess_S2 > 0, \
                        f"Excess S={excess_S}, Excess S2={excess_S2}. Both should be positive to continue the distribution."
                    new_eligible_deme_1 = [d for d in new_eligible_demes if d.N > 0]
                    new_eligible_deme_2 = [d for d in new_eligible_demes if d.N2 > 0]

                print(f"The {iter} th iteration: Distributing excess S={excess_S} and excess S2={excess_S2} to {len(new_eligible_demes)} eligible demes.")

                if new_eligible_demes and (excess_S > 0 or excess_S2 > 0):
                    # the probability for multinomial distribution: p_i = N_i / S_eligible
                    if excess_S > 0:
                        p = np.array([d.N / S_eligible for d in new_eligible_demes])
                        growth_per_deme = multinomial_rvs(excess_S, p.reshape(-1, 1), self.rng).flatten()
                    else:
                        p = np.zeros(len(new_eligible_demes))
                        growth_per_deme = np.zeros(len(new_eligible_demes))
                    if excess_S2 > 0:
                        p2 = np.array([d.N2 / S2_eligible for d in new_eligible_demes])
                        growth_per_deme2 = multinomial_rvs(excess_S2, p2.reshape(-1, 1), self.rng).flatten()
                    else:
                        p2 = np.zeros(len(new_eligible_demes))
                        growth_per_deme2 = np.zeros(len(new_eligible_demes))

                    # Update the populations of the eligible demes: because of the deme fission later,
                    # the methyltion states will be updated for N+N2<=G
                    for i, deme in enumerate(new_eligible_demes):
                        growth = growth_per_deme[i]
                        growth2 = growth_per_deme2[i]
                        if growth == 0 and growth2 == 0: continue
                        
                        old_N = deme.N
                        new_N = deme.N+growth
                        old_N2 = deme.N2
                        new_N2 = deme.N2+growth2
                        print(f"Distributing growth: Deme at ({deme.x}, {deme.y}) with old N={old_N}, old N2={old_N2}, growth={growth}, growth2={growth2}.")
                        # Grow the N and N2：
                        if old_N+old_N2> 0 :
                            if new_N+new_N2 > G and deme not in boundary_demes:
                                # grow the N and N2 of two CLONES according to the proportions of the two clones,
                                # to control the number of cells up to G.
                                remain_part = G - old_N - old_N2
                                methylation_N = old_N + round(remain_part* (growth / (growth + growth2)))
                                methylation_N2 = G- methylation_N
                                deme.N = (methylation_N).astype(int)
                                deme.N2 = methylation_N2.astype(int)
                                # update the remain excess S and excess S2
                                excess_S -= methylation_N - old_N
                                excess_S2 -= methylation_N2 - old_N2
                            else:
                                deme.N = new_N
                                deme.N2 = new_N2
                                # update the remain excess S and excess S2
                                excess_S -= growth
                                excess_S2 -= growth2

            # Break the loop if both excess_S and excess_S2 are zero
            if excess_S == 0 and excess_S2 == 0:
                print("Excess population fully distributed. ")
                new_eligible_demes = None
                break
        
        print(f"Growth distributed. Now methylation states updated.")
        # for deme in self.grid.demes.values():
        #     print(f"Deme at ({deme.x}, {deme.y}) has N={deme.N}.")


        # for deme in self.grid.demes.values():
        #     if deme.N>G and deme not in fission_candidates:
        #         print(f"Deme at ({deme.x}, {deme.y}) has N={deme.N} > G={G}. And it isnot fission demes.")
        #     if deme not in fission_candidates:
        #         assert deme.N <= G, \
        #             f"Deme at ({deme.x}, {deme.y}) has N={deme.N} > G={G}. It should not be a fission candidate."

        # if new_eligible_demes is None:
        #     new_eligible_demes = eligible_demes.copy()
        if S_eligible + S2_eligible >0:
            for i, deme in enumerate(eligible_demes):
                old_N = old_N_eligible[i]
                old_N2 = old_N2_eligible[i]
                new_N = deme.N
                new_N2 = deme.N2
                # Grow the methylation states： 
    
                if old_N+old_N2 > 0 and new_N+new_N2 > old_N+old_N2:
                    assert np.all(deme.m + deme.k + deme.w == old_N), \
                        f"Deme at ({deme.x}, {deme.y}) has inconsistent methylation states of old_N={old_N}: m={deme.m}, k={deme.k}, w={deme.w}."
                    assert np.all(deme.m2 + deme.k2 + deme.w2 == old_N2), \
                        f"Deme at ({deme.x}, {deme.y}) has inconsistent methylation states of old_N2={old_N2}: m2={deme.m2}, k2={deme.k2}, w2={deme.w2}."
                    
                    if new_N + new_N2 > G :
                        # grow the methylation states of two CLONES according to the proportions of the two clones.
                        growth = new_N - old_N
                        growth2 = new_N2 - old_N2
                        remain_part = G - old_N - old_N2
                        methylation_N = old_N + round(remain_part* (growth / (growth + growth2)))
                        methylation_N2 = G- methylation_N
                        
                        if old_N>0:
                            m, k, w = grow_cancer(deme.m, deme.k, deme.w, old_N, methylation_N, deme.rng)
                            deme.m = m
                            deme.k = k
                            deme.w = w
                        if old_N2>0:
                            m2, k2, w2 = grow_cancer(deme.m2, deme.k2, deme.w2, old_N2, methylation_N2, deme.rng)
                            deme.m2 = m2
                            deme.k2 = k2
                            deme.w2 = w2

                        deme.N = new_N
                        deme.N2 = new_N2
                    
                        assert np.all(deme.m + deme.k + deme.w + deme.m2 + deme.k2 +deme.w2 == G), \
                            f"Deme at ({deme.x}, {deme.y}) has inconsistent methylation states of G after growth: m={deme.m}, k={deme.k}, w={deme.w}.\
                                m2={deme.m2}, k2={deme.k2}, w2={deme.w2}.\
                                N={deme.N}, N2={deme.N2}."

                        
                    else:

                        m, k, w = grow_cancer(deme.m, deme.k, deme.w, old_N, new_N, deme.rng)
                        deme.m = m
                        deme.k = k
                        deme.w = w
                        m2, k2, w2 = grow_cancer(deme.m2, deme.k2, deme.w2, old_N2, new_N2, deme.rng)
                        deme.m2 = m2
                        deme.k2 = k2
                        deme.w2 = w2

                        deme.N = new_N
                        deme.N2 = new_N2
                        
                        assert np.all(deme.m + deme.k + deme.w == new_N), \
                            f"Deme at ({deme.x}, {deme.y}) has inconsistent methylation states of new_N={new_N} after growth: m={deme.m}, k={deme.k}, w={deme.w}."
                    
                        assert np.all(deme.m2 + deme.k2 + deme.w2 == new_N2), \
                            f"Deme at ({deme.x}, {deme.y}) has inconsistent methylation states of new_N2={new_N2} after growth: m2={deme.m2}, k2={deme.k2}, w2={deme.w2}." 
                else:   
                    print(f"Deme at ({deme.x}, {deme.y}) has no cells to grow from. Skipping growth.")



        if self.params['exit_code']==1:
            print("Exit code is 1, exit the big while loop.")
            for deme in self.grid.demes.values():
                if deme.N+deme.N2 > G:
                    deme.N = deme.m[0] + deme.k[0] + deme.w[0]
                    deme.N2 = deme.m2[0] + deme.k2[0] + deme.w2[0]
                    assert deme.N+deme.N2==G, \
                        f"Deme at ({deme.x}, {deme.y}) has N+N2={deme.N+deme.N2} > G={G} when exit. It should be G."
            return
        if erosion:
            boundary_demes = self.find_boundary_demes()
            fission_candidates = [
                d for d in boundary_demes
                if d.N+d.N2 >= G 
            ]
        else:
            fission_candidates = [
                d for d in self.grid.demes.values() 
                if d.N+d.N2 >= G and self.is_deme_at_boundary(d)
            ]
        # No eligible demes to distribute excess population, fission the full demes
        print("Before fission:")
        old_fission_candidates = fission_candidates.copy()
        for deme in fission_candidates:
            print(f"Before: Fission candidate at ({deme.x}, {deme.y}) with N={deme.N}, N2={deme.N2}.")

        
        iter=0
        grid_boundary = self.grid.dim_grid
        # fission demes iteratively until no more fission candidates are found.
        # This is to ensure that all demes that can be split are split, even if it takes multiple iterations.
        while len(fission_candidates) > 0:  

            # Before fission demes at the edge, first check if there are demes reaching the grid boundary 
            for deme in self.grid.demes.values():
                if deme.x == 0  or deme.x == grid_boundary - 1 or deme.y == 0 or deme.y == grid_boundary - 1:
                    print("Deme at ({deme.x}, {deme.y}) has reached the grid boundary when deme fission on the edge.")
                    self.params['exit_code'] = 1
                    break
            if self.params['exit_code'] == 1:
                print("Exit code is 1, stop the deme fission.")
                for deme in self.grid.demes.values():
                    if deme.N+deme.N2 > G:
                        deme.N = deme.m[0] + deme.k[0] + deme.w[0]
                        deme.N2 = deme.m2[0] + deme.k2[0] + deme.w2[0]
                        assert deme.N+deme.N2==G, \
                            f"Deme at ({deme.x}, {deme.y}) has N+N2={deme.N+deme.N2} > G={G} when exit. It should be G."
                break

            print(f"Found {len(fission_candidates)} fission candidates at the boundary with N+N2 >= G={G}.")
            iter += 1
            print(f"Iteration {iter}: Fissioning demes at the boundary.")
            assert iter<10, \
                f"Too many iterations ({iter}) for fissioning demes at the boundary. Check the logic."
            for deme_to_split in fission_candidates:
                
                new_N = deme_to_split.N
                new_N2= deme_to_split.N2

                excess_population_1 = new_N - deme_to_split.N
                excess_population_2 = new_N2 - deme_to_split.N2
                excess_population = excess_population_1 + excess_population_2
            
                if np.any(deme_to_split.m + deme_to_split.k + deme_to_split.w +deme_to_split.m2 + deme_to_split.k2 + deme_to_split.w2 != G):
                    print("m:", deme_to_split.m)
                    print("k:", deme_to_split.k)
                    print("w:", deme_to_split.w)
                    print("m2:", deme_to_split.m2)
                    print("k2:", deme_to_split.k2)
                    print("w2:", deme_to_split.w2)
                    print("deme to split:", deme_to_split.x, deme_to_split.y)
                    print("N:", deme_to_split.N, "N2:", deme_to_split.N2)

                # Set the deme to size G for the split
                deme_to_split.N = (deme_to_split.m[0] + deme_to_split.k[0] + deme_to_split.w[0]).astype(int)
                deme_to_split.N2 = (deme_to_split.m2[0] + deme_to_split.k2[0] + deme_to_split.w2[0]).astype(int)
                assert deme_to_split.N + deme_to_split.N2 <= G, \
                    f"Deme to split at ({deme_to_split.x}, {deme_to_split.y}) has N={deme_to_split.N} + N2={deme_to_split.N2} > G={G}."
                assert deme_to_split.N + deme_to_split.N2 == G, \
                    f"When fission, deme_to_split.N + deme_to_split.N2 should be G, but got {deme_to_split.N} + {deme_to_split.N2} = {deme_to_split.N + deme_to_split.N2}."
                
                # Execute the deme fission
                new_deme = self.deme_fission(deme_to_split)
                
                if new_deme:
                    print(f"Fissioned deme on the boundary at ({deme_to_split.x}, {deme_to_split.y}) into new deme at ({new_deme.x}, {new_deme.y}) with N={new_deme.N}, N2={new_deme.N2}.")
                else:
                    print(f"Fission failed for deme on the boundary at ({deme_to_split.x}, {deme_to_split.y}). No valid location for new deme.")
                # add the excess population equally to the origin and new deme
                if excess_population > 0:
                    excess_per_deme_1 = excess_population_1 // 2
                    excess_per_deme_2 = excess_population_2 // 2
                    excess_origin_deme_1 = excess_per_deme_1
                    excess_new_deme_1 = excess_per_deme_1
                    excess_origin_deme_2 = excess_per_deme_2
                    excess_new_deme_2 = excess_per_deme_2
                    
                    if excess_population_1 % 2 != 0:
                        # Randomly assign the extra cell to one of the demes
                        if self.rng.random() < 0.5:
                            excess_origin_deme_1 +=1
                        else:
                            excess_new_deme_1 +=1
                    
                    if excess_population_2 % 2 != 0:
                        if self.rng.random() < 0.5:
                            excess_origin_deme_2 +=1
                        else:
                            excess_new_deme_2 +=1
                    
                    if new_deme:
                        new_deme.N += excess_new_deme_1
                        new_deme.N2 += excess_new_deme_2
                        deme_to_split.N += excess_origin_deme_1
                        deme_to_split.N2 += excess_origin_deme_2
                    else:
                        deme_to_split.N += excess_population_1
                        deme_to_split.N2 += excess_population_2
                    
                    if new_deme:

                        assert np.all(deme_to_split.m + deme_to_split.k + deme_to_split.w == deme_to_split.N-excess_origin_deme_1), \
                            f"Deme at ({deme_to_split.x}, {deme_to_split.y}) has inconsistent methylation states of clone 1 before excess population distribution: m={deme_to_split.m}, k={deme_to_split.k}, w={deme_to_split.w}."
                        assert np.all(deme_to_split.m2 + deme_to_split.k2 + deme_to_split.w2 == deme_to_split.N2-excess_origin_deme_2), \
                            f"Deme at ({deme_to_split.x}, {deme_to_split.y}) has inconsistent methylation states of clone 2 before excess population distribution: m2={deme_to_split.m2}, k2={deme_to_split.k2}, w2={deme_to_split.w2}."
                        if deme_to_split.N + deme_to_split.N2 <= G: 
                            # update methylation state for the original deme
                            m,k,w = grow_cancer(
                                deme_to_split.m, deme_to_split.k, deme_to_split.w,
                                deme_to_split.N - excess_origin_deme_1, deme_to_split.N, self.rng)
                            deme_to_split.m = m
                            deme_to_split.k = k
                            deme_to_split.w = w

                            m2, k2, w2 = grow_cancer(
                                deme_to_split.m2, deme_to_split.k2, deme_to_split.w2,
                                deme_to_split.N2 - excess_origin_deme_2, deme_to_split.N2, self.rng)
                            deme_to_split.m2 = m2
                            deme_to_split.k2 = k2
                            deme_to_split.w2 = w2

                            assert np.all(deme_to_split.m + deme_to_split.k + deme_to_split.w == deme_to_split.N), \
                                f"Deme at ({deme_to_split.x}, {deme_to_split.y}) of N={deme_to_split.N} has inconsistent methylation states after excess population distribution: m={deme_to_split.m}, k={deme_to_split.k}, w={deme_to_split.w}."
                            assert np.all(deme_to_split.m2 + deme_to_split.k2 + deme_to_split.w2 == deme_to_split.N2), \
                                f"Deme at ({deme_to_split.x}, {deme_to_split.y}) of N2={deme_to_split.N2} has inconsistent methylation states after excess population distribution: m2={deme_to_split.m2}, k2={deme_to_split.k2}, w2={deme_to_split.w2}."
                        
                        else:
                            # control the deme size to be G
                            growth = excess_origin_deme_1
                            growth2 = excess_origin_deme_2
                            old_N = deme_to_split.N - growth
                            old_N2 = deme_to_split.N2 - growth2
                            remain_part = G - old_N - old_N2
                            methylation_N = old_N + round(remain_part* (growth / (growth + growth2)))
                            methylation_N2 = G- methylation_N
                            
                            m,k,w = grow_cancer(
                                deme_to_split.m, deme_to_split.k, deme_to_split.w,
                                old_N, methylation_N, self.rng)
                            deme_to_split.m = m
                            deme_to_split.k = k
                            deme_to_split.w = w

                            m2, k2, w2 = grow_cancer(
                                deme_to_split.m2, deme_to_split.k2, deme_to_split.w2,
                                old_N2, methylation_N2, self.rng)
                            deme_to_split.m2 = m2
                            deme_to_split.k2 = k2
                            deme_to_split.w2 = w2

                            assert np.all(deme_to_split.m + deme_to_split.k + deme_to_split.w + deme_to_split.m2 + deme_to_split.k2 + deme_to_split.w2 == G), \
                                f"Deme at ({deme_to_split.x}, {deme_to_split.y}) of G has inconsistent methylation states after excess population distribution: m={deme_to_split.m}, k={deme_to_split.k}, w={deme_to_split.w}, \
                                    m2 = {deme_to_split.m2}, k2 = {deme_to_split.k2}, w2 = {deme_to_split.w2}."

                        assert np.all(new_deme.m + new_deme.k + new_deme.w == new_deme.N-excess_new_deme_1), \
                            f"New deme at ({new_deme.x}, {new_deme.y}) has inconsistent methylation states before excess population distribution: m={new_deme.m}, k={new_deme.k}, w={new_deme.w}."
                        assert np.all(new_deme.m2 + new_deme.k2 + new_deme.w2 == new_deme.N2-excess_new_deme_2), \
                            f"New deme at ({new_deme.x}, {new_deme.y}) has inconsistent methylation states before excess population distribution: m2={new_deme.m2}, k2={new_deme.k2}, w2={new_deme.w2}."
                        if new_deme.N <= G:    
                            # update methylation state for the new deme
                            m, k, w = grow_cancer(
                                new_deme.m, new_deme.k, new_deme.w,
                                new_deme.N - excess_new_deme_1, new_deme.N, self.rng)
                            new_deme.m = m
                            new_deme.k = k
                            new_deme.w = w
                            
                            m2, k2, w2 = grow_cancer(
                                new_deme.m2, new_deme.k2, new_deme.w2,
                                new_deme.N2 - excess_new_deme_2, new_deme.N2, self.rng)
                            new_deme.m2 = m2
                            new_deme.k2 = k2
                            new_deme.w2 = w2
                            assert np.all(new_deme.m + new_deme.k + new_deme.w == new_deme.N), \
                                f"New deme at ({new_deme.x}, {new_deme.y}) of N={new_deme.N} has inconsistent methylation states after excess population distribution: m={new_deme.m}, k={new_deme.k}, w={new_deme.w}."
                            assert np.all(new_deme.m2 + new_deme.k2 + new_deme.w2 == new_deme.N2), \
                                f"New deme at ({new_deme.x}, {new_deme.y}) of N2={new_deme.N2} has inconsistent methylation states after excess population distribution: m2={new_deme.m2}, k2={new_deme.k2}, w2={new_deme.w2}."
                        
                        else:
                            # control the deme size to be G
                            growth = excess_new_deme_1
                            growth2 = excess_new_deme_2
                            old_N = new_deme.N - growth
                            old_N2 = new_deme.N2 - growth2
                            remain_part = G - old_N - old_N2
                            methylation_N = old_N + round(remain_part* (growth / (growth + growth2)))
                            methylation_N2 = G- methylation_N
                            # update the methylation states of the new deme

                            m, k, w = grow_cancer(
                                new_deme.m, new_deme.k, new_deme.w,
                                old_N, methylation_N, self.rng)
                            new_deme.m = m
                            new_deme.k = k
                            new_deme.w = w

                            m2, k2, w2 = grow_cancer(
                                new_deme.m2, new_deme.k2, new_deme.w2,
                                old_N2, methylation_N2, self.rng)
                            new_deme.m2 = m2
                            new_deme.k2 = k2
                            new_deme.w2 = w2

                            assert np.all(new_deme.m + new_deme.k + new_deme.w + new_deme.m2 + new_deme.k2 + new_deme.w2 == G), \
                                f"New deme at ({new_deme.x}, {new_deme.y}) of G has inconsistent methylation states after excess population distribution: m={new_deme.m}, k={new_deme.k}, w={new_deme.w}.\
                                    m2 = {new_deme.m2}, k2 = {new_deme.k2}, w2 = {new_deme.w2}."

            #update the fission candidates after the fission
            if erosion:
                boundary_demes = self.find_boundary_demes()
                fission_candidates = [
                    d for d in boundary_demes
                    if d.N+d.N2 >= G 
                ]
                print("I am updating the fission candidates after the fission")
            else:
                fission_candidates = [
                    d for d in self.grid.demes.values() 
                    if d.N+d.N2 >= G and self.is_deme_at_boundary(d)
                ]
                print("I am updating the fission candidates after the fission")
        
        print("After fissioning demes:")
        for deme in self.grid.demes.values():
            print(f"After fission on the edge: Deme at ({deme.x}, {deme.y}) has N={deme.N}, N2={deme.N2}.")
        for deme in old_fission_candidates:
            print(f"NEW OLD Fission candidate at ({deme.x}, {deme.y}) with N={deme.N}, N2={deme.N2}.")

            # # Break the loop if both excess_S and excess_S2 are zero
            # if excess_S <= 0 and excess_S2 <= 0:
            #     print("Excess population fully distributed. Exiting loop.")
            #     break
            # if self.params['exit_code']==1:
            #     print("Exit code is 1, stop the expansion of boundary growth, exit the loop of distributing excess S and S2.")
            #     break
        
        if self.params['exit_code'] == 1:
            print("Exit code is 1, stop the deme fission for remaining full demes after distributing excess S and S2.")
            for deme in self.grid.demes.values():
                if deme.N+deme.N2>G:
                    deme.N = deme.m[0] + deme.k[0] + deme.w[0]
                    deme.N2 = deme.m2[0] + deme.k2[0] + deme.w2[0]
                    assert deme.N+deme.N2==G, \
                        f"Deme at ({deme.x}, {deme.y}) has N+N2={deme.N+deme.N2} > G={G} when exit. It should be G."
            return 
        
        # Check for any demes that are still full. 
        # These are typically internal demes or boundary demes that had no empty neighbors.
        remaining_full_demes = [d for d in self.grid.demes.values() if d.N+d.N2 > G]
        original_setting = self.params['migration_edge_only']

        iter=0
        while len(remaining_full_demes)>0:
            iter+=1
            assert iter<10,\
                f"Too many iterations ({iter}) for fissioning remaining full demes. Check the logic."
            
            if self.params['exit_code'] == 1:
                print("Exit code is 1, stop the deme fission for remaining full demes.")
                for deme in self.grid.demes.values():
                    if deme.N+deme.N2>G:
                        deme.N = deme.m[0] + deme.k[0] + deme.w[0]
                    deme.N2 = deme.m2[0] + deme.k2[0] + deme.w2[0]
                    assert deme.N+deme.N2==G, \
                        f"Deme at ({deme.x}, {deme.y}) has N+N2={deme.N+deme.N2} > G={G} when exit. It should be G."
                break

            # Temporarily change the simulation mode to allow the deme_fission function to use budging.
            self.params['migration_edge_only'] = False
        
            for deme_to_split in remaining_full_demes:
                # Check the condition again, as a previous budge may have altered the grid
                if deme_to_split.N > G:
                    oldN = deme_to_split.N
                    print(f"Fissioning remaining full deme at ({deme_to_split.x}, {deme_to_split.y}) with N={oldN}.")
                    new_deme = self.update_deme_fission_subclonal(deme_to_split, boundary = True)
                    if new_deme is None:
                        print(f"Fission failed for remaining full deme at ({deme_to_split.x}, {deme_to_split.y}). No valid location for new deme. Exit the loop.")
                        self.params['exit_code'] = 1
                        break

            remaining_full_demes = [d for d in self.grid.demes.values() if d.N+d.N2 > G]
        # Restore the original simulation mode
        self.params['migration_edge_only'] = original_setting

        if erosion:
            boundary_demes = self.find_boundary_demes()
        else:
            boundary_demes = [
                d for d in self.grid.demes.values() if self.is_deme_at_boundary(d)
            ]
        #print coordinates of boundary demes
        # for deme in boundary_demes:
        #     print(f"Boundary deme at ({deme.x}, {deme.y}) with N={deme.N}.")
        fission_candidates = [
            d for d in boundary_demes
            if d.N+d.N2 >= G
        ]
        for deme in fission_candidates:
            print(f"Fission candidate at ({deme.x}, {deme.y}) with N={deme.N}, after allll the process")
        # assert len(fission_candidates) == 0, \
        #     f"There are still fission candidates with N >= G: {[f'({d.x}, {d.y})' for d in fission_candidates]}."
        while len(fission_candidates) > 0:
            if self.params['exit_code'] == 1:
                print("Exit code is 1, stop the deme fission for remaining full demes after alll the process.")
                break
            for deme in fission_candidates:
                new_deme = self.deme_fission(deme)
                if new_deme:
                    print(f"Fissioned deme at ({deme.x}, {deme.y}) into new deme at ({new_deme.x}, {new_deme.y}) with N={new_deme.N}. After alll the process.")
                else:
                    self.params['exit_code'] = 1
                    print(f"Fission failed for deme at ({deme.x}, {deme.y}). No valid location for new deme. After alll the process.")
                    break
            if erosion:
                boundary_demes = self.find_boundary_demes()
            else:
                boundary_demes = [
                    d for d in self.grid.demes.values() if self.is_deme_at_boundary(d)
                ]
            fission_candidates = [
                d for d in boundary_demes
                if d.N+d.N2 >= G
            ]
        for deme_to_check in self.grid.demes.values():
            if deme_to_check.N+deme_to_check.N2 > G:
                print(f"Deme at ({deme_to_check.x}, {deme_to_check.y}) has N={deme_to_check.N}+N2={deme_to_check.N2} > G={G}.")
                #print(f"m: {deme_to_check.m}, k: {deme_to_check.k}, w: {deme_to_check.w}")
                boundary_check = self.is_deme_at_boundary(deme_to_check)
                if deme_to_check in fission_candidates:
                    fission_check = True
                else:
                    fission_check = False
                print(f"is at the boundary:{boundary_check}")
                print(f"is a fission candidate: {fission_check}")
            assert deme_to_check.N+deme_to_check.N2 <= G, \
                f"Deme at ({deme_to_check.x}, {deme_to_check.y}) has N={deme_to_check.N} > G={G}. After alll the process."
            assert np.all(deme_to_check.m + deme_to_check.k + deme_to_check.w == deme_to_check.N), \
                f"Deme at ({deme_to_check.x}, {deme_to_check.y}) has inconsistent methylation states: m={deme_to_check.m}, k={deme_to_check.k}, w={deme_to_check.w} != N={deme_to_check.N}."
            assert np.all(deme_to_check.m2 + deme_to_check.k2 + deme_to_check.w2 == deme_to_check.N2), \
                f"Deme at ({deme_to_check.x}, {deme_to_check.y}) has inconsistent methylation states: m2={deme_to_check.m2}, k2={deme_to_check.k2}, w2={deme_to_check.w2} != N2={deme_to_check.N2}."

    def cell_death(self, deme):
        """Simulates the death of a cell from a chosen deme"""
        return 

    def plot_grid(self, output_filename):
        """
        use plot_demes_from_data to plot the grid of demes.
        each pixel represent a deme, the color demonstrate the fullness of the deme.
        """
        all_demes = list(self.grid.demes.values())
        if not all_demes:
            return

        # Prepare data lists
        x_coords = [d.x for d in all_demes]
        y_coords = [d.y for d in all_demes]
        

        # Calculate the fractional fullness (N / G) for each deme, used for the sequential colormap.
        color_values = []
        G = self.params['G']
        for d in all_demes:
            # The value is capped at 1.0 to handle cases where a deme
            # might temporarily exceed its capacity before fissioning.
            fullness = min((d.N+d.N2) / G, 1.0)
            color_values.append(fullness)

        # Call the plotting function with the fixed grid dimension
        plot_demes_from_data(x_coords, y_coords, color_values, self.grid.dim_grid, output_filename)



    def plot_grid_subclonal(self, output_filename):
        """
        Prepares bivariate color data for each deme to visualize the
        proportion of two different clones and calls the plotting function.
        """
        all_demes = list(self.grid.demes.values())
        if not all_demes:
            # If there are no demes, create an empty plot
            plot_demes_from_data([], [], [], self.grid.dim_grid, output_filename)
            return

        # Prepare data lists
        x_coords = [d.x for d in all_demes]
        y_coords = [d.y for d in all_demes]
        
        # # Create a list of (R, G, B) color tuples for each deme.
        # # We map the subclone (N2) to RED and the original clone (N) to BLUE.
        rgb_color_values = []
        G = self.params['G']
        # #blue and red version:
        # for d in all_demes:
        #     if G > 0:
        #         # Normalize each population by the carrying capacity
        #         # red: origin clone (N), blue: subclone (N2)
        #         red_val = d.N / G
        #         blue_val = d.N2 / G
        #     else:
        #         red_val, blue_val = 0, 0

        #     # Create the RGB tuple, ensuring values are capped at 1.0
        #     # Format: (Red, Green, Blue)
        #     color_tuple = (min(red_val, 1.0), 0, min(blue_val, 1.0))
        #     rgb_color_values.append(color_tuple)

        # 1. Define the base colors for each clone (normalized from 0-255 to 0-1)
        # Sublone: Green
        color_subclone = np.array([0/255, 77/255, 64/255]) 
        # Origin Clone: Yellow
        color_origin = np.array([255/255, 193/255, 7/255])   
        for d in all_demes:
            if G > 0:
                # 2. Calculate the fraction of each clone relative to the carrying capacity.
                # This weights the color brightness by the deme's fullness.
                frac_origin = d.N / G
                frac_subclone = d.N2 / G
            else:
                frac_origin, frac_subclone = 0, 0

            # 3. Linearly blend the two colors based on the clone fractions.
            # The final color is a weighted sum of the two base colors.
            final_color_vector = (color_origin * frac_origin) + (color_subclone * frac_subclone)
            
            # Ensure values are capped at 1.0 and convert to a tuple
            final_color_tuple = tuple(np.clip(final_color_vector, 0, 1))    
            rgb_color_values.append(final_color_tuple) 
             
            
            # # hsv version
            # total_pop = d.N + d.N2
            # if total_pop == 0 or G == 0:
            #     # Append white for empty demes
            #     rgb_color_values.append((1.0, 1.0, 1.0))
            #     continue

            # # HUE: Determined by the ratio of the subclone to the total population.
            # # 0.66 corresponds to blue (0% subclone). 0.0 corresponds to red (100% subclone).
            # subclone_fraction = d.N2 / total_pop
            # hue = subclone_fraction * 0.66
            
            # # SATURATION: Kept at maximum for vibrant colors.
            # saturation = 0.8
            
            # # VALUE (BRIGHTNESS): Determined by the deme's total fullness.
            # # We apply a square root (power of 0.5) to make lower values brighter.
            # total_fullness = total_pop / G
            # value = min(total_fullness, 1.0)**0.5
            
            # # Convert the HSV color to an RGB tuple for plotting
            # color_tuple = colorsys.hsv_to_rgb(hue, saturation, value)
            

        # Call the plotting function with the RGB color data
        plot_demes_from_data_subclone(x_coords, y_coords, rgb_color_values, self.grid.dim_grid, output_filename)


    def plot_beta_histogram(self, output_filename):
        """
        Calculates the mean beta value for each CpG site across all demes
        and plots the distribution as a histogram.
        """
       
        all_demes = list(self.grid.demes.values())
        if len(all_demes) < 1:
            print("Warning: No demes available to generate a histogram.")
            return

        list_of_beta_arrays = [d.beta_value for d in all_demes]
        
        # Stack the arrays into a single 2D NumPy array: shape (number_of_demes, NSIM).
        beta_matrix = np.stack(list_of_beta_arrays)
        
        # the representative value for each site (locus): average across all demes.
        representative_betas = np.mean(beta_matrix, axis=0)
        # take the first NSIM values from the beta_matrix.
        #representative_betas = beta_matrix[0,:]
        assert len(representative_betas) == self.params['NSIM'], \
            f"Expected {self.params['NSIM']} beta values, got {len(representative_betas)}."
        # Create and save the histogram plot.
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(representative_betas, bins=100, color='#2878b5', edgecolor='black')
        
        ax.set_title(f'Distribution of Mean Beta Values at Time {self.tau:.2f} of {len(all_demes)} Demes', fontsize=14)
        ax.set_xlabel(f'Mean Beta Value per Locus (Averaged Across All Demes)')
        ax.set_ylabel('Frequency (Number of Loci)')
        ax.set_xlim(0, 1) 
        ax.grid(axis='y', alpha=0.75)

        try:
            plt.savefig(output_filename, dpi=150, bbox_inches='tight')
            print(f"Saved histogram to {output_filename}")
        except Exception as e:
            print(f"Error saving histogram: {e}")
        plt.close(fig)

    def plot_beta_colormap_pdf(self, output_filename):
        """
        Generates a multi-page PDF of heatmaps, with each page showing the
        evolution of beta values for a single CpG site over time and across demes.
        """
        print(f"Generating multi-page beta value heatmap PDF to {output_filename}...")
        
        if not self.beta_history:
            print("Warning: No beta value history to plot.")
            return

        # 1. Prepare the data into a structured format.
        # Get the unique, ordered list of all demes that ever existed.
        all_deme_ids = self.ordered_deme_ids
        num_timepoints = len(self.beta_history)
        NSIM = self.params['NSIM']

        # 2. Set up the custom colormap.
        # Beta from 0 (blue) to 1 (red), with ~0.5 as the center.
        cmap = sns.diverging_palette(253, 11, s=60, l=40, sep=80, as_cmap=True)

        # 3. Create the multi-page PDF object.
        with PdfPages(output_filename) as pdf:
            # 4. Loop through each CpG site (locus) to create one page per site.
            for i in range(NSIM):
                # Create a data matrix for this specific locus.
                # Rows: demes, Columns: timepoints
                # Default value is -1 to represent a non-existent deme, which we'll color black.
                locus_data = np.full((len(all_deme_ids), num_timepoints,), -1.0)
                
                for t, beta_snapshot in enumerate(self.beta_history):
                    for j, deme_id in enumerate(all_deme_ids):
                        if deme_id in beta_snapshot:
                            locus_data[j, t] = beta_snapshot[deme_id][i]
                
                # 5. Create the plot for the current locus.
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Use imshow for direct grid plotting.
                mesh = ax.imshow(locus_data, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='none')
                
                # Set a specific color for non-existent demes (-1).
                ax.set_facecolor('black')
                
                # 6. Add labels, titles, and a color bar.
                ax.set_title(f'Beta Value Evolution for CpG Site #{i+1}')
                ax.set_xlabel('Deme (in order of appearance)')
                ax.set_ylabel('Timepoint')
                
                # Set y-axis ticks to show actual timepoints
                tick_indices = np.linspace(0, num_timepoints - 1, num=min(10, num_timepoints), dtype=int)
                ax.set_yticks(tick_indices)
                ax.set_yticklabels([f't={t}' for t in tick_indices])
                
                cbar = fig.colorbar(mesh, ax=ax)
                cbar.set_label('Beta Value (Methylation Fraction)')
                
                # 7. Save the current figure to the PDF.
                pdf.savefig(fig)
                plt.close(fig) # Close the figure to free memory

        print("Finished generating PDF.")


    # def plot_beta_heatmap_pdf(self, output_filename):
    #     """
    #     Generates a multi-page PDF of heatmaps:
    #     Rows: Demes (ordered by appearance)
    #     Columns: Timepoints
    #     """
    #     print(f"Generating re-oriented multi-page PDF to {output_filename}...")
        
    #     if not self.beta_history:
    #         print("Warning: No beta value history to plot.")
    #         return

    #     # --- REVISED SECTION 1: Data Preparation ---
        
    #     # 1. Determine the unique, ordered list of all demes by their appearance time.
    #     # Use the definitive list of integer IDs built during the simulation
    #     ordered_deme_ids = self.ordered_deme_ids
    #     num_demes = len(ordered_deme_ids)
     

    #     num_timepoints = len(self.beta_history)
    #     num_demes = len(ordered_deme_ids)
    #     NSIM = self.params['NSIM']

    #     # 2. Set up the custom colormap.
    #     cmap = sns.diverging_palette(253, 11, s=60, l=40, sep=80, as_cmap=True)

    #     # 3. Create the multi-page PDF object.
    #     with PdfPages(output_filename) as pdf:
    #         # Loop through each CpG site to create one page per site.
    #         for i in range(NSIM):
    #             # Create a data matrix for this specific locus.
    #             # Shape: (Number of Demes, Number of Timepoints)
    #             locus_data = np.full((num_demes, num_timepoints), -1.0) # -1 for black
                
    #             # Populate the matrix
    #             for t, beta_snapshot in enumerate(self.beta_history): # Iterate through columns (time)
    #                 for j, deme_id in enumerate(ordered_deme_ids):   # Iterate through rows (demes)
    #                     if deme_id in beta_snapshot:
    #                         # If the deme exists at this time, record its beta value
    #                         locus_data[j, t] = beta_snapshot[deme_id][i]
    #                         # If a deme is not present at a timepoint, it remains -1 (black).
         
    #             # --- END REVISED SECTION 1 ---

    #             # --- REVISED SECTION 2: Plotting and Labeling ---

    #             fig, ax = plt.subplots(figsize=(12, 8))
                
    #             mesh = ax.imshow(locus_data, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='none')
    #             ax.set_facecolor('black') # Set background for non-existent demes
                
    #             # Set the new labels and title
    #             ax.set_title(f'Beta Value Evolution for CpG Site #{i+1}')
    #             ax.set_xlabel('Timepoint')
    #             ax.set_ylabel('Deme (in order of appearance)')
                
    #             # Set ticks to be more informative
    #             # X-axis ticks for time
    #             x_tick_indices = np.linspace(0, num_timepoints - 1, num=min(10, num_timepoints), dtype=int)
    #             ax.set_xticks(x_tick_indices)
    #             ax.set_xticklabels([f't={t}' for t in x_tick_indices])

    #             # Y-axis ticks for demes, showing their coordinates
    #             y_tick_indices = np.linspace(0, num_demes - 1, num=min(20, num_demes), dtype=int)
    #             ax.set_yticks(y_tick_indices)
    #             ax.set_yticklabels([f'Deme {ordered_deme_ids[idx]}' for idx in y_tick_indices])
                
    #             cbar = fig.colorbar(mesh, ax=ax)
    #             cbar.set_label('Beta Value (Methylation Fraction)')
                
    #             plt.tight_layout()
    #             pdf.savefig(fig)
    #             plt.close(fig)
    #             # --- END REVISED SECTION 2 ---

    #     print("Finished generating PDF.")


    # def plot_beta_heatmap_pdf(self, output_filename):
    #     """
    #     Generates a multi-page PDF of heatmaps from the ordered history.
    #     Rows: Demes (ordered by appearance)
    #     Columns: Timepoints
    #     """
    #     print(f"Generating ordered multi-page PDF to {output_filename}...")
        
    #     if len(self.beta_history)==0:
    #         print("Warning: No beta value history to plot.")
    #         return

    #     # --- SIMPLIFIED DATA PREPARATION ---
    #     # The history is already ordered by demes. We just need to stack and transpose it.
    #     # The result is a 3D array: (num_demes, num_timepoints, NSIM)
    #     # We convert None to np.nan for plotting the black background.
    #     full_history_array = np.array(self.beta_history, dtype=float)
    #     print("Full history array shape:", full_history_array.shape)
    #     num_demes, num_timepoints = full_history_array.shape[0], full_history_array.shape[1]
    #     NSIM = self.params['NSIM']
    #     # --- END SIMPLIFICATION ---

    #     cmap = sns.diverging_palette(253, 11, s=60, l=40, sep=80, as_cmap=True)

    #     with PdfPages(output_filename) as pdf:
    #         for i in range(NSIM):
    #             # Extract the data for this specific CpG site
    #             # This is a 2D array of (num_demes, num_timepoints)
    #             locus_data = full_history_array[:, :, i]
    #             print("Locus number of nans:", np.sum(np.isnan(locus_data)))
    #             print("Shape of locus data:", locus_data.shape)
    #             # Use np.nan as a sentinel for missing data to be colored black
    #             locus_data[np.isnan(locus_data)] = -1.0 

    #             # --- Plotting logic is now much cleaner ---
    #             fig, ax = plt.subplots(figsize=(12, 8))
    #             mesh = ax.imshow(locus_data, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='none')
    #             ax.set_facecolor('black')
                
    #             # ... (the rest of the labeling and saving logic is the same) ...
    #             ax.set_title(f'Beta Value Evolution for CpG Site #{i+1}')
    #             ax.set_xlabel('Timepoint')
    #             ax.set_ylabel('Deme (in order of appearance)')

    #             pdf.savefig(fig)
    #             plt.close(fig)

    #     print("Finished generating PDF.")


    # def plot_beta_heatmap_pdf(self, output_filename):
    #     """
    #     Generates a multi-page PDF of heatmaps from the pre-allocated history array.
    #     Each row in the plot corresponds to a flattened grid coordinate, which is one deme.
    #     """
    #     print(f"Generating full grid heatmap PDF to {output_filename}...")
        
    #     # --- SIMPLIFIED DATA PREPARATION ---
    #     # The beta_history array is already in the correct format.
    #     # Shape is (max_demes, num_timepoints, NSIM).
    #     max_demes, num_timepoints, NSIM = self.beta_history.shape
        
    #     if max_demes == 0:
    #         print("Warning: No data in history to plot.")
    #         return

    #     # --- END SIMPLIFICATION ---

    #     # Set up the colormap as requested.
    #     cmap = sns.diverging_palette(253, 11, s=60, l=40, sep=80, as_cmap=True)
    #     cmap.set_bad(color='black') # Set color for NaN values to black.

    #     with PdfPages(output_filename) as pdf:
    #         # Loop through each CpG site (locus) to create one page per site.
    #         for i in range(NSIM):
    #             # Slice the history array to get the 2D data for the current locus.
    #             # The shape is already (max_demes, num_timepoints).
    #             locus_data = self.beta_history[:, :, i]
                
    #             # Create the plot for the current locus.
    #             fig, ax = plt.subplots(figsize=(12, 16))
                
    #             # Display the data. `imshow` will render NaN values using the `cmap.set_bad` color.
    #             mesh = ax.imshow(locus_data, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='none')
                
    #             # Add labels and title.
    #             ax.set_title(f'Beta Value Evolution for CpG Site #{i+1}')
    #             ax.set_xlabel('Timepoint')
    #             ax.set_ylabel('Deme Location Index (Flattened Grid)')
                
    #             # Set informative ticks for the axes.
    #             x_tick_indices = np.linspace(0, num_timepoints - 1, num=min(10, num_timepoints), dtype=int)
    #             ax.set_xticks(x_tick_indices)
    #             ax.set_xticklabels([f't={t}' for t in x_tick_indices])
                
    #             cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    #             cbar.set_label('Beta Value (Methylation Fraction)')
                
    #             plt.tight_layout()
    #             pdf.savefig(fig)
    #             plt.close(fig) # Close the figure to free memory

    #     print("Finished generating PDF.")


    def plot_beta_clustermap_pdf(self, output_filename):
        """
        Generates a multi-page PDF of clustermaps showing the total beta value
        at each timepoint after the deme count exceeds 100.
        """
        print(f"Generating beta clustermap to {output_filename}...")
        
        output_dir = "Cluster"
        if self.params['subclone']:
            output_dir += "_Subclonal"
        if self.params['migration_edge_only']:
            if self.params['erosion']:
                output_dir += "_boundary_erosion"
            else:
                output_dir += "_boundary"
        else:
            output_dir += "_gland_fission"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. Find the first timepoint with > 100 demes
        start_timepoint = -1
        for t, snapshot in enumerate(self.simulation_history):
            if len(snapshot) > 100:
                start_timepoint = t
                break
        
        if start_timepoint == -1:
            print("Simulation did not reach 100 demes. No plot generated.")
            return

            
        cmap = sns.diverging_palette(253, 11, s=60, l=40, sep=80, as_cmap=True)

        for t in range(start_timepoint, len(self.simulation_history)):
            if t % 20 == 0:
                print(f"Processing timepoint {t}...")
                snapshot = self.simulation_history[t]
                
                # Prepare beta data DataFrame (same as before)
                df_beta = pd.DataFrame({
                    deme_id: data['beta'] for deme_id, data in snapshot.items()
                }).dropna(axis=0, how='all')
                df_beta = df_beta.loc[df_beta.var(axis=1) > 0]
                if df_beta.empty: continue

                grid_center = self.params['dim_grid'] // 2
                
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
                g.fig.savefig(f"{(os.path.join(output_dir, output_filename))}_t{t}.png", dpi=150, bbox_inches='tight')
                plt.close(g.fig)
                
        print("Finished generating clustermap series.")

    
    # visualizing the subclone
    def plot_beta_clustermap_pdf_subclone(self, output_filename):
        """
        Generates a multi-page PDF of clustermaps showing total beta value,
        annotated by radius, angle, and subclonal fraction.
        """
        print(f"Generating annotated clustermap series to {output_filename}...")
        # ... (code to create the output directory is the same) ...
        output_dir = "Cluster"
        if self.params['subclone']:
            output_dir += "_Subclonal"
        if self.params['migration_edge_only']:
            if self.params['erosion']:
                output_dir += "_boundary_erosion"
            else:
                output_dir += "_boundary"
        else:
            output_dir += "_gland_fission"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        start_timepoint = -1
        for t, snapshot in enumerate(self.simulation_history):
            if len(snapshot) > 100:
                start_timepoint = t
                break
        
        if start_timepoint == -1:
            print("Simulation did not reach 100 demes. No plot generated.")
            return

        cmap = sns.diverging_palette(253, 11, s=60, l=40, sep=80, as_cmap=True)

        for t in range(start_timepoint, len(self.simulation_history)):
            if t % 20 == 0:
                print(f"Processing timepoint {t}...")
                snapshot = self.simulation_history[t]
                
                # Prepare beta data DataFrame (same as before)
                df_beta = pd.DataFrame({
                    deme_id: data['beta'] for deme_id, data in snapshot.items()
                }).dropna(axis=0, how='all')
                df_beta = df_beta.loc[df_beta.var(axis=1) > 0]
                if df_beta.empty: continue

                grid_center = self.params['dim_grid'] // 2
                
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
                g.fig.savefig(f"{(os.path.join(output_dir, output_filename))}_t{t}.png", dpi=150, bbox_inches='tight')
                plt.close(g.fig)
                
        print("Finished generating clustermap series.")

    def plot_beta_heatmap_pdf(self, output_filename):
        """
        Generates a multi-page PDF of heatmaps. Each page shows the beta value
        evolution of the ORIGINAL CLONE for a single CpG site.
        Rows: Demes (ordered by appearance)
        Columns: Timepoints (starting after 100 demes exist)
        """
        print(f"Generating single-clone beta heatmap PDF to {output_filename}...")
        
        # 1. Find the first timepoint with > 100 demes
        start_timepoint = -1
        for t, snapshot in enumerate(self.simulation_history):
            if len(snapshot) > 100:
                start_timepoint = t
                break
        
        if start_timepoint == -1:
            print("Simulation did not reach 100 demes. No plot generated.")
            return
            
        # 2. Prepare data structures
        ordered_deme_ids = self.ordered_deme_ids # Assumes this is tracked during the run
        num_demes = len(ordered_deme_ids)
        timepoints_to_plot = self.simulation_history[start_timepoint:]
        num_timepoints = len(timepoints_to_plot)
        NSIM = self.params['NSIM']
        
        cmap = sns.diverging_palette(253, 11, s=60, l=40, sep=80, as_cmap=True)
        cmap.set_bad(color='black')
        sites_numbers = np.random.choice(range(NSIM), size=min(50, NSIM), replace=False)

        with PdfPages(output_filename) as pdf:
            # 3. Loop through each CpG site to create one page
            # random pick 50 sites from range(NSIM)
            
            
            for i in sites_numbers:
                # Create the data matrix for this site
                locus_data = np.full((num_demes, num_timepoints), np.nan)
                
                for t, snapshot in enumerate(timepoints_to_plot):
                    for j, deme_id in enumerate(ordered_deme_ids):
                        if deme_id in snapshot:
                            locus_data[j, t] = snapshot[deme_id]['beta'][i]

                # 4. Create the plot
                fig, ax = plt.subplots(figsize=(15, 10))
                mesh = ax.imshow(locus_data, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='none')
                
                ax.set_title(f'Original Clone Beta Value Evolution for CpG Site #{i+1}')
                ax.set_xlabel(f'Timepoint (starting from t={start_timepoint})')
                ax.set_ylabel('Deme (in order of appearance)')
                
                cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
                cbar.set_label('Beta Value')
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
        print("Finished generating single-clone PDF.")

    def plot_beta_heatmap_pdf_subclone(self, output_filename):
        """
        Generates a multi-page PDF of heatmaps showing both TOTAL beta value (as brightness)
        and SUBCLONAL FRACTION (as color) for each CpG site.
        """
        print(f"Generating subclonal beta heatmap PDF to {output_filename}...")
        
        # 1. Find the first timepoint with > 100 demes (same as above)
        start_timepoint = -1
        for t, snapshot in enumerate(self.simulation_history):
            if len(snapshot) > 100:
                start_timepoint = t
                break
        
        if start_timepoint == -1:
            print("Simulation did not reach 100 demes. No plot generated.")
            return
            
        ordered_deme_ids = self.ordered_deme_ids
        num_demes = len(ordered_deme_ids)
        timepoints_to_plot = self.simulation_history[start_timepoint:]
        num_timepoints = len(timepoints_to_plot)
        NSIM = self.params['NSIM']
        sites_numbers = np.random.choice(range(NSIM), size=min(50, NSIM), replace=False)

        with PdfPages(output_filename) as pdf:
            for i in sites_numbers:
                # Create a data matrix to hold RGB values
                locus_data_rgb = np.full((num_demes, num_timepoints, 3), 0.0) # Black background
                
                for t, snapshot in enumerate(timepoints_to_plot):
                    for j, deme_id in enumerate(ordered_deme_ids):
                        if deme_id in snapshot:
                            # Get data for this deme at this time
                            total_beta = snapshot[deme_id]['beta'][i]
                            subclone_frac = snapshot[deme_id]['subclonal_fraction']
                            
                            # Convert to HSV and then to RGB
                            hue = 0.66 - (subclone_frac * 0.66) # Blue to Red
                            saturation = 1.0
                            value = total_beta # Brightness = total methylation
                            
                            locus_data_rgb[j, t, :] = colorsys.hsv_to_rgb(hue, saturation, value)

                # Create the plot
                fig, ax = plt.subplots(figsize=(15, 10))
                ax.imshow(locus_data_rgb, aspect='auto', interpolation='none', origin='lower')
                
                ax.set_title(f'Total Beta & Subclone Fraction for CpG Site #{i+1}')
                ax.set_xlabel(f'Timepoint (starting from t={start_timepoint})')
                ax.set_ylabel('Deme (in order of appearance)')
                
                # Add a custom 2D legend for this color scheme
                # ... (code for the custom 2D legend, as provided previously) ...

                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
        print("Finished generating subclonal PDF.")

    def perform_virtual_biopsy(self, center_x, center_y, size):
        """
        Extracts a virtual bulk biopsy from the simulated grid.

        Args:
            center_x (int): The x-coordinate of the center of the biopsy.
            center_y (int): The y-coordinate of the center of the biopsy.
            size (int): The width and height of the square biopsy region.

        Returns:
            A dictionary containing the biopsy's name and its aggregate
            beta value array, or None if the biopsy region is empty.
        """
        half_size = size // 2
        x_start, x_end = center_x - half_size, center_x + half_size
        y_start, y_end = center_y - half_size, center_y + half_size
        
        biopsy_demes = []
        for x in range(x_start, x_end + 1):
            for y in range(y_start, y_end + 1):
                deme = self.grid.get_deme(x, y)
                if deme and deme.N+deme.N2 > 0:
                    biopsy_demes.append(deme)
        
        if not biopsy_demes:
            return None

        list_of_beta_arrays = [d.beta_value for d in biopsy_demes]
        
        # Stack the arrays into a single 2D NumPy array: shape (number_of_demes, NSIM).
        beta_matrix = np.stack(list_of_beta_arrays)
        
        # the representative value for each site (locus): average across all demes.
        biopsy_betas = np.mean(beta_matrix, axis=0)


        biopsy_name = f"Biopsy_({center_x},{center_y})"
        return {"name": biopsy_name, "beta_values": biopsy_betas}


    def plot_biopsy_correlation(self, biopsy_results, output_filename):
        """
        Creates a pair plot using the plot_longitudinal function to show
        the correlation of beta values between multiple virtual biopsies.
        """
        if not biopsy_results or len(biopsy_results) < 2:
            print("Warning: Need at least two biopsies to create a correlation plot.")
            return

        # Create a directory for the output images if it doesn't exist
        output_dir = "simulation_plots"
        if self.params['migration_edge_only']:
            if self.params['erosion']:
                output_dir += "_boundary_erosion"
            else:
                output_dir += "_boundary"
        else:
            output_dir += "_gland_fission"
        if self.params['subclone']:
            output_dir = "Subclonal_" + output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create a pandas DataFrame from the biopsy results.
        # Each column is a biopsy, each row is a CpG site.
        data_for_df = {
            result["name"]: result["beta_values"] for result in biopsy_results
        }
        df = pd.DataFrame(data_for_df)

        # Callthe plotting function with the prepared data: biopsy beta values
        print(f"Generating biopsy correlation plot to {output_filename}...")
        plot_longitudinal(
            df=df,
            outpath=(os.path.join(output_dir,output_filename)),
            title="Correlation of Beta Values Between Virtual Biopsies",
            color=True,      # Enable custom coloring
            corner=True,     # Create a lower-triangle plot
            cmap='viridis_r' # Use reversed viridis for coloring (high diff = dark)
        )
        # g = sns.pairplot(
        #     df,
        #     corner=True,  # Create a lower-triangle plot
        #     plot_kws={"alpha": 0.5},  # Transparency for scatter plots
        # )

        # # Apply the custom function to the lower triangle
        # g.map_lower(lower_triangle_wasserstein)

        # # Add a title to the plot
        # g.fig.suptitle("Correlation of Beta Values Between Virtual Biopsies", y=1.02)

        # Save the plot to the specified output file
        
    def plot_budging_trajectories(self, output_filename, num_to_plot=100, sampling_interval=5):
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
        if not self.simulation_history:
            print("Warning: No simulation history to plot.")
            return
        # Find the Start and End Timepoints for Plotting
        start_t = -1
        end_t = -1
        last_deme_count = 0

        for t, snapshot in enumerate(self.simulation_history):
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
            end_t = len(self.simulation_history)

        print(f"Plotting trajectories from timepoint {start_t} to {end_t}...")
        
        # Slice the history to the desired time window
        history_slice = self.simulation_history[start_t:end_t]

        output_dir = "simulation_plots"
        if self.params['migration_edge_only']:
            if self.params['erosion']:
                output_dir += "_boundary_erosion"
            else:
                output_dir += "_boundary"
        else:
            output_dir += "_gland_fission"
        if self.params['subclone']:
            output_dir = "Subclonal_" + output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

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
        
        plt.savefig((os.path.join(output_dir,output_filename)), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def run_gland(self):
        """
        Runs the main time-driven simulation loop for the Gland Fission model.
        """
        print("\n--- Starting Evoflux-DEMON Simulation (Gland Fission Model) ---")
        params = self.params
        migration_edge_only = params['migration_edge_only']
        erosion = params['erosion']
        gamma= params['gamma']
        mu = params['mu']
        nu = params['nu']
        zeta = params['zeta']
        theta = params['theta']
        T = params['T']

        # 1. Setup time steps for the simulation
        # The time step `dt` is chosen to be small enough that transition probabilities
        # in the methylation model do not exceed 10%, ensuring stability.
        max_rate = max(2*gamma, 2*mu, 2*nu, 2*zeta, theta)
        dt_max = 0.1 / max_rate
        n = int((T- self.tau) / dt_max) + 2  # Number of time steps.
        time_points = np.linspace(self.tau, T, n) 
        step_dt = time_points[1] - time_points[0]
        
        S_cancer = np.exp(theta* (time_points-params['tau'])).astype(int)

        if np.any(S_cancer < 0):
            raise(OverflowError('overflow encountered for S_cancer'))
        
        # Create a directory for the output images if it doesn't exist
        output_dir = "simulation_plots"
        if migration_edge_only:
            if erosion:
                output_dir += "_boundary_erosion"
            else:
                output_dir += "_boundary"
        else:
            output_dir += "_gland_fission"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initial plot at time zero
        self.plot_grid(os.path.join(output_dir, f"grid_t_{self.tau:.2f}.png"))

        beta_hist_dir = "beta_histogram"
        if migration_edge_only:
            if erosion:
                beta_hist_dir += "_boundary_erosion"
            else:
                beta_hist_dir += "_boundary"
        else:
            beta_hist_dir += "_gland_fission"

        if not os.path.exists(beta_hist_dir):
            os.makedirs(beta_hist_dir)

        # Initial plot at time zero
        self.plot_grid(os.path.join(output_dir, f"grid_t_{self.tau:.2f}.png"))
        self.plot_beta_histogram(os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}.png"))

        
        NSIM = params['NSIM']
        max_demes = self.grid.dim_grid * self.grid.dim_grid
        initial_time= self.tau

        # Shape: (Max Possible Demes, Number of Timepoints, Number of CpG sites)
        # self.beta_history = np.full((max_demes, len(time_points), NSIM), np.nan)
        self.simulation_history = []

        iter_stop_expansion = 0
        # 2. Main Simulation Loop
        for i in range(len(time_points) - 1):
            self.tau = time_points[i+1]
            terminate_cell = S_cancer[i]
            if self.params['exit_code']==0:
            # the process including expansion of demes (number of cells growing and fission),
            # and methylation states transition and beta_value updating
                self.calculate_deme_pop_glandfission_mno(step_dt, params, S_cancer[i], S_cancer[i+1])
                
                for deme in self.grid.demes.values():
                    
                    deme.update_methylation(step_dt, params)
                    # deme.beta_value = (deme.k + 2*deme.m) / (2*deme.N)
                    deme.calculate_beta_value()
                    deme.calculate_subclonal_fraction()
                    # check if there is any NaN value in beta_value
                    assert not np.any(np.isnan(deme.beta_value)), \
                        f"NaN found in beta_value for deme {deme.unique_id} at time {self.tau:.2f}"

                    if deme.unique_id not in self.seen_deme_ids:
                        self.ordered_deme_ids.append(deme.unique_id)
                        self.seen_deme_ids.add(deme.unique_id)
                
                # # Directly populate the pre-allocated history array for the current time step `i`
                # for deme in self.grid.demes.values():
                #     iter=0
                #     # Find the row index for this deme based on its appearance order
                #     if deme.unique_id in self.ordered_deme_ids:
                #         row_index = self.ordered_deme_ids.index(deme.unique_id)
                        
                #         # Assign the beta value array to the correct [row, column, :] slice
                #         self.beta_history[row_index, i, :] = deme.beta_value.reshape(1, -1)
                #         iter+=1
                # print(f"Time {self.tau:.2f}: {iter} demes updated with beta values.")

                # Create a temporary dictionary for efficient lookup of the current state
                current_betas_dict = {
                    deme.unique_id: deme.beta_value 
                    for deme in self.grid.demes.values()
                }
                print(f"Time {self.tau:.2f}: {len(current_betas_dict)} demes updated with beta values.")
                # Build an ordered list for the history.
                # The list is based on self.ordered_deme_ids to guarantee the correct order.
                # We use .get(deme_id, None) to handle cases where a deme might not exist at the current time step (though this is unlikely with the current logic).
                ordered_beta_snapshot = [
                    current_betas_dict.get(deme_id) for deme_id in self.ordered_deme_ids
                ]
                l = len(ordered_beta_snapshot)
                print("shape of ordered_beta_snapshot:", np.array(ordered_beta_snapshot).shape)
                print("number of nan in ordered_beta_snapshot:", np.sum(np.isnan(ordered_beta_snapshot)))
                assert np.array(ordered_beta_snapshot).shape==(l, NSIM), \
                    f"Expected ordered_beta_snapshot shape {(l, NSIM)}, got {np.array(ordered_beta_snapshot).shape}"
                # self.beta_history[:l, i, :] = np.array(ordered_beta_snapshot)
                # print("number of nan in beta history:", np.sum(np.isnan(self.beta_history[:, i, :])))

                current_demes_dict = {
                    deme.unique_id: deme for deme in self.grid.demes.values()
                }
                ordered_demes = [
                    current_demes_dict.get(deme_id) for deme_id in self.ordered_deme_ids
                ]
                snapshot = {
                    deme.unique_id: {
                        'coords': (deme.x, deme.y),
                        'beta': deme.beta_value,
                        'subclonal_fraction': deme.subclonal_fraction
                    } for deme in ordered_demes
                }
                self.simulation_history.append(snapshot)

                # # Ensure the snapshot has the shape of (number_of_demes, NSIM)
                # # print("shape of ordered_beta_snapshot:", len(ordered_beta_snapshot))
                # # print("shape of ordered_beta_snapshot ahhha:", len(ordered_beta_snapshot[0]))
                # # Append the ordered list to the history
                # self.beta_history.append(ordered_beta_snapshot)


                # Periodically print the state of the simulation and draw the color plots.
                if i % 10 == 0:
                    # Generate and save the plot for the current state
                    output_filename = os.path.join(output_dir, f"grid_t_{self.tau:.2f}.png")
                    self.plot_grid(output_filename)

                    # Generate and save the histogram plot
                    hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}.png")
                    self.plot_beta_histogram(hist_filename)  
            
                    total_cells = sum(d.N for d in self.grid.demes.values())
                    print(f"Time: {self.tau:.2f} | Demes: {len(self.grid.demes)} | Total Cells: {total_cells}")

            else:
                # when exit_code==1 (deme reach the grid boundary), only methylation states transion and update beta_value
                # as the time goes by until reach the terminal time T
                iter_stop_expansion += 1
                if iter_stop_expansion == 1:
                    print("\nDeme is reaching the edge of grid. Halting expansion. Only methylation transition.")
                    terminate_cell = S_cancer[i]

                    output_filename = os.path.join(output_dir, f"grid_t_{self.tau:.2f}_stop_expansion.png")
                    self.plot_grid(output_filename)
                    
                    hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}_stop_expansion.png")
                    self.plot_beta_histogram(hist_filename)
                
                # Update methylation states and beta values for all demes
                for deme in self.grid.demes.values():
                    deme.update_methylation(step_dt, params)
                    deme.calculate_beta_value()
                
                current_demes_dict = {
                    deme.unique_id: deme for deme in self.grid.demes.values()
                }
                ordered_demes = [
                    current_demes_dict.get(deme_id) for deme_id in self.ordered_deme_ids
                ]
                snapshot = {
                    deme.unique_id: {
                        'coords': (deme.x, deme.y),
                        'beta': deme.beta_value,
                        'subclonal_fraction': deme.subclonal_fraction
                    } for deme in ordered_demes
                }
                self.simulation_history.append(snapshot)

                if i%10 ==0:     
                    hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}_stop_expansion.png")
                    self.plot_beta_histogram(hist_filename)

            

            num_demes = len(self.grid.demes)
            #print(f"Current Deme Count: {num_demes}")

            # if num_demes >= max_demes:
            #     print("\nGrid is full. No more space for fission. Halting simulation.")
            #     output_filename = os.path.join(output_dir, f"grid_t_{self.tau:.2f}.png")
            #     self.plot_grid(output_filename)
                
            #     hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}.png")
            #     self.plot_beta_histogram(hist_filename)  
            #     break
              

                
        
        #print the number of nan value in each site of beta history
        # for k in range(NSIM):
        #     print(f"Number of NaN values in beta history for CpG site {k+1}: {np.sum(np.isnan(self.beta_history[:,:,k]))}")

        # 3. Final Simulation State
        print("\n--- Simulation Finished ---")
        
        total_cells = sum(d.N for d in self.grid.demes.values())
        print(f"Final Time: {self.tau:.2f}")
        print(f"Final Deme Count: {len(self.grid.demes)}")
        print(f"Final Total Cell Count: {total_cells}")
        print(f"Expected Total Cells: {terminate_cell}")
        print(f"Expected Total Cells by exponential: {np.exp(params['theta'] * (self.tau - initial_time))}")
        hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}.png")
        #print beta history to see the structure
        #print("Beta history shape:", self.beta_history.shape)
        # for i, beta_snapshot in enumerate(self.beta_history):
        #     print(f"Time {i}: {beta_snapshot.keys()}")
        #     print(f"Time {i}: {len(beta_snapshot)} demes")
        self.plot_beta_histogram(hist_filename) 
        # self.plot_beta_heatmap_pdf("beta_evolution_summary.pdf")
    
        all_demes = list(self.grid.demes.values())
        if len(all_demes) < 1:
            print("Warning: No demes available to generate a histogram.")
            return

        list_of_beta_arrays = [d.beta_value for d in all_demes]
        
        # Stack the arrays into a single 2D NumPy array: shape (number_of_demes, NSIM).
        beta_matrix = np.stack(list_of_beta_arrays)
        
        # the representative value for each site (locus): average across all demes.
        representative_betas = np.mean(beta_matrix, axis=0)
        
        # self.save_state("Gland_fission_simulation_final_state.pkl")

        return representative_betas
        

    def find_boundary_demes(self):
        """
        Uses morphological operations to efficiently find all demes on the
        boundary of the tumour cluster.

        Returns:
            list[Deme]: A list of deme objects that are on the boundary.
        """
        dim = self.grid.dim_grid
        
        # 1. Create a binary grid representing the tumour's current shape.
        #    1 where a deme exists, 0 for empty space.
        tumour_grid = np.zeros((dim, dim), dtype=np.uint8)
        for x, y in self.grid.demes.keys():
            tumour_grid[y, x] = 1

        # 2. Define the structuring element (a 3x3 square for 8-connectivity,
        #    or can create a cross shape for 4-connectivity).
        structuring_element = square(3)
        # For 4-connectivity (up, down, left, right), would use:
        # structuring_element = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        
        # 3. Dilate the tumour grid. This expands the shape by one pixel.
        dilated_grid = binary_dilation(tumour_grid, footprint=structuring_element)
        
        # 4. Find the boundary by subtracting the original from the dilated grid.
        #    The result is a grid where only the empty neighbors are marked as 1.
        boundary_pixels = dilated_grid - tumour_grid
        
        # 5. Identify which demes are adjacent to these boundary pixels.
        boundary_demes = []
        # We can do this by dilating the boundary pixels and seeing where it overlaps
        # with the original tumour grid.
        boundary_touching_grid = binary_dilation(boundary_pixels, footprint=structuring_element)
        
        # Find the coordinates where the original tumour and the boundary-touching grid overlap.
        final_boundary_coords_y, final_boundary_coords_x = np.where(
            (tumour_grid == 1) & (boundary_touching_grid == 1)
        )

        for x, y in zip(final_boundary_coords_x, final_boundary_coords_y):
            boundary_demes.append(self.grid.get_deme(x, y))
            
        return boundary_demes

    def find_and_assign_fission_targets(self):
        """
        Finds all boundary demes and assigns a unique, empty neighboring spot
        to each one for potential fission.

        Returns:
            dict: A dictionary mapping each boundary Deme object to a unique
                tuple (x, y) representing its assigned fission target coordinate.
        """
        dim = self.grid.dim_grid
        diagonal = self.params['diagonal']
        
        # 1. Create a binary grid representing the tumour's current shape.
        #    1 where a deme exists, 0 for empty space.
        tumour_grid = np.zeros((dim, dim), dtype=np.uint8)
        for x, y in self.grid.demes.keys():
            tumour_grid[y, x] = 1

        # 2. Define the structuring element (a 3x3 square for 8-connectivity,
        #    or can create a cross shape for 4-connectivity).
        if diagonal:
            structuring_element = square(3)
        else:
        # For 4-connectivity (up, down, left, right), would use:
            structuring_element = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        
        # 3. Dilate the tumour grid. This expands the shape by one pixel.
        dilated_grid = binary_dilation(tumour_grid, footprint=structuring_element)
        
        # 4. Find the boundary by subtracting the original from the dilated grid.
        #    The result is a grid where only the empty neighbors are marked as 1.
        boundary_pixels = dilated_grid - tumour_grid

        # 5. Identify which demes are adjacent to these boundary pixels.
        boundary_demes = []
        # We can do this by dilating the boundary pixels and seeing where it overlaps
        # with the original tumour grid.
        boundary_touching_grid = binary_dilation(boundary_pixels, footprint=structuring_element)
        
        # Find the coordinates where the original tumour and the boundary-touching grid overlap.
        final_boundary_coords_y, final_boundary_coords_x = np.where(
            (tumour_grid == 1) & (boundary_touching_grid == 1)
        )

        for x, y in zip(final_boundary_coords_x, final_boundary_coords_y):
            boundary_demes.append(self.grid.get_deme(x, y))
    
        # Get the coordinates of all available empty spots： from boundary_pixels, minus those get demes
        available_spots_y, available_spots_x = np.where(boundary_pixels == 1)
        available_spots = list(zip(available_spots_x, available_spots_y))
        for x,y in zip(available_spots_x, available_spots_y):
            if (x,y) in self.grid.demes:
                # If the spot is occupied by a deme, remove it from available spots.
                available_spots.remove((x,y))
        
        if not available_spots:
            return {} # No place to grow

        if not boundary_demes:
            return {}

        # 5. Assign a unique empty spot to each boundary deme.
        #    We use a KDTree for a very efficient way to find the nearest empty spot for each boundary deme.
        
        boundary_deme_coords = np.array([(d.x, d.y) for d in boundary_demes])
        available_spots_tree = KDTree(available_spots)
        
        fission_assignments = {}
        
        # Find the nearest available spot for each boundary deme.
        distances, indices = available_spots_tree.query(boundary_deme_coords, k=1)
        
        # Create a list of (deme, target_spot, distance) to sort and assign.
        potential_assignments = []
        for i, deme in enumerate(boundary_demes):
            target_spot = tuple(available_spots[indices[i]])
            potential_assignments.append((deme, target_spot, distances[i]))
            
        # Sort by distance to prioritize demes that are closest to an empty spot.
        potential_assignments.sort(key=lambda item: item[2])
        
        assigned_spots = set()
        for deme, target_spot, dist in potential_assignments:
            if target_spot not in assigned_spots:
                fission_assignments[deme] = target_spot
                assigned_spots.add(target_spot)
                
        return fission_assignments

    def calculate_deme_pop_boundary_growth(self, dt, params, S_i, S_iplus1):

        
        print(f"Beginning:")
        for deme in self.grid.demes.values():
            print(f"At the beginning, Deme at ({deme.x}, {deme.y}) has N={deme.N}.")

        delta_S = S_iplus1 - S_i
        assert delta_S >= 0, "Population growth must be non-negative."

        G = params['G']
        erosion = params['erosion']

        # Distribute growth multinomially to elligible demes that are not yet full.
        eligible_demes = [d for d in self.grid.demes.values() if d.N < G]
        if not eligible_demes:
            print("No eligible demes to grow")
            return
        old_N_eligible = np.zeros(len(eligible_demes))

        if eligible_demes:
            # Calculate the total population of only the eligible demes.
            S_eligible = sum(d.N for d in eligible_demes)
        
        if S_eligible > 0:
            # the probability for multinomial distribution: p_i = N_i / S_eligible
            p = np.array([d.N / S_eligible for d in eligible_demes])
            growth_per_deme = multinomial_rvs(delta_S, p.reshape(-1, 1), self.rng).flatten()
            
            # Update the populations of the eligible demes: because of the deme fission later,
            # the methyltion states will be updated for N<=G
            for i, deme in enumerate(eligible_demes):
                old_N_eligible[i] = deme.N
                growth = growth_per_deme[i]
                if growth == 0: continue
                
                deme.N +=growth

        print(f"Growth distributed.")
        # for deme in self.grid.demes.values():
        #     print(f"Deme at ({deme.x}, {deme.y}) has N={deme.N}.")

        # Fission of Full Demes at the Boundary: identify all demes that are full and have space to divide.
        if erosion:
            boundary_demes = self.find_boundary_demes()
            fission_candidates = [
                d for d in boundary_demes
                if d.N >= G 
            ]
        else:
            fission_candidates = [
                d for d in self.grid.demes.values() 
                if d.N >= G and self.is_deme_at_boundary(d)
            ]
        
        # Different to gland fission model: the demes that are not fission candidates but N>G should only have N=G, 
        # and the excess population will be sum up and multinomial distributed to those eligible demes again.
        not_fission_full_demes =[d for d in self.grid.demes.values() if d.N>G and d not in fission_candidates]
        old_not_fission_full_demes = not_fission_full_demes.copy()

        
        for deme in old_not_fission_full_demes:
            print(f"OLD Deme at ({deme.x}, {deme.y}) has N={deme.N} > G={G}.")
        for deme in fission_candidates:
            print(f"Before distribute excess S, Fission candidate at ({deme.x}, {deme.y}) has N={deme.N} >= G={G}.")

        excess_S = sum(
            d.N - G for d in not_fission_full_demes)
        for deme in not_fission_full_demes:
            deme.N = G


        while len(not_fission_full_demes) > 0:
            new_eligible_demes = [d for d in self.grid.demes.values() if d.N < G]
            if new_eligible_demes:
                new_S_eligible = sum(d.N for d in new_eligible_demes)
                if new_S_eligible > 0:
                    p = np.array([d.N / new_S_eligible for d in new_eligible_demes])
                    excess_growth_per_deme = multinomial_rvs(excess_S, p.reshape(-1, 1), self.rng).flatten()
                    
                    for i, deme in enumerate(new_eligible_demes):
                        excess_growth = excess_growth_per_deme[i]
                        if excess_growth == 0: continue
                        deme.N += round(excess_growth)
            else:
                print("No eligible demes to distribute excess population.")
                break

            if erosion:
                boundary_demes = self.find_boundary_demes()
                fission_candidates = [
                    d for d in boundary_demes
                    if d.N >= G 
                ]
            else:
                fission_candidates = [
                    d for d in self.grid.demes.values() 
                    if d.N >= G and self.is_deme_at_boundary(d)
                ]

            not_fission_full_demes =[d for d in self.grid.demes.values() if d.N > G and d not in fission_candidates]
            excess_S = sum(
                d.N - G for d in not_fission_full_demes)
            for deme in not_fission_full_demes:
                deme.N = G

        print("After distributing excess population:")
        for deme in self.grid.demes.values():
            print(f"After distributing excess S: Deme at ({deme.x}, {deme.y}) has N={deme.N}.")

        not_fission_full_demes =[d for d in self.grid.demes.values() if d.N > G and d not in fission_candidates]

        for deme in old_not_fission_full_demes:
            print(f"NEW OLD Deme at ({deme.x}, {deme.y}) has N={deme.N}.")

        for deme in not_fission_full_demes:
            print(f"Deme at ({deme.x}, {deme.y}) has N={deme.N} > G={G}. It should not be a fission candidate from list.")

        assert len(not_fission_full_demes) == 0, \
            f"There are still demes with N > G that are not fission candidates: {[f'({d.x}, {d.y})' for d in not_fission_full_demes]}."
        # for deme in self.grid.demes.values():
        #     if deme.N>G and deme not in fission_candidates:
        #         print(f"Deme at ({deme.x}, {deme.y}) has N={deme.N} > G={G}. And it isnot fission demes.")
        #     if deme not in fission_candidates:
        #         assert deme.N <= G, \
        #             f"Deme at ({deme.x}, {deme.y}) has N={deme.N} > G={G}. It should not be a fission candidate."


        if S_eligible >0:
            for i, deme in enumerate(eligible_demes):
                old_N = old_N_eligible[i]
                new_N = deme.N
                # Grow the methylation states：
    
                if old_N > 0 and new_N > old_N:
                    if new_N >= G :
                        assert np.all(deme.m + deme.k + deme.w == old_N), \
                            f"Deme at ({deme.x}, {deme.y}) has inconsistent methylation states of old_N={old_N}: m={deme.m}, k={deme.k}, w={deme.w}."
                        m, k, w = grow_cancer(deme.m, deme.k, deme.w, old_N, G, deme.rng)
                        deme.m = m
                        deme.k = k
                        deme.w = w
                        deme.N = new_N
                
                        if np.any(deme.m + deme.k + deme.w != G):
                            print("old N:", old_N)
                            print("new N:", new_N)
                            print("m:", deme.m)
                            print("k:", deme.k)
                            print("w:", deme.w)

                        assert np.all(deme.m + deme.k + deme.w == G), \
                            f"Deme at ({deme.x}, {deme.y}) has inconsistent methylation states of G after growth: m={deme.m}, k={deme.k}, w={deme.w}."
                        
                    else:
                        assert np.all(deme.m + deme.k + deme.w == old_N), \
                            f"Deme at ({deme.x}, {deme.y}) has inconsistent methylation states of old_N={old_N}: m={deme.m}, k={deme.k}, w={deme.w}."
                        m, k, w = grow_cancer(deme.m, deme.k, deme.w, old_N, new_N, deme.rng)
                        deme.m = m
                        deme.k = k
                        deme.w = w
                        if np.any(deme.m + deme.k + deme.w != new_N):
                            print("old N:", old_N)
                            print("new N:", new_N)
                            print("m:", deme.m)
                            print("k:", deme.k)
                            print("w:", deme.w)
                        assert np.all(deme.m + deme.k + deme.w == new_N), \
                            f"Deme at ({deme.x}, {deme.y}) has inconsistent methylation states of new_N={new_N}: m={deme.m}, k={deme.k}, w={deme.w}."
                        deme.N = new_N
                       
                        assert np.all(deme.m + deme.k + deme.w == new_N), \
                            f"Deme at ({deme.x}, {deme.y}) has inconsistent methylation states of N={new_N} after growth: m={deme.m}, k={deme.k}, w={deme.w}." 
                else:   
                    print(f"Deme at ({deme.x}, {deme.y}) has no cells to grow from. Skipping growth.")

        if erosion:
            boundary_demes = self.find_boundary_demes()
            fission_candidates = [
                d for d in boundary_demes
                if d.N >= G 
            ]
        else:
            fission_candidates = [
                d for d in self.grid.demes.values() 
                if d.N >= G and self.is_deme_at_boundary(d)
            ]

        print("Before fission:")
        old_fission_candidates = fission_candidates.copy()
        for deme in fission_candidates:
            print(f"Before: Fission candidate at ({deme.x}, {deme.y}) with N={deme.N}.")

        
        iter=0
        grid_boundary = self.grid.dim_grid
        # fission demes iteratively until no more fission candidates are found.
        # This is to ensure that all demes that can be split are split, even if it takes multiple iterations.
        while len(fission_candidates) > 0:  

            # Before fission demes at the edge, first check if there are demes reaching the grid boundary 
            for deme in self.grid.demes.values():
                if deme.x == 0  or deme.x == grid_boundary - 1 or deme.y == 0 or deme.y == grid_boundary - 1:
                    print("Deme at ({deme.x}, {deme.y}) has reached the grid boundary when deme fission on the edge.")
                    self.params['exit_code'] = 1
                    break
            if self.params['exit_code'] == 1:
                print("Exit code is 1, stop the deme fission.")
                for deme in self.grid.demes.values():
                    if deme.N>G:
                        deme.N = G
                break

            print(f"Found {len(fission_candidates)} fission candidates at the boundary with N >= G={G}.")
            iter += 1
            print(f"Iteration {iter}: Fissioning demes at the boundary.")
            assert iter<10, \
                f"Too many iterations ({iter}) for fissioning demes at the boundary. Check the logic."
            for deme_to_split in fission_candidates:
                # if deme_to_split.N >= 2*G:
                #     f"Deme at ({deme_to_split.x}, {deme_to_split.y}) has N={deme_to_split.N} >= 2*G={2*G}. Cannot split."
                # assert deme_to_split.N < 2*G, \
                #     f"Deme at ({deme_to_split.x}, {deme_to_split.y}) has N={deme_to_split.N} >= 2*G={2*G}. Cannot split."
                excess_population = deme_to_split.N - G
                if np.any(deme_to_split.m + deme_to_split.k + deme_to_split.w != G):
                    print("m:", deme_to_split.m)
                    print("k:", deme_to_split.k)
                    print("w:", deme_to_split.w)
                    print("deme to split:", deme_to_split.x, deme_to_split.y)
                    print("N:", deme_to_split.N)
                # Set the deme to size G for the split
                deme_to_split.N = G
                
                # Execute the deme fission
                new_deme = self.deme_fission(deme_to_split)
                
                if new_deme:
                    print(f"Fissioned deme on the boundary at ({deme_to_split.x}, {deme_to_split.y}) into new deme at ({new_deme.x}, {new_deme.y}) with N={new_deme.N}.")
                else:
                    print(f"Fission failed for deme on the boundary at ({deme_to_split.x}, {deme_to_split.y}). No valid location for new deme.")
                # add the excess population equally to the origin and new deme
                if excess_population > 0:
                    excess_per_deme = excess_population // 2
                    excess_origin_deme = excess_per_deme
                    excess_new_deme = excess_per_deme
                    if excess_population % 2 != 0:
                        # Randomly assign the extra cell to one of the demes
                        if self.rng.random() < 0.5:
                            excess_origin_deme +=1
                        else:
                            excess_new_deme +=1
                    
                    if new_deme:
                        new_deme.N += excess_new_deme
                        deme_to_split.N += excess_origin_deme
                    else:
                        deme_to_split.N += excess_population
                    
                    if new_deme:

                        assert np.all(deme_to_split.m + deme_to_split.k + deme_to_split.w == deme_to_split.N-excess_origin_deme), \
                            f"Deme at ({deme_to_split.x}, {deme_to_split.y}) has inconsistent methylation states before excess population distribution: m={deme_to_split.m}, k={deme_to_split.k}, w={deme_to_split.w}."
                        if deme_to_split.N <= G: 
                            # update methylation state for the original deme
                            m,k,w = grow_cancer(
                                deme_to_split.m, deme_to_split.k, deme_to_split.w,
                                deme_to_split.N - excess_origin_deme, deme_to_split.N, self.rng)
                            deme_to_split.m = m
                            deme_to_split.k = k
                            deme_to_split.w = w
                            #deme_to_split.calculate_beta_value()
                            assert np.all(deme_to_split.m + deme_to_split.k + deme_to_split.w == deme_to_split.N), \
                                f"Deme at ({deme_to_split.x}, {deme_to_split.y}) of N={deme_to_split.N} has inconsistent methylation states after excess population distribution: m={deme_to_split.m}, k={deme_to_split.k}, w={deme_to_split.w}."
                        else:
                            m,k,w = grow_cancer(
                                deme_to_split.m, deme_to_split.k, deme_to_split.w,
                                deme_to_split.N - excess_origin_deme, G, self.rng)
                            deme_to_split.m = m
                            deme_to_split.k = k
                            deme_to_split.w = w
                            assert np.all(deme_to_split.m + deme_to_split.k + deme_to_split.w == G), \
                                f"Deme at ({deme_to_split.x}, {deme_to_split.y}) of G has inconsistent methylation states after excess population distribution: m={deme_to_split.m}, k={deme_to_split.k}, w={deme_to_split.w}."
                    
                        assert np.all(new_deme.m + new_deme.k + new_deme.w == new_deme.N-excess_new_deme), \
                            f"New deme at ({new_deme.x}, {new_deme.y}) has inconsistent methylation states before excess population distribution: m={new_deme.m}, k={new_deme.k}, w={new_deme.w}."
                        if new_deme.N <= G:    
                            # update methylation state for the new deme
                            m, k, w = grow_cancer(
                                new_deme.m, new_deme.k, new_deme.w,
                                new_deme.N - excess_new_deme, new_deme.N, self.rng)
                            new_deme.m = m
                            new_deme.k = k
                            new_deme.w = w
                            #new_deme.calculate_beta_value()
                            assert np.all(new_deme.m + new_deme.k + new_deme.w == new_deme.N), \
                                f"New deme at ({new_deme.x}, {new_deme.y}) of N={new_deme.N} has inconsistent methylation states after excess population distribution: m={new_deme.m}, k={new_deme.k}, w={new_deme.w}."
                        else:
                            m, k, w = grow_cancer(
                                new_deme.m, new_deme.k, new_deme.w,
                                new_deme.N - excess_new_deme, G, self.rng)
                            new_deme.m = m
                            new_deme.k = k
                            new_deme.w = w
                            assert np.all(new_deme.m + new_deme.k + new_deme.w == G), \
                                f"New deme at ({new_deme.x}, {new_deme.y}) of G has inconsistent methylation states after excess population distribution: m={new_deme.m}, k={new_deme.k}, w={new_deme.w}."
        
            #update the fission candidates after the fission
            if erosion:
                boundary_demes = self.find_boundary_demes()
                fission_candidates = [
                    d for d in boundary_demes
                    if d.N >= G 
                ]
                print("I am updating the fission candidates after the fission")
            else:
                fission_candidates = [
                    d for d in self.grid.demes.values() 
                    if d.N >= G and self.is_deme_at_boundary(d)
                ]
                print("I am updating the fission candidates after the fission")
        
        print("After fissioning demes:")
        for deme in self.grid.demes.values():
            print(f"After fission on the edge: Deme at ({deme.x}, {deme.y}) has N={deme.N}.")
        for deme in old_fission_candidates:
            print(f"NEW OLD Fission candidate at ({deme.x}, {deme.y}) with N={deme.N}.")

        if self.params['exit_code']==1:
            print("Exit code is 1, stop the expansion of boundary growth.")
            for deme in self.grid.demes.values():
                if deme.N>G:
                    deme.N = G
            return
        

        # Check for any demes that are still full. 
        # These are typically internal demes or boundary demes that had no empty neighbors.
        remaining_full_demes = [d for d in self.grid.demes.values() if d.N > G]
        original_setting = self.params['migration_edge_only']

        iter=0
        while len(remaining_full_demes)>0:
            iter+=1
            assert iter<10,\
                f"Too many iterations ({iter}) for fissioning remaining full demes. Check the logic."
            
            if self.params['exit_code'] == 1:
                print("Exit code is 1, stop the deme fission for remaining full demes.")
                for deme in self.grid.demes.values():
                    if deme.N>G:
                        deme.N = G
                break

            # Temporarily change the simulation mode to allow the deme_fission function to use budging.
            self.params['migration_edge_only'] = False
        
            for deme_to_split in remaining_full_demes:
                # Check the condition again, as a previous budge may have altered the grid
                if deme_to_split.N > G:
                    oldN = deme_to_split.N
                    print(f"Fissioning remaining full deme at ({deme_to_split.x}, {deme_to_split.y}) with N={oldN}.")
                    excess_population = deme_to_split.N - G
                    deme_to_split.N = G
                    # Execute the deme fission
                    new_deme = self.deme_fission_boundary_budge(deme_to_split)

                    if new_deme is None:
                        self.params['exit_code'] = 1
                        print(f"The new deme when fissioning remain full deme at ({deme_to_split.x}, {deme_to_split.y}) is None, stop the deme fission.")
                        break
                    
                    # if not new_deme:
                    #     print(f"Fission failed for deme at ({deme_to_split.x}, {deme_to_split.y}). No valid location for new deme.")
                    #     continue
                    # add the excess population equally to the origin and new deme
                    if new_deme and excess_population > 0:
                        excess_per_deme = excess_population // 2
                        excess_origin_deme = excess_per_deme
                        excess_new_deme = excess_per_deme
                        if excess_population % 2 != 0:
                            # Randomly assign the extra cell to one of the demes
                            if self.rng.random() < 0.5:
                                excess_origin_deme +=1
                            else:
                                excess_new_deme +=1
                        deme_to_split.N += excess_origin_deme
                        new_deme.N += excess_new_deme

                        assert np.all(deme_to_split.m + deme_to_split.k + deme_to_split.w == deme_to_split.N-excess_origin_deme), \
                                f"Remain Full Deme at ({deme_to_split.x}, {deme_to_split.y}) has inconsistent methylation states before excess population distribution: m={deme_to_split.m}, k={deme_to_split.k}, w={deme_to_split.w}."
                        if deme_to_split.N <= G:     
                            # update methylation state for the original deme
                            m,k,w = grow_cancer(
                                deme_to_split.m, deme_to_split.k, deme_to_split.w,
                                deme_to_split.N - excess_origin_deme, deme_to_split.N, self.rng)
                            deme_to_split.m = m
                            deme_to_split.k = k
                            deme_to_split.w = w
                            assert np.all(deme_to_split.m + deme_to_split.k + deme_to_split.w == deme_to_split.N), \
                                f"Remain Full Deme at ({deme_to_split.x}, {deme_to_split.y}) of N={deme_to_split.N} has inconsistent methylation states after excess population distribution: m={deme_to_split.m}, k={deme_to_split.k}, w={deme_to_split.w}."
                        else:
                            m,k,w = grow_cancer(
                                deme_to_split.m, deme_to_split.k, deme_to_split.w,
                                deme_to_split.N - excess_origin_deme, G, self.rng)
                            deme_to_split.m = m
                            deme_to_split.k = k
                            deme_to_split.w = w
                            assert np.all(deme_to_split.m + deme_to_split.k + deme_to_split.w == G), \
                                f"Remain Full Deme at ({deme_to_split.x}, {deme_to_split.y}) of G has inconsistent methylation states after excess population distribution: m={deme_to_split.m}, k={deme_to_split.k}, w={deme_to_split.w}."

                        assert np.all(new_deme.m + new_deme.k + new_deme.w == new_deme.N-excess_new_deme), \
                            f"New deme at ({new_deme.x}, {new_deme.y}) has inconsistent methylation states before excess population distribution: m={new_deme.m}, k={new_deme.k}, w={new_deme.w}."
                        if new_deme.N <= G:   
                            # update methylation state for the new deme
                            m, k, w = grow_cancer(
                                new_deme.m, new_deme.k, new_deme.w,
                                new_deme.N - excess_new_deme, new_deme.N, self.rng)
                            new_deme.m = m
                            new_deme.k = k
                            new_deme.w = w
                            assert np.all(new_deme.m + new_deme.k + new_deme.w == new_deme.N), \
                                f"New deme at ({new_deme.x}, {new_deme.y}) of N={new_deme.N} has inconsistent methylation states after excess population distribution: m={new_deme.m}, k={new_deme.k}, w={new_deme.w}."
                        else:
                            m, k, w = grow_cancer(
                                new_deme.m, new_deme.k, new_deme.w,
                                new_deme.N - excess_new_deme, G, self.rng)
                            new_deme.m = m
                            new_deme.k = k
                            new_deme.w = w
                            assert np.all(new_deme.m + new_deme.k + new_deme.w == G), \
                                f"New deme at ({new_deme.x}, {new_deme.y}) of G has inconsistent methylation states after excess population distribution: m={new_deme.m}, k={new_deme.k}, w={new_deme.w}."
        
            remaining_full_demes = [d for d in self.grid.demes.values() if d.N > G]
        # Restore the original simulation mode
        self.params['migration_edge_only'] = original_setting


        boundary_demes = self.find_boundary_demes()
        #print coordinates of boundary demes
        # for deme in boundary_demes:
        #     print(f"Boundary deme at ({deme.x}, {deme.y}) with N={deme.N}.")
        fission_candidates = [
            d for d in boundary_demes
            if d.N >= G
        ]
        for deme in fission_candidates:
            print(f"Fission candidate at ({deme.x}, {deme.y}) with N={deme.N}, after allll the process")
        # assert len(fission_candidates) == 0, \
        #     f"There are still fission candidates with N >= G: {[f'({d.x}, {d.y})' for d in fission_candidates]}."
        while len(fission_candidates) > 0:
            if self.params['exit_code'] == 1:
                print("Exit code is 1, stop the deme fission for remaining full demes.")
                break
            for deme in fission_candidates:
                new_deme = self.deme_fission(deme)
                if new_deme:
                    print(f"Fissioned deme at ({deme.x}, {deme.y}) into new deme at ({new_deme.x}, {new_deme.y}) with N={new_deme.N}. After alll the process.")
                else:
                    self.params['exit_code'] = 1
                    print(f"Fission failed for deme at ({deme.x}, {deme.y}). No valid location for new deme. After alll the process.")
                boundary_demes = self.find_boundary_demes()
                fission_candidates = [
                    d for d in boundary_demes
                    if d.N >= G
                ]
        for deme_to_check in self.grid.demes.values():
            if deme_to_check.N > G:
                print(f"Deme at ({deme_to_check.x}, {deme_to_check.y}) has N={deme_to_check.N} > G={G}.")
                #print(f"m: {deme_to_check.m}, k: {deme_to_check.k}, w: {deme_to_check.w}")
                boundary_check = self.is_deme_at_boundary(deme_to_check)
                if deme_to_check in fission_candidates:
                    
                    fission_check = True
                else:
                    fission_check = False
                print(f"is at the boundary:{boundary_check}")
                print(f"is a fission candidate: {fission_check}")
            assert deme_to_check.N <= G, \
                f"Deme at ({deme_to_check.x}, {deme_to_check.y}) has N={deme_to_check.N} > G={G}. After alll the process."
            assert np.all(deme_to_check.m + deme_to_check.k + deme_to_check.w == deme_to_check.N), \
                f"Deme at ({deme_to_check.x}, {deme_to_check.y}) has inconsistent methylation states: m={deme_to_check.m}, k={deme_to_check.k}, w={deme_to_check.w} != N={deme_to_check.N}."


    
    def run_boundary_growth_model(self):
        """
        A dedicated simulation loop for the Boundary Growth model.
        Growth occurs only when full demes at the boundary divide into empty space.
        """
        print("\n--- Running Simulation: Boundary Growth (Erosion) Model ---")
        params = self.params
        G = params['G']
        T = params['T']
        gamma = params['gamma']
        mu = params['mu']
        nu = params['nu']
        zeta = params['zeta']
        theta = params['theta']
        erosion = params['erosion']

        # Setup time steps
        max_rate = max(2*gamma, 2*mu, 2*nu, 2*zeta, theta)
        dt_max = 0.1 / max_rate
        # dt_max/=5  #not work
        n = int((T- self.tau) / dt_max) + 2  # Number of time steps.
        time_points = np.linspace(self.tau, T, n) 
        step_dt = time_points[1] - time_points[0]
        S_cancer = np.exp(theta * (time_points - params['tau'])).astype(int)
        if np.any(S_cancer < 0):
            raise OverflowError('Overflow encountered for S_cancer')
        

        # Create a directory for the output images if it doesn't exist
        output_dir = "simulation_plots_boundary"
        beta_hist_dir = "beta_histogram_boundary"
        if erosion:
            output_dir += "_erosion"
            beta_hist_dir += "_erosion"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(beta_hist_dir):
            os.makedirs(beta_hist_dir)

        # Initial plot at time zero
        self.plot_grid(os.path.join(output_dir, f"grid_t_{self.tau:.2f}.png"))
        self.plot_beta_histogram(os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}.png"))
        max_num_demes = self.grid.dim_grid * self.grid.dim_grid
        step_dt = time_points[1] - time_points[0]

        self.simulation_history = []
        iter_stop_expansion = 0
        initial_time= self.tau
        # Main Simulation Loop
        for i in range(len(time_points) - 1):
            self.tau += step_dt
            # if erosion:
            #     self.calculate_deme_pop_boundary_growth_erosion(step_dt, params, S_cancer[i], S_cancer[i+1])
            # else:
            #     self.calculate_deme_pop_boundary_growth(step_dt, params, S_cancer[i], S_cancer[i+1])

            if self.params['exit_code'] == 0:
                self.calculate_deme_pop_boundary_growth(step_dt, params, S_cancer[i], S_cancer[i+1])
                # Update the beta values and methylation states for each deme
                for deme in self.grid.demes.values():
                    deme.update_methylation(step_dt, params)
                    deme.beta_value = (deme.k + 2*deme.m) / (2*deme.N)

                    if deme.unique_id not in self.seen_deme_ids:
                        self.ordered_deme_ids.append(deme.unique_id)
                        self.seen_deme_ids.add(deme.unique_id)
                    
                # plot the grid every 10 iterations, when expansion is still ongoing
                if i % 10 == 0:
                    total_cells = sum(d.N for d in self.grid.demes.values())
                    print(f"Time: {self.tau:.2f} | Demes: {len(self.grid.demes)} | Total Cells: {total_cells}")
                    
                    output_filename = os.path.join(output_dir, f"grid_t_{self.tau:.2f}.png")
                    self.plot_grid(output_filename)

                    hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}.png")
                    self.plot_beta_histogram(hist_filename)


                current_demes_dict = {
                    deme.unique_id: deme for deme in self.grid.demes.values()
                }
                ordered_demes = [
                    current_demes_dict.get(deme_id) for deme_id in self.ordered_deme_ids
                ]
                snapshot = {
                    deme.unique_id: {
                        'coords': (deme.x, deme.y),
                        'beta': deme.beta_value,
                        'subclonal_fraction': deme.subclonal_fraction
                    } for deme in ordered_demes
                }
                self.simulation_history.append(snapshot)
            else:
                iter_stop_expansion += 1
                if iter_stop_expansion == 1:
                    print("\nDeme is reaching the edge of grid. Halting expansion. Only methylation transition.")
                    terminate_cell = S_cancer[i]
                    output_filename = os.path.join(output_dir, f"grid_t_{self.tau:.2f}_stop_expansion.png")
                    self.plot_grid(output_filename)
                    hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}_stop_expansion.png")
                    self.plot_beta_histogram(hist_filename)

                for deme in self.grid.demes.values():
                    deme.update_methylation(step_dt, params)
                    deme.beta_value = (deme.k + 2*deme.m) / (2*deme.N)
                
                current_demes_dict = {
                    deme.unique_id: deme for deme in self.grid.demes.values()
                }
                ordered_demes = [
                    current_demes_dict.get(deme_id) for deme_id in self.ordered_deme_ids
                ]
                snapshot = {
                    deme.unique_id: {
                        'coords': (deme.x, deme.y),
                        'beta': deme.beta_value,
                        'subclonal_fraction': deme.subclonal_fraction
                    } for deme in ordered_demes
                }
                self.simulation_history.append(snapshot)

                # plot the beta histogram every 10 iterations after the expansion stopped
                if i % 10 == 0:

                    hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}_stop_expansion.png")
                    self.plot_beta_histogram(hist_filename)

            # if len(self.grid.demes) >= max_num_demes:
            #     break

                
        print("\n--- Simulation Finished ---")
        # Final print statements
        total_cells = sum(d.N for d in self.grid.demes.values())
        print(f"Final Time: {self.tau:.2f}")
        print(f"Final Deme Count: {len(self.grid.demes)}")
        print(f"Final Total Cell Count: {total_cells}")
        print(f"Expected Total Cells: {terminate_cell}")
        print(f"Expected Total Cells by exponential: {np.exp(params['theta'] * (self.tau - initial_time))}")
        hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}.png")
        self.plot_beta_histogram(hist_filename)
        output_filename = os.path.join(output_dir, f"grid_t_{self.tau:.2f}.png")
        self.plot_grid(output_filename)

        all_demes = list(self.grid.demes.values())
        if len(all_demes) < 1:
            print("Warning: No demes available to generate a histogram.")
            return

        list_of_beta_arrays = [d.beta_value for d in all_demes]
        
        # Stack the arrays into a single 2D NumPy array: shape (number_of_demes, NSIM).
        beta_matrix = np.stack(list_of_beta_arrays)
        
        # the representative value for each site (locus): average across all demes.
        representative_betas = np.mean(beta_matrix, axis=0)

        # self.save_state("Boundary_growth_simulation_final_state.pkl")

        return representative_betas

    def run_gland_subclonal(self):
        """
        Runs the main time-driven simulation loop for the Gland Fission model, and there is SUBCLONE.
        """
        print("\n--- Starting Evoflux-DEMON Simulation (Spatial Subclonal Model) ---")
        params = self.params
        # Unpack parameters for clarity
        migration_edge_only = params['migration_edge_only']
        erosion = params['erosion']
        theta1, theta2 = params['theta'], params['theta2']
        tau1, tau2, T = params['tau'], params['tau2'], params['T']
        gamma = params['gamma']
        mu = params['mu']
        nu = params['nu']
        zeta = params['zeta']

        # --- SETUP TIME STEPS FOR BOTH PHASES ---
        dt_max1 = 0.1 / np.max([2*gamma, 2*mu, 2*nu, 2*zeta, theta1])
        dt_max2 = 0.1 / np.max([2*gamma, 2*mu, 2*nu, 2*zeta, theta1, theta2])
        
        n1 = int((tau2 - tau1) / dt_max1) + 2
        n2 = int((T - tau2) / dt_max2) + 2
        time_points1 = np.linspace(tau1, tau2, n1)
        time_points2 = np.linspace(tau2, T, n2)
        dt1, dt2 = time_points1[1] - time_points1[0], time_points2[1] - time_points2[0]
        S_cancer = np.exp(theta1 * (time_points1-tau1)).astype(int)
        S_cancer1 = np.exp(theta1 * (time_points2-tau1)).astype(int)
        S_cancer2 = np.exp(theta2 * (time_points2-tau2)).astype(int)


        if np.any(S_cancer < 0):
            raise(OverflowError('overflow encountered for S_cancer'))
        
        # Create a directory for the output images if it doesn't exist
        output_dir = "Subclonal_simulation_plots"

        if migration_edge_only:
            if erosion:
                output_dir += "_boundary_erosion"
            else:
                output_dir += "_boundary"
        else:
            output_dir += "_gland_fission"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initial plot at time zero
        self.plot_grid(os.path.join(output_dir, f"grid_t_{self.tau:.2f}.png"))

        beta_hist_dir = "Subclonal_beta_histogram"
        if migration_edge_only:
            if erosion:
                beta_hist_dir += "_boundary_erosion"
            else:
                beta_hist_dir += "_boundary"
        else:
            beta_hist_dir += "_gland_fission"

        if not os.path.exists(beta_hist_dir):
            os.makedirs(beta_hist_dir)

        # Initial plot at time zero
        self.plot_grid(os.path.join(output_dir, f"grid_t_{self.tau:.2f}.png"))
        self.plot_beta_histogram(os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}.png"))

        gamma= params['gamma']
        mu = params['mu']
        nu = params['nu']
        zeta = params['zeta']
        NSIM = params['NSIM']
        max_demes = self.grid.dim_grid * self.grid.dim_grid
        initial_time= self.tau

        # Shape: (Max Possible Demes, Number of Timepoints(tau1 to tau2 to T), Number of CpG sites)
        #self.beta_history = np.full((max_demes, len(time_points1)+len(time_points2), NSIM), np.nan)
        self.simulation_history = []

        iter_stop_expansion = 0
        # Main Simulation Loop for the first clone (tau1 to tau2)
        for i in range(len(time_points1) - 1):
            self.tau = time_points1[i+1]
            self.calculate_deme_pop_glandfission_mno(dt1, params, S_cancer[i], S_cancer[i+1])
            
            assert self.params['exit_code'] == 0, \
                f"Simulation exited at time {self.tau:.2f}. Deme reach the boundary before Subclone growth."
            
            for deme in self.grid.demes.values():

                deme.update_methylation(dt1, params)
                # deme.beta_value = (deme.k + 2*deme.m) / (2*deme.N)
                deme.calculate_beta_value()
                deme.calculate_subclonal_fraction()
                # check if there is any NaN value in beta_value
                assert not np.any(np.isnan(deme.beta_value)), \
                    f"NaN found in beta_value for deme {deme.unique_id} at time {self.tau:.2f}"
                
                # rng = self.rng
                # NSIM = len(deme.m)
                # old_w = deme.w
                # # Use sequential rounds of binomial sampling to calculate how many cells transition between each state
                # m_to_k, k_out, w_to_k = rng.binomial(
                #                     n = (deme.m, deme.k, deme.w), 
                #                     p = np.tile([2*gamma*step_dt, 
                #                         (nu + zeta)*step_dt, 2*mu*step_dt], [NSIM, 1]).T)

                # k_to_m = rng.binomial(n=k_out, p = np.repeat(nu / (nu + zeta), NSIM))

                # deme.m = deme.m - m_to_k + k_to_m
                # deme.k = deme.k - k_out + m_to_k + w_to_k
                # deme.w = deme.N - deme.m - deme.k
                # k_to_w = k_out - k_to_m
                # # check if w==w+ k_to_w-w_to_k
                # new_w = old_w + k_to_w - w_to_k
                # if np.any(new_w!=deme.w):
                #     print("m:", deme.m)
                #     print("k:", deme.k)
                #     print("w:", deme.w)
                #     print("m_to_k:", m_to_k)
                #     print("k_out:", k_out)
                #     print("k_to_m:", k_to_m)
                #     print("k_to_w:", k_to_w)
                #     print("w_to_k:", w_to_k)
                # assert np.all(new_w==deme.w),\
                #     f"w is not conserved: {old_w} + {k_to_w} - {w_to_k} != {deme.w}"
                
                # assert np.all(m == deme.m) and np.all(k == deme.k) and np.all(w == deme.w), \
                #     f"Methylation states mismatch for deme {deme.unique_id} at time {self.tau:.2f}"

                if deme.unique_id not in self.seen_deme_ids:
                    self.ordered_deme_ids.append(deme.unique_id)
                    self.seen_deme_ids.add(deme.unique_id)
            
            # # Directly populate the pre-allocated history array for the current time step `i`
            # for deme in self.grid.demes.values():
            #     iter=0
            #     # Find the row index for this deme based on its appearance order
            #     if deme.unique_id in self.ordered_deme_ids:
            #         row_index = self.ordered_deme_ids.index(deme.unique_id)
                    
            #         # Assign the beta value array to the correct [row, column, :] slice
            #         self.beta_history[row_index, i, :] = deme.beta_value.reshape(1, -1)
            #         iter+=1
            # print(f"Time {self.tau:.2f}: {iter} demes updated with beta values.")

            # Create a temporary dictionary for efficient lookup of the current state
            current_betas_dict = {
                deme.unique_id: deme.beta_value 
                for deme in self.grid.demes.values()
            }
            print(f"Time {self.tau:.2f}: {len(current_betas_dict)} demes updated with beta values.")
            # Build an ordered list for the history.
            # The list is based on self.ordered_deme_ids to guarantee the correct order.
            # We use .get(deme_id, None) to handle cases where a deme might not exist at the current time step (though this is unlikely with the current logic).
            ordered_beta_snapshot = [
                current_betas_dict.get(deme_id) for deme_id in self.ordered_deme_ids
            ]
            l = len(ordered_beta_snapshot)
            print("shape of ordered_beta_snapshot:", np.array(ordered_beta_snapshot).shape)
            print("number of nan in ordered_beta_snapshot:", np.sum(np.isnan(ordered_beta_snapshot)))
            assert np.array(ordered_beta_snapshot).shape==(l, NSIM), \
                f"Expected ordered_beta_snapshot shape {(l, NSIM)}, got {np.array(ordered_beta_snapshot).shape}"
            #self.beta_history[:l, i, :] = np.array(ordered_beta_snapshot)
            #print("number of nan in beta history:", np.sum(np.isnan(self.beta_history[:, i, :])))

            current_demes_dict = {
                    deme.unique_id: deme for deme in self.grid.demes.values()
                }
            ordered_demes = [
                current_demes_dict.get(deme_id) for deme_id in self.ordered_deme_ids
            ]
            snapshot = {
                deme.unique_id: {
                    'coords': (deme.x, deme.y),
                    'beta': deme.beta_value,
                    'subclonal_fraction': deme.subclonal_fraction
                } for deme in ordered_demes
            }
            self.simulation_history.append(snapshot)
            # # Ensure the snapshot has the shape of (number_of_demes, NSIM)
            # # print("shape of ordered_beta_snapshot:", len(ordered_beta_snapshot))
            # # print("shape of ordered_beta_snapshot ahhha:", len(ordered_beta_snapshot[0]))
            # # Append the ordered list to the history
            # self.beta_history.append(ordered_beta_snapshot)

            # Periodically print the state of the simulation and draw the color plots.
            if i % 10 == 0:
                # Generate and save the plot for the current state
                output_filename = os.path.join(output_dir, f"grid_t_{self.tau:.2f}.png")
                self.plot_grid_subclonal(output_filename)

                # Generate and save the histogram plot
                hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}.png")
                self.plot_beta_histogram(hist_filename)  
        
                total_cells = sum(d.N for d in self.grid.demes.values())
                print(f"Time: {self.tau:.2f} | Demes: {len(self.grid.demes)} | Total Cells: {total_cells}")

            num_demes = len(self.grid.demes)
            #print(f"Current Deme Count: {num_demes}")
            terminate_cell=S_cancer[i+1]

            # if num_demes >= max_demes:
            #     print("\nGrid is full. No more space for fission. Halting simulation.")
            #     output_filename = os.path.join(output_dir, f"grid_t_{self.tau:.2f}.png")
            #     self.plot_grid(output_filename)
                
            #     hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}.png")
            #     self.plot_beta_histogram(hist_filename)  
            #     break

        
        # End of first phase (tau1 to tau2), print some summary statistics
        print("\n--- Phase 1 Finished ---")
        total_cells = sum(d.N for d in self.grid.demes.values())
        print(f"First Clone Time: {self.tau:.2f}")
        print(f"First Clone Deme Count: {len(self.grid.demes)}")
        print(f"First Clone Total Cell Count: {total_cells}")
        hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}_FirstClone.png")
        self.plot_beta_histogram(hist_filename)   

        # --- Phase 2: Second Clone (tau2 to T) ---
        print("\n--- Starting Phase 2: SubClone (tau2 to T) ---")
        
        demeslist = list(self.grid.demes.values())
        # select a random deme as the center deme for the subclonal expansion
        center_deme = self.rng.choice(demeslist)
        center_deme.N2 =1
        prob_matrix = np.stack((center_deme.m, center_deme.k, center_deme.w )) / center_deme.N
        center_deme.m2, center_deme.k2, center_deme.w2 = multinomial_rvs(1, prob_matrix, self.rng)

        for i in range(len(time_points2) - 1):
            self.tau = time_points2[i+1]

            if self.params['exit_code'] == 0:
                self.calculate_deme_pop_glandfission_mno_subclonal(dt1, params, S_cancer1[i], S_cancer1[i+1], S_cancer2[i], S_cancer2[i+1])
                
                for deme in self.grid.demes.values():
                    
                    deme.update_methylation(dt2, params, dt2=dt2)
                    # deme.beta_value = (deme.k + 2*deme.m) / (2*deme.N)
                    deme.calculate_beta_value()
                    deme.calculate_subclonal_fraction()
                    # check if there is any NaN value in beta_value
                    assert not np.any(np.isnan(deme.beta_value)), \
                        f"NaN found in beta_value for deme {deme.unique_id} at time {self.tau:.2f}"
                    
                    if deme.unique_id not in self.seen_deme_ids:
                        self.ordered_deme_ids.append(deme.unique_id)
                        self.seen_deme_ids.add(deme.unique_id)
                
            
                # Create a temporary dictionary for efficient lookup of the current state
                current_betas_dict = {
                    deme.unique_id: deme.beta_value 
                    for deme in self.grid.demes.values()
                }
                print(f"Time {self.tau:.2f}: {len(current_betas_dict)} demes updated with beta values.")
                # Build an ordered list for the history.
                # The list is based on self.ordered_deme_ids to guarantee the correct order.
                # We use .get(deme_id, None) to handle cases where a deme might not exist at the current time step (though this is unlikely with the current logic).
                ordered_beta_snapshot = [
                    current_betas_dict.get(deme_id) for deme_id in self.ordered_deme_ids
                ]
                l = len(ordered_beta_snapshot)
                print("shape of ordered_beta_snapshot:", np.array(ordered_beta_snapshot).shape)
                print("number of nan in ordered_beta_snapshot:", np.sum(np.isnan(ordered_beta_snapshot)))
                assert np.array(ordered_beta_snapshot).shape==(l, NSIM), \
                    f"Expected ordered_beta_snapshot shape {(l, NSIM)}, got {np.array(ordered_beta_snapshot).shape}"
                #self.beta_history[:l, i, :] = np.array(ordered_beta_snapshot)
                #print("number of nan in beta history:", np.sum(np.isnan(self.beta_history[:, i, :])))

                current_demes_dict = {
                    deme.unique_id: deme for deme in self.grid.demes.values()
                }
                ordered_demes = [
                    current_demes_dict.get(deme_id) for deme_id in self.ordered_deme_ids
                ]
                snapshot = {
                    deme.unique_id: {
                        'coords': (deme.x, deme.y),
                        'beta': deme.beta_value,
                        'subclonal_fraction': deme.subclonal_fraction
                    } for deme in ordered_demes
                }
                self.simulation_history.append(snapshot)

                # Periodically print the state of the simulation and draw the color plots.
                if i % 10 == 0:
                    # Generate and save the plot for the current state
                    output_filename = os.path.join(output_dir, f"grid_t_{self.tau:.2f}.png")
                    self.plot_grid_subclonal(output_filename)

                    # Generate and save the histogram plot
                    hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}.png")
                    self.plot_beta_histogram(hist_filename)  
            
                    total_cells = sum(d.N for d in self.grid.demes.values())
                    print(f"Time: {self.tau:.2f} | Demes: {len(self.grid.demes)} | Total Cells: {total_cells}")

          
            else:
                iter_stop_expansion += 1
                if iter_stop_expansion == 1:
                    print("\nDeme is reaching the edge of grid. Halting expansion. Only methylation transition.")
                    terminate_cell = S_cancer1[i] + S_cancer2[i]

                    total_cells = sum((d.N+d.N2) for d in self.grid.demes.values())
                    print(f"Time: {self.tau:.2f} | Demes: {len(self.grid.demes)} | Total Cells When Stop Expansion: {total_cells}")
                    output_filename = os.path.join(output_dir, f"grid_t_{self.tau:.2f}_stop_expansion.png")
                    self.plot_grid_subclonal(output_filename)

                    # Generate and save the histogram plot
                    hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}_stop_expansion.png")
                    self.plot_beta_histogram(hist_filename)

                for deme in self.grid.demes.values():
                    
                    deme.update_methylation(dt2, params, dt2=dt2)
                    # deme.beta_value = (deme.k + 2*deme.m) / (2*deme.N)
                    deme.calculate_beta_value()
                    # check if there is any NaN value in beta_value
                    assert not np.any(np.isnan(deme.beta_value)), \
                        f"NaN found in beta_value for deme {deme.unique_id} at time {self.tau:.2f}"

                current_demes_dict = {
                    deme.unique_id: deme for deme in self.grid.demes.values()
                }
                ordered_demes = [
                    current_demes_dict.get(deme_id) for deme_id in self.ordered_deme_ids
                ]
                snapshot = {
                    deme.unique_id: {
                        'coords': (deme.x, deme.y),
                        'beta': deme.beta_value,
                        'subclonal_fraction': deme.subclonal_fraction
                    } for deme in ordered_demes
                }
                self.simulation_history.append(snapshot)

                # Periodically draw the beta histogram, no need for grid as the expansion stopped.
                if i % 10 == 0:
                    # Generate and save the histogram plot
                    hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}.png")
                    self.plot_beta_histogram(hist_filename)  
            

            num_demes = len(self.grid.demes)
            #print(f"Current Deme Count: {num_demes}")

            # if num_demes >= max_demes:
            #     print("\nGrid is full. No more space for fission. Halting simulation.")
            #     output_filename = os.path.join(output_dir, f"grid_t_{self.tau:.2f}.png")
            #     self.plot_grid(output_filename)
                
            #     hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}.png")
            #     self.plot_beta_histogram(hist_filename)  
            #     break

           
        # 3. Final Simulation State
        print("\n--- Simulation Finished ---")
        total_cells = sum(d.N for d in self.grid.demes.values())
        print(f"Final Time: {self.tau:.2f}")
        print(f"Final Deme Count: {num_demes}")
        print(f"Final Total Cell Count: {total_cells}")
        print(f"Expected Total Cells: {terminate_cell}")
        print(f"Expected Total Cells by exponential: {np.exp(params['theta'] * (self.tau - initial_time))}")
        hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}.png")
        #print beta history to see the structure
        #print("Beta history shape:", self.beta_history.shape)
        # for i, beta_snapshot in enumerate(self.beta_history):
        #     print(f"Time {i}: {beta_snapshot.keys()}")
        #     print(f"Time {i}: {len(beta_snapshot)} demes")
        self.plot_beta_histogram(hist_filename) 
        #self.plot_beta_heatmap_pdf("beta_evolution_summary_subclonal.pdf")

        
        all_demes = list(self.grid.demes.values())
        if len(all_demes) < 1:
            print("Warning: No demes available to generate a histogram.")
            return

        list_of_beta_arrays = [d.beta_value for d in all_demes]
        
        # Stack the arrays into a single 2D NumPy array: shape (number_of_demes, NSIM).
        beta_matrix = np.stack(list_of_beta_arrays)
        
        # the representative value for each site (locus): average across all demes.
        representative_betas = np.mean(beta_matrix, axis=0)

        # self.save_state("Subclonal_gland_fission_simulation_final_state.pkl")

        return representative_betas

    
    def run_boundary_subclonal(self):
        """
        Runs the main time-driven simulation loop for the Boundary Growth model, and there is SUBCLONE.
        """
        print("\n--- Starting Evoflux-DEMON Simulation (Spatial Subclonal Model) ---")
        params = self.params
        # Unpack parameters for clarity
        migration_edge_only = params['migration_edge_only']
        erosion = params['erosion']
        theta1, theta2 = params['theta'], params['theta2']
        tau1, tau2, T = params['tau'], params['tau2'], params['T']
        gamma = params['gamma']
        mu = params['mu']
        nu = params['nu']
        zeta = params['zeta']
        NSIM = params['NSIM']

        # --- SETUP TIME STEPS FOR BOTH PHASES ---
        dt_max1 = 0.1 / np.max([2*gamma, 2*mu, 2*nu, 2*zeta, theta1])
        dt_max2 = 0.1 / np.max([2*gamma, 2*mu, 2*nu, 2*zeta, theta1, theta2])
        
        n1 = int((tau2 - tau1) / dt_max1) + 2
        n2 = int((T - tau2) / dt_max2) + 2
        time_points1 = np.linspace(tau1, tau2, n1)
        time_points2 = np.linspace(tau2, T, n2)
        dt1, dt2 = time_points1[1] - time_points1[0], time_points2[1] - time_points2[0]
        S_cancer = np.exp(theta1 * (time_points1-tau1)).astype(int)
        S_cancer1 = np.exp(theta1 * (time_points2-tau1)).astype(int)
        S_cancer2 = np.exp(theta2 * (time_points2-tau2)).astype(int)


        if np.any(S_cancer < 0):
            raise(OverflowError('overflow encountered for S_cancer'))
        
        # Create a directory for the output images if it doesn't exist
        output_dir = "Subclonal_simulation_plots"

        if migration_edge_only:
            if erosion:
                output_dir += "_boundary_erosion"
            else:
                output_dir += "_boundary"
        else:
            output_dir += "_gland_fission"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initial plot at time zero
        self.plot_grid(os.path.join(output_dir, f"grid_t_{self.tau:.2f}.png"))

        beta_hist_dir = "Subclonal_beta_histogram"
        if migration_edge_only:
            if erosion:
                beta_hist_dir += "_boundary_erosion"
            else:
                beta_hist_dir += "_boundary"
        else:
            beta_hist_dir += "_gland_fission"

        if not os.path.exists(beta_hist_dir):
            os.makedirs(beta_hist_dir)

        # Initial plot at time zero
        self.plot_grid(os.path.join(output_dir, f"grid_t_{self.tau:.2f}.png"))
        self.plot_beta_histogram(os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}.png"))

        max_demes = self.grid.dim_grid * self.grid.dim_grid
        initial_time= self.tau

        # Shape: (Max Possible Demes, Number of Timepoints(tau1 to tau2 to T), Number of CpG sites)
        #self.beta_history = np.full((max_demes, len(time_points1)+len(time_points2), NSIM), np.nan)
        self.simulation_history = []

        iter_stop_expansion = 0
        # Main Simulation Loop for the first clone (tau1 to tau2)
        for i in range(len(time_points1) - 1):
            self.tau = time_points1[i+1]
            self.calculate_deme_pop_boundary_growth(dt1, params, S_cancer[i], S_cancer[i+1])
            
            assert self.params['exit_code'] == 0, \
                f"Simulation exited at time {self.tau:.2f}. Deme reach the boundary before Subclone growth."
            
            for deme in self.grid.demes.values():

                deme.update_methylation(dt1, params)
                # deme.beta_value = (deme.k + 2*deme.m) / (2*deme.N)
                deme.calculate_beta_value()
                deme.calculate_subclonal_fraction()
                # check if there is any NaN value in beta_value
                assert not np.any(np.isnan(deme.beta_value)), \
                    f"NaN found in beta_value for deme {deme.unique_id} at time {self.tau:.2f}"

                if deme.unique_id not in self.seen_deme_ids:
                    self.ordered_deme_ids.append(deme.unique_id)
                    self.seen_deme_ids.add(deme.unique_id)
            
            # # Directly populate the pre-allocated history array for the current time step `i`
            # for deme in self.grid.demes.values():
            #     iter=0
            #     # Find the row index for this deme based on its appearance order
            #     if deme.unique_id in self.ordered_deme_ids:
            #         row_index = self.ordered_deme_ids.index(deme.unique_id)
                    
            #         # Assign the beta value array to the correct [row, column, :] slice
            #         self.beta_history[row_index, i, :] = deme.beta_value.reshape(1, -1)
            #         iter+=1
            # print(f"Time {self.tau:.2f}: {iter} demes updated with beta values.")

            # Create a temporary dictionary for efficient lookup of the current state
            current_betas_dict = {
                deme.unique_id: deme.beta_value 
                for deme in self.grid.demes.values()
            }
            print(f"Time {self.tau:.2f}: {len(current_betas_dict)} demes updated with beta values.")
            # Build an ordered list for the history.
            # The list is based on self.ordered_deme_ids to guarantee the correct order.
            # We use .get(deme_id, None) to handle cases where a deme might not exist at the current time step (though this is unlikely with the current logic).
            ordered_beta_snapshot = [
                current_betas_dict.get(deme_id) for deme_id in self.ordered_deme_ids
            ]
            l = len(ordered_beta_snapshot)
            print("shape of ordered_beta_snapshot:", np.array(ordered_beta_snapshot).shape)
            print("number of nan in ordered_beta_snapshot:", np.sum(np.isnan(ordered_beta_snapshot)))
            assert np.array(ordered_beta_snapshot).shape==(l, NSIM), \
                f"Expected ordered_beta_snapshot shape {(l, NSIM)}, got {np.array(ordered_beta_snapshot).shape}"
            #self.beta_history[:l, i, :] = np.array(ordered_beta_snapshot)
            #print("number of nan in beta history:", np.sum(np.isnan(self.beta_history[:, i, :])))

            current_demes_dict = {
                    deme.unique_id: deme for deme in self.grid.demes.values()
                }
            ordered_demes = [
                current_demes_dict.get(deme_id) for deme_id in self.ordered_deme_ids
            ]
            snapshot = {
                deme.unique_id: {
                    'coords': (deme.x, deme.y),
                    'beta': deme.beta_value,
                    'subclonal_fraction': deme.subclonal_fraction
                } for deme in ordered_demes
            }
            self.simulation_history.append(snapshot)
            # # Ensure the snapshot has the shape of (number_of_demes, NSIM)
            # # print("shape of ordered_beta_snapshot:", len(ordered_beta_snapshot))
            # # print("shape of ordered_beta_snapshot ahhha:", len(ordered_beta_snapshot[0]))
            # # Append the ordered list to the history
            # self.beta_history.append(ordered_beta_snapshot)

            # Periodically print the state of the simulation and draw the color plots.
            if i % 10 == 0:
                # Generate and save the plot for the current state
                output_filename = os.path.join(output_dir, f"grid_t_{self.tau:.2f}.png")
                self.plot_grid_subclonal(output_filename)

                # Generate and save the histogram plot
                hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}.png")
                self.plot_beta_histogram(hist_filename)  
        
                total_cells = sum(d.N for d in self.grid.demes.values())
                print(f"Time: {self.tau:.2f} | Demes: {len(self.grid.demes)} | Total Cells: {total_cells}")

            num_demes = len(self.grid.demes)
            #print(f"Current Deme Count: {num_demes}")
            terminate_cell=S_cancer[i+1]

            # if num_demes >= max_demes:
            #     print("\nGrid is full. No more space for fission. Halting simulation.")
            #     output_filename = os.path.join(output_dir, f"grid_t_{self.tau:.2f}.png")
            #     self.plot_grid(output_filename)
                
            #     hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}.png")
            #     self.plot_beta_histogram(hist_filename)  
            #     break

        
        # End of first phase (tau1 to tau2), print some summary statistics
        print("\n--- Phase 1 Finished ---")
        total_cells = sum(d.N for d in self.grid.demes.values())
        print(f"First Clone Time: {self.tau:.2f}")
        print(f"First Clone Deme Count: {len(self.grid.demes)}")
        print(f"First Clone Total Cell Count: {total_cells}")
        hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}_FirstClone.png")
        self.plot_beta_histogram(hist_filename)   

        # --- Phase 2: Second Clone (tau2 to T) ---
        print("\n--- Starting Phase 2: SubClone (tau2 to T) ---")
        
        demeslist = list(self.grid.demes.values())
        # select a random deme as the center deme for the subclonal expansion
        center_deme = self.rng.choice(demeslist)
        center_deme.N2 =1
        prob_matrix = np.stack((center_deme.m, center_deme.k, center_deme.w )) / center_deme.N
        center_deme.m2, center_deme.k2, center_deme.w2 = multinomial_rvs(1, prob_matrix, self.rng)

        for i in range(len(time_points2) - 1):
            self.tau = time_points2[i+1]

            if self.params['exit_code'] == 0:
                self.calculate_deme_pop_boundary_growth_subclonal(dt1, params, S_cancer1[i], S_cancer1[i+1], S_cancer2[i], S_cancer2[i+1])
                
                for deme in self.grid.demes.values():
                    
                    deme.update_methylation(dt2, params, dt2=dt2)
                    # deme.beta_value = (deme.k + 2*deme.m) / (2*deme.N)
                    deme.calculate_beta_value()
                    deme.calculate_subclonal_fraction()
                    # check if there is any NaN value in beta_value
                    assert not np.any(np.isnan(deme.beta_value)), \
                        f"NaN found in beta_value for deme {deme.unique_id} at time {self.tau:.2f}"
                    
                    if deme.unique_id not in self.seen_deme_ids:
                        self.ordered_deme_ids.append(deme.unique_id)
                        self.seen_deme_ids.add(deme.unique_id)
                
            
                # Create a temporary dictionary for efficient lookup of the current state
                current_betas_dict = {
                    deme.unique_id: deme.beta_value 
                    for deme in self.grid.demes.values()
                }
                print(f"Time {self.tau:.2f}: {len(current_betas_dict)} demes updated with beta values.")
                # Build an ordered list for the history.
                # The list is based on self.ordered_deme_ids to guarantee the correct order.
                # We use .get(deme_id, None) to handle cases where a deme might not exist at the current time step (though this is unlikely with the current logic).
                ordered_beta_snapshot = [
                    current_betas_dict.get(deme_id) for deme_id in self.ordered_deme_ids
                ]
                l = len(ordered_beta_snapshot)
                print("shape of ordered_beta_snapshot:", np.array(ordered_beta_snapshot).shape)
                print("number of nan in ordered_beta_snapshot:", np.sum(np.isnan(ordered_beta_snapshot)))
                assert np.array(ordered_beta_snapshot).shape==(l, NSIM), \
                    f"Expected ordered_beta_snapshot shape {(l, NSIM)}, got {np.array(ordered_beta_snapshot).shape}"
                #self.beta_history[:l, i, :] = np.array(ordered_beta_snapshot)
                #print("number of nan in beta history:", np.sum(np.isnan(self.beta_history[:, i, :])))

                current_demes_dict = {
                    deme.unique_id: deme for deme in self.grid.demes.values()
                }
                ordered_demes = [
                    current_demes_dict.get(deme_id) for deme_id in self.ordered_deme_ids
                ]
                snapshot = {
                    deme.unique_id: {
                        'coords': (deme.x, deme.y),
                        'beta': deme.beta_value,
                        'subclonal_fraction': deme.subclonal_fraction
                    } for deme in ordered_demes
                }
                self.simulation_history.append(snapshot)

                # Periodically print the state of the simulation and draw the color plots.
                if i % 10 == 0:
                    # Generate and save the plot for the current state
                    output_filename = os.path.join(output_dir, f"grid_t_{self.tau:.2f}.png")
                    self.plot_grid_subclonal(output_filename)

                    # Generate and save the histogram plot
                    hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}.png")
                    self.plot_beta_histogram(hist_filename)  
            
                    total_cells = sum(d.N for d in self.grid.demes.values())
                    print(f"Time: {self.tau:.2f} | Demes: {len(self.grid.demes)} | Total Cells: {total_cells}")

          
            else:
                iter_stop_expansion += 1
                if iter_stop_expansion == 1:
                    print("\nDeme is reaching the edge of grid. Halting expansion. Only methylation transition.")
                    terminate_cell = S_cancer1[i] + S_cancer2[i]

                    total_cells = sum((d.N+d.N2) for d in self.grid.demes.values())
                    print(f"Time: {self.tau:.2f} | Demes: {len(self.grid.demes)} | Total Cells When Stop Expansion: {total_cells}")
                    output_filename = os.path.join(output_dir, f"grid_t_{self.tau:.2f}_stop_expansion.png")
                    self.plot_grid_subclonal(output_filename)

                    # Generate and save the histogram plot
                    hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}_stop_expansion.png")
                    self.plot_beta_histogram(hist_filename)

                for deme in self.grid.demes.values():
                    
                    deme.update_methylation(dt2, params, dt2=dt2)
                    # deme.beta_value = (deme.k + 2*deme.m) / (2*deme.N)
                    deme.calculate_beta_value()
                    # check if there is any NaN value in beta_value
                    assert not np.any(np.isnan(deme.beta_value)), \
                        f"NaN found in beta_value for deme {deme.unique_id} at time {self.tau:.2f}"

                current_demes_dict = {
                    deme.unique_id: deme for deme in self.grid.demes.values()
                }
                ordered_demes = [
                    current_demes_dict.get(deme_id) for deme_id in self.ordered_deme_ids
                ]
                snapshot = {
                    deme.unique_id: {
                        'coords': (deme.x, deme.y),
                        'beta': deme.beta_value,
                        'subclonal_fraction': deme.subclonal_fraction
                    } for deme in ordered_demes
                }
                self.simulation_history.append(snapshot)

                # Periodically draw the beta histogram, no need for grid as the expansion stopped.
                if i % 10 == 0:
                    # Generate and save the histogram plot
                    hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}.png")
                    self.plot_beta_histogram(hist_filename)  
            

            num_demes = len(self.grid.demes)
            #print(f"Current Deme Count: {num_demes}")

            # if num_demes >= max_demes:
            #     print("\nGrid is full. No more space for fission. Halting simulation.")
            #     output_filename = os.path.join(output_dir, f"grid_t_{self.tau:.2f}.png")
            #     self.plot_grid(output_filename)
                
            #     hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}.png")
            #     self.plot_beta_histogram(hist_filename)  
            #     break

           
        # 3. Final Simulation State
        print("\n--- Simulation Finished ---")
        total_cells = sum(d.N for d in self.grid.demes.values())
        print(f"Final Time: {self.tau:.2f}")
        print(f"Final Deme Count: {num_demes}")
        print(f"Final Total Cell Count: {total_cells}")
        print(f"Expected Total Cells: {terminate_cell}")
        print(f"Expected Total Cells by exponential: {np.exp(params['theta'] * (self.tau - initial_time))}")
        hist_filename = os.path.join(beta_hist_dir, f"hist_t_{self.tau:.2f}.png")
        #print beta history to see the structure
        #print("Beta history shape:", self.beta_history.shape)
        # for i, beta_snapshot in enumerate(self.beta_history):
        #     print(f"Time {i}: {beta_snapshot.keys()}")
        #     print(f"Time {i}: {len(beta_snapshot)} demes")
        self.plot_beta_histogram(hist_filename) 
        #self.plot_beta_heatmap_pdf("beta_evolution_summary_subclonal.pdf")

        
        all_demes = list(self.grid.demes.values())
        if len(all_demes) < 1:
            print("Warning: No demes available to generate a histogram.")
            return

        list_of_beta_arrays = [d.beta_value for d in all_demes]
        
        # Stack the arrays into a single 2D NumPy array: shape (number_of_demes, NSIM).
        beta_matrix = np.stack(list_of_beta_arrays)
        
        # the representative value for each site (locus): average across all demes.
        representative_betas = np.mean(beta_matrix, axis=0)

        # self.save_state("Subclonal_boundary_simulation_final_state.pkl")

        return representative_betas
    
    def plot_kymograph(self, output_filename, xslice=True):
        """
        Generates a kymograph showing clonal expansion across a central
        slice of the grid over time.
        """
        num_timepoints = len(self.simulation_history)
        center_y = self.grid.dim_grid // 2
        
        # Rows: Timepoints, Columns: X-coordinate of the grid
        kymograph_data = np.full((num_timepoints, self.grid.dim_grid), -1.0)

        if xslice:
            for t, snapshot in enumerate(self.simulation_history):
                for deme_id, data in snapshot.items():
                    x, y = data['coords']
                    if y == center_y:
                        # Use the subclonal fraction as the value to plot
                        kymograph_data[t, x] = data['subclonal_fraction']
        else:
            for t, snapshot in enumerate(self.simulation_history):
                for deme_id, data in snapshot.items():
                    x, y = data['coords']
                    if x == center_y:
                    # Use the subclonal fraction as the value to plot
                        kymograph_data[t, y] = data['subclonal_fraction']

        plt.figure(figsize=(12, 8))
        # Use a perceptually uniform colormap
        cmap = sns.color_palette("viridis", as_cmap=True)
        cmap.set_bad(color='black') # For unpopulated areas
        
        # Replace -1 with NaN for plotting
        kymograph_data[kymograph_data == -1] = np.nan
        
        plt.imshow(kymograph_data, aspect='auto', cmap=cmap, interpolation='none')
        plt.title('Spatio-Temporal Kymograph of Clonal Expansion')
        plt.xlabel('Spatial Position (X-axis at grid center)')
        plt.ylabel('Time (Simulation Step)')
        
        cbar = plt.colorbar()
        cbar.set_label('Subclonal Fraction')
        
        # Create a directory for the output images if it doesn't exist
        output_dir = "simulation_plots"
        if self.params['migration_edge_only']:
            if self.params['erosion']:
                output_dir += "_boundary_erosion"
            else:
                output_dir += "_boundary"
        else:
            output_dir += "_gland_fission"
        if self.params['subclone']:
            output_dir = "Subclonal_" + output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.savefig(os.path.join(output_dir,output_filename), dpi=150)
        plt.close()


    def plot_kymograph_2d(self, output_filename):
        """
        Generates a kymograph showing clonal expansion across a central
        slice of the grid over time.
        """
        num_timepoints = len(self.simulation_history)
        dim_grid = self.grid.dim_grid
        center_y = dim_grid // 2
        center_x = dim_grid // 2

        # Rows: Timepoints, Columns: X-coordinate of the grid
        kymograph_data = np.full((num_timepoints, dim_grid*dim_grid), -1.0)

        for t, snapshot in enumerate(self.simulation_history):
            for deme_id, data in snapshot.items():
                x, y = data['coords']
                xy_index = y * dim_grid + x  # Convert 2D coordinates to 1D index
                # Use the subclonal fraction as the value to plot
                kymograph_data[t, xy_index] = data['subclonal_fraction']

        plt.figure(figsize=(20, 12))
        # Use a perceptually uniform colormap
        cmap = sns.color_palette("viridis", as_cmap=True)
        cmap.set_bad(color='black') # For unpopulated areas
        
        # Replace -1 with NaN for plotting
        kymograph_data[kymograph_data == -1] = np.nan
        
        plt.imshow(kymograph_data, aspect='auto', cmap=cmap, interpolation='none')
        plt.title('Spatio-Temporal Kymograph of Clonal Expansion')
        plt.xlabel('Spatial Position (XY-axis flattened)')
        plt.ylabel('Time (Simulation Step)')
        
        cbar = plt.colorbar()
        cbar.set_label('Subclonal Fraction')
        
        # Create a directory for the output images if it doesn't exist
        output_dir = "simulation_plots"
        if self.params['migration_edge_only']:
            if self.params['erosion']:
                output_dir += "_boundary_erosion"
            else:
                output_dir += "_boundary"
        else:
            output_dir += "_gland_fission"
        if self.params['subclone']:
            output_dir = "Subclonal_" + output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.savefig(os.path.join(output_dir,output_filename), dpi=150)
        plt.close()

    def save_state(self, output_filename):
        """
        Saves the entire simulation object to a file using pickle.

        Args:
            output_filename (str): The path to the file to save the state to.
        """
        temp_filename = output_filename + ".tmp"
        print(f"Saving simulation state to {output_filename}...")
        try:
            with open(temp_filename, 'wb') as f:
                pickle.dump(self.simulation_history, f)
            
            # If the pickle.dump was successful, rename the temp file
            os.replace(temp_filename, output_filename)
            print("Save successful.")

        except Exception as e:
            print(f"Error saving simulation state: {e}")
            # Clean up the temporary file if it exists
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    def run_gland_onlybeta(self):
        """
        Runs the main time-driven simulation loop for the Gland Fission model.
        Only want the return beta value at the end of simulation.
        """
        print("\n--- Starting Evoflux-DEMON Simulation (Gland Fission Model) ---")
        params = self.params
        migration_edge_only = params['migration_edge_only']
        erosion = params['erosion']
        gamma= params['gamma']
        mu = params['mu']
        nu = params['nu']
        zeta = params['zeta']
        theta = params['theta']
        T = params['T']

        # 1. Setup time steps for the simulation
        # The time step `dt` is chosen to be small enough that transition probabilities
        # in the methylation model do not exceed 10%, ensuring stability.
        max_rate = max(2*gamma, 2*mu, 2*nu, 2*zeta, theta)
        dt_max = 0.1 / max_rate
        n = int((T- self.tau) / dt_max) + 2  # Number of time steps.
        time_points = np.linspace(self.tau, T, n) 
        step_dt = time_points[1] - time_points[0]
        
        S_cancer = np.exp(theta* (time_points-params['tau'])).astype(int)

        if np.any(S_cancer < 0):
            raise(OverflowError('overflow encountered for S_cancer'))
        
        NSIM = params['NSIM']
        max_demes = self.grid.dim_grid * self.grid.dim_grid
        initial_time= self.tau
        
        iter_stop_expansion = 0
        # 2. Main Simulation Loop
        for i in range(len(time_points) - 1):
            self.tau = time_points[i+1]
            terminate_cell = S_cancer[i]
            if self.params['exit_code']==0:
            # the process including expansion of demes (number of cells growing and fission),
            # and methylation states transition and beta_value updating
                self.calculate_deme_pop_glandfission_mno(step_dt, params, S_cancer[i], S_cancer[i+1])
                
                for deme in self.grid.demes.values():
                    
                    deme.update_methylation(step_dt, params)
                    # deme.beta_value = (deme.k + 2*deme.m) / (2*deme.N)
                    deme.calculate_beta_value()
                    deme.calculate_subclonal_fraction()
                    # check if there is any NaN value in beta_value
                    assert not np.any(np.isnan(deme.beta_value)), \
                        f"NaN found in beta_value for deme {deme.unique_id} at time {self.tau:.2f}"


            else:
                # when exit_code==1 (deme reach the grid boundary), only methylation states transion and update beta_value
                # as the time goes by until reach the terminal time T
                iter_stop_expansion += 1
                if iter_stop_expansion == 1:
                    print("\nDeme is reaching the edge of grid. Halting expansion. Only methylation transition.")
                    terminate_cell = S_cancer[i]

                # Update methylation states and beta values for all demes
                for deme in self.grid.demes.values():
                    deme.update_methylation(step_dt, params)
                    deme.calculate_beta_value()
                
        print("\n--- Simulation Finished ---")
        
        total_cells = sum(d.N for d in self.grid.demes.values())
        print(f"Final Time: {self.tau:.2f}")
        print(f"Final Deme Count: {len(self.grid.demes)}")
        print(f"Final Total Cell Count: {total_cells}")
        print(f"Expected Total Cells: {terminate_cell}")
        print(f"Expected Total Cells by exponential: {np.exp(params['theta'] * (self.tau - initial_time))}")

        all_demes = list(self.grid.demes.values())
        if len(all_demes) < 1:
            print("Warning: No demes available to generate a histogram.")
            return

        list_of_beta_arrays = [d.beta_value for d in all_demes]
        # Stack the arrays into a single 2D NumPy array: shape (number_of_demes, NSIM).
        beta_matrix = np.stack(list_of_beta_arrays)  
        # the representative value for each site (locus): average across all demes.
        representative_betas = np.mean(beta_matrix, axis=0)    
        return representative_betas

    def run_boundary_growth_model_onlybeta(self):
        """
        A dedicated simulation loop for the Boundary Growth model.
        Growth occurs only when full demes at the boundary divide into empty space.
        """
        print("\n--- Running Simulation: Boundary Growth (Erosion) Model ---")
        params = self.params
        G = params['G']
        T = params['T']
        gamma = params['gamma']
        mu = params['mu']
        nu = params['nu']
        zeta = params['zeta']
        theta = params['theta']
        erosion = params['erosion']

        # Setup time steps
        max_rate = max(2*gamma, 2*mu, 2*nu, 2*zeta, theta)
        dt_max = 0.1 / max_rate
        # dt_max/=5  #not work
        n = int((T- self.tau) / dt_max) + 2  # Number of time steps.
        time_points = np.linspace(self.tau, T, n) 
        step_dt = time_points[1] - time_points[0]
        S_cancer = np.exp(theta * (time_points - params['tau'])).astype(int)
        if np.any(S_cancer < 0):
            raise OverflowError('Overflow encountered for S_cancer')
        

        step_dt = time_points[1] - time_points[0]

        iter_stop_expansion = 0
        initial_time= self.tau
        # Main Simulation Loop
        for i in range(len(time_points) - 1):
            self.tau += step_dt
            terminate_cell = S_cancer[i+1]
            if self.params['exit_code'] == 0:
                self.calculate_deme_pop_boundary_growth(step_dt, params, S_cancer[i], S_cancer[i+1])
                # Update the beta values and methylation states for each deme
                for deme in self.grid.demes.values():
                    deme.update_methylation(step_dt, params)
                    deme.beta_value = (deme.k + 2*deme.m) / (2*deme.N)

            else:
                iter_stop_expansion += 1
                if iter_stop_expansion == 1:
                    print("\nDeme is reaching the edge of grid. Halting expansion. Only methylation transition.")
                    terminate_cell = S_cancer[i]
                    
                for deme in self.grid.demes.values():
                    deme.update_methylation(step_dt, params)
                    deme.beta_value = (deme.k + 2*deme.m) / (2*deme.N)
                

                
        print("\n--- Simulation Finished ---")
        # Final print statements
        total_cells = sum(d.N for d in self.grid.demes.values())
        print(f"Final Time: {self.tau:.2f}")
        print(f"Final Deme Count: {len(self.grid.demes)}")
        print(f"Final Total Cell Count: {total_cells}")
        print(f"Expected Total Cells: {terminate_cell}")
        print(f"Expected Total Cells by exponential: {np.exp(params['theta'] * (self.tau - initial_time))}")

        all_demes = list(self.grid.demes.values())
        if len(all_demes) < 1:
            print("Warning: No demes available to generate a histogram.")
            return

        list_of_beta_arrays = [d.beta_value for d in all_demes]
        # Stack the arrays into a single 2D NumPy array: shape (number_of_demes, NSIM).
        beta_matrix = np.stack(list_of_beta_arrays)
        # the representative value for each site (locus): average across all demes.
        representative_betas = np.mean(beta_matrix, axis=0)

        return representative_betas

    def run_gland_subclonal_onlybeta(self):
        """
        Runs the main time-driven simulation loop for the Gland Fission model, and there is SUBCLONE.
        """
        print("\n--- Starting Evoflux-DEMON Simulation (Spatial Subclonal Model) ---")
        params = self.params
        # Unpack parameters for clarity
        migration_edge_only = params['migration_edge_only']
        erosion = params['erosion']
        theta1, theta2 = params['theta'], params['theta2']
        tau1, tau2, T = params['tau'], params['tau2'], params['T']
        gamma = params['gamma']
        mu = params['mu']
        nu = params['nu']
        zeta = params['zeta']

        # --- SETUP TIME STEPS FOR BOTH PHASES ---
        dt_max1 = 0.1 / np.max([2*gamma, 2*mu, 2*nu, 2*zeta, theta1])
        dt_max2 = 0.1 / np.max([2*gamma, 2*mu, 2*nu, 2*zeta, theta1, theta2])
        
        n1 = int((tau2 - tau1) / dt_max1) + 2
        n2 = int((T - tau2) / dt_max2) + 2
        time_points1 = np.linspace(tau1, tau2, n1)
        time_points2 = np.linspace(tau2, T, n2)
        dt1, dt2 = time_points1[1] - time_points1[0], time_points2[1] - time_points2[0]
        S_cancer = np.exp(theta1 * (time_points1-tau1)).astype(int)
        S_cancer1 = np.exp(theta1 * (time_points2-tau1)).astype(int)
        S_cancer2 = np.exp(theta2 * (time_points2-tau2)).astype(int)


        if np.any(S_cancer < 0):
            raise(OverflowError('overflow encountered for S_cancer'))
        
        NSIM = params['NSIM']
        max_demes = self.grid.dim_grid * self.grid.dim_grid
        initial_time= self.tau

        iter_stop_expansion = 0
        # Main Simulation Loop for the first clone (tau1 to tau2)
        for i in range(len(time_points1) - 1):
            self.tau = time_points1[i+1]
            self.calculate_deme_pop_glandfission_mno(dt1, params, S_cancer[i], S_cancer[i+1])
            
            assert self.params['exit_code'] == 0, \
                f"Simulation exited at time {self.tau:.2f}. Deme reach the boundary before Subclone growth."
            
            for deme in self.grid.demes.values():

                deme.update_methylation(dt1, params)
                # deme.beta_value = (deme.k + 2*deme.m) / (2*deme.N)
                deme.calculate_beta_value()
                deme.calculate_subclonal_fraction()
                # check if there is any NaN value in beta_value
                assert not np.any(np.isnan(deme.beta_value)), \
                    f"NaN found in beta_value for deme {deme.unique_id} at time {self.tau:.2f}"
                
            num_demes = len(self.grid.demes)
            #print(f"Current Deme Count: {num_demes}")
            terminate_cell=S_cancer[i+1]

        
        # End of first phase (tau1 to tau2), print some summary statistics
        print("\n--- Phase 1 Finished ---")
        total_cells = sum(d.N for d in self.grid.demes.values())
        print(f"First Clone Time: {self.tau:.2f}")
        print(f"First Clone Deme Count: {len(self.grid.demes)}")
        print(f"First Clone Total Cell Count: {total_cells}")  

        # --- Phase 2: Second Clone (tau2 to T) ---
        print("\n--- Starting Phase 2: SubClone (tau2 to T) ---")
        
        demeslist = list(self.grid.demes.values())
        # select a random deme as the center deme for the subclonal expansion
        center_deme = self.rng.choice(demeslist)
        center_deme.N2 =1
        prob_matrix = np.stack((center_deme.m, center_deme.k, center_deme.w )) / center_deme.N
        center_deme.m2, center_deme.k2, center_deme.w2 = multinomial_rvs(1, prob_matrix, self.rng)

        for i in range(len(time_points2) - 1):
            self.tau = time_points2[i+1]

            if self.params['exit_code'] == 0:
                self.calculate_deme_pop_glandfission_mno_subclonal(dt1, params, S_cancer1[i], S_cancer1[i+1], S_cancer2[i], S_cancer2[i+1])
                terminate_cell = S_cancer1[i+1] + S_cancer2[i+1]
                for deme in self.grid.demes.values():
                    
                    deme.update_methylation(dt2, params, dt2=dt2)
                    # deme.beta_value = (deme.k + 2*deme.m) / (2*deme.N)
                    deme.calculate_beta_value()
                    deme.calculate_subclonal_fraction()
                    # check if there is any NaN value in beta_value
                    assert not np.any(np.isnan(deme.beta_value)), \
                        f"NaN found in beta_value for deme {deme.unique_id} at time {self.tau:.2f}"
          
            else:
                iter_stop_expansion += 1
                if iter_stop_expansion == 1:
                    print("\nDeme is reaching the edge of grid. Halting expansion. Only methylation transition.")
                    terminate_cell = S_cancer1[i] + S_cancer2[i]

                    total_cells = sum((d.N+d.N2) for d in self.grid.demes.values())
                    print(f"Time: {self.tau:.2f} | Demes: {len(self.grid.demes)} | Total Cells When Stop Expansion: {total_cells}")

                for deme in self.grid.demes.values():
                    
                    deme.update_methylation(dt2, params, dt2=dt2)
                    # deme.beta_value = (deme.k + 2*deme.m) / (2*deme.N)
                    deme.calculate_beta_value()
                    # check if there is any NaN value in beta_value
                    assert not np.any(np.isnan(deme.beta_value)), \
                        f"NaN found in beta_value for deme {deme.unique_id} at time {self.tau:.2f}"
            num_demes = len(self.grid.demes)
        
        # 3. Final Simulation State
        print("\n--- Simulation Finished ---")
        total_cells = sum(d.N+d.N2 for d in self.grid.demes.values())
        print(f"Final Time: {self.tau:.2f}")
        print(f"Final Deme Count: {num_demes}")
        print(f"Final Total Cell Count: {total_cells}")
        print(f"Expected Total Cells: {terminate_cell}")
        print(f"Expected Total Cells by exponential: {np.exp(params['theta'] * (self.tau - initial_time))+np.exp(params['theta2'] * (self.tau - tau2))}")
        
        
        all_demes = list(self.grid.demes.values())
        if len(all_demes) < 1:
            print("Warning: No demes available to generate a histogram.")
            return

        list_of_beta_arrays = [d.beta_value for d in all_demes]
        
        # Stack the arrays into a single 2D NumPy array: shape (number_of_demes, NSIM).
        beta_matrix = np.stack(list_of_beta_arrays)
        
        # the representative value for each site (locus): average across all demes.
        representative_betas = np.mean(beta_matrix, axis=0)
        return representative_betas

    
    def run_boundary_subclonal_onlybeta(self):
        """
        Runs the main time-driven simulation loop for the Boundary Growth model, and there is SUBCLONE.
        """
        print("\n--- Starting Evoflux-DEMON Simulation (Spatial Subclonal Model) ---")
        params = self.params
        # Unpack parameters for clarity
        migration_edge_only = params['migration_edge_only']
        erosion = params['erosion']
        theta1, theta2 = params['theta'], params['theta2']
        tau1, tau2, T = params['tau'], params['tau2'], params['T']
        gamma = params['gamma']
        mu = params['mu']
        nu = params['nu']
        zeta = params['zeta']
        NSIM = params['NSIM']

        # --- SETUP TIME STEPS FOR BOTH PHASES ---
        dt_max1 = 0.1 / np.max([2*gamma, 2*mu, 2*nu, 2*zeta, theta1])
        dt_max2 = 0.1 / np.max([2*gamma, 2*mu, 2*nu, 2*zeta, theta1, theta2])
        
        n1 = int((tau2 - tau1) / dt_max1) + 2
        n2 = int((T - tau2) / dt_max2) + 2
        time_points1 = np.linspace(tau1, tau2, n1)
        time_points2 = np.linspace(tau2, T, n2)
        dt1, dt2 = time_points1[1] - time_points1[0], time_points2[1] - time_points2[0]
        S_cancer = np.exp(theta1 * (time_points1-tau1)).astype(int)
        S_cancer1 = np.exp(theta1 * (time_points2-tau1)).astype(int)
        S_cancer2 = np.exp(theta2 * (time_points2-tau2)).astype(int)


        if np.any(S_cancer < 0):
            raise(OverflowError('overflow encountered for S_cancer'))
        

        max_demes = self.grid.dim_grid * self.grid.dim_grid
        initial_time= self.tau

        iter_stop_expansion = 0
        # Main Simulation Loop for the first clone (tau1 to tau2)
        for i in range(len(time_points1) - 1):
            self.tau = time_points1[i+1]
            self.calculate_deme_pop_boundary_growth(dt1, params, S_cancer[i], S_cancer[i+1])
            
            assert self.params['exit_code'] == 0, \
                f"Simulation exited at time {self.tau:.2f}. Deme reach the boundary before Subclone growth."
            
            for deme in self.grid.demes.values():

                deme.update_methylation(dt1, params)
                # deme.beta_value = (deme.k + 2*deme.m) / (2*deme.N)
                deme.calculate_beta_value()
                deme.calculate_subclonal_fraction()
                # check if there is any NaN value in beta_value
                assert not np.any(np.isnan(deme.beta_value)), \
                    f"NaN found in beta_value for deme {deme.unique_id} at time {self.tau:.2f}"

            num_demes = len(self.grid.demes)
            #print(f"Current Deme Count: {num_demes}")
            terminate_cell=S_cancer[i+1]

        # End of first phase (tau1 to tau2), print some summary statistics
        print("\n--- Phase 1 Finished ---")
        total_cells = sum(d.N for d in self.grid.demes.values())
        print(f"First Clone Time: {self.tau:.2f}")
        print(f"First Clone Deme Count: {len(self.grid.demes)}")
        print(f"First Clone Total Cell Count: {total_cells}")
        
        # --- Phase 2: Second Clone (tau2 to T) ---
        print("\n--- Starting Phase 2: SubClone (tau2 to T) ---")
        
        demeslist = list(self.grid.demes.values())
        # select a random deme as the center deme for the subclonal expansion
        center_deme = self.rng.choice(demeslist)
        center_deme.N2 =1
        prob_matrix = np.stack((center_deme.m, center_deme.k, center_deme.w )) / center_deme.N
        center_deme.m2, center_deme.k2, center_deme.w2 = multinomial_rvs(1, prob_matrix, self.rng)

        for i in range(len(time_points2) - 1):
            self.tau = time_points2[i+1]

            if self.params['exit_code'] == 0:
                self.calculate_deme_pop_boundary_growth_subclonal(dt1, params, S_cancer1[i], S_cancer1[i+1], S_cancer2[i], S_cancer2[i+1])
                terminate_cell = S_cancer1[i+1] + S_cancer2[i+1]
                for deme in self.grid.demes.values():
                    
                    deme.update_methylation(dt2, params, dt2=dt2)
                    # deme.beta_value = (deme.k + 2*deme.m) / (2*deme.N)
                    deme.calculate_beta_value()
                    deme.calculate_subclonal_fraction()
                    # check if there is any NaN value in beta_value
                    assert not np.any(np.isnan(deme.beta_value)), \
                        f"NaN found in beta_value for deme {deme.unique_id} at time {self.tau:.2f}"
                    
            else:
                iter_stop_expansion += 1
                if iter_stop_expansion == 1:
                    print("\nDeme is reaching the edge of grid. Halting expansion. Only methylation transition.")
                    terminate_cell = S_cancer1[i] + S_cancer2[i]

                    total_cells = sum((d.N+d.N2) for d in self.grid.demes.values())
                    print(f"Time: {self.tau:.2f} | Demes: {len(self.grid.demes)} | Total Cells When Stop Expansion: {total_cells}")
                    
                for deme in self.grid.demes.values():
                    
                    deme.update_methylation(dt2, params, dt2=dt2)
                    # deme.beta_value = (deme.k + 2*deme.m) / (2*deme.N)
                    deme.calculate_beta_value()
                    # check if there is any NaN value in beta_value
                    assert not np.any(np.isnan(deme.beta_value)), \
                        f"NaN found in beta_value for deme {deme.unique_id} at time {self.tau:.2f}"

        # 3. Final Simulation State
        print("\n--- Simulation Finished ---")
        total_cells = sum(d.N+d.N2 for d in self.grid.demes.values())
        print(f"Final Time: {self.tau:.2f}")
        print(f"Final Deme Count: {num_demes}")
        print(f"Final Total Cell Count: {total_cells}")
        print(f"Expected Total Cells: {terminate_cell}")
        print(f"Expected Total Cells by exponential: {np.exp(params['theta'] * (self.tau - initial_time))+np.exp(params['theta2'] * (self.tau - tau2))}")
        
        all_demes = list(self.grid.demes.values())
        if len(all_demes) < 1:
            print("Warning: No demes available to generate a histogram.")
            return

        list_of_beta_arrays = [d.beta_value for d in all_demes]
        
        # Stack the arrays into a single 2D NumPy array: shape (number_of_demes, NSIM).
        beta_matrix = np.stack(list_of_beta_arrays)
        
        # the representative value for each site (locus): average across all demes.
        representative_betas = np.mean(beta_matrix, axis=0)
        return representative_betas

# --- Main Execution Block ---

if __name__ == "__main__":
    # Define parameters for a simple simulation run
    base_params = {
        'dim_grid': 150,
        'G': 128,  # Carrying capacity per deme
        'init_pop': 1,
        'NSIM': 500,
        'tau': 40.0,  # Start time
        'tau2': 44.0,  # Transition time to second clone
        'T': 50,  # End time
        'mu': 0.01,    # Transition rate from homozygous demethylated to heterozygous
        'gamma': 0.02, # Transition rate from homozygous methylated to heterozygous
        'nu': 0.015,   # Transition rate from heterozygous to homozygous methylated
        'zeta': 0.01,  # Transition rate from heterozygous to homozygous demethylated
        'theta': 1.64,  # Global growth rate for deme fission
        'theta2': 4,  # Global growth rate for subclonal expansion
        'subclone': False, # Enable subclonal expansion in the simulation
        'migration_edge_only': True,
        'migration_diagonal': True,
        'diagonal': True,  # Allow diagonal movement in the grid
        'erosion':True,  # Erosion mode for boundary growth model

        'normal_birth_rate': 0.1,
        'baseline_death_rate': 0.1,
        'density_dept_death_rate': 10.0,
        'init_migration_rate': 0.01,
        
        # Mutation rates and effects
        'mu_passenger': 1e-2,
        'mu_driver_birth': 1e-5,
        's_passenger': 0.01, # 1% cost
        's_driver_birth': 0.1, # 10% average benefit
        
        # Simulation controls
        'max_pop': 5000000,
        'max_generations': 2000,
        
        # This parameter is crucial for DEMON's different modes
        'migration_type': 2, # 0=cell migration, 2=deme fission etc.

        'exit_code': 0
    }
    
    # num_replicates = 50  # Number of independent simulation runs
    # # Run the simulation multiple times and collect results: beta value at the last time point
    # all_final_betas = []
    # print(f"--- Running {num_replicates} Replicates for Boundary Growth Model ---")
    # params_Boundary = base_params.copy()
    # params_Boundary['migration_edge_only'] = True
    # params_Boundary['subclone'] = False  #subclonal expansion
    # params_Boundary['G'] = 128 # Set a very high carrying capacity to simulate non-spatial conditions
    # for i in range(num_replicates):
        # Create a new Simulation object for each independent run
    #     params_Boundary['exit_code'] = 0  # Reset exit code for each run
    #     params_Boundary['tau'] = 40.0  # Reset start time for each run
    #     params_Boundary['tau2'] = 44.0
    #     sim = Simulation(params_Boundary)
        
    #     # Run the simulation and get only the final beta values
    #     final_betas = sim.run_boundary_growth_model_onlybeta()
        
    #     # Add the result to our list
    #     all_final_betas.append(final_betas)
    #     print(f"Completed run {i+1}/{num_replicates}")

    # # Save the entire list of results to a single file
    # output_filename = "Boundary_Growth_new.pkl"
    # print(f"\nSaving all {num_replicates} results to {output_filename}...")
    # try:
    #     with open(output_filename, 'wb') as f:
    #         pickle.dump(all_final_betas, f)
    #     print("Save successful.")
    # except Exception as e:
    #     print(f"Error saving results: {e}")

    biopsy_locations = [
        (45, 45),
        (45, 105),
        (75, 75),
        (105, 45),
        (105, 105)  # Center of the grid
    ]
    biopsy_size = 15 # A deme square
    
    # Create and run the simulation
    # simulation = Simulation(base_params)
    # if simulation.params['migration_edge_only']:
    #     simulation.run_boundary_growth_model()
        
    # else:
    #     simulation.run_gland_subclonal()

   
    # Parameters for the Gland Fission model
    # params_gland = base_params.copy()
    # params_gland['migration_edge_only'] = False
    # params_gland['subclone'] = False  # Disable subclonal expansion

    # print("---  RUNNING GLAND FISSION MODEL ---")
    # sim_gland = Simulation(params_gland)
    # betas_gland = sim_gland.run_gland()

    # # Perform the biopsies and collect the results
    # biopsy_data = []
    # for x, y in biopsy_locations:
    #     result = sim_gland.perform_virtual_biopsy(center_x=x, center_y=y, size=biopsy_size)
    #     if result:
    #         biopsy_data.append(result)

    # # Generate the correlation plot from the biopsy data
    # if biopsy_data:
    #     sim_gland.plot_biopsy_correlation(biopsy_data, "biopsy_correlation_gland_fission.png")
    # sim_gland.plot_kymograph("kymograph_gland_fission.png")
    # sim_gland.plot_budging_trajectories("budging_trajectories_gland_fission.png", num_to_plot=1000, sampling_interval=1)
    # # sim_gland.plot_beta_clustermap_pdf("beta_evolution_summary_gland_fission_clustermap")

    # Parameters for the Boundary Growth model
    # params_boundary = base_params.copy()
    # params_boundary['migration_edge_only'] = True
    # params_boundary['subclone'] = False
 
    # print("\n--- RUNNING BOUNDARY GROWTH MODEL ---")
    # sim_boundary = Simulation(params_boundary)
    # betas_boundary = sim_boundary.run_boundary_growth_model()

    # # Perform the biopsies and collect the results
    # biopsy_data = []
    # for x, y in biopsy_locations:
    #     result = sim_boundary.perform_virtual_biopsy(center_x=x, center_y=y, size=biopsy_size)
    #     if result:
    #         biopsy_data.append(result)

    # # Generate the correlation plot from the biopsy data
    # if biopsy_data:
    #     sim_boundary.plot_biopsy_correlation(biopsy_data, "biopsy_correlation_boundary.png")
    
    # sim_boundary.plot_kymograph("kymograph_boundary.png")
    # sim_boundary.plot_budging_trajectories("budging_trajectories_boundary.png", num_to_plot=1000, sampling_interval=1)
    # sim_boundary.plot_beta_clustermap_pdf("beta_evolution_summary_boundary_clustermap")

    # # Parameters for the Non_spatial model
    # params_non = base_params.copy()
    # params_non['G'] = 20000000  # Set a very high carrying capacity to simulate non-spatial conditions
    # params_non['migration_edge_only'] = False
    # print("\n--- RUNNING NON-SPATIAL MODEL ---")
    # sim_nonspatial = Simulation(params_non)
    # betas_nonspatial = sim_nonspatial.run_gland()

    # # plot the histogram of gland fission, boundary growth and non-spatial on the same plot with different color
    # plot_beta_histogram_comparison(
    #     betas_gland=betas_gland,
    #     betas_boundary=betas_boundary,
    #     betas_nonspatial=betas_nonspatial,
    #     output_filename="beta_comparison_gland_boundary_nonspatial.png"  
    # )
    # plot_beta_comparison(betas_group1= betas_gland,
    #                      betas_group2=betas_nonspatial,
    #                      group1_label='Gland Fission',
    #                      group2_label='Non-Spatial',
    #                      title = 'Beta Comparison: Gland Fission vs Non-Spatial',
    #                      output_filename='beta_comparison_gland_vs_non_spatial.png')
    # plot_beta_comparison(betas_group1= betas_boundary,
    #                         betas_group2=betas_nonspatial,
    #                         group1_label='Boundary Growth',
    #                         group2_label='Non-Spatial',
    #                         title= 'Beta Comparison: Boundary Growth vs Non-Spatial',
    #                         output_filename='beta_comparison_boundary_vs_non_spatial.png')
    # plot_beta_comparison(betas_group1 = betas_boundary,
    #                         betas_group2=betas_gland,
    #                         group1_label='Boundary Growth',
    #                         group2_label='Gland Fission',
    #                         title= 'Beta Comparison: Boundary Growth vs Gland Fission',
    #                         output_filename='beta_comparison_boundary_vs_gland.png')
    


    # Parameters for the subclonal model
    # params_subclonal = base_params.copy()
    # params_subclonal['migration_edge_only'] = False
    # params_subclonal['subclone'] = True  # Enable subclonal expansion

    # print("\n--- RUNNING SUBCLONAL MODEL ---")
    # sim_subclonal = Simulation(params_subclonal)
    # betas_subclonal = sim_subclonal.run_gland_subclonal()
    # # sim_subclonal.plot_beta_heatmap_pdf("beta_evolution_summary_subclonal.pdf")
    # # sim_subclonal.plot_beta_heatmap_pdf_subclone("beta_evolution_summary_subclonal_forclone.pdf")

    # # Perform the biopsies and collect the results
    # biopsy_data = []
    # for x, y in biopsy_locations:
    #     result = sim_subclonal.perform_virtual_biopsy(center_x=x, center_y=y, size=biopsy_size)
    #     if result:
    #         biopsy_data.append(result)

    # # Generate the correlation plot from the biopsy data
    # if biopsy_data:
    #     sim_subclonal.plot_biopsy_correlation(biopsy_data, "biopsy_correlation_subclonal_gland_fission.png")

    # sim_subclonal.plot_kymograph("x_kymograph_subclonal_gland_fission.png", xslice=True)
    # sim_subclonal.plot_kymograph("y_kymograph_subclonal_gland_fission.png", xslice=False)
    # sim_subclonal.plot_budging_trajectories("budging_trajectories_subclonal_gland_fission.png", num_to_plot=1000, sampling_interval=1)
    # sim_subclonal.plot_beta_clustermap_pdf_subclone("beta_evolution_summary_subclonal_clustermap_gland_fission")

    # Parameters for the Subclonal Boundary growth model
    params_subclonal_boundary = base_params.copy()
    params_subclonal_boundary['migration_edge_only'] = True
    params_subclonal_boundary['subclone'] = True  
    print("\n--- RUNNING SUBCLONAL BOUNDARY GROWTH MODEL ---")
    sim_subclonal_boundary = Simulation(params_subclonal_boundary)
    betas_subclonal_boundary = sim_subclonal_boundary.run_boundary_subclonal()
    
    # Perform the biopsies and collect the results
    biopsy_data = []
    for x, y in biopsy_locations:
        result = sim_subclonal_boundary.perform_virtual_biopsy(center_x=x, center_y=y, size=biopsy_size)
        if result:
            biopsy_data.append(result)
    # Generate the correlation plot from the biopsy data
    if biopsy_data:
        sim_subclonal_boundary.plot_biopsy_correlation(biopsy_data, "biopsy_correlation_subclonal_boundary.png")
    # Generate the kymograph for subclonal boundary growth
    sim_subclonal_boundary.plot_kymograph("x_kymograph_subclonal_boundary.png", xslice=True)
    sim_subclonal_boundary.plot_kymograph("y_kymograph_subclonal_boundary.png", xslice=False)
    sim_subclonal_boundary.plot_budging_trajectories("budging_trajectories_subclonal_boundary.png", num_to_plot=1000, sampling_interval=1)
    sim_subclonal_boundary.plot_beta_clustermap_pdf_subclone("beta_evolution_summary_subclonal_boundary_clustermap")

    # # --- Comparison of Gland Fission and Boundary Growth ---
    
    # Calculate Wasserstein Distance:
    # This measures the "work" required to transform one distribution into the other.
    # A smaller value means the distributions are more similar.
    

    #Comparison of these spatial models
    # print("---  Comparison of Spatial Models ---")
    
    # plot_beta_comparison(betas_group1 = betas_subclonal,
    #                         betas_group2=betas_subclonal_boundary,
    #                         group1_label='Subclonal Gland Fission',
    #                         group2_label='Subclonal Boundary Growth',
    #                         title='Beta Comparison: Subclonal Gland Fission vs Subclonal Boundary Growth',
    #                         output_filename='beta_comparison_subclonal_gland_vs_subclonal_boundary.png')
    # plot_beta_comparison(betas_group1 = betas_subclonal,
    #                         betas_group2=betas_nonspatial,
    #                         group1_label='Subclonal Gland Fission',
    #                         group2_label='Non-Spatial',
    #                         title='Beta Comparison: Subclonal Gland Fission vs Non-Spatial',
    #                         output_filename='beta_comparison_subclonal_gland_vs_non_spatial.png')
    # plot_beta_comparison(betas_group1 = betas_subclonal_boundary,
    #                         betas_group2=betas_nonspatial,
    #                         group1_label='Subclonal Boundary Growth',
    #                         group2_label='Non-Spatial',
    #                         title= 'Beta Comparison: Subclonal Boundary Growth vs Non-Spatial',
    #                         output_filename='beta_comparison_subclonal_boundary_vs_non_spatial.png')
    # plot_beta_comparison(betas_group1 = betas_subclonal,
    #                         betas_group2=betas_gland,
    #                         group1_label='Subclonal Gland Fission',
    #                         group2_label='Gland Fission',
    #                         title= 'Beta Comparison: Subclonal Gland Fission vs Gland Fission',
    #                         output_filename='beta_comparison_subclonal_gland_vs_gland.png')
    # plot_beta_comparison(betas_group1 = betas_subclonal_boundary,
    #                         betas_group2=betas_boundary,
    #                         group1_label='Subclonal Boundary Growth',
    #                         group2_label='Boundary Growth',
    #                         title= 'Beta Comparison: Subclonal Boundary Growth vs Boundary Growth',
    #                         output_filename='beta_comparison_subclonal_boundary_vs_boundary.png')
    


