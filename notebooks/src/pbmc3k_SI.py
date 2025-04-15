import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
import pyswarms as ps
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from pyswarms.utils.plotters.formatters import Mesher, Designer
import matplotlib.animation as animation

# -------------------------------
# 1. Define the clustering objective function
# -------------------------------
def clustering_objective_function(particles, data, n_clusters):
    """
    For each candidate solution (particle), compute the sum of squared distances
    of each data point to its nearest cluster center.
    Each particle is a flattened array of cluster centers.
    """
    n_particles = particles.shape[0]
    n_features = data.shape[1]
    cost = np.zeros(n_particles)
    
    for i in range(n_particles):
        centers = particles[i].reshape((n_clusters, n_features))
        distances = pairwise_distances(data, centers, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        cost[i] = np.sum(min_distances**2)
    return cost

# -------------------------------
# 2. PSO Implementation (using pyswarms)
# -------------------------------
def run_pso(data_tsne, n_clusters, iters=100, n_particles=30, record_history=False):
    """
    Run PSO to cluster the t-SNE data and optionally record the movement of the particles.
    
    Parameters:
        data_tsne (np.array): The t-SNE 2D data.
        n_clusters (int): Number of clusters to form.
        iters (int): Number of iterations.
        n_particles (int): Number of particles in the swarm.
        record_history (bool): Whether to record the positions of particles at each iteration.
    
    Returns:
        best_cost (float): Best cost (objective value).
        best_centers (np.array): Best cluster centers found.
        labels (np.array): Cluster labels for data points.
        sil_score (float): Silhouette score.
        pos_history (list or None): If record_history is True, a list where each element is an array of particle positions 
                                    (shape: [n_particles, n_clusters*2]) at each iteration; otherwise, None.
    """
    n_features = data_tsne.shape[1]  # should be 2 for t-SNE
    dimensions = n_clusters * n_features
    # Define bounds based on the t-SNE data range
    lb = np.min(data_tsne) * np.ones(dimensions)
    ub = np.max(data_tsne) * np.ones(dimensions)
    bounds = (lb, ub)
    
    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dimensions,
                                        options=options, bounds=bounds)
    best_cost, best_pos = optimizer.optimize(clustering_objective_function,
                                             iters=iters,
                                             data=data_tsne,
                                             n_clusters=n_clusters,
                                             verbose=False)
    # Here we assume pyswarms is recording the position history.
    # (If your pyswarms version does not automatically store history,
    #  you might need to implement a callback or customize the update loop.)
    pos_history = optimizer.pos_history if record_history else None
    cost_history = optimizer.cost_history if record_history else None

    best_centers = best_pos.reshape((n_clusters, n_features))
    distances = cdist(data_tsne, best_centers, metric='euclidean')
    labels = np.argmin(distances, axis=1)
    sil_score = silhouette_score(data_tsne, labels)
    return best_cost, best_centers, labels, sil_score, pos_history, cost_history

def run_abc(data_tsne, n_clusters, iters=100, n_food_sources=30, limit=20, record_history=False):
    """
    Run ABC to cluster the t-SNE data and optionally record the movement of the food sources
    and the corresponding cost history.
    
    Parameters:
        data_tsne (np.array): The t-SNE 2D data.
        n_clusters (int): Number of clusters to form.
        iters (int): Number of iterations.
        n_food_sources (int): Number of food sources.
        limit (int): Trial limit for scout phase.
        record_history (bool): Whether to record the positions (food sources) and costs at each iteration.
    
    Returns:
        best_cost (float): Best cost found.
        best_centers (np.array): Best cluster centers found.
        labels (np.array): Cluster labels for data points.
        sil_score (float): Silhouette score.
        history (dict or None): Dictionary with keys 'positions' (list of food source positions recorded each iteration)
                                and 'costs' (list of best cost at the end of each iteration).
    """
    n_features = data_tsne.shape[1]
    dimensions = n_clusters * n_features
    lb = np.min(data_tsne) * np.ones(dimensions)
    ub = np.max(data_tsne) * np.ones(dimensions)
    
    # Initialize food sources randomly
    food_sources = np.random.uniform(low=lb, high=ub, size=(n_food_sources, dimensions))
    fitness = clustering_objective_function(food_sources, data_tsne, n_clusters)
    trial_counters = np.zeros(n_food_sources)
    
    best_index = np.argmin(fitness)
    best_solution = food_sources[best_index].copy()
    best_cost = fitness[best_index]
    
    # Initialize history as a dictionary if required
    if record_history:
        history = {
            'positions': [food_sources.copy()],
            'costs': [best_cost]
        }
    else:
        history = None
    
    for t in range(iters):
        # Employed bees phase
        for i in range(n_food_sources):
            j = np.random.randint(0, n_food_sources)
            while j == i:
                j = np.random.randint(0, n_food_sources)
            phi = np.random.uniform(-1, 1, size=dimensions)
            candidate = food_sources[i] + phi * (food_sources[i] - food_sources[j])
            candidate = np.clip(candidate, lb, ub)
            candidate_fitness = clustering_objective_function(candidate[np.newaxis, :], data_tsne, n_clusters)[0]
            if candidate_fitness < fitness[i]:
                food_sources[i] = candidate
                fitness[i] = candidate_fitness
                trial_counters[i] = 0
            else:
                trial_counters[i] += 1
        
        # Onlooker bees phase
        prob = (1.0 / (fitness + 1e-10))
        prob = prob / np.sum(prob)
        for i in range(n_food_sources):
            if np.random.rand() < prob[i]:
                j = np.random.randint(0, n_food_sources)
                while j == i:
                    j = np.random.randint(0, n_food_sources)
                phi = np.random.uniform(-1, 1, size=dimensions)
                candidate = food_sources[i] + phi * (food_sources[i] - food_sources[j])
                candidate = np.clip(candidate, lb, ub)
                candidate_fitness = clustering_objective_function(candidate[np.newaxis, :], data_tsne, n_clusters)[0]
                if candidate_fitness < fitness[i]:
                    food_sources[i] = candidate
                    fitness[i] = candidate_fitness
                    trial_counters[i] = 0
                else:
                    trial_counters[i] += 1
        
        # Scout phase: reinitialize sources that haven't improved
        for i in range(n_food_sources):
            if trial_counters[i] > limit:
                food_sources[i] = np.random.uniform(low=lb, high=ub, size=dimensions)
                fitness[i] = clustering_objective_function(food_sources[i][np.newaxis, :], data_tsne, n_clusters)[0]
                trial_counters[i] = 0
        
        # Update best solution after iteration
        current_best_index = np.argmin(fitness)
        if fitness[current_best_index] < best_cost:
            best_cost = fitness[current_best_index]
            best_solution = food_sources[current_best_index].copy()
        
        # Record history for this iteration if requested
        if record_history:
            history['positions'].append(food_sources.copy())
            history['costs'].append(best_cost)
    
    best_centers = best_solution.reshape((n_clusters, n_features))
    distances = cdist(data_tsne, best_centers, metric='euclidean')
    labels = np.argmin(distances, axis=1)
    sil_score = silhouette_score(data_tsne, labels)
    return best_cost, best_centers, labels, sil_score, history

def run_acor(data_tsne, n_clusters, iters=100, archive_size=30, ants=30, q=0.5, xi=0.85, record_history=False):
    """
    Run ACOR to cluster the t-SNE data and optionally record the archive evolution 
    as well as the cost history.
    
    Parameters:
        data_tsne (np.array): The t-SNE 2D data.
        n_clusters (int): Number of clusters.
        iters (int): Number of iterations.
        archive_size (int): Size of the solution archive.
        ants (int): Number of ants used to generate new solutions.
        q (float): Weighting factor for the Gaussian kernel.
        xi (float): Factor to control the standard deviation in sampling.
        record_history (bool): Whether to record the archive and corresponding cost 
                               at each iteration.
    
    Returns:
        best_cost (float): Best cost found.
        best_centers (np.array): Best cluster centers found.
        labels (np.array): Cluster labels for data points.
        sil_score (float): Silhouette score.
        archive_history (dict or None): If record_history is True, a dictionary with keys:
            - 'archives': list of archives at each iteration
            - 'costs': list of best cost for each iteration
    """
    n_features = data_tsne.shape[1]
    dimensions = n_clusters * n_features
    lb = np.min(data_tsne) * np.ones(dimensions)
    ub = np.max(data_tsne) * np.ones(dimensions)
    
    # Initialize archive with random solutions
    archive = np.random.uniform(low=lb, high=ub, size=(archive_size, dimensions))
    fitness = clustering_objective_function(archive, data_tsne, n_clusters)
    
    # Initialize archive history as a dictionary if required
    if record_history:
        best_cost_initial = np.min(fitness)
        archive_history = {
            'archives': [archive.copy()],
            'costs': [best_cost_initial]
        }
    else:
        archive_history = None
    
    for t in range(iters):
        # Sort archive (best fitness first)
        sorted_idx = np.argsort(fitness)
        archive = archive[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # Compute weights (Gaussian kernel based on rank)
        ranks = np.arange(archive_size)
        weights = (1 / (q * archive_size * np.sqrt(2 * np.pi))) * \
                  np.exp(- (ranks ** 2) / (2 * (q * archive_size) ** 2))
        weights = weights / np.sum(weights)
        
        # Compute standard deviations for Gaussian sampling
        sigma = np.zeros((archive_size, dimensions))
        for i in range(archive_size):
            diff = archive - archive[i]
            sigma[i] = xi * np.std(diff, axis=0)
            sigma[i][sigma[i] == 0] = 1e-6  # Prevent zero std
        
        # Generate new candidate solutions
        new_solutions = np.zeros((ants, dimensions))
        for i in range(ants):
            idx = np.random.choice(np.arange(archive_size), p=weights)
            new_solution = np.random.normal(loc=archive[idx], scale=sigma[idx])
            new_solution = np.clip(new_solution, lb, ub)
            new_solutions[i] = new_solution
        
        new_fitness = clustering_objective_function(new_solutions, data_tsne, n_clusters)
        
        # Merge archive and new solutions, then keep the best 'archive_size' solutions
        combined = np.vstack((archive, new_solutions))
        combined_fitness = np.concatenate((fitness, new_fitness))
        sorted_idx = np.argsort(combined_fitness)
        archive = combined[sorted_idx][:archive_size]
        fitness = combined_fitness[sorted_idx][:archive_size]
        
        # Record archive and best cost (i.e. first entry after sort) for this iteration
        if record_history:
            archive_history['archives'].append(archive.copy())
            archive_history['costs'].append(fitness[0])
    
    best_solution = archive[0]
    best_cost = fitness[0]
    best_centers = best_solution.reshape((n_clusters, n_features))
    distances = cdist(data_tsne, best_centers, metric='euclidean')
    labels = np.argmin(distances, axis=1)
    sil_score = silhouette_score(data_tsne, labels)
    return best_cost, best_centers, labels, sil_score, archive_history

def animate_sa(data_tsne, pos_history, centers, interval=200, figsize=(8,6), margin=5):
    """
    Create an animation of PSO particle movement over iterations.

    Parameters:
    - data_tsne (np.array): The t-SNE 2D data with shape (n_cells, 2).
    - pos_history (array-like): Particle history of shape (n_iters, n_particles, 2).
    - centers (np.array): Final cluster centers with shape (n_clusters, 2).
    - interval (int): Time between frames in milliseconds. Default is 100.
    - figsize (tuple): Figure size for the plot. Default is (8, 6).
    - margin (float): Extra space to add/subtract to the data limits. Default is 5.

    Returns:
    - ani (FuncAnimation): A Matplotlib animation object.
    """
    # Set axis limits based on the data range and margin
    x_min, x_max = np.min(data_tsne[:, 0]) - margin, np.max(data_tsne[:, 0]) + margin
    y_min, y_max = np.min(data_tsne[:, 1]) - margin, np.max(data_tsne[:, 1]) + margin
    
    # Create a new figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x-coordinate')
    ax.set_ylabel('y-coordinate')
    ax.set_title('Particle Movement over Iterations')
    
    # Create a scatter plot for the particles.
    # pso_history should be of shape (n_iters, n_particles, 2).
    scat = ax.scatter([], [], c='blue', label='Particles', s=50)
    
    # Plot the final cluster centers as static markers.
    # pso_centers should be of shape (n_clusters, 2).
    ax.scatter(centers[:, 0], centers[:, 1],
               c='red', marker='x', s=100, label='Final Cluster Centers')
    
    ax.legend()
    
    # Define the update function for the animation
    def update(frame):
        # Update particle positions for the current frame
        positions = pos_history[frame]
        scat.set_offsets(positions)
        ax.set_title(f'Iteration: {frame + 1}')
        return scat,
    
    # Create the animation object
    ani = animation.FuncAnimation(fig, update, frames=len(pos_history),
                                  interval=interval, blit=True, repeat=False)
    return ani

def plot_clusters(data, labels, sil_score, title, centers=None,
                  x_label="t-SNE 1", y_label="t-SNE 2", 
                  cmap='viridis', alpha=0.7, 
                  center_color='red', center_marker='X', center_size=200):
    """
    Plots clustering results on a 2D scatter plot.

    Parameters:
        data (np.array): Array of data points with shape (n_samples, 2).
        labels (np.array): Cluster labels for each data point.
        sil_score (float): Silhouette score to be displayed in the title.
        title (str): A format string for the title (should include a placeholder for sil_score).
        centers (np.array, optional): Array of cluster center coordinates with shape (n_clusters, 2).
                                      If None, cluster centers are omitted.
        x_label (str): Label for the x-axis. Default is "t-SNE 1".
        y_label (str): Label for the y-axis. Default is "t-SNE 2".
        cmap (str): Colormap for the data points. Default is 'viridis'.
        alpha (float): Transparency for data points. Default is 0.7.
        center_color (str): Color for the cluster centers. Default is 'red'.
        center_marker (str): Marker style for the cluster centers. Default is 'X'.
        center_size (int): Size of the marker for the cluster centers. Default is 200.

    Returns:
        None. Displays the plot.
    """
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, alpha=alpha)
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c=center_color, s=center_size, marker=center_marker)
    plt.title(title.format(sil_score))
    plt.xlabel(x_label)
    plt.ylabel(y_label)