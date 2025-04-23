import numpy as np
import pandas as pd
import pyswarms as ps
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    pairwise_distances,
)
from scipy.spatial.distance import cdist
from matplotlib.animation import FuncAnimation
from pyswarms.utils.plotters.formatters import Mesher, Designer


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
        distances = pairwise_distances(data, centers, metric="euclidean")
        min_distances = np.min(distances, axis=1)
        cost[i] = np.sum(min_distances**2)
    return cost


def calculate_clustering_scores(data, labels):
    """
    Calculate clustering scores for the given data and labels.

    Parameters:
        data (array-like of shape (n_samples, n_features)):
            The input data points.
        labels (array-like of shape (n_samples,)):
            Cluster labels for each data point.

    Returns:
        sil_score (float):
            Mean Silhouette Coefficient (−1 to 1, higher = better).
        db_score (float):
            Davies–Bouldin Index (lower = better).
        ch_score (float):
            Calinski–Harabasz Index (higher = better).
    """
    # Silhouette: cohesion vs. separation
    sil_score = round(silhouette_score(data, labels), 3)

    # Davies–Bouldin: average cluster “worst-case” similarity
    db_score = round(davies_bouldin_score(data, labels), 3)

    # Calinski–Harabasz: variance ratio criterion
    ch_score = round(calinski_harabasz_score(data, labels), 1)

    return sil_score, db_score, ch_score


# -------------------------------
# 2. PSO Implementation (using pyswarms)
# -------------------------------
def run_pso(
    data_tsne,
    n_clusters,
    iters=100,
    n_particles=30,
    options={"c1": 1.5, "c2": 1.5, "w": 0.7},
    record_history=False,
):
    """
    Run PSO to cluster the data and optionally record the movement
    of the particles and the best centers at each iteration.

    Parameters:
        data_tsne (np.array): The input data, typically a 2D array of shape (n_samples, n_features).
                              For t-SNE data, n_features should be 2.
        n_clusters (int): Number of clusters to form.
        iters (int): Number of iterations for the PSO algorithm.
        n_particles (int): Number of particles in the swarm.
        options (dict): PSO hyperparameters, including:
                        - 'c1': Cognitive parameter (influence of personal best).
                        - 'c2': Social parameter (influence of global best).
                        - 'w': Inertia weight (influence of previous velocity).
        record_history (bool): Whether to record the positions of particles
                               and the best centers at each iteration.

    Returns:
        best_cost (float): The best cost (objective value) achieved by the swarm.
        best_centers (np.array): The best cluster centers found, with shape (n_clusters, n_features).
                                 Each row represents the coordinates of a cluster center.
        labels (np.array): Cluster labels for each data point, with shape (n_samples,).
                           Each element is an integer representing the assigned cluster.
        pos_history (list or None): If record_history is True, a list of particle positions at each iteration.
                                    Each element is a 2D array of shape (n_particles, n_clusters * n_features),
                                    where each row represents a particle's flattened cluster center positions.
                                    If record_history is False, this is None.
        cost_history (list or None): If record_history is True, a list of the best cost at each iteration.
                                     Each element is a float representing the best cost at that iteration.
                                     If record_history is False, this is None.
        best_centers_history (np.array or None): If record_history is True, an array of the best cluster centers
                                                 at each iteration, with shape (iters, n_clusters, n_features).
                                                 Each element is a 2D array representing the cluster centers
                                                 at a specific iteration. If record_history is False, this is None.
    """
    n_features = data_tsne.shape[1]  # should be 2 for t-SNE
    dimensions = n_clusters * n_features

    # Define bounds based on the t-SNE data range
    lb = np.min(data_tsne) * np.ones(dimensions)
    ub = np.max(data_tsne) * np.ones(dimensions)
    bounds = (lb, ub)

    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles, dimensions=dimensions, options=options, bounds=bounds
    )

    # always turn on verbose so that rep.hook() is called each iter
    best_cost, best_pos = optimizer.optimize(
        clustering_objective_function,
        iters=iters,
        data=data_tsne,
        n_clusters=n_clusters,
        verbose=False,
    )

    # fetch raw histories if requested
    pos_history = optimizer.pos_history if record_history else None
    cost_history = optimizer.cost_history if record_history else None

    # reshape the final best position into cluster centers
    best_centers = best_pos.reshape((n_clusters, n_features))

    # assign labels for the final solution
    distances = cdist(data_tsne, best_centers, metric="euclidean")
    labels = np.argmin(distances, axis=1)

    # now build best_centers_history by re-evaluating each snapshot
    if record_history:
        best_centers_history = []
        for snapshot in pos_history:
            # snapshot: shape (n_particles, dimensions)
            # evaluate all particles at this iter
            costs = clustering_objective_function(snapshot, data_tsne, n_clusters)
            # pick the particle with minimal cost
            idx = np.argmin(costs)
            # reshape its position into (n_clusters, n_features)
            best_centers_history.append(snapshot[idx].reshape((n_clusters, n_features)))
        best_centers_history = np.array(best_centers_history)
    else:
        best_centers_history = None

    return (
        best_cost,
        best_centers,
        labels,
        pos_history,
        cost_history,
        best_centers_history,
    )


def run_abc(
    data_tsne, n_clusters, iters=100, n_food_sources=30, limit=20, record_history=False
):
    """
    Run ABC to cluster the t-SNE data and optionally record the movement of the food sources,
    the corresponding cost history, and the best centers per iteration.

    Parameters:
        data_tsne (np.array): The t-SNE 2D data.
        n_clusters (int): Number of clusters to form.
        iters (int): Number of iterations.
        n_food_sources (int): Number of food sources.
        limit (int): Trial limit for scout phase.
        record_history (bool): Whether to record the positions (food sources), costs, and best centers at each iteration.

    Returns:
        best_cost (float): Best cost found.
        best_centers (np.array): Best cluster centers found.
        labels (np.array): Cluster labels for data points.
        history (dict or None): Dictionary with keys:
            'positions' (list of food source positions per iteration),
            'costs' (list of best cost per iteration),
            'best_centers' (array of best centers per iteration, shape (iters+1, n_clusters, 2)).
    """
    n_features = data_tsne.shape[1]
    dimensions = n_clusters * n_features
    lb = np.min(data_tsne) * np.ones(dimensions)
    ub = np.max(data_tsne) * np.ones(dimensions)

    # Initialize food sources randomly
    food_sources = np.random.uniform(low=lb, high=ub, size=(n_food_sources, dimensions))
    fitness = clustering_objective_function(food_sources, data_tsne, n_clusters)
    trial_counters = np.zeros(n_food_sources, dtype=int)

    # Initialize best solution
    best_index = np.argmin(fitness)
    best_solution = food_sources[best_index].copy()
    best_cost = fitness[best_index]

    # Prepare history storage
    if record_history:
        history = {
            "positions": [food_sources.copy()],
            "costs": [best_cost],
            "best_centers": [best_solution.reshape((n_clusters, n_features))],
        }
    else:
        history = None

    # Main ABC loop
    for t in range(iters):
        # Employed bees phase
        for i in range(n_food_sources):
            j = np.random.choice([idx for idx in range(n_food_sources) if idx != i])
            phi = np.random.uniform(-1, 1, size=dimensions)
            candidate = food_sources[i] + phi * (food_sources[i] - food_sources[j])
            candidate = np.clip(candidate, lb, ub)
            candidate_fitness = clustering_objective_function(
                candidate[np.newaxis, :], data_tsne, n_clusters
            )[0]
            if candidate_fitness < fitness[i]:
                food_sources[i] = candidate
                fitness[i] = candidate_fitness
                trial_counters[i] = 0
            else:
                trial_counters[i] += 1

        # Onlooker bees phase
        probabilities = 1.0 / (fitness + 1e-10)
        probabilities /= probabilities.sum()
        for i in range(n_food_sources):
            if np.random.rand() < probabilities[i]:
                j = np.random.choice([idx for idx in range(n_food_sources) if idx != i])
                phi = np.random.uniform(-1, 1, size=dimensions)
                candidate = food_sources[i] + phi * (food_sources[i] - food_sources[j])
                candidate = np.clip(candidate, lb, ub)
                candidate_fitness = clustering_objective_function(
                    candidate[np.newaxis, :], data_tsne, n_clusters
                )[0]
                if candidate_fitness < fitness[i]:
                    food_sources[i] = candidate
                    fitness[i] = candidate_fitness
                    trial_counters[i] = 0
                else:
                    trial_counters[i] += 1

        # Scout phase
        for i in range(n_food_sources):
            if trial_counters[i] > limit:
                food_sources[i] = np.random.uniform(low=lb, high=ub, size=dimensions)
                fitness[i] = clustering_objective_function(
                    food_sources[i][np.newaxis, :], data_tsne, n_clusters
                )[0]
                trial_counters[i] = 0

        # Update best solution of this iteration
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_cost:
            best_cost = fitness[current_best_idx]
            best_solution = food_sources[current_best_idx].copy()

        # Record history
        if record_history:
            history["positions"].append(food_sources.copy())
            history["costs"].append(best_cost)
            history["best_centers"].append(
                best_solution.reshape((n_clusters, n_features))
            )

    # Final assignments
    best_centers = best_solution.reshape((n_clusters, n_features))
    distances = cdist(data_tsne, best_centers, metric="euclidean")
    labels = np.argmin(distances, axis=1)

    # Convert best_centers history to array if recorded
    if record_history:
        history["best_centers"] = np.array(history["best_centers"])
    return best_cost, best_centers, labels, history


def run_acor(
    data_tsne,
    n_clusters,
    iters=100,
    archive_size=30,
    ants=30,
    q=0.5,
    xi=0.85,
    record_history=False,
):
    """
    Run ACOR to cluster the t-SNE data and optionally record the archive evolution, cost history,
    and the best centers at each iteration.

    Parameters:
        data_tsne (np.array): The t-SNE 2D data.
        n_clusters (int): Number of clusters.
        iters (int): Number of iterations.
        archive_size (int): Size of the solution archive.
        ants (int): Number of ants used to generate new solutions.
        q (float): Weighting factor for the Gaussian kernel.
        xi (float): Factor to control the standard deviation in sampling.
        record_history (bool): Whether to record the archive, costs, and best centers at each iteration.

    Returns:
        best_cost (float): Best cost found.
        best_centers (np.array): Best cluster centers found.
        labels (np.array): Cluster labels for data points.
        archive_history (dict or None): If record_history is True, a dictionary with keys:
            - 'archives': list of archives at each iteration
            - 'costs': list of best cost per iteration
            - 'best_centers': array of best centers per iteration, shape (iters+1, n_clusters, 2)
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
        best_idx0 = np.argmin(fitness)
        archive_history = {
            "archives": [archive.copy()],
            "costs": [fitness[best_idx0]],
            "best_centers": [archive[best_idx0].reshape((n_clusters, n_features))],
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
        weights = (1 / (q * archive_size * np.sqrt(2 * np.pi))) * np.exp(
            -(ranks**2) / (2 * (q * archive_size) ** 2)
        )
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

        new_fitness = clustering_objective_function(
            new_solutions, data_tsne, n_clusters
        )

        # Merge archive and new solutions, then keep the best 'archive_size' solutions
        combined = np.vstack((archive, new_solutions))
        combined_fitness = np.concatenate((fitness, new_fitness))
        sorted_idx = np.argsort(combined_fitness)
        archive = combined[sorted_idx][:archive_size]
        fitness = combined_fitness[sorted_idx][:archive_size]

        # Record archive, best cost, and best centers for this iteration
        if record_history:
            best_idx = np.argmin(fitness)
            archive_history["archives"].append(archive.copy())
            archive_history["costs"].append(fitness[best_idx])
            archive_history["best_centers"].append(
                archive[best_idx].reshape((n_clusters, n_features))
            )

    # Final solution
    best_solution = archive[np.argmin(fitness)]
    best_cost = np.min(fitness)
    best_centers = best_solution.reshape((n_clusters, n_features))
    distances = cdist(data_tsne, best_centers, metric="euclidean")
    labels = np.argmin(distances, axis=1)

    # Convert best_centers history to array if recorded
    if record_history:
        archive_history["best_centers"] = np.array(archive_history["best_centers"])
    return best_cost, best_centers, labels, archive_history


def animate_best_center_history(
    data,
    pos_history,
    best_centers_history,
    center_idx,
    interval=200,
    figsize=(8, 6),
    margin=5,
):
    """
    Animate the search of a single cluster center, showing both all particle positions
    and the current best particle's center per iteration.

    Parameters:
        data_tsne (np.ndarray): t-SNE embedding, shape (n_samples, 2).
        pos_history (list of np.ndarray): List of particle positions each iteration,
            each array shape (n_particles, n_clusters*2).
        best_centers_history (np.ndarray): Array of best centers per iteration,
            shape (n_iters, n_clusters, 2).
        center_idx (int): Index of the cluster center to animate (0 <= center_idx < n_clusters).
        interval (int): Delay between frames in milliseconds.
        figsize (tuple): Figure size.
        margin (float): Margin to pad plot limits.

    Returns:
        matplotlib.animation.FuncAnimation: The animation object.
    """
    # Dimensions and sanity checks
    n_iters = len(pos_history)
    first = pos_history[0]
    n_particles, dim = first.shape
    assert dim % 2 == 0, "last dim must be even"
    n_clusters = dim // 2
    assert 0 <= center_idx < n_clusters, "center_idx out of range"

    # Slice indices for this center in the flat position vector
    i0, i1 = 2 * center_idx, 2 * center_idx + 2

    # Set plot limits
    x_min, x_max = data[:, 0].min() - margin, data[:, 0].max() + margin
    y_min, y_max = data[:, 1].min() - margin, data[:, 1].max() + margin

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title(f"Cluster center #{center_idx} search")

    # Scatter of all particle positions for this center
    scat = ax.scatter([], [], c="blue", s=50, label=f"Particles C{center_idx}")
    # Scatter for the current best center (will update each frame)
    scat_best = ax.scatter(
        [], [], c="red", marker="X", s=100, label=f"Best Center C{center_idx}"
    )
    ax.legend(loc="upper right")

    def update(frame):
        # Update particle positions
        pts = pos_history[frame]  # shape (n_particles, dim)
        center_pts = pts[:, i0:i1]  # shape (n_particles, 2)
        scat.set_offsets(center_pts)
        # Update current best center
        best_pt = best_centers_history[frame, center_idx]
        scat_best.set_offsets(best_pt.reshape(1, 2))
        # Update title
        title.set_text(f"Iteration {frame + 1}/{n_iters}")
        return scat, scat_best, title

    ani = animation.FuncAnimation(
        fig, update, frames=n_iters, interval=interval, blit=True, repeat=False
    )
    return ani


def plot_clusters(
    data,
    labels,
    title,
    centers=None,
    x_label="x",
    y_label="y",
    cmap="viridis",
    alpha=0.7,
    center_color="red",
    center_marker="X",
    center_size=200,
):
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
    # Convert DataFrame to NumPy array if necessary
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, alpha=alpha)
    if centers is not None:
        plt.scatter(
            centers[:, 0],
            centers[:, 1],
            c=center_color,
            s=center_size,
            marker=center_marker,
        )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
