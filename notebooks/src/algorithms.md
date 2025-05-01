## Overview

This document provides a unified, well-structured walkthrough of three metaheuristic clustering routines—**Particle Swarm Optimization (PSO)**, **Artificial Bee Colony (ABC)**, and **Ant Colony Optimization for continuous domains (ACOR)**. Each section describes the algorithm’s purpose, parameters, initialization, iterative update logic, history recording (if enabled), and final assignment of data points to clusters.

---

## 1. PSO-Based Clustering

### 1.1 Purpose and High-Level Flow
- Uses a swarm of particles to search for optimal cluster centers.
- Each particle encodes all centroids in a flat vector, iteratively updating velocities and positions.

### 1.2 Function Signature and Parameters
```python
GlobalBestPSO(
    n_particles,        # number of particles
    dimensions,         # k * n_features
    options,            # dict with c1, c2, w
    bounds=(lb, ub)     # search space limits
)
```
- `n_particles`: swarm size.
- `dimensions`: total degrees of freedom (`n_clusters * n_features`).
- `options`: inertia and acceleration coefficients.
- `bounds`: lower/upper limits based on data.

### 1.3 Initialization
```python
lb = np.min(data_tsne)
ub = np.max(data_tsne)
optimizer = ps.single.GlobalBestPSO(..., bounds=(lb, ub))
```

### 1.4 Iterative Updates
1. **Velocity**: mix of inertia, personal best attraction, global best attraction.
2. **Position**: add new velocity to current position.
3. **Cost Evaluation**: compute clustering objective for each particle.
4. **Best Updates**: update personal and global bests.

### 1.5 History Recording
- `pos_history`: particle positions per iteration.
- `cost_history`: global best cost per iteration.
- Reconstruct `best_centers_history` by selecting best particle each iteration.

### 1.6 Final Assignment
```python
best_centers = best_pos.reshape((k, n_features))
labels = assign_to_nearest(data_tsne, best_centers)
```

---

## 2. ABC-Based Clustering

### 2.1 Purpose and Flow
- Mimics foraging of honey bees: employed, onlooker, and scout phases.
- Food sources represent candidate centroid sets.

### 2.2 Function Signature and Parameters
```python
def run_abc(
    data,
    n_clusters,
    iters=100,
    n_food_sources=30,
    limit=20,
    record_history=False
):
```
- `n_food_sources`: population size.
- `limit`: scout reinitialization threshold.

### 2.3 Initialization
```python
lb = np.min(data)
ub = np.max(data)
food_sources = random_uniform(lb, ub, (n_food_sources, dims))
fitness = evaluate(food_sources)
```

### 2.4 Employed, Onlooker, and Scout Phases
1. **Employed Bees**: local neighbor perturbations.
2. **Onlooker Bees**: probabilistic selection based on fitness.
3. **Scout Bees**: replace stagnated sources.

### 2.5 Global Best Update and Recording
- Track best solution across iterations.
- If `record_history`, append positions, costs, centers.

### 2.6 Final Assignment
```python
best_centers = best_solution.reshape((k, n_features))
labels = assign_to_nearest(data, best_centers)
```

---

## 3. ACOR-Based Clustering

### 3.1 Purpose and Flow
- Adapts ant colony optimization to continuous search spaces.
- Maintains an archive of top solutions and samples new ants around them.

### 3.2 Function Signature and Parameters
```python
def run_acor(
    data,
    n_clusters,
    iters=100,
    archive_size=30,
    ants=30,
    q=0.5,
    xi=0.85,
    record_history=False
):
```
- `archive_size`: number of elite solutions retained.
- `ants`: new solutions per iteration.
- `q`: kernel weight factor.
- `xi`: sampling spread scaler.

### 3.3 Initialization
```python
archive = random_uniform(lb, ub, (archive_size, dims))
fitness = evaluate(archive)
```

### 3.4 Iterative Updates
1. **Sort Archive** by fitness.
2. **Compute Weights** via Gaussian kernel on ranks.
3. **Compute σ** for each archive member across dimensions.
4. **Generate Ant Solutions** by sampling Gaussians.
5. **Merge & Prune**: keep top `archive_size`.
6. **Record** if requested.

### 3.5 Final Assignment
```python
best_centers = best_solution.reshape((k, n_features))
labels = assign_to_nearest(data, best_centers)
```

---

## 4. Common Components

- **Cost Function**: typically sum of squared Euclidean distances to nearest centroid.
- **Label Assignment**: `labels[i] = argmin_j ||x_i - center_j||`.
- **History Structures**: optional dictionaries for post-hoc analysis and visualization.

---

## 5. Summary

All three routines encode cluster centers as continuous vectors and optimize a clustering objective via different bio-inspired metaphors:

- **PSO**: social–cognitive particle dynamics.
- **ABC**: division of labor among employed, onlooker, and scout bees.
- **ACOR**: probabilistic sampling around an elite archive.

Each can be equipped with history recording for convergence analysis and easily adapted to different distance metrics or objective functions.

