## Overview

This document provides a unified, well-structured walkthrough of three metaheuristic clustering routines—**Particle Swarm Optimization (PSO)**, **Artificial Bee Colony (ABC)**, and **Ant Colony Optimization for continuous domains (ACOR)**. Each section describes the algorithm’s purpose, parameters, initialization, iterative update logic, history recording (if enabled), and final assignment of data points to clusters.

## 1. PSO-Based Clustering

**Note:** This implementation uses [PySwarms](https://pyswarms.readthedocs.io/en/latest/) for Particle Swarm Optimization (PSO) in Python ([pyswarms.readthedocs.io](https://pyswarms.readthedocs.io/), [github.com](https://github.com/ljvmiranda921/pyswarms?)).

> Miranda, L. J. V. (2018). *PySwarms, a research-toolkit for Particle Swarm Optimization in Python*. _Journal of Open Source Software, 3_(21). https://doi.org/10.21105/joss.00433

### 1.1 Purpose and High-Level Flow

- Uses a swarm of particles to search for optimal cluster centers.
- Each particle encodes all centroids in a flat vector, iteratively updating velocities and positions.

### 1.2 Function Signature and Parameters

```python
GlobalBestPSO(
    n_particles,        # number of particles in the swarm
    dimensions,         # total degrees of freedom (e.g., n_clusters * n_features)
    options,            # dict with keys 'c1', 'c2', 'w' for cognitive, social, and inertia weights
    bounds=(lb, ub)     # tuple of arrays/vectors specifying lower and upper limits per dimension
)
```

This class implements the **global-best** (gbest) variant of Particle Swarm Optimization in PySwarms, using a star-topology where every particle is influenced by the single best solution found so far. At each iteration, particles update their velocities and positions via

$$
v_i \leftarrow w\,v_i + c_1\,r_1\,(p_i - x_i) + c_2\,r_2\,(g - x_i),\quad
x_i \leftarrow x_i + v_i
$$

where  
- $x_i$ is the particle’s position,  
- $v_i$ its velocity,  
- $p_i$ its personal best,  
- $g$ the global best across all particles,  
- $r_1,r_2\sim U(0,1)$, and  
- $w,c_1,c_2$ are taken from `options`.  ([pyswarms.single package](https://pyswarms.readthedocs.io/en/latest/api/pyswarms.single.html?), [Source code for pyswarms.single.global_best](https://pyswarms.readthedocs.io/en/latest/_modules/pyswarms/single/global_best.html?))

In the context of the GlobalBestPSO algorithm in PySwarms, three key hyperparameters govern the behavior of the swarm during optimization. These each govern the attractive vectorized force that the current trajectory, the personal best and global best exert on the particle, respectively:

- **Inertia Weight ($w$)**: This parameter controls the influence of a particle's previous velocity on its current one. A higher inertia weight encourages global exploration of the search space, helping particles to avoid local minima. Conversely, a lower inertia weight promotes local exploitation, allowing particles to fine-tune their positions around promising areas.

- **Cognitive Coefficient ($c_1$)**: This coefficient determines the degree to which a particle is influenced by its own best-known position. A higher $c_1$ value encourages particles to rely more on their personal experiences, enhancing individual exploration.

- **Social Coefficient ($c_2$)**: This coefficient dictates the extent to which a particle is influenced by the best-known position of the entire swarm. A higher $c_2$ value promotes collective behavior, guiding particles toward the global best solution found so far.

After initialization (random positions in `bounds`, zero or small random velocities), you call

```python
best_cost, best_pos = optimizer.optimize(objective_func, iters=100)
```

to run 100 iterations, returning the lowest objective value found and its corresponding position vector.  ([[PDF] PySwarms Documentation - Read the Docs](https://media.readthedocs.org/pdf/pyswarms/development/pyswarms.pdf?))

### 1.3 Iterative Updates

During the optimization process, the following steps occur at each iteration:

1. **Objective Evaluation**:
   - Each particle's current position $ x_i(t) $ is evaluated using the objective function to obtain its cost.

2. **Personal Best Update**:
   - If a particle's current cost is better than its previous personal best $ p_i $, update $ p_i $ to the current position.

3. **Global Best Update**:
   - If any particle's personal best $ p_i $ is better than the current global best $ g $, update $ g $ accordingly.

4. **Velocity Update**:
   - Each particle's velocity is updated using the formula:
     $$
     v_i(t+1) = w \cdot v_i(t) + c_1 \cdot r_1 \cdot (p_i - x_i(t)) + c_2 \cdot r_2 \cdot (g - x_i(t))
     $$

5. **Position Update**:
   - Each particle's position is updated:
     $$
     x_i(t+1) = x_i(t) + v_i(t+1)
     $$
   - If any component of $ x_i(t+1) $ falls outside the specified `bounds`, it is clipped to lie within them.

### 1.4 History Recording
- `pos_history`: particle positions per iteration.
- `cost_history`: global best cost per iteration.
- Reconstruct `best_centers_history` by selecting best particle each iteration.

### 1.5 Final Assignment

After the PSO optimization concludes, the best position vector `best_pos`—representing the optimal cluster centers—is reshaped into a two-dimensional array:

```python
best_centers = best_pos.reshape((n_clusters, n_features))
```

Each data point $ x_i $ is then assigned to the nearest cluster center by computing the Euclidean distance to each center $ c_j $ and selecting the one with the minimum distance:

$$
\text{label}_i = \arg\min_j \| x_i - c_j \|_2
$$

This results in a label vector:

```python
labels = np.argmin(cdist(data, best_centers, metric="euclidean"), axis=1)
```

Here, `cdist` computes the pairwise Euclidean distances between data points and cluster centers, and `np.argmin` identifies the index of the nearest center for each data point. This assignment partitions the dataset into clusters based on proximity to the optimized centers.

## 2. ABC-Based Clustering

**Note:** This custom implementation of the Artificial Bee Colony (ABC) algorithm for clustering was inspired by Pedro Buarque's article, *A Modified Artificial Bee Colony Algorithm to Solve Clustering Problems*. In this article, Buarque demonstrates how the ABC algorithm can be adapted for clustering tasks by representing each bee as a potential set of cluster centroids and using the Sum of Squared Errors (SSE) as the objective function to minimize. This approach effectively transforms the clustering problem into an optimization task, aligning well with the structure of this implementation. ([A modified Artificial Bee Colony algorithm to solve Clustering problems | by Pedro Buarque | TDS Archive | Medium](https://medium.com/towards-data-science/a-modified-artificial-bee-colony-algorithm-to-solve-clustering-problems-fc0b69bd0788?))

### 2.1 Purpose and High-Level Flow

- ABC is a swarm intelligence algorithm inspired by the foraging behavior of honey bees.
- In the context of clustering, each food source represents a potential set of cluster centers.
- The algorithm iteratively improves these food sources through the actions of employed, onlooker, and scout bees. ([A novel chaotic and neighborhood search-based artificial bee ...](https://www.nature.com/articles/s41598-023-44770-8?))

### 2.2 Function Signature and Parameters

```python
def run_abc(
    data,
    n_clusters,
    iters=100,
    n_food_sources=30,
    limit=20,
    record_history=False
)
```

- `data`: Input dataset as a NumPy array of shape `(n_samples, n_features)`.
- `n_clusters`: Number of clusters to form.
- `iters`: Number of iterations to run the algorithm.
- `n_food_sources`: Number of food sources (candidate solutions) in the population.
- `limit`: Trial limit for scout phase; if a food source isn't improved within this number of trials, it's reinitialized.
- `record_history`: Boolean flag to record the positions, costs, and best centers at each iteration. ([Swarm intelligence](https://en.wikipedia.org/wiki/Swarm_intelligence?))

### 2.3 Initialization

- **Data Dimensions**: Determine `n_features` from the input data.
- **Solution Representation**: Each food source is a flat vector of length `dimensions = n_clusters * n_features`, representing all cluster centers.
- **Bounds**: Set lower (`lb`) and upper (`ub`) bounds for each dimension based on the minimum and maximum values in the dataset.
- **Food Sources**: Initialize `n_food_sources` food sources randomly within the specified bounds.
- **Fitness Evaluation**: Compute the initial fitness (objective function value) for each food source.
- **Trial Counters**: Initialize a trial counter for each food source to zero.

### 2.4 Iterative Updates

For each iteration from `t = 1` to `iters`, perform the following phases:

1. **Employed Bees Phase**:
   - For each food source `i`:
     - Select a randomly chosen peer food source `j ≠ i`.
     - Generate a perturbation vector `φ` with elements uniformly distributed in `[-1, 1]`.
     - Create a new candidate solution:
       $$
       v_i = x_i + \phi \cdot (x_i - x_j)
       $$
     - Clip `v_i` to the bounds `[lb, ub]`.
     - Evaluate the fitness of `v_i`.
     - If the fitness of `v_i` is better than `x_i`, replace `x_i` with `v_i` and reset its trial counter; otherwise, increment the trial counter.

2. **Onlooker Bees Phase**:
   - Compute selection probabilities for each food source based on their fitness:
     $$
     p_i = \frac{1 / (f_i + \epsilon)}{\sum_{k=1}^{n_{\text{food\_sources}}} 1 / (f_k + \epsilon)}
     $$
     where `f_i` is the fitness of food source `i` and `ε` is a small constant to avoid division by zero.
     
     Food sources will later be selected by onlooker bees, with probability of being picked proportional to 1/fitness, making better sources more likely to be picked.
   - For each food source chosen by an onlooker bee, perform the same steps as in employment phase:
     - Select a food source `i` with probability `p_i`.
     - Select a randomly chosen peer food source `j ≠ i`.
     - Generate a perturbation vector `φ` as in the employed bees phase.
     - Create a new candidate solution and evaluate its fitness.
     - Apply the greedy selection as in the employed bees phase.

3. **Scout Bees Phase**:
   - For each food source `i`:
     - If its trial counter exceeds `limit` (no improvement in fitness for `n_iterations` > `limit`), reinitialize `x_i` randomly within the bounds and reset its trial counter.

4. **Global Best Update**:
   - Identify the food source with the best fitness and update the global best solution and cost accordingly.

5. **History Recording** (if `record_history` is `True`):
   - Append the current food sources, best cost, and best centers to the history records.

### 2.5 Objective Function

The objective function aims to minimize the sum of squared Euclidean distances between each data point and its nearest cluster center. Given a food source vector `x` reshaped into cluster centers `C = {c_1, c_2, ..., c_k}`, the objective function is:

$$
J(C) = \sum_{i=1}^{n} \min_{j \in \{1, ..., k\}} \| x_i - c_j \|^2
$$

where `x_i` is the `i`-th data point.

### 2.6 Final Assignment

After completing all iterations, the best solution `best_solution` is reshaped into cluster centers:

```python
best_centers = best_solution.reshape((n_clusters, n_features))
```


Each data point is then assigned to the nearest cluster center:

```python
distances = cdist(data, best_centers, metric="euclidean")
labels = np.argmin(distances, axis=1)
```


This results in a label vector `labels` indicating the cluster assignment for each data point.

### 2.7 History Recording

If `record_history` is set to `True`, the function returns a `history` dictionary containing:

- `positions`: List of food source positions at each iteration.
- `costs`: List of best cost values at each iteration.
- `best_centers`: Array of best centers at each iteration, with shape `(iters+1, n_clusters, n_features)`.

This allows for analysis of the algorithm's convergence behavior over time.

## 3. ACOR-Based Clustering

**Note:** This implementation of the Ant Colony Optimization for Clustering (ACOR) algorithm in Python was inspired by the Ant Clustering Algorithm developed by Luis M. Rocha, as detailed in the lab exercises provided by the Binghamton University's Computational Ant Systems and Intelligence (CASI) lab. In these exercises, Rocha presents a method where ants interact with data items to form clusters based on proximity and similarity, utilizing concepts like neighborhood perception and probabilistic decision-making to guide the clustering process. This approach emphasizes the use of an archive to store potential solutions, the generation of new solutions through Gaussian sampling influenced by the archive's diversity, and the iterative refinement of solutions based on fitness evaluations. These principles directly influenced the design of this ACOR implementation, which aims to replicate and extend the concepts introduced by Rocha.

For further details on the original Ant Clustering Algorithm, refer to the CASI lab's resources:

- *Ant Clustering Algorithm* by Luis M. Rocha: [https://casci.binghamton.edu/academics/i-bic/lab5/](https://casci.binghamton.edu/academics/i-bic/lab5/) 

### 3.1 Purpose and High-Level Flow

- **Objective:** Employs Ant Colony Optimization (ACO) to identify optimal cluster centers in continuous spaces.
- **Mechanism:** Utilizes a population of artificial ants that iteratively adjust positions based on pheromone trails and heuristic information, guiding them towards optimal clustering solutions.

## 3.2 Function Signature and Parameters

[[PDF] Ant colony optimization for continuous domains - IRIDIA](https://iridia.ulb.ac.be/~mdorigo/Published_papers/All_Dorigo_papers/SocDor2008ejor.pdf?)

```python
def run_acoc(
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

1. **archive_size** (`K`)  
   How many elite solutions you keep after each loop. A small `K` zeroes in on the very best points (high exploitation), while a larger `K` keeps more variety in play (high exploration).

2. **ants** (`m`)  
   The number of new samples (or “ants”) you send out each iteration. More ants give you a denser look at promising regions but cost more to evaluate.

3. **q** (kernel weight factor)  
   Governs how strongly you prefer your top archive members when choosing where to sample next. A low `q` heightens focus on the very best solutions; a higher `q` spreads your attention more evenly across all elites.

4. **xi** (spread scaler)  
   Sets how broadly you explore around each chosen solution. A larger `xi` means wider sampling clouds (more exploration); a smaller `xi` tightens your search around each point (more exploitation).

5. **Candidate generation**  
   At each step, you:
   - Pick one of the `K` archive solutions, with better-ranked ones more likely (controlled by `q`).  
   - Draw a brand-new point by adding Gaussian “jitter” around that solution, where the jitter size depends on `xi` and how far apart your archive points are.  
   - Clip to ensure it stays within bounds, then evaluate and fold back into the archive—keeping the best `K` again.

### 3.3 Initialization

```python
archive = random_uniform(lb, ub, (archive_size, dims))
fitness = evaluate(archive)
```

- **`archive`**: A matrix of size `(archive_size, dims)` initialized randomly within the bounds `lb` and `ub`.
- **`fitness`**: Evaluates the quality of each solution in the archive.

### 3.4 Iterative Updates

At iteration $t$, given archive $T^{(t-1)} = \{\mathbf{x}_1,\dots,\mathbf{x}_K\}$ with corresponding fitnesses $f(\mathbf{x}_l)$:

1. **Sort Archive**  
   - Compute permutation $\pi$ such that  
     $
       f\bigl(\mathbf{x}_{\pi(1)}\bigr)\le f\bigl(\mathbf{x}_{\pi(2)}\bigr)\le\cdots\le f\bigl(\mathbf{x}_{\pi(K)}\bigr).
     $  
   - Reorder: $\mathbf{x}_{(l)} = \mathbf{x}_{\pi(l)},\;\;l=1\dots K$.

2. **Compute Component Weights**  
   - Raw weight for rank $l$:  
     $
       W_l \;=\; \frac{1}{q\,K\sqrt{2\pi}}\;\exp\!\Bigl(-\tfrac{(l-1)^2}{2\,(qK)^2}\Bigr).
     $  
   - Normalize:  
     $
       p_l = \frac{W_l}{\sum_{j=1}^K W_j},\quad \sum_{l=1}^K p_l = 1.
     $

3. **Compute Sampling Spreads**  
   - For each elite $\mathbf{x}_{(l)}$ and each dimension $d$,  
     $
       \sigma_{l,d}
       = \xi \;\frac{1}{K-1}
         \sum_{\substack{j=1 \\ j\neq l}}^{K}
         \bigl|\;x_{(j),d} - x_{(l),d}\bigr|.
     $

4. **Generate $m$ New Ants**  
   For each ant $i=1,\dots,m$:
   1. Draw component index $L_i\sim\text{Categorical}(p_1,\dots,p_K)$.  
   2. Sample each coordinate:  
      $
        s_{i,d}
        \;\sim\;
        \mathcal{N}\bigl(\mu_{L_i,d}=x_{(L_i),d},\;\sigma_{L_i,d}^2\bigr).
      $  
   3. Clip: $s_{i,d}\leftarrow\min\{\max\{s_{i,d},\,\text{lb}_d\},\,\text{ub}_d\}$.

5. **Evaluate New Solutions**  
   - Compute fitness vector $\mathbf{f}_{\text{new}} = [\,f(\mathbf{s}_1),\dots,f(\mathbf{s}_m)\,]$.

6. **Merge & Prune**  
   - Stack  
     $\tilde T = [\,\mathbf{x}_{(1)},\dots,\mathbf{x}_{(K)},\,\mathbf{s}_1,\dots,\mathbf{s}_m\,]$  
     with fitness $\tilde{\mathbf{f}}=[f(\mathbf{x}_{(l)})]_{l=1}^K\|\mathbf{f}_{\text{new}}$.  
   - Sort $\tilde T$ by $\tilde{\mathbf{f}}$ ascending and truncate to the best $K$:
     $
       T^{(t)} = \bigl\{\tilde T_{(1)},\dots,\tilde T_{(K)}\bigr\}.
     $

### 3.5 Final Assignment

```python
best_centers = best_solution.reshape((n_clusters, n_features))
labels = assign_to_nearest(data, best_centers)
```

- **`best_centers`**: The optimal cluster centers obtained from the best solution.
- **`labels`**: Assign each data point to the nearest cluster center by computing the Euclidean distance and selecting the minimum.

## 4. Common Components

- **Cost Function**: typically sum of squared Euclidean distances to nearest centroid.
- **Label Assignment**: `labels[i] = argmin_j ||x_i - center_j||`.
- **History Structures**: optional dictionaries for post-hoc analysis and visualization.

## 5. Summary

All three routines encode cluster centers as continuous vectors and optimize a clustering objective via different bio-inspired metaphors:

- **PSO**: social–cognitive particle dynamics.
- **ABC**: division of labor among employed, onlooker, and scout bees.
- **ACOR**: probabilistic sampling around an elite archive.

Each can be equipped with history recording for convergence analysis and easily adapted to different distance metrics or objective functions.

## Example: PSO for Clustering in 2D Space

### Dataset

Imagine a dataset comprising 100 samples, each with two features. This can be visualized as 100 points scattered in a 2D space.

### Objective

The goal is to partition these data points into a predetermined number of clusters, say $ k = 3 $, by identifying optimal cluster centers that minimize the sum of squared distances between each data point and its nearest cluster center.

### Particle Representation

In PSO, each particle represents a potential solution. For clustering, a particle encodes the positions of all $ k $ cluster centers. Given that each center has two coordinates (due to the 2D nature of the data), a particle's position vector has $ 2 \times k = 6 $ dimensions.

For example, a particle's position vector might look like:


$$
\mathbf{x}_i = [x_{1}^{(1)}, x_{1}^{(2)}, x_{2}^{(1)}, x_{2}^{(2)}, x_{3}^{(1)}, x_{3}^{(2)}]
$$

where $ x_{j}^{(d)} $ denotes the $ d $-th coordinate of the $ j $-th cluster center.

### Optimization Process

1. **Initialization**:
   - A swarm of particles is initialized with random positions within the data space. Each particle's position corresponds to a set of potential cluster centers.

2. **Evaluation**:
   - At each iteration, the clustering objective function evaluates each particle by:
     - Reshaping the particle's position vector into $ k $ cluster centers.
     - Assigning each data point to the nearest cluster center.
     - Calculating the sum of squared distances between data points and their assigned centers.

   The objective function $ J $ for a particle is given by:

   $$
   J = \sum_{i=1}^{N} \min_{j \in \{1, \ldots, k\}} \| \mathbf{x}_i - \mathbf{c}_j \|^2
   $$

   where:
   - $ N $ is the number of data points (here, 100),
   - $ \mathbf{x}_i $ is the $ i $-th data point, and
   - $ \mathbf{c}_j $ is the $ j $-th cluster center.

3. **Update**:
   - Particles adjust their velocities and positions based on their own best-known positions and the swarm's global best-known position, aiming to find better clustering configurations.

4. **Iteration**:
   - Steps 2 and 3 are repeated for a set number of iterations or until convergence.

### Outcome

After optimization, the particle with the best position vector represents the set of cluster centers that most effectively partition the data, according to the objective function. Each data point is then assigned to the nearest of these optimized centers, resulting in the final clustering.

This approach allows PSO to effectively search for cluster configurations that minimize intra-cluster distances, leading to meaningful groupings in the data.

--- 
