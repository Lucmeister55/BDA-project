# SI Algorithm Clustering Benchmark

This repository provides a benchmarking framework for three Swarm Intelligence (SI) clustering algorithms—Particle Swarm Optimization (PSO), Artificial Bee Colony (ABC), and Ant Colony Optimization for Real-valued problems (ACOR)—alongside a baseline K-Means implementation. Each method is evaluated across three distinct datasets, with implementations in `SI.py`, optional hyperparameter tuning, and visualization tools to compare performance.

![Alt Text](results\pbmc10k\pso\pso_animation_center0.gif)

## Datasets

### PBMC 10k

A single-cell RNA sequencing dataset of 10,000 peripheral blood mononuclear cells (PBMCs). We reduce high-dimensional expression profiles to 2D via PCA and then cluster the first 2 components.

- **Ground truth:** None used.
- **Parameter tuning:** Algorithm hyperparameters were tuned via grid search on a held-out subset.  
  **Note:** 2D input enables animated visualization of search dynamics (`record_history` flag).

### Iris Dataset

The iris dataset consists of 150 samples from three species of the Iris flower (Iris setosa, Iris versicolor, and Iris virginica), with four features measured for each sample: sepal length, sepal width, petal length, and petal width, all in centimeters.

- **Ground truth:** Labels are available.

### Speeches Clustering Dataset

A textual dataset of speeches represented as frequencies of 181 different 3/4-grams transformed by Discriminant Correspondence Analysis

- **Ground truth:** Grouped into 4 different categories:
 1. **NAT_Dut** (Native Dutch speeches from the Belgian National Parliament)
 2. **MEP_Dut** (Native Dutch speeches from the European Parliament)
 3. **Fra_Dut** (Translated French speeches from the European Parliament)
 4. **Eng_Dut** (Translated Egnlish speeches from the European Parliament)

- **Parameter tuning:** Manually adjusted but the default values were often fine.
- **Note:** Part of the research on this dataset is to see whether these groups can be differentiated.

## Clustering Methods

### 1. K-Means (Baseline)

**Objective:** Find $k$ centroids $\{\boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_k\}$ that minimize the within-cluster sum of squares:

$$
J(\{\boldsymbol{\mu}_j\}) = \sum_{i=1}^n \min_{1 \le j \le k} \|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2.
$$

**Steps:**

1. **Assignment:**
   $$
   z_i = \underset{1 \le j \le k}{\arg\min}\; \|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2.
   $$
2. **Update:**
   $$
   \boldsymbol{\mu}_j = \frac{1}{|C_j|} \sum_{i:\,z_i=j} \mathbf{x}_i,
   $$
   where $C_j = \{i:\,z_i = j\}$.

**Code reference:** Uses `sklearn.cluster.KMeans`:

```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=k, init='k-means++', max_iter=300)
labels = km.fit_predict(data)
centers = km.cluster_centers_
```

---

### 2. Particle Swarm Optimization (PSO)

**Idea:** Treat each set of $k$ centers as a “particle.” Particles move in the search space, influenced by their own best-known position (pbest) and the swarm’s global best (gbest).

**Updates:**

$$
\begin{aligned}
v_{t+1} &= w\,v_t + c_1\,r_1\,(\mathrm{pbest}_i - x_t^i) + c_2\,r_2\,(\mathrm{gbest} - x_t^i),\\
x_{t+1}^i &= x_t^i + v_{t+1},
\end{aligned}
$$

where:

- $w$ = inertia weight,
- $c_1,c_2$ = cognitive/social coefficients,
- $r_1,r_2 \sim \mathcal{U}(0,1)$.

**Cost function:** Same K-Means objective $J$, implemented in:

```python
def clustering_objective_function(particles, data, n_clusters):
    # particles: shape (n_particles, k * d)
    # returns cost array of length n_particles
```

**Invocation:**

```python
best_cost, best_pos = optimizer.optimize(
    clustering_objective_function,
    iters=iters,
    data=data_tsne,
    n_clusters=n_clusters,
)
```

then reshape `best_pos` → `(k, d)` and assign labels by nearest center.

---

### 3. Artificial Bee Colony (ABC)

**Metaphor:** A colony of bees explores solution “food sources” (candidate centers) through three phases:

1. **Employed bees:** For each source $x_i$, pick another $x_k$ and perturb:
   $$
   x'_{ij} = x_{ij} + \phi_{ij}(x_{ij} - x_{kj}),\quad \phi_{ij} \sim \mathcal{U}(-1,1).
   $$
2. **Onlooker bees:** Choose sources with probability $p_i \propto 1/(f_i+\epsilon)$, and apply the same update.
3. **Scout bees:** Replace any source that hasn’t improved in `limit` trials with a new random point in bounds.

**Implementation:** `run_abc` initializes `food_sources` (shape `(n_food_sources, k*d)`), tracks `fitness`, and loops through the three phases, recording history if `record_history=True`.

---

### 4. Ant Colony Optimization for Real-valued Problems (ACOR)

**Key steps:** Maintain an **archive** of the best $N$ solutions.

1. **Rank** archive by fitness.
2. Compute sampling weights:
   $$
   w_i = \frac{1}{qN\sqrt{2\pi}}\exp\Bigl(-\frac{i^2}{2(qN)^2}\Bigr),\quad i = 0,\dots,N-1,
   $$
   then normalize so $\sum_i w_i = 1$.
3. For each of `ants` new solutions:
   - Sample index $i$ with probability $w_i$.
   - Draw a new solution from $\mathcal{N}(\mu=\mathrm{archive}[i],\,\sigma_i^2)$, with
     $$
     \sigma_i = \xi \times \mathrm{std}(\mathrm{archive} - \mathrm{archive}[i]).
     $$
4. Merge new solutions into the archive, keep top $N$.

**Implementation:** in `run_acor`, with parameters `q`, `xi`, `archive_size`, and `ants`. Records `archive_history` when `record_history=True`.

---

### 5. Evaluation Metrics

#### Internal (no ground truth)

- **Silhouette Coefficient** $s(i) \in [-1, 1]$:

  $$
  s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}},
  $$

  where $a(i)$ = mean intra-cluster distance, $b(i)$ = mean nearest-cluster distance.

- **Davies–Bouldin Index (DB)** (lower is better):

  $$
  \mathrm{DB} = \frac{1}{k}\sum_{i=1}^k \max_{j \neq i} \frac{S_i + S_j}{\|\mu_i - \mu_j\|},
  $$

  with $S_i$ = average distance of points in cluster $i$ to $\mu_i$.

- **Calinski–Harabasz Index (CH)** (higher is better):
  $$
  \mathrm{CH} = \frac{\mathrm{tr}(B)/(k-1)}{\mathrm{tr}(W)/(n-k)},
  $$
  where $B$ = between-cluster dispersion, $W$ = within-cluster dispersion.

Computed by `calculate_clustering_scores(data, labels)`.

#### External (with ground truth)

- **Accuracy** (when labels align exactly).
- **Adjusted Rand Index (ARI)**: corrected-for-chance measure of agreement.

Computed by `evaluate_labels(data, true_labels, predicted_labels)`.

---

Refer to `SI.py` for full implementation and parameter details. Visualization functions enable inspection of algorithm search trajectories.
