```python
function RUN_PSO(data, k, iters, N, options):
    // data: points, k: clusters
    // iters: max iterations, N: number of particles

    initialize N particles randomly within bounds
    evaluate fitness for each particle
    set best ← best of particles

    for t = 1 to iters do
        for i = 1 to N do
            if fitness(current[i]) < fitness(personal_best[i]):
                personal_best[i] ← current[i]

        global_best ← best of personal_best

        for i = 1 to N do
            update velocity[i] using inertia, cognitive, and social terms
            update position[i] ← position[i] + velocity[i]

            evaluate fitness of position[i]
            if fitness(position[i]) < fitness(best):
                best ← position[i]

    end for

    centers ← reshape(best, (k, dims))
    labels ← assign each data point to nearest center
```

```python
function RUN_ABC(data, k, iters, N, limit):
    // data: points, k: clusters, iters: iterations
    // N: number of food sources, limit: scout threshold

    initialize N food_sources randomly
    compute fitness for each source
    set best ← best of food_sources

    for t = 1 to iters do
        // Employed bees
        for each source i:
            generate candidate by perturbing source i
            if fitness(candidate) < fitness(i):
                replace source i

        // Onlooker bees
        compute selection probabilities ∝ 1/fitness
        for each source i:
            if random() < prob(i):
                generate & test candidate as above

        // Scout bees
        for each source i with trials > limit:
            reinitialize source i

        update best if any source is better
    end for

    centers ← reshape(best, (k, dims))
    labels ← assign each data point to nearest center
```

```python
function RUN_ACOR(data, k, M, m, q, ξ, iters):
    // data: points, k: clusters
    // M: archive size, m: ants per iter
    // q: kernel weight factor, ξ: std scaling

    archive ← random_solutions(M, k × dims)
    fitness ← evaluate(archive, data, k)

    for t = 1 to iters do
        sort archive by ascending fitness

        for i = 1 to M:
            weight[i] ← gaussian_weight(i, M, q)

        for i = 1 to M:
            σ[i] ← ξ × std_dev(archive, i)

        for a = 1 to m:
            idx ← sample_index(weights)
            sol ← sample_gaussian(archive[idx], σ[idx])
            sol ← clamp(sol, data_min, data_max)
            new_solutions[a] ← sol

        new_fitness ← evaluate(new_solutions, data, k)

        combine archive and new_solutions
        keep best M solutions as new archive
    end for

    best ← archive[1]
    centers ← reshape(best, (k, dims))
    labels ← assign_to_nearest(data, centers)
```