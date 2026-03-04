EGME Experiments Package
=======================

This package contains:
1) Code used to generate the simulation datasets and graph instances.
2) CSV datasets (entropy grids, discrimination demo, tree robustness).
3) Graph edge-lists used/produced during experiments.

Folders
-------
- code/egme_experiments.py
  Reproducible script that regenerates all CSVs and graphs (uses fixed random seed by default).

- data/
  * entropy_comparison_grid.csv
  * discrimination_demo_summary.csv
  * discrimination_demo_per_node.csv
  * tree_topology_robustness.csv
  * random_tree_samples_meta.csv

- graphs/
  * K20_edgelist.tsv
  * tree_{Family}_n{n}.tsv for deterministic families
  * tree_RandomTree_n{n}_sampleXX_seedYYYY.tsv for random trees

Definitions
-----------
- Cost-induced probability: p(v) = c_v / sum_u c_u.
- Distance cost: sum of shortest-path lengths from v to all nodes.
- Edge cost proxy:
    - default: alpha * degree(v)
    - discrimination demo: alpha * buyers[v] (ownership proxy)

Reproducibility
---------------
Run:
    python code/egme_experiments.py

Outputs will be written under ./outputs (relative to where you run it).
