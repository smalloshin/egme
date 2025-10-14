EGME Networks Export
====================
Structure:
  n{20|50|100}_alpha{0.5|1|2|5|10}/
    run_{k}_edges.csv   # u,v,owned_by=min(u,v)
    run_{k}_nodes.csv   # node-level: owned_edges, dist_sum, cost, p
    run_{k}.graphml     # GraphML with attributes; graph attrs alpha, graph_type

Costs: cost_i = alpha*owned_edges_i + sum_j dist(i,j)
Probs: p_i = cost_i / sum_k cost_k
EGME can be recomputed from p: H_shannon - H_min
