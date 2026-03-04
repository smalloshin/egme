# EGME Experiments (Reproducible Script)
# -------------------------------------
# This script reproduces:
#   (A) Entropy comparison grid across (n, alpha)
#   (B) Identical-structure discrimination demo on K20 with different ownership patterns
#   (C) Tree-topology robustness across multiple tree families (star/path/balanced/broom/random)
#
# Notes:
# - Distance cost: sum of shortest path lengths from v to all nodes.
# - Edge cost proxy: alpha * degree(v) unless an explicit 'buyers' dict is provided.
# - Discrimination demo: 'buyers' models who pays for edges (ownership proxy).
#
# Run:
#   python egme_experiments.py
#
import os, random
import numpy as np
import pandas as pd
import networkx as nx

def cost_induced_p(G: nx.Graph, alpha: float, buyers=None):
    dist = dict(nx.all_pairs_shortest_path_length(G))
    dist_sums = {u: sum(dist[u].values()) for u in G.nodes()}
    if buyers is None:
        edge_costs = {u: alpha * G.degree[u] for u in G.nodes()}
    else:
        edge_costs = {u: alpha * buyers.get(u, 0) for u in G.nodes()}
    c = {u: edge_costs[u] + dist_sums[u] for u in G.nodes()}
    total = sum(c.values())
    p = np.array([c[u] / total for u in G.nodes()], dtype=float)
    return p, c

def EGME_from_p(p): return float(-np.log2(np.max(p)))

def Dehmer_from_p(p):
    p = np.array(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def degree_entropy(G: nx.Graph):
    degs = np.array([d for _, d in G.degree()], dtype=int)
    _, counts = np.unique(degs, return_counts=True)
    q = counts / counts.sum()
    return float(-np.sum(q * np.log2(q)))

def save_edgelist(G: nx.Graph, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for u, v in G.edges():
            f.write(f"{u}\t{v}\n")

def tree_star(n): return nx.star_graph(n-1)
def tree_path(n): return nx.path_graph(n)

def tree_balanced_binary(n):
    r, h = 2, 1
    while True:
        T = nx.balanced_tree(r, h)
        if T.number_of_nodes() >= n:
            break
        h += 1
    bfs = list(nx.bfs_tree(T, 0).nodes())
    keep = bfs[:n]
    return T.subgraph(keep).copy()

def tree_broom(n, handle_frac=0.4):
    handle = max(2, int(round(n * handle_frac)))
    brush = n - handle
    P = nx.path_graph(handle)
    if brush <= 1:
        return P
    S = nx.star_graph(brush)
    S = nx.relabel_nodes(S, {i: i+handle for i in S.nodes()})
    G = nx.disjoint_union(P, S)
    G.add_edge(handle-1, handle)
    return G

def run_all(out_dir="./outputs", seed=42):
    os.makedirs(out_dir, exist_ok=True)
    data_dir = os.path.join(out_dir, "data"); os.makedirs(data_dir, exist_ok=True)
    graphs_dir = os.path.join(out_dir, "graphs"); os.makedirs(graphs_dir, exist_ok=True)

    np.random.seed(seed); random.seed(seed)

    # (A) Entropy comparison grid
    node_sizes = [20, 50, 100]
    alpha_values = [0.5, 1, 2, 5, 10]
    R = 30
    rows = []
    for n in node_sizes:
        for alpha in alpha_values:
            egmes, dehmers, degents = [], [], []
            for _ in range(R):
                if alpha < 1:
                    G = nx.complete_graph(n)
                elif 1 <= alpha < 2:
                    G = nx.star_graph(n-1)
                else:
                    G = nx.random_tree(n, seed=int(np.random.randint(0, 10**9)))
                p, _ = cost_induced_p(G, alpha, buyers=None)
                egmes.append(EGME_from_p(p))
                dehmers.append(Dehmer_from_p(p))
                degents.append(degree_entropy(G))
            rows.append({
                "n": n, "alpha": alpha, "repeats": R,
                "EGME_mean": float(np.mean(egmes)), "EGME_std": float(np.std(egmes)),
                "Dehmer_mean": float(np.mean(dehmers)), "Dehmer_std": float(np.std(dehmers)),
                "DegreeEntropy_mean": float(np.mean(degents)), "DegreeEntropy_std": float(np.std(degents)),
            })
    pd.DataFrame(rows).to_csv(os.path.join(data_dir, "entropy_comparison_grid.csv"), index=False)

    # (B) Discrimination demo
    n_demo, alpha_demo = 20, 0.5
    Gk = nx.complete_graph(n_demo)
    buyers_one = {0: n_demo - 1}
    buyers_bal = {u: (n_demo - 1)//2 for u in Gk.nodes()}

    p_one, c_one = cost_induced_p(Gk, alpha_demo, buyers=buyers_one)
    p_bal, c_bal = cost_induced_p(Gk, alpha_demo, buyers=buyers_bal)

    pd.DataFrame([
        {"variant":"K20_one_buyer", "n":n_demo, "alpha":alpha_demo,
         "EGME":EGME_from_p(p_one), "Dehmer":Dehmer_from_p(p_one), "DegreeEntropy":degree_entropy(Gk)},
        {"variant":"K20_balanced_buyers", "n":n_demo, "alpha":alpha_demo,
         "EGME":EGME_from_p(p_bal), "Dehmer":Dehmer_from_p(p_bal), "DegreeEntropy":degree_entropy(Gk)},
    ]).to_csv(os.path.join(data_dir, "discrimination_demo_summary.csv"), index=False)

    pd.DataFrame({
        "node": list(Gk.nodes()),
        "p_one_buyer": p_one,
        "p_balanced_buyers": p_bal,
        "cost_one_buyer": [c_one[u] for u in Gk.nodes()],
        "cost_balanced_buyers": [c_bal[u] for u in Gk.nodes()],
        "buyers_one_buyer": [buyers_one.get(u,0) for u in Gk.nodes()],
        "buyers_balanced": [buyers_bal.get(u,0) for u in Gk.nodes()],
    }).to_csv(os.path.join(data_dir, "discrimination_demo_per_node.csv"), index=False)

    save_edgelist(Gk, os.path.join(graphs_dir, "K20_edgelist.tsv"))

    # (C) Tree-topology robustness
    tree_families = {
        "Star": tree_star,
        "Path": tree_path,
        "BalancedBinary": tree_balanced_binary,
        "Broom": tree_broom,
    }
    ns = [50, 100]
    alphas = [2, 5, 10]
    Rr = 50

    for n in ns:
        for fam, fn in tree_families.items():
            save_edgelist(fn(n), os.path.join(graphs_dir, f"tree_{fam}_n{n}.tsv"))

    meta = []
    for n in ns:
        for i in range(Rr):
            sd = int(np.random.randint(0, 10**9))
            G = nx.random_tree(n, seed=sd)
            fn = f"tree_RandomTree_n{n}_sample{i:02d}_seed{sd}.tsv"
            save_edgelist(G, os.path.join(graphs_dir, fn))
            meta.append({"family":"RandomTree", "n":n, "sample_id":i, "seed":sd, "edgelist":fn})
    pd.DataFrame(meta).to_csv(os.path.join(data_dir, "random_tree_samples_meta.csv"), index=False)

    out = []
    for n in ns:
        for alpha in alphas:
            for fam, fn in tree_families.items():
                G = fn(n)
                p, _ = cost_induced_p(G, alpha, buyers=None)
                out.append({
                    "n": n, "alpha": alpha, "family": fam, "repeats": 1,
                    "EGME_mean": EGME_from_p(p), "EGME_std": 0.0,
                    "Dehmer_mean": Dehmer_from_p(p), "Dehmer_std": 0.0,
                    "DegreeEntropy_mean": degree_entropy(G), "DegreeEntropy_std": 0.0,
                })

            eg_list, de_list, dg_list = [], [], []
            for rec in [r for r in meta if r["n"] == n]:
                G = nx.read_edgelist(os.path.join(graphs_dir, rec["edgelist"]), nodetype=int)
                p, _ = cost_induced_p(G, alpha, buyers=None)
                eg_list.append(EGME_from_p(p))
                de_list.append(Dehmer_from_p(p))
                dg_list.append(degree_entropy(G))

            out.append({
                "n": n, "alpha": alpha, "family": "RandomTree", "repeats": Rr,
                "EGME_mean": float(np.mean(eg_list)), "EGME_std": float(np.std(eg_list)),
                "Dehmer_mean": float(np.mean(de_list)), "Dehmer_std": float(np.std(de_list)),
                "DegreeEntropy_mean": float(np.mean(dg_list)), "DegreeEntropy_std": float(np.std(dg_list)),
            })

    pd.DataFrame(out).to_csv(os.path.join(data_dir, "tree_topology_robustness.csv"), index=False)
    print("Done. Outputs written to:", out_dir)

if __name__ == "__main__":
    run_all(out_dir="./outputs", seed=42)
