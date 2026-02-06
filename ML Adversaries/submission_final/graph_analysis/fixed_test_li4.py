#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seedless, structure-aware de-anonymization with Fused Gromov–Wasserstein (FGW)
- Stage-1: fast global prefilter 
- Stage-2: FGW OT between query vs. top-K gallery candidates
- Fully seedless, structure-only.

Run CLI:
python3 sda_fgw.py \
  --data_dir ~/data \
  --prepared_subdir prepared_dataset_er3_sk3 \
  --scales 5 10 100 1000 10000 \
  --max_nodes 192 \
  --prefilter_k 100 \
  --alpha 0.5 \
  --num_procs 10
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time
import math
import pickle
import random
import hashlib
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter, OrderedDict
from tqdm import tqdm
from multiprocessing import get_context, cpu_count

from numpy.linalg import eigvalsh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

try:
    import ot  # POT: Python Optimal Transport
except ImportError as e:
    print("POT (Python Optimal Transport) is required. Install: pip install POT", file=sys.stderr)
    raise

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ------------------------ Config defaults ------------------------
DATA_DIR = "~/data"
PREP_SUBDIR = "prepared_dataset_er3_sk4"
TOP_M_SHORTLIST = 100
MAX_NODES = 192
FGW_ALPHA = 0.5           # [0,1]: 0 -> pure features, 1 -> pure structure
FGW_LOSS = "square_loss"  # GW loss
N_PROCS = min(10, max(1, cpu_count() - 1))
ADJ_CACHE_DIR = None      # set later from DATA_DIR
LRU_CAP = 1024            # per-worker gallery items

# ------------------------ Utils ------------------------
def ts(msg: str) -> None:
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def gid(rec: Dict) -> str:
    return f"{rec['graph_type']}_{rec['graph_id']}"

def graph_type_from_id(graph_id: str) -> str:
    return graph_id.split('_', 1)[0] if '_' in graph_id else graph_id

def load_qubo(path: str) -> Dict:
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            if "h" in data and "J" in data:
                return data
            if "ising_list" in data and data["ising_list"]:
                L = data["ising_list"]
                if isinstance(L, list) and isinstance(L[0], dict):
                    return L[0]
        return {}
    except Exception:
        return {}

def qubo_to_adj(qubo: Dict, edge_mode: str = "binary") -> np.ndarray:
    """
    Convert Ising-format QUBO to symmetric adjacency matrix.
    edge_mode: 'binary' (1 for nonzero), 'abs' (|J|), 'signed' (J)
    """
    if not qubo or "J" not in qubo:
        return np.zeros((1, 1), dtype=np.float32)

    # infer N from h and J keys
    N = 0
    h = qubo.get("h", None)
    if isinstance(h, dict):
        for k in h.keys():
            try:
                N = max(N, int(k) + 1)
            except: pass
    elif isinstance(h, (list, np.ndarray)):
        N = max(N, len(h))
    J = qubo.get("J", {})
    if isinstance(J, dict):
        for k in J.keys():
            try:
                if isinstance(k, tuple) and len(k) == 2:
                    i, j = int(k[0]), int(k[1])
                elif isinstance(k, str) and ',' in k:
                    i, j = map(int, k.split(','))
                else:
                    continue
                N = max(N, i + 1, j + 1)
            except: continue
    if N <= 0:
        return np.zeros((1,1), dtype=np.float32)

    A = np.zeros((N, N), dtype=np.float32)
    if isinstance(J, dict):
        for key, val in J.items():
            try:
                if isinstance(key, tuple) and len(key) == 2:
                    i, j = int(key[0]), int(key[1])
                elif isinstance(key, str) and ',' in key:
                    i, j = map(int, key.split(','))
                else:
                    continue
                if i == j:
                    continue
                v = float(val)
                if not np.isfinite(v):
                    v = 0.0
                if edge_mode == "binary":
                    w = 1.0 if abs(v) > 0 else 0.0
                elif edge_mode == "abs":
                    w = abs(v)
                else:
                    w = v
                A[i, j] = w
                A[j, i] = w
            except Exception:
                continue
    return A

def cap_nodes(A: np.ndarray, max_nodes: int) -> np.ndarray:
    n = A.shape[0]
    if n <= max_nodes:
        return A
    deg = (A != 0).sum(axis=1).astype(np.int32)
    order = np.argsort(-deg, kind="mergesort")
    idx = np.sort(order[:max_nodes])
    return A[np.ix_(idx, idx)]

# ------------------------ Global sketches (prefilter) ------------------------
def degree_histogram(A_bin: np.ndarray, bins=(0,1,2,3,4,6,8,12,9999)) -> np.ndarray:
    deg = (A_bin != 0).sum(axis=1).astype(np.int32)
    h = np.zeros(len(bins), dtype=np.float32)
    for d in deg:
        for b, ub in enumerate(bins):
            if d <= ub:
                h[b] += 1.0
                break
    s = h.sum()
    if s > 0: h /= s
    return h

def heat_trace_signature(A_bin: np.ndarray, T: int = 16) -> np.ndarray:
    """
    NetLSD-style heat trace: tr(exp(-t L_norm)) at log-spaced times.
    For our node caps (~<=192), a full eigen solve is fine and fast.
    """
    n = A_bin.shape[0]
    d = A_bin.sum(axis=1)
    # normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
    with np.errstate(divide='ignore'):
        inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-9))
    Dm = np.diag(inv_sqrt)
    L = np.eye(n, dtype=np.float64) - (Dm @ A_bin.astype(np.float64) @ Dm)
    evals = np.clip(eigvalsh(L), 0.0, None)  # shape (n,)
    ts = np.logspace(-2, 2, num=T)          # multi-scale
    sig = np.array([np.sum(np.exp(-t * evals)) for t in ts], dtype=np.float64)
    # size-invariant normalization (divide by n)
    return (sig / max(1, n)).astype(np.float32)

def chi2(a: np.ndarray, b: np.ndarray, eps=1e-9) -> float:
    x = a - b
    s = a + b + eps
    return 0.5 * float(np.sum((x * x) / s))

# ------------------------ Node features (per node) ------------------------
_HDIM = 64  # hashed WL one-hot dimension (fixed)

def wl_hash_features(A_bin: np.ndarray, iters: int = 2) -> np.ndarray:
    """
    1–2 iterations of Weisfeiler-Lehman node coloring, hashed to fixed _HDIM dims.
    """
    n = A_bin.shape[0]
    # initial colors = degree
    colors = (A_bin.sum(axis=1)).astype(np.int32).tolist()
    rng_mod = np.uint64(2**64-59)

    def h64(x: str) -> int:
        # stable 64-bit hash -> bucket in [0, _HDIM)
        h = hashlib.blake2b(x.encode('utf-8'), digest_size=8).digest()
        v = int.from_bytes(h, byteorder='little', signed=False)
        return (v % _HDIM)

    X = np.zeros((n, _HDIM), dtype=np.float32)
    for it in range(iters):
        new_colors = []
        for i in range(n):
            neigh = np.nonzero(A_bin[i])[0]
            multiset = ",".join(map(str, sorted(colors[j] for j in neigh)))
            sig = f"{colors[i]}|{multiset}"
            new_colors.append(sig)
        colors = new_colors

    # final one-hot hashed encoding
    for i in range(n):
        b = h64(str(colors[i]))
        X[i, b] = 1.0
    # L2 normalize rows (avoid scale issues)
    rown = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(1e-8, rown)
    return X

def structural_feats(A_bin: np.ndarray) -> np.ndarray:
    """
    Compact structural node features: [deg, 2hop_deg_mean, 2hop_deg_max]
    """
    n = A_bin.shape[0]
    deg = A_bin.sum(axis=1).astype(np.float32)
    # 2-hop via A^2 in binary
    A2 = (A_bin @ A_bin)
    np.fill_diagonal(A2, 0.0)
    # neighbors of neighbors degrees (approx): sum over rows of A2
    twohop = (A2 > 0).sum(axis=1).astype(np.float32)
    # simple stats relative to degree neighborhood
    denom = np.maximum(1.0, deg)
    mean2 = twohop / denom
    max2 = np.maximum(twohop, deg)
    F = np.stack([deg, mean2, max2], axis=1).astype(np.float32)
    # per-graph min-max scale
    mn = F.min(axis=0); mx = F.max(axis=0); sc = np.where((mx - mn) < 1e-6, 1.0, (mx - mn))
    return ((F - mn) / sc).astype(np.float32)

def node_features(A: np.ndarray) -> np.ndarray:
    A_bin = (A != 0).astype(np.float32)
    return np.concatenate([structural_feats(A_bin), wl_hash_features(A_bin, iters=2)], axis=1)

# ------------------------ Structure matrices (per graph) ------------------------
def struct_matrix(A: np.ndarray) -> np.ndarray:
    """
    Unweighted shortest-path distance matrix (float32), finite with large penalty for disconnected.
    """
    A_bin = (A != 0).astype(np.float32)
    G = csr_matrix(A_bin)
    D = shortest_path(G, directed=False, unweighted=True)  # (n,n) float64
    # replace inf by 2*diameter (robust cap)
    fin = np.isfinite(D); ifnot = ~fin
    if np.any(ifnot):
        diam = np.max(D[fin]) if np.any(fin) else 1.0
        D[ifnot] = 2.0 * max(1.0, diam)
    return D.astype(np.float32)

# ------------------------ Persistent cache for gallery ------------------------
def _adj_cache_dir(data_dir: str) -> Path:
    p = Path(data_dir) / PREP_SUBDIR / "_cache" / "fgw_adj"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _key(gpath: str) -> Path:
    h = hashlib.sha1(gpath.encode("utf-8")).hexdigest()[:16]
    return ADJ_CACHE_DIR / f"{h}.pkl"

def build_graph_record_from_qubo(gpath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build per-graph objects once:
      - A_bin, features Y (n x d), structure matrix C (n x n),
      - global sketches: deg-hist (h) and heat trace (s)
    """
    qubo = load_qubo(gpath)
    A = qubo_to_adj(qubo, edge_mode="binary").astype(np.float32)
    if A.size == 0:
        return np.zeros((1,1), dtype=np.float32), np.zeros((1, _HDIM+3), dtype=np.float32), np.ones((1,1), dtype=np.float32), (np.zeros(9, dtype=np.float32), np.zeros(16, dtype=np.float32))
    A = cap_nodes(A, MAX_NODES)
    A_bin = (A != 0).astype(np.float32)
    Y = node_features(A)                        # (n, d)
    C = struct_matrix(A)                        # (n, n)
    h = degree_histogram(A_bin)                 # (9,)
    s = heat_trace_signature(A_bin, T=16)       # (16,)
    return A_bin, Y, C, (h, s)

def load_or_build_cached(gpath: str):
    cpath = _key(gpath)
    if cpath.exists():
        with open(cpath, "rb") as f:
            return pickle.load(f)
    rec = build_graph_record_from_qubo(gpath)
    with open(cpath, "wb") as f:
        pickle.dump(rec, f, protocol=pickle.HIGHEST_PROTOCOL)
    return rec

# ------------------------ LRU cache (per worker) ------------------------
class LRU:
    def __init__(self, cap: int = 1024):
        self.cap = cap
        self.store = OrderedDict()
    def get(self, k):
        v = self.store.get(k)
        if v is not None:
            self.store.move_to_end(k)
        return v
    def put(self, k, v):
        self.store[k] = v
        self.store.move_to_end(k)
        if len(self.store) > self.cap:
            self.store.popitem(last=False)

_worker_cache = None
def _worker_init(cap):
    global _worker_cache
    _worker_cache = LRU(cap)

# ------------------------ Data prep ------------------------
def prepare_data(data_dir: str, n_graphs: int):
    train_file = Path(data_dir) / PREP_SUBDIR / "pairs_option2_train.pkl"
    test_file  = Path(data_dir) / PREP_SUBDIR / "pairs_option2_test.pkl"
    with open(train_file, "rb") as f:
        train_records = pickle.load(f)
    with open(test_file, "rb") as f:
        test_records  = pickle.load(f)

    train_set = {gid(r) for r in train_records}
    overlap_ids = []
    seen=set()
    for r in test_records:
        i = gid(r)
        if i in train_set and i not in seen:
            seen.add(i)
            overlap_ids.append(i)

    # choose gallery train ids (balanced, include all overlaps if n_graphs>overlaps)
    if n_graphs <= len(overlap_ids):
        selected_train_ids = overlap_ids[:n_graphs]
        selected_test_ids = set(selected_train_ids)
    else:
        selected_train_ids = list(overlap_ids)
        have = set(selected_train_ids)
        for r in train_records:
            i = gid(r)
            if i not in have:
                selected_train_ids.append(i)
                have.add(i)
            if len(selected_train_ids) >= n_graphs:
                break
        selected_test_ids = set(overlap_ids)

    id2orig = {gid(r): r["original_path"]   for r in train_records}
    id2e2   = {gid(r): r["obfuscated_path"] for r in test_records}

    gallery_ids   = [i for i in selected_train_ids if i in id2orig]
    gallery_paths = [id2orig[i] for i in gallery_ids]

    query_ids   = [i for i in selected_test_ids if i in id2e2]
    query_paths = [id2e2[i] for i in query_ids]

    per_type = Counter(graph_type_from_id(x) for x in query_ids)
    ts(f"Test queries per type: {dict(sorted(per_type.items()))}")

    return {"gallery_ids": gallery_ids, "gallery_paths": gallery_paths,
            "query_ids": query_ids, "query_paths": query_paths}

# ------------------------ Prefilter scoring ------------------------
def prefilter_scores_for_query(q_sig, gal_sigs):
    """Combine χ²(deg-hist) + cosine(heat-trace) for shortlist."""
    qh, qs = q_sig
    scores = []
    for (gid, (gh, gs)) in gal_sigs.items():
        d = chi2(qh, gh)
        cs = float(np.dot(qs, gs) / (np.linalg.norm(qs) * np.linalg.norm(gs) + 1e-9))
        # lower χ² is better; higher cosine is better -> combine as:
        s = (1.0 - cs) + d  # simple monotone blend
        scores.append((s, gid))
    scores.sort(key=lambda x: x[0])  # smaller is better
    return [g for _, g in scores]

# ------------------------ FGW pair score ------------------------
def fgw_pair_score(Y1, C1, Y2, C2, alpha=FGW_ALPHA) -> float:
    """
    Return negative FGW loss (higher is better).
    POT expects node weights p,q (uniform), feature cost M computed inside.
    """
    n1, n2 = Y1.shape[0], Y2.shape[0]
    p = np.ones(n1) / max(1, n1)
    q = np.ones(n2) / max(1, n2)
    # POT fused_gromov_wasserstein2 returns the FGW loss value
    loss = ot.gromov.fused_gromov_wasserstein2(X=Y1, Y=Y2,
                                               C1=C1, C2=C2,
                                               p=p, q=q,
                                               loss_fun=FGW_LOSS,
                                               alpha=alpha)
    return -float(loss)
def fgw_pair_score(Y1, C1, Y2, C2, alpha=FGW_ALPHA) -> float:
    """
    Return negative FGW loss (higher is better).
    Works across POT versions:
      - legacy: fused_gromov_wasserstein2(M, C1, C2, p, q, ...)
      - newer : fused_gromov_wasserstein2(X=..., Y=..., C1=..., C2=..., ...)
    """
    Y1 = np.asarray(Y1, dtype=np.float64)
    Y2 = np.asarray(Y2, dtype=np.float64)
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)

    n1, n2 = Y1.shape[0], Y2.shape[0]
    p = np.ones(n1, dtype=np.float64) / max(1, n1)
    q = np.ones(n2, dtype=np.float64) / max(1, n2)

    # Feature cost matrix (Euclidean). Square it to match 'square_loss'.
    M = ot.dist(Y1, Y2)          # (n1, n2) Euclidean
    M = (M * M).astype(np.float64)

    try:
        # Legacy POT signature
        loss = ot.gromov.fused_gromov_wasserstein2(
            M, C1, C2, p, q,
            loss_fun=FGW_LOSS, alpha=alpha
            # you can add: verbose=False, tol=1e-6, numItermax=50
        )
    except TypeError:
        # Newer POT signature (if available)
        loss = ot.gromov.fused_gromov_wasserstein2(
            X=Y1, Y=Y2, C1=C1, C2=C2, p=p, q=q,
            loss_fun=FGW_LOSS, alpha=alpha
        )

    return -float(loss)

# ------------------------ Query worker ------------------------
def _score_query(job):
    """Process one query: compute its signature & FGW vs. top-K candidates."""
    (qpath, qid, cand_paths, cand_ids, alpha) = job
    global _worker_cache

    # build query graph objects ad-hoc (no persistent cache for queries)
    qubo_q = load_qubo(qpath)
    A_q = qubo_to_adj(qubo_q, edge_mode="binary").astype(np.float32)
    if A_q.size == 0:
        return (qid, 0)
    A_q = cap_nodes(A_q, MAX_NODES)
    A_q_bin = (A_q != 0).astype(np.float32)
    Yq = node_features(A_q)
    Cq = struct_matrix(A_q)
    q_sig = (degree_histogram(A_q_bin), heat_trace_signature(A_q_bin, 16))

    # shortlist ordering: recompute vs precomputed gallery sigs in parent (passed via cand order)
    sims = np.empty(len(cand_paths), dtype=np.float32)
    for i, gp in enumerate(cand_paths):
        item = _worker_cache.get(gp)
        if item is None:
            # Load from disk cache (build once)
            A_bin, Y, C, _ = load_or_build_cached(gp)
            item = (A_bin, Y, C)
            _worker_cache.put(gp, item)
        else:
            A_bin, Y, C = item
        sims[i] = fgw_pair_score(Yq, Cq, Y, C, alpha=alpha)

    order = np.argsort(-sims)
    ranked_ids = [cand_ids[k] for k in order]
    rank = (ranked_ids.index(qid) + 1) if qid in ranked_ids else 0
    return (qid, rank)

# ------------------------ Evaluation ------------------------
def evaluate_fgw(data: Dict, alpha: float, prefilter_k: int, num_procs: int) -> Dict:
    g_ids, g_paths = data["gallery_ids"], data["gallery_paths"]
    q_ids, q_paths = data["query_ids"], data["query_paths"]

    # Build gallery global sketches (degree-hist + heat trace) with cache
    ts("Preparing gallery cache + global sketches...")
    gal_sigs = {}
    for gid, gp in tqdm(list(zip(g_ids, g_paths)), desc="Gallery precompute"):
        A_bin, Y, C, (h, s) = load_or_build_cached(gp)
        gal_sigs[gid] = (h, s)

    # Prebuild set of all candidates (union of top-K is unknown; we cache lazily per worker)
    # Prepare jobs
    jobs = []
    ts("Preparing Stage-2 jobs (prefilter per query)...")
    for qid, qpath in zip(q_ids, q_paths):
        # Build query sig once (cheap)
        A_q = cap_nodes(qubo_to_adj(load_qubo(qpath), "binary").astype(np.float32), MAX_NODES)
        A_q_bin = (A_q != 0).astype(np.float32)
        q_sig = (degree_histogram(A_q_bin), heat_trace_signature(A_q_bin, 16))
        # Rank gallery by blended prefilter score
        ranked_g = prefilter_scores_for_query(q_sig, gal_sigs)
        cand_ids = ranked_g[:min(prefilter_k, len(ranked_g))]
        idxs = [g_ids.index(g) for g in cand_ids]    # gallery already small; O(G) okay
        cand_paths = [g_paths[i] for i in idxs]
        jobs.append((qpath, qid, cand_paths, cand_ids, alpha))

    # Parallel over queries
    ranks = []
    recall_at = {1: 0, 5: 0, 10: 0}
    ts("Stage-2 FGW (parallel over queries)...")
    with get_context("fork").Pool(
        processes=num_procs,
        initializer=_worker_init, initargs=(LRU_CAP,),
        maxtasksperchild=100
    ) as pool:
        for (qid, rank) in tqdm(pool.imap_unordered(_score_query, jobs, chunksize=1),
                                total=len(jobs), desc="Queries"):
            if rank > 0:
                ranks.append(rank)
                if rank <= 1:  recall_at[1]  += 1
                if rank <= 5:  recall_at[5]  += 1
                if rank <= 10: recall_at[10] += 1

    n = max(1, len(q_ids))
    mrr = np.mean([1.0 / r for r in ranks]) if ranks else 0.0
    return {
        "recall@1": recall_at[1] / n,
        "recall@5": recall_at[5] / n,
        "recall@10": recall_at[10] / n,
        "mrr": mrr,
    }

def evaluate_fgw(data: Dict, alpha: float, prefilter_k: int, num_procs: int) -> Dict:
    g_ids, g_paths = data["gallery_ids"], data["gallery_paths"]
    q_ids, q_paths = data["query_ids"], data["query_paths"]

    # Build gallery global sketches (degree-hist + heat trace) with cache
    ts("Preparing gallery cache + global sketches...")
    gal_sigs = {}
    for gid, gp in tqdm(list(zip(g_ids, g_paths)), desc="Gallery precompute"):
        A_bin, Y, C, (h, s) = load_or_build_cached(gp)
        gal_sigs[gid] = (h, s)

    # Prepare jobs (prefilter per query)
    ts("Preparing Stage-2 jobs (prefilter per query)...")
    gid2idx = {gid: i for i, gid in enumerate(g_ids)}  # speedup
    jobs = []
    for qid, qpath in zip(q_ids, q_paths):
        A_q = cap_nodes(qubo_to_adj(load_qubo(qpath), "binary").astype(np.float32), MAX_NODES)
        A_q_bin = (A_q != 0).astype(np.float32)
        q_sig = (degree_histogram(A_q_bin), heat_trace_signature(A_q_bin, 16))
        ranked_g = prefilter_scores_for_query(q_sig, gal_sigs)
        cand_ids = ranked_g[:min(prefilter_k, len(ranked_g))]
        idxs = [gid2idx[g] for g in cand_ids]
        cand_paths = [g_paths[i] for i in idxs]
        jobs.append((qpath, qid, cand_paths, cand_ids, alpha))

    # Overall + per-type accumulators
    recall_at = {1: 0, 5: 0, 10: 0}
    ranks_all = []
    qtype = {qid: graph_type_from_id(qid) for qid in q_ids}
    from collections import defaultdict
    per_type = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0, "r10": 0, "ranks": []})

    # Parallel over queries
    ts("Stage-2 FGW (parallel over queries)...")
    with get_context("fork").Pool(
        processes=num_procs,
        initializer=_worker_init, initargs=(LRU_CAP,),
        maxtasksperchild=100
    ) as pool:
        for (qid, rank) in tqdm(pool.imap_unordered(_score_query, jobs, chunksize=1),
                                total=len(jobs), desc="Queries"):
            t = qtype[qid]
            per_type[t]["n"] += 1
            if rank > 0:
                ranks_all.append(rank)
                per_type[t]["ranks"].append(rank)
                if rank <= 1:
                    recall_at[1] += 1
                    per_type[t]["r1"] += 1
                if rank <= 5:
                    recall_at[5] += 1
                    per_type[t]["r5"] += 1
                if rank <= 10:
                    recall_at[10] += 1
                    per_type[t]["r10"] += 1

    # Summaries
    n = max(1, len(q_ids))
    mrr = np.mean([1.0 / r for r in ranks_all]) if ranks_all else 0.0
    by_type = {}
    for t, acc in per_type.items():
        n_t = max(1, acc["n"])
        by_type[t] = {
            "n": acc["n"],
            "recall@1": acc["r1"] / n_t,
            "recall@5": acc["r5"] / n_t,
            "recall@10": acc["r10"] / n_t,
            "mrr": (np.mean([1.0 / r for r in acc["ranks"]]) if acc["ranks"] else 0.0),
        }

    return {
        "recall@1": recall_at[1] / n,
        "recall@5": recall_at[5] / n,
        "recall@10": recall_at[10] / n,
        "mrr": mrr,
        "by_type": by_type,
    }

# ------------------------ Main ------------------------
def main():
    global PREP_SUBDIR, MAX_NODES, TOP_M_SHORTLIST, FGW_ALPHA, N_PROCS, ADJ_CACHE_DIR
    
    ap = argparse.ArgumentParser(description="Seedless FGW SDA (structure-aware, fast)")
    ap.add_argument("--data_dir", type=str, default=DATA_DIR)
    ap.add_argument("--prepared_subdir", type=str, default=PREP_SUBDIR)
    ap.add_argument("--scales", type=int, nargs="+", default=[5,10,100,1000,2000,5000,10000])
    ap.add_argument("--max_nodes", type=int, default=MAX_NODES)
    ap.add_argument("--prefilter_k", type=int, default=TOP_M_SHORTLIST)
    ap.add_argument("--alpha", type=float, default=FGW_ALPHA, help="FGW alpha (0=features,1=structure)")
    ap.add_argument("--num_procs", type=int, default=N_PROCS)

    args = ap.parse_args()
    PREP_SUBDIR = args.prepared_subdir
    MAX_NODES   = int(args.max_nodes)
    TOP_M_SHORTLIST = int(args.prefilter_k)
    FGW_ALPHA   = float(args.alpha)
    N_PROCS     = int(args.num_procs)
    ADJ_CACHE_DIR = _adj_cache_dir(args.data_dir)

    print("SEEDLESS STRUCTURE-AWARE SDA — FGW OT (prefilter + parallel Stage-2)")
    print(f"Data dir: {args.data_dir}")
    print(f"Subdir:   {PREP_SUBDIR}")
    print(f"Scales:   {args.scales}")
    print(f"Prefilter top-K: {TOP_M_SHORTLIST}  | alpha={FGW_ALPHA}  | procs={N_PROCS}")
    print(f"Node cap: {MAX_NODES}  | Cache: {ADJ_CACHE_DIR}\n")

    results = {}
    for n in args.scales:
        print("\n" + "-" * 70)
        ts(f"SCALE: {n}")
        data = prepare_data(args.data_dir, n_graphs=n)
        if len(data["gallery_ids"]) == 0 or len(data["query_ids"]) == 0:
            ts("⚠️  Skipping (no usable data).")
            continue

        start = time.time()
        metrics = evaluate_fgw(data, alpha=FGW_ALPHA, prefilter_k=TOP_M_SHORTLIST, num_procs=N_PROCS)
        elapsed = time.time() - start

        Gsz = max(1, len(data["gallery_ids"]))
        rand_r5 = min(5 / Gsz, 1.0)
        vs_rand = metrics["recall@5"] / rand_r5 if rand_r5 > 0 else 0.0

        print("Per-type:")
        for t, mt in sorted(metrics["by_type"].items()):
            print(f"  {t:>6s}  n={mt['n']:3d}  "
                f"R@1={mt['recall@1']:.3f}  R@5={mt['recall@5']:.3f}  "
                f"R@10={mt['recall@10']:.3f}  MRR={mt['mrr']:.3f}")

        print(f"Results: R@1={metrics['recall@1']:.3f}  R@5={metrics['recall@5']:.3f}  "
              f"R@10={metrics['recall@10']:.3f}  MRR={metrics['mrr']:.3f}  "
              f"vs_random={vs_rand:.2f}x  time={elapsed/60:.1f} min")


        results[n] = metrics

    print("\n" + "=" * 80)
    ts("SUMMARY")
    for n in args.scales:
        if n in results:
            m = results[n]
            print(f"N={n:5d}  R@5={m['recall@5']:.3f}  MRR={m['mrr']:.3f}")

if __name__ == "__main__":
    sys.exit(main())
