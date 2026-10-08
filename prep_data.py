#!/usr/bin/env python3
"""Graph + query preparation for the subgraph-extraction project.
Graph file format: line 1 "N M", then M lines "u v" (0-based, undirected).
Commands: synthetic | graphrag | cooccur | queries   (see --help)
"""
import argparse, os, sys
import numpy as np


def write_graph(path, n, edges):
    with open(path, "w") as f:
        f.write(f"{n} {len(edges)}\n")
        np.savetxt(f, edges, fmt="%d")
    print(f"wrote {path}: {n} nodes, {len(edges)} edges")


def cmd_synthetic(a):
    rng = np.random.default_rng(a.seed)
    n, m = a.nodes, int(a.nodes * a.avg_deg / 2)
    # Chung-Lu: heavy-tailed degrees (hubs) -> realistic/hard case for GPU load balance
    w = np.arange(1, n + 1, dtype=np.float64) ** (-a.alpha)
    w /= w.sum()
    perm = rng.permutation(n)
    u = perm[rng.choice(n, size=m, p=w)]
    v = perm[rng.choice(n, size=m, p=w)]
    keep = u != v
    edges = np.stack([u[keep], v[keep]], 1)
    write_graph(a.out, n, edges)


def cmd_graphrag(a):
    import pandas as pd
    cand = ["create_final_relationships.parquet", "relationships.parquet"]
    rel = next((os.path.join(a.dir, c) for c in cand if os.path.exists(os.path.join(a.dir, c))), None)
    if rel is None:
        sys.exit(f"no relationships parquet in {a.dir} (looked for {cand}). "
                 "Run GraphRAG indexing first, or check the dataset repo layout.")
    df = pd.read_parquet(rel)
    names = pd.unique(pd.concat([df["source"], df["target"]]))
    idx = {s: i for i, s in enumerate(names)}
    edges = np.stack([df["source"].map(idx).to_numpy(), df["target"].map(idx).to_numpy()], 1)
    write_graph(a.out, len(names), edges)
    if a.names:
        with open(a.names, "w", encoding="utf-8") as f:
            for s in names:
                f.write(str(s).replace("\n", " ") + "\n")
        print(f"wrote {a.names} (line i = entity name of node i)")


def cmd_cooccur(a):
    import re
    from collections import Counter
    from itertools import combinations
    ent_re = re.compile(r"\b[A-Z][\w'&-]+(?:\s+[A-Z][\w'&-]+){0,3}")
    stop = {"The", "And", "But", "So", "That", "This", "It", "We", "You", "I", "He", "She", "They",
            "What", "When", "Where", "Why", "How", "If", "In", "On", "At", "As", "Yeah", "Well", "Oh",
            "Right", "Okay", "Thanks", "Thank", "Yes", "No", "A", "An", "Of", "To", "For", "My", "Our"}
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(a.dir) for f in fs
                   if f.lower().endswith((".txt", ".md")))
    if not files:
        sys.exit(f"no .txt/.md under {a.dir}")
    windows = []
    for fp in files:
        words = open(fp, encoding="utf-8", errors="ignore").read().split()
        for i in range(0, len(words), a.window):
            windows.append(" ".join(words[i:i + a.window]))
    per = []
    freq = Counter()
    for w in windows:
        ents = {m.strip() for m in ent_re.findall(w)} - stop
        per.append(ents)
        freq.update(ents)
    keep = {e for e, c in freq.items() if c >= a.min_count}
    idx, edges = {}, set()
    for ents in per:
        es = sorted(e for e in ents if e in keep)[: a.max_per_window]
        for e in es:
            idx.setdefault(e, len(idx))
        for x, y in combinations(es, 2):
            i, j = idx[x], idx[y]
            edges.add((min(i, j), max(i, j)))
    write_graph(a.out, len(idx), np.array(sorted(edges), dtype=np.int64).reshape(-1, 2))
    if a.names:
        with open(a.names, "w", encoding="utf-8") as f:
            for e, _ in sorted(idx.items(), key=lambda t: t[1]):
                f.write(e + "\n")
    print(f"{len(files)} files, {len(windows)} windows, {len(idx)} entities")


def cmd_queries(a):
    rng = np.random.default_rng(a.seed)
    with open(a.graph) as f:
        n, m = map(int, f.readline().split())
        e = np.loadtxt(f, dtype=np.int64, max_rows=m) if a.degree_biased else None
    if a.degree_biased:
        deg = np.bincount(e.ravel(), minlength=n).astype(np.float64)
        p = deg / deg.sum()
        seeds = rng.choice(n, size=(a.n, a.seeds), p=p)
    else:
        seeds = rng.integers(0, n, size=(a.n, a.seeds))
    np.savetxt(a.out, seeds, fmt="%d")
    print(f"wrote {a.out}: {a.n} queries x {a.seeds} seeds")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("synthetic"); s.add_argument("--nodes", type=int, required=True)
    s.add_argument("--avg-deg", type=float, default=10); s.add_argument("--alpha", type=float, default=0.8)
    s.add_argument("--seed", type=int, default=1); s.add_argument("--out", required=True)
    s.set_defaults(f=cmd_synthetic)
    g = sp.add_parser("graphrag"); g.add_argument("--dir", required=True)
    g.add_argument("--out", required=True); g.add_argument("--names"); g.set_defaults(f=cmd_graphrag)
    c = sp.add_parser("cooccur"); c.add_argument("--dir", required=True)
    c.add_argument("--out", required=True); c.add_argument("--names")
    c.add_argument("--window", type=int, default=120); c.add_argument("--min-count", type=int, default=3)
    c.add_argument("--max-per-window", type=int, default=25); c.set_defaults(f=cmd_cooccur)
    q = sp.add_parser("queries"); q.add_argument("--graph", required=True)
    q.add_argument("--n", type=int, default=5000); q.add_argument("--seeds", type=int, default=3)
    q.add_argument("--seed", type=int, default=7); q.add_argument("--degree-biased", action="store_true")
    q.add_argument("--out", required=True); q.set_defaults(f=cmd_queries)
    a = ap.parse_args()
    a.f(a)
