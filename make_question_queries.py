#!/usr/bin/env python3
"""Seed queries from the dataset's own questions CSV.
python make_question_queries.py --csv Q.csv --graph g.txt --names names.txt --out q.txt"""
import argparse, csv, re
from collections import defaultdict

STOP = set("the a an of to in on and or for with is are was were what how why who which do does did "
           "about from by as at it its that this these those their they them be been has have had can "
           "could would should will not more most than into over between across other each any some "
           "such tech technology leaders guests".split())
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True); ap.add_argument("--graph", required=True)
ap.add_argument("--names", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--seeds", type=int, default=3)
a = ap.parse_args()

names = [l.rstrip("\n") for l in open(a.names, encoding="utf-8")]
with open(a.graph) as f:
    n, m = map(int, f.readline().split())
    deg = [0] * n
    for line in f:
        u, v = map(int, line.split()); deg[u] += 1; deg[v] += 1
index = defaultdict(list)                       # token -> entity ids
for i, nm in enumerate(names):
    for t in set(tok(nm)) - STOP:
        index[t].append(i)

rows = list(csv.DictReader(open(a.csv, encoding="utf-8-sig", errors="ignore")))
col = next((c for c in rows[0] if "question" in c.lower()), list(rows[0])[0])
print(f"{len(rows)} questions, column used: {col!r}")

out, hit = [], 0
for r in rows:
    q = r[col] or ""
    ql, qt = q.lower(), set(tok(q)) - STOP
    score = defaultdict(float)
    for t in qt:
        for i in index.get(t, []):
            score[i] += 1.0 / len(set(tok(names[i])) - STOP or {1})
    for i in list(score):
        if len(names[i]) > 3 and names[i].lower() in ql:
            score[i] += 10                      # full-name match beats partial token match
    best = sorted(score, key=lambda i: (-score[i], -deg[i], i))[: a.seeds]
    if best:
        hit += 1; out.append(best)
with open(a.out, "w") as f:
    for s in out:
        f.write(" ".join(map(str, s)) + "\n")
print(f"{hit}/{len(rows)} questions matched at least one entity -> {a.out}")
print("example:", [names[i] for i in out[0]] if out else "none")
