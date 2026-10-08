#!/usr/bin/env bash
# A1 runner: build -> run (1-hop, 2-hop) -> gprof profile -> save everything in results/
# usage: ./run_a1.sh <graph.txt> <queries.txt>
set -e
G="${1:?graph file}"; Q="${2:?queries file}"
mkdir -p results

echo "== system info =="
{
  echo "date: $(date)"
  echo "--- CPU ---";  lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)|Thread|Core|L2|L3|MHz" || true
  echo "--- RAM ---";  free -h 2>/dev/null || true
  echo "--- compiler ---"; g++ --version | head -1
  echo "--- GPU ---";  (nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null) || echo "no NVIDIA GPU / driver visible"
} | tee results/system_info.txt

echo "== build =="
g++ -O3 -std=c++17 subgraph_seq.cpp -o subgraph_seq
g++ -O2 -g -pg -std=c++17 subgraph_seq.cpp -o subgraph_seq_pg

echo "== timing runs (-O3) =="
for H in 1 2; do
  ./subgraph_seq --graph "$G" --queries "$Q" --hops $H --repeat 3 \
      --dump results/seq_hops${H}.tsv --out results/seq_hops${H}.json > /dev/null
  echo "-- hops=$H"
  grep -E "nodes\"|phase_pct|latency_ms|throughput|MTEPS|avg_subgraph|peak_rss" results/seq_hops${H}.json
done

echo "== gprof profile (hops=2) =="
./subgraph_seq_pg --graph "$G" --queries "$Q" --hops 2 --repeat "${PROF_REPEAT:-20}" > /dev/null
gprof -b -p ./subgraph_seq_pg gmon.out 2>/dev/null | head -25 > results/gprof_flat.txt
gprof -b -q ./subgraph_seq_pg gmon.out 2>/dev/null | head -60 > results/gprof_callgraph.txt
cat results/gprof_flat.txt | cut -c1-110

echo
echo "DONE. Files in results/: system_info.txt seq_hops1.json seq_hops2.json gprof_flat.txt gprof_callgraph.txt"
