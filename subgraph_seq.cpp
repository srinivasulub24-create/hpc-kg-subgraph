// Sequential baseline: k-hop knowledge-graph subgraph extraction (agentic RAG)
// Graph file : first line "N M", then M lines "u v" (0-based, undirected)
// Query file : one query per line, whitespace-separated seed node ids
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/resource.h>

using namespace std;
using clk = chrono::steady_clock;
typedef uint32_t u32;
typedef uint64_t u64;

static inline double secs(clk::time_point a, clk::time_point b) {
    return chrono::duration<double>(b - a).count();
}

struct CSR {
    u32 n = 0;
    vector<u64> off;  // n+1
    vector<u32> adj;  // sorted, de-duplicated neighbours, symmetric
};

static bool load_edges(const string& path, u32& n, vector<pair<u32, u32>>& edges) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;
    unsigned long long N, M;
    if (fscanf(f, "%llu %llu", &N, &M) != 2) return false;
    n = (u32)N;
    edges.reserve(M);
    unsigned long long a, b;
    while (fscanf(f, "%llu %llu", &a, &b) == 2) edges.emplace_back((u32)a, (u32)b);
    fclose(f);
    return true;
}

static void build_csr(u32 n, const vector<pair<u32, u32>>& edges, CSR& g) {
    g.n = n;
    vector<u64> deg(n + 1, 0);
    for (auto& e : edges)
        if (e.first != e.second) { deg[e.first]++; deg[e.second]++; }
    vector<u64> off(n + 1, 0);
    for (u32 i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    vector<u32> adj(off[n]);
    vector<u64> pos(off.begin(), off.end() - 1);
    for (auto& e : edges)
        if (e.first != e.second) { adj[pos[e.first]++] = e.second; adj[pos[e.second]++] = e.first; }
    g.off.assign(n + 1, 0);
    g.adj.clear();
    g.adj.reserve(adj.size());
    for (u32 i = 0; i < n; i++) {
        auto b = adj.begin() + off[i], e = adj.begin() + off[i + 1];
        sort(b, e);
        e = unique(b, e);
        g.adj.insert(g.adj.end(), b, e);
        g.off[i + 1] = g.adj.size();
    }
}

// stamp[v] == qid  <=>  v belongs to the current query's subgraph (no O(N) reset)
__attribute__((noinline)) static void bfs_khop(const CSR& g, const vector<u32>& seeds, int hops,
                                               vector<u32>& stamp, u32 qid, vector<u32>& nodes,
                                               u64& scanned) {
    nodes.clear();
    vector<u32> frontier, next;
    for (u32 s : seeds)
        if (s < g.n && stamp[s] != qid) { stamp[s] = qid; nodes.push_back(s); frontier.push_back(s); }
    for (int h = 0; h < hops && !frontier.empty(); h++) {
        next.clear();
        for (u32 u : frontier) {
            for (u64 e = g.off[u]; e < g.off[u + 1]; e++) {
                u32 v = g.adj[e];
                scanned++;
                if (stamp[v] != qid) { stamp[v] = qid; nodes.push_back(v); next.push_back(v); }
            }
        }
        frontier.swap(next);
    }
}

__attribute__((noinline)) static void extract_induced(const CSR& g, const vector<u32>& nodes,
                                                      const vector<u32>& stamp, u32 qid,
                                                      vector<pair<u32, u32>>& edges, u64& scanned) {
    edges.clear();
    for (u32 u : nodes)
        for (u64 e = g.off[u]; e < g.off[u + 1]; e++) {
            u32 v = g.adj[e];
            scanned++;
            if (u < v && stamp[v] == qid) edges.emplace_back(u, v);
        }
}

static inline u64 mix(u64 h, u64 x) {
    h ^= x;
    return h * 1099511628211ULL;
}

__attribute__((noinline)) static u64 finalize(vector<u32>& nodes, vector<pair<u32, u32>>& edges) {
    sort(nodes.begin(), nodes.end());
    sort(edges.begin(), edges.end());
    u64 h = 1469598103934665603ULL;
    for (u32 v : nodes) h = mix(h, v);
    for (auto& e : edges) h = mix(mix(h, e.first), e.second);
    return h;
}

int main(int argc, char** argv) {
    string gpath, qpath, dump, out;
    int hops = 2, repeat = 1;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        auto nxt = [&]() { return string(argv[++i]); };
        if (a == "--graph") gpath = nxt();
        else if (a == "--queries") qpath = nxt();
        else if (a == "--hops") hops = stoi(nxt());
        else if (a == "--repeat") repeat = stoi(nxt());
        else if (a == "--dump") dump = nxt();
        else if (a == "--out") out = nxt();
    }
    if (gpath.empty() || qpath.empty()) {
        fprintf(stderr, "usage: %s --graph g.txt --queries q.txt [--hops 2] [--repeat 1] [--dump f.tsv] [--out f.json]\n", argv[0]);
        return 1;
    }

    auto t0 = clk::now();
    u32 n;
    vector<pair<u32, u32>> edges;
    if (!load_edges(gpath, n, edges)) { fprintf(stderr, "cannot read graph\n"); return 1; }
    auto t1 = clk::now();
    CSR g;
    build_csr(n, edges, g);
    vector<pair<u32, u32>>().swap(edges);
    auto t2 = clk::now();

    vector<vector<u32>> queries;
    {
        ifstream qf(qpath);
        string line;
        while (getline(qf, line)) {
            istringstream ss(line);
            vector<u32> q;
            unsigned long long x;
            while (ss >> x) q.push_back((u32)x);
            if (!q.empty()) queries.push_back(q);
        }
    }

    vector<u32> stamp(g.n, 0), nodes;
    vector<pair<u32, u32>> sub_edges;
    vector<double> lat;
    double t_bfs = 0, t_ind = 0, t_fin = 0;
    u64 scan_bfs = 0, scan_ind = 0, tot_nodes = 0, tot_edges = 0, global_ck = 0;
    vector<array<u64, 3>> rows;
    u32 qid = 0;

    auto tq0 = clk::now();
    for (int r = 0; r < repeat; r++) {
        for (size_t qi = 0; qi < queries.size(); qi++) {
            ++qid;
            auto a = clk::now();
            bfs_khop(g, queries[qi], hops, stamp, qid, nodes, scan_bfs);
            auto b = clk::now();
            extract_induced(g, nodes, stamp, qid, sub_edges, scan_ind);
            auto c = clk::now();
            u64 ck = finalize(nodes, sub_edges);
            auto d = clk::now();
            t_bfs += secs(a, b); t_ind += secs(b, c); t_fin += secs(c, d);
            lat.push_back(secs(a, d));
            tot_nodes += nodes.size(); tot_edges += sub_edges.size();
            global_ck ^= ck + qi;
            if (r == 0) rows.push_back({nodes.size(), sub_edges.size(), ck});
        }
    }
    double total_q = secs(tq0, clk::now());

    if (!dump.empty()) {
        FILE* f = fopen(dump.c_str(), "w");
        for (size_t i = 0; i < rows.size(); i++)
            fprintf(f, "%zu\t%llu\t%llu\t%llu\n", i, (unsigned long long)rows[i][0],
                    (unsigned long long)rows[i][1], (unsigned long long)rows[i][2]);
        fclose(f);
    }

    sort(lat.begin(), lat.end());
    auto pct = [&](double p) { return lat[min(lat.size() - 1, (size_t)(p * lat.size()))] * 1e3; };
    double mean_ms = 0;
    for (double x : lat) mean_ms += x;
    mean_ms = mean_ms / lat.size() * 1e3;
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    u64 scanned = scan_bfs + scan_ind;
    size_t nq = lat.size();

    char buf[2048];
    snprintf(buf, sizeof buf,
        "{\n"
        "  \"impl\": \"sequential_cpp\", \"hops\": %d, \"queries\": %zu, \"repeat\": %d,\n"
        "  \"graph\": {\"nodes\": %u, \"directed_edges_csr\": %zu},\n"
        "  \"setup_s\": {\"load\": %.4f, \"csr_build\": %.4f},\n"
        "  \"phase_s\": {\"bfs_khop\": %.4f, \"extract_induced\": %.4f, \"finalize_sort\": %.4f},\n"
        "  \"phase_pct\": {\"bfs_khop\": %.1f, \"extract_induced\": %.1f, \"finalize_sort\": %.1f},\n"
        "  \"query_total_s\": %.4f,\n"
        "  \"latency_ms\": {\"mean\": %.4f, \"p50\": %.4f, \"p95\": %.4f, \"p99\": %.4f},\n"
        "  \"throughput_qps\": %.1f,\n"
        "  \"edges_scanned\": %llu, \"MTEPS\": %.1f,\n"
        "  \"avg_subgraph\": {\"nodes\": %.1f, \"edges\": %.1f},\n"
        "  \"peak_rss_mb\": %.1f,\n"
        "  \"checksum\": %llu\n}\n",
        hops, nq, repeat, g.n, g.adj.size(), secs(t0, t1), secs(t1, t2), t_bfs, t_ind, t_fin,
        100 * t_bfs / (t_bfs + t_ind + t_fin), 100 * t_ind / (t_bfs + t_ind + t_fin),
        100 * t_fin / (t_bfs + t_ind + t_fin), total_q, mean_ms, pct(0.5), pct(0.95), pct(0.99),
        nq / total_q, (unsigned long long)scanned, scanned / total_q / 1e6,
        (double)tot_nodes / nq, (double)tot_edges / nq, ru.ru_maxrss / 1024.0,
        (unsigned long long)global_ck);
    fputs(buf, stdout);
    if (!out.empty()) { FILE* f = fopen(out.c_str(), "w"); fputs(buf, f); fclose(f); }
    return 0;
}
