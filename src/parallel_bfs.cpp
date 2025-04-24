#include "parallel_bfs.h"

#include <iostream>
#include <fstream>
#include <omp.h>
#include <algorithm>
#include <climits>    // For INT_MAX
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <mutex>

// ——— Graph member function implementations ———
std::vector<int> Graph::neighbors(int u) const {
    if (u < 0 || u >= static_cast<int>(offsets.size() - 1)) {
        throw std::out_of_range("Vertex index out of range");
    }
    return { edges.begin() + offsets[u],
             edges.begin() + offsets[u+1] };
}

// ——— GraphGenerator::from_file ———
Graph GraphGenerator::from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<std::pair<int,int>> edge_list;
    std::unordered_set<int> unique_vertices;
    int u, v;
    while (file >> u >> v) {
        edge_list.emplace_back(u, v);
        unique_vertices.insert(u);
        unique_vertices.insert(v);
    }

    // map original IDs → [0..V)
    std::unordered_map<int,int> vertex_map;
    vertex_map.reserve(unique_vertices.size());
    int next_id = 0;
    for (int w : unique_vertices) {
        vertex_map[w] = next_id++;
    }

    size_t V = unique_vertices.size();
    std::vector<int> offsets(V+1, 0);
    std::vector<int> edges;
    edges.reserve(edge_list.size());

    // count degrees
    for (auto &e : edge_list) {
        offsets[ vertex_map[e.first] + 1 ]++;
    }
    // prefix‐sum
    for (size_t i = 1; i <= V; ++i) {
        offsets[i] += offsets[i-1];
    }

    // fill edges[]
    std::vector<int> cursor(offsets.begin(), offsets.begin()+V);
    edges.resize(edge_list.size());
    for (auto &e : edge_list) {
        int from = vertex_map[e.first];
        int to   = vertex_map[e.second];
        edges[ cursor[from]++ ] = to;
    }

    return Graph(std::move(offsets), std::move(edges));
}

// ——— Single‐source “optimized” (top‐down ↔ bottom‐up) ———
// (unchanged from before)
namespace ParallelBFS {
void optimized(const Graph& g, int source, std::vector<std::atomic<int>>& dist) {
    // … your existing optimized() implementation …
}

// ——— Baseline (serial) ———
void baseline(const Graph& g, int source, std::vector<std::atomic<int>>& dist) {
    // … unchanged …
}

// ——— Multi‐source queue‐based (optional) ———
void optimized_multi_source(const Graph& g, std::vector<std::atomic<int>>& dist) {
    // … unchanged …
}

// ——— Frontier‐based hybrid multi‐source ———
void optimized_hybrid(const Graph& g, std::vector<std::atomic<int>>& dist) {
    const size_t V = g.vertex_count();
    const float alpha = g.avg_degree;

    // Reset distances
    #pragma omp parallel for
    for (size_t i = 0; i < V; ++i)
        dist[i].store(INT_MAX);

    // Build initial frontier = all non‐isolated vertices
    std::vector<int> frontier;
    frontier.reserve(V);
    for (size_t u = 0; u < V; ++u) {
        if (g.offsets[u] < g.offsets[u+1]) {
            dist[u].store(0);
            frontier.push_back((int)u);
        }
    }
    size_t total_visited = frontier.size();

    std::vector<int> remainder;
    bool remainder_initialized = false;
    int iteration = 0;

    while (!frontier.empty()) {
        std::vector<int> next;
        size_t work_est = frontier.size() * alpha;
        bool use_bottom_up = false;
        if (remainder_initialized) {
            use_bottom_up = (work_est > remainder.size()) ||
                            (iteration > 10 && frontier.size() < 100);
        }

        if (use_bottom_up) {
            // BOTTOM‐UP: scan remainder
            next.reserve(remainder.size());
            #pragma omp parallel
            {
                std::vector<int> priv;
                #pragma omp for schedule(dynamic,32) nowait
                for (size_t i = 0; i < remainder.size(); ++i) {
                    int u = remainder[i];
                    for (int v : g.neighbors(u)) {
                        if (dist[v].load() != INT_MAX) {
                            int nd = dist[v].load() + 1;
                            int exp = INT_MAX;
                            if (dist[u].compare_exchange_strong(exp, nd)) {
                                priv.push_back(u);
                                break;
                            }
                        }
                    }
                }
                #pragma omp critical
                next.insert(next.end(), priv.begin(), priv.end());
            }
            // rebuild remainder
            std::vector<int> new_rem;
            new_rem.reserve(remainder.size());
            for (int u : remainder)
                if (dist[u].load() == INT_MAX)
                    new_rem.push_back(u);
            remainder.swap(new_rem);

        } else {
            // TOP‐DOWN: expand frontier
            if (!remainder_initialized && work_est > V/4) {
                remainder.reserve(V - frontier.size());
                #pragma omp parallel for schedule(static)
                for (size_t u = 0; u < V; ++u)
                    if (dist[u].load() == INT_MAX)
                        #pragma omp critical
                        remainder.push_back((int)u);
                remainder_initialized = true;
            }

            next.reserve(work_est);
            #pragma omp parallel
            {
                std::vector<int> priv;
                #pragma omp for schedule(static,4) nowait
                for (size_t i = 0; i < frontier.size(); ++i) {
                    int u = frontier[i];
                    int d = dist[u].load();
                    for (int v : g.neighbors(u)) {
                        int exp = INT_MAX;
                        if (dist[v].compare_exchange_strong(exp, d+1))
                            priv.push_back(v);
                    }
                }
                #pragma omp critical
                next.insert(next.end(), priv.begin(), priv.end());
            }
        }

        // dedupe + count
        if (!next.empty()) {
            std::sort(next.begin(), next.end());
            next.erase(std::unique(next.begin(), next.end()), next.end());
            total_visited += next.size();
        }
        frontier.swap(next);
        iteration++;

        if (iteration % 10 == 0) {
            std::cout << "Iteration " << iteration
                      << ": Mode=" << (use_bottom_up ? "BOTTOM-UP" : "TOP-DOWN")
                      << ", Frontier=" << frontier.size()
                      << ", Remainder=" << remainder.size()
                      << ", Visited=" << total_visited << "\n";
        }
    }

    std::cout << "BFS completed in " << iteration
              << " iterations. Total vertices visited: "
              << total_visited << "\n";
}

} // namespace ParallelBFS
