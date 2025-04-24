#pragma once
#include <vector>
#include <atomic>
#include <queue>
#include <cstddef>
#include <random>
#include <stdexcept>

// Forward declaration for graph generators
struct Graph;

// Graph generation functions
namespace GraphGenerator {
    Graph random(size_t V, float density, unsigned seed = std::random_device{}());
    Graph random_undirected(size_t V, float density, unsigned seed = std::random_device{}());
    Graph scale_free(size_t V, size_t E, unsigned seed = std::random_device{}());
    Graph rmat(size_t scale, size_t E, float a = 0.57, float b = 0.19, float c = 0.19, unsigned seed = std::random_device{}());
}

struct Graph {
    std::vector<int> offsets;
    std::vector<int> edges;
    const float avg_degree;
    
    Graph(std::vector<int>&& off, std::vector<int>&& e)
        : offsets(std::move(off)), edges(std::move(e)),
          avg_degree(edges.size() / static_cast<float>(std::max<size_t>(1, offsets.size() - 1))) {
        if (offsets.size() < 2) throw std::invalid_argument("Graph must have at least 1 vertex");
    }
    
    // Change this from implementation to declaration only:
    std::vector<int> neighbors(int u) const;
    
    // Keep these inline implementations:
    size_t vertex_count() const noexcept { return offsets.size() - 1; }
    size_t edge_count() const noexcept { return edges.size(); }
    
    bool validate() const {
        if (offsets.empty() || offsets.back() != edges.size()) return false;
        for (int v : edges) {
            if (v < 0 || v >= static_cast<int>(offsets.size() - 1)) return false;
        }
        return true;
    }
};

// Parallel BFS functions
namespace ParallelBFS {
    void optimized(const Graph& g, int source, std::vector<std::atomic<int>>& dist);
    void baseline(const Graph& g, int source, std::vector<std::atomic<int>>& dist);
    
    // Utility functions
    bool validate_result(const Graph& g, int source, const std::vector<std::atomic<int>>& dist);
    std::vector<int> get_distances(const std::vector<std::atomic<int>>& dist);
}

// Implementation of graph generators
namespace GraphGenerator {
    inline Graph random(size_t V, float density, unsigned seed) {
        if (V == 0) throw std::invalid_argument("Graph must have at least 1 vertex");
        if (density < 0 || density > 1) throw std::invalid_argument("Density must be between 0 and 1");

        std::vector<int> offsets(V+1);
        std::vector<int> edges;
        edges.reserve(static_cast<size_t>(V * V * density));

        std::mt19937 gen(seed);
        std::uniform_real_distribution<> dis(0.0, 1.0);

        size_t edge_pos = 0;
        for (size_t u = 0; u < V; ++u) {
            offsets[u] = edge_pos;
            for (size_t v = 0; v < V; ++v) {
                if (u != v && dis(gen) < density) {
                    edges.push_back(static_cast<int>(v));
                    edge_pos++;
                }
            }
        }
        offsets[V] = edge_pos;

        return Graph(std::move(offsets), std::move(edges));
    }

    Graph from_file(const std::string& filename);

}