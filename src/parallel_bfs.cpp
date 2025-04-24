#include "parallel_bfs.h"
#include <iostream>
#include <fstream>
#include <omp.h>
#include <algorithm>
#include <climits>
#include <unordered_map>
#include <queue>
#include <stdexcept>
#include <unordered_set>

// Graph member function implementations
std::vector<int> Graph::neighbors(int u) const {
    if (u < 0 || u >= static_cast<int>(offsets.size() - 1)) {
        throw std::out_of_range("Vertex index out of range");
    }
    return {edges.begin() + offsets[u], edges.begin() + offsets[u+1]};
}

Graph GraphGenerator::from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<std::pair<int, int>> edge_list;
    std::unordered_set<int> unique_vertices;
    int u, v;

    // First pass: collect all unique vertices
    while (file >> u >> v) {
        edge_list.emplace_back(u, v);
        unique_vertices.insert(u);
        unique_vertices.insert(v);
    }

    // Create mapping from original IDs to contiguous IDs (0-based)
    std::unordered_map<int, int> vertex_map;
    std::vector<int> original_ids;
    int current_id = 0;
    for (int vertex : unique_vertices) {
        vertex_map[vertex] = current_id++;
        original_ids.push_back(vertex);
    }

    const size_t V = unique_vertices.size();
    std::vector<int> offsets(V + 1, 0);
    std::vector<int> edges;
    edges.reserve(edge_list.size());

    // Count degrees
    for (const auto& edge : edge_list) {
        offsets[vertex_map[edge.first] + 1]++;
    }

    // Compute offsets (prefix sum)
    for (size_t i = 1; i <= V; ++i) {
        offsets[i] += offsets[i - 1];
    }

    // Sort edges and build adjacency list
    edges.resize(edge_list.size());
    std::vector<int> pos(offsets.begin(), offsets.begin() + V);
    
    for (const auto& edge : edge_list) {
        edges[pos[vertex_map[edge.first]]++] = vertex_map[edge.second];
    }

    return Graph(std::move(offsets), std::move(edges));
}

// Parallel BFS implementations
namespace ParallelBFS {

void optimized(const Graph& g, int source, std::vector<std::atomic<int>>& dist) {
    const size_t V = g.vertex_count();
    
    // Initialize distances
    #pragma omp parallel for
    for (size_t i = 0; i < V; ++i) {
        dist[i].store(INT_MAX);
    }
    dist[source].store(0);
    
    std::vector<int> current_frontier = {source};
    size_t total_visited = 1;
    
    while (!current_frontier.empty()) {
        std::vector<int> next_frontier;
        next_frontier.reserve(current_frontier.size() * 2);
        
        #pragma omp parallel
        {
            std::vector<int> private_next;
            #pragma omp for nowait
            for (size_t i = 0; i < current_frontier.size(); ++i) {
                int u = current_frontier[i];
                int current_dist = dist[u].load();
                
                for (int v : g.neighbors(u)) {
                    int expected = INT_MAX;
                    if (dist[v].compare_exchange_strong(expected, current_dist + 1)) {
                        private_next.push_back(v);
                        #pragma omp atomic
                        total_visited++;
                    }
                }
            }
            
            #pragma omp critical
            next_frontier.insert(next_frontier.end(), private_next.begin(), private_next.end());
        }
        
        std::cout << "Frontier: " << current_frontier.size() 
                  << " vertices | Total visited: " << total_visited << "\n";
        current_frontier = std::move(next_frontier);
    }
    
    std::cout << "BFS completed. Total vertices visited: " << total_visited << "\n";
}

void baseline(const Graph& g, int source, std::vector<std::atomic<int>>& dist) {
    for (auto& d : dist) d.store(INT_MAX);
    dist[source].store(0);
    
    std::queue<int> q;
    q.push(source);
    
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        
        for (int v : g.neighbors(u)) {
            int expected = INT_MAX;
            if (dist[v].compare_exchange_strong(expected, dist[u].load() + 1)) {
                q.push(v);
            }
        }
    }
}

bool validate_result(const Graph& g, int source, const std::vector<std::atomic<int>>& dist) {
    std::vector<std::atomic<int>> reference(dist.size());
    baseline(g, source, reference);
    
    for (size_t i = 0; i < dist.size(); ++i) {
        if (dist[i].load() != reference[i].load()) {
            std::cerr << "Validation failed at vertex " << i 
                      << ": expected " << reference[i].load() 
                      << ", got " << dist[i].load() << "\n";
            return false;
        }
    }
    return true;
}

std::vector<int> get_distances(const std::vector<std::atomic<int>>& dist) {
    std::vector<int> result(dist.size());
    for (size_t i = 0; i < dist.size(); ++i) {
        result[i] = dist[i].load();
    }
    return result;
}

void validate_graph_structure(const Graph& g) {
    std::cout << "\nGraph Structure Validation:\n";
    size_t total_edges = 0;
    size_t min_edges = SIZE_MAX;
    size_t max_edges = 0;
    size_t isolated_vertices = 0;

    for (size_t i = 0; i < g.vertex_count(); i++) {
        size_t degree = g.neighbors(i).size();
        total_edges += degree;
        min_edges = std::min(min_edges, degree);
        max_edges = std::max(max_edges, degree);
        if (degree == 0) isolated_vertices++;
    }

    std::cout << "Total edges (directed count): " << total_edges << "\n";
    std::cout << "Min degree: " << min_edges << "\n";
    std::cout << "Max degree: " << max_edges << "\n";
    std::cout << "Isolated vertices: " << isolated_vertices << "\n";
    std::cout << "Average degree: " << static_cast<double>(total_edges) / g.vertex_count() << "\n";

    // Verify edge targets are valid
    size_t invalid_edges = 0;
    for (size_t u = 0; u < std::min(g.vertex_count(), static_cast<size_t>(1000)); u++) {
        for (int v : g.neighbors(u)) {
            if (v < 0 || v >= static_cast<int>(g.vertex_count())) {
                invalid_edges++;
            }
        }
    }
    std::cout << "Invalid edge targets found: " << invalid_edges << "\n";
}

} // namespace ParallelBFS