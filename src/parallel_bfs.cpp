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
#include <mutex> // Include mutex for thread safety

// Graph member function implementations
std::vector<int> Graph::neighbors(int u) const {
    if (u < 0 || u >= static_cast<int>(offsets.size() - 1)) {
        throw std::out_of_range("Vertex index out of range");
    }
    return {edges.begin() + offsets[u], edges.begin() + offsets[u+1]};
}

Graph GraphGenerator::from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Could not open file: " + filename);

    // First pass: count edges and find max vertex ID
    size_t edge_count = 0;
    int max_vertex = 0;
    int u, v;
    while (file >> u >> v) {
        edge_count++;
        max_vertex = std::max({max_vertex, u, v});
    }

    // Second pass: build the graph with memory efficiency
    file.clear();
    file.seekg(0);
    
    std::vector<int> offsets(max_vertex + 2, 0); // +2 for 1-based indexing and sentinel
    std::vector<int> edges;
    edges.reserve(edge_count);

    while (file >> u >> v) {
        offsets[u+1]++; // Count degrees
        edges.push_back(v);
    }

    // Prefix sum to get offsets
    for (size_t i = 1; i < offsets.size(); ++i) {
        offsets[i] += offsets[i-1];
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
    std::atomic<size_t> total_visited{1};
    int iteration = 0;
    
    while (!current_frontier.empty()) {
        std::vector<int> next_frontier;
        next_frontier.reserve(current_frontier.size() * g.avg_degree);
        
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
                        total_visited++; // Increment using atomic<size_t>
                    }
                }
            }
            
            #pragma omp critical
            next_frontier.insert(next_frontier.end(), private_next.begin(), private_next.end());
        }
        
        // Remove duplicates efficiently
        if (!next_frontier.empty()) {
            std::sort(next_frontier.begin(), next_frontier.end());
            next_frontier.erase(std::unique(next_frontier.begin(), next_frontier.end()), next_frontier.end());
        }
        
        current_frontier = std::move(next_frontier);
        iteration++;
        
        // Progress reporting
        if (iteration % 10 == 0) {
            std::cout << "Iteration " << iteration 
                      << ": Frontier=" << current_frontier.size()
                      << ", Visited=" << total_visited << "\n";
        }
    }
    
    std::cout << "BFS completed in " << iteration << " iterations. "
              << "Total vertices visited: " << total_visited << "\n";
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

void optimized_multi_source(const Graph& g, std::vector<std::atomic<int>>& dist) {
    const size_t V = g.vertex_count();
    std::atomic<size_t> total_visited{0};
    
    #pragma omp parallel
    {
        std::vector<int> local_sources;
        
        // First find all potential sources in parallel
        #pragma omp for schedule(static)
        for (size_t i = 0; i < V; ++i) {
            if (dist[i].load() == INT_MAX && !g.neighbors(i).empty()) {
                local_sources.push_back(i);
            }
        }
        
        // Process each potential source
        for (int source : local_sources) {
            int expected = INT_MAX;
            if (dist[source].compare_exchange_strong(expected, 0)) {
                // Local BFS
                std::queue<int> q;
                q.push(source);
                size_t local_visited = 1;
                
                while (!q.empty()) {
                    int u = q.front();
                    q.pop();
                    
                    for (int v : g.neighbors(u)) {
                        int expected = INT_MAX;
                        if (dist[v].compare_exchange_strong(expected, dist[u].load() + 1)) {
                            q.push(v);
                            local_visited++;
                        }
                    }
                }
                
                total_visited.fetch_add(local_visited, std::memory_order_relaxed);
            }
        }
    }
    
    std::cout << "BFS completed. Total vertices visited: " << total_visited.load() << "\n";
}

} // namespace ParallelBFS
