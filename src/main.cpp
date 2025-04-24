#include "parallel_bfs.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <string>
#include <climits>  // Add this for INT_MAX

void print_usage() {
    std::cout << "Usage: ./parallel_bfs [vertices=1000] [density=0.01] [seed=42]\n"
              << "Safe test examples:\n"
              << "  ./parallel_bfs 100 0.1      # Tiny test (100 vertices, 10% density)\n"
              << "  ./parallel_bfs 1000 0.01    # Small test (default)\n"
              << "  ./parallel_bfs 10000 0.001  # Medium test\n"
              << "  ./parallel_bfs 100000 0.0001 # Large test\n"
              << "Original test (1M vertices):\n"
              << "  ./parallel_bfs 1000000 0.0001\n";
}

int main(int argc, char* argv[]) {
    // Default safe values
    size_t V = 1000;
    float density = 0.01f;
    unsigned seed = 42;
    std::string graph_file;
    bool from_file = false;

    // Parse command-line arguments
    if (argc > 1) {
        if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
            print_usage();
            return 0;
        }
        
        // Check if argument is a file (ends with .txt)
        std::string first_arg = argv[1];
        if (first_arg.size() >= 4 && first_arg.substr(first_arg.size() - 4) == ".txt") {
            graph_file = first_arg;
            from_file = true;
        } else {
            try {
                V = std::stoul(argv[1]);
                if (argc > 2) density = std::stof(argv[2]);
                if (argc > 3) seed = std::stoul(argv[3]);
            } catch (...) {
                std::cerr << "Invalid arguments!\n";
                print_usage();
                return 1;
            }
        }
    }

    try {
        // Initialize graph based on input
        Graph g = from_file ? GraphGenerator::from_file(graph_file) 
                           : GraphGenerator::random(V, density, seed);

        // Safety check for synthetic graph
        if (!from_file && V > 10000) {
            std::cout << "Warning: Large graph size (" << V << " vertices). Continue? (y/n): ";
            char response;
            std::cin >> response;
            if (response != 'y' && response != 'Y') return 0;
        }
        
        if (!from_file) {
            std::cout << "Generating synthetic graph with:\n"
                      << "  Vertices: " << V << "\n"
                      << "  Density:  " << density << "\n"
                      << "  Seed:     " << seed << "\n";
        }

        std::cout << "Graph stats:\n"
                  << "  Vertices: " << g.vertex_count() << "\n"
                  << "  Edges:    " << g.edge_count() << "\n"
                  << "  Avg deg:  " << g.avg_degree << "\n";

        // Prepare distance array
        std::vector<std::atomic<int>> dist(g.vertex_count());
        
        // Initialize distances
        for (auto& d : dist) {
            d.store(INT_MAX);
        }

        // Find the first vertex with outgoing edges as source
        int source = 0;
        for (size_t i = 0; i < g.vertex_count(); ++i) {
            if (!g.neighbors(i).empty()) {
                source = i;
                break;
            }
        }

        std::cout << "Running BFS from source vertex \n";
        auto start = std::chrono::high_resolution_clock::now();
        ParallelBFS::optimized(g, 0, dist);
        auto end = std::chrono::high_resolution_clock::now();

        // Results
        std::chrono::duration<double> elapsed = end - start;
        size_t reachable = std::count_if(dist.begin(), dist.end(),
            [](const std::atomic<int>& d) { return d.load() != INT_MAX; });

        std::cout << "\nResults:\n"
                  << "  Time:      " << elapsed.count() << " s\n"
                  << "  Throughput: " << (g.edge_count() / elapsed.count() / 1e6) << " M edges/s\n"
                  << "  Reachable: " << reachable << "/" << g.vertex_count() << " vertices\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}