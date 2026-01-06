#include "parallel_bfs.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <string>
#include <climits>   // For INT_MAX
#include <omp.h>     // For OpenMP

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
    // Use all available cores

    // omp_set_num_threads(4);
    omp_set_num_threads(omp_get_max_threads());// maximum number of threads

    // Default safe values
    size_t V = 1000;
    float density = 0.01f;
    unsigned seed = 42;
    std::string graph_file;
    bool from_file = false;

    std::cout << "With Dynamic Switching\n"
              << "========================\n"
              << "Hybrid frontier-based multi-source BFS\n\n";

    // Parse command-line arguments
    if (argc > 1) {
        if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
            print_usage();
            return 0;
        }
        
        // File?
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
        // Build graph
        Graph g = from_file
                  ? GraphGenerator::from_file(graph_file)
                  : GraphGenerator::random(V, density, seed);

        if (!from_file && V > 10000) {
            std::cout << "Warning: Large graph (" << V << " vertices). Continue? (y/n): ";
            char resp; std::cin >> resp;
            if (resp != 'y' && resp != 'Y') return 0;
        }

        std::cout << "Graph stats:\n"
                  << "  Vertices: " << g.vertex_count() << "\n"
                  << "  Edges:    " << g.edge_count()   << "\n"
                  << "  Avg deg:  " << g.avg_degree     << "\n\n";

        // Distance array (will be initialized inside optimized_hybrid)
        std::vector<std::atomic<int>> dist(g.vertex_count());

        std::cout << "Running hybrid frontier-based multi-source BFS\n";
        auto start = std::chrono::high_resolution_clock::now();
        ParallelBFS::optimized_hybrid(g, dist);
        auto end   = std::chrono::high_resolution_clock::now();

        // Count how many got visited
        size_t reachable = 0;
        #pragma omp parallel for reduction(+:reachable)
        for (size_t i = 0; i < dist.size(); ++i)
            if (dist[i].load() != INT_MAX)
                ++reachable;

        double sec = std::chrono::duration<double>(end - start).count();
        std::cout << "\nFinal Results:\n"
                  << "  Time:       " << sec << " s\n"
                  << "  Throughput: " << (g.edge_count() / sec / 1e6) << " M edges/s\n"
                  << "  Reachable:  " << reachable << "/" << g.vertex_count() << " vertices\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
