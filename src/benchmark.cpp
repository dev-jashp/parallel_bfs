#include "parallel_bfs.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <omp.h>
#include <climits>

class Timer {
    std::chrono::high_resolution_clock::time_point start;
public:
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    double elapsed() const {
        return std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count();
    }
};

struct BenchmarkResult {
    std::string graph_name;
    size_t vertex_count;
    size_t edge_count;
    double avg_time_sec;
    double throughput_mega_edges_sec;
    double speedup;
    size_t reachable_vertices;
};

void run_benchmark(const Graph& g, const std::string& graph_name, 
                  int num_threads, BenchmarkResult& result) {
    std::vector<std::atomic<int>> dist(g.vertex_count());
    
    // Warmup run
    {
        for (auto& d : dist) d.store(INT_MAX);
        ParallelBFS::optimized(g, 0, dist);
    }

    // Main benchmark
    const int runs = 5;
    double total_time = 0;
    size_t reachable = 0;

    for (int i = 0; i < runs; ++i) {
        // Reset distances
        for (auto& d : dist) d.store(INT_MAX);
        
        Timer timer;
        ParallelBFS::optimized(g, 0, dist);
        double elapsed = timer.elapsed();
        total_time += elapsed;

        // Count reachable nodes only on first run
        if (i == 0) {
            reachable = std::count_if(dist.begin(), dist.end(),
                [](const auto& d) { return d.load() != INT_MAX; });
        }
    }

    // Calculate baseline (single-threaded) performance
    double baseline_time = 0;
    if (num_threads > 1) {
        omp_set_num_threads(1);
        for (auto& d : dist) d.store(INT_MAX);
        Timer timer;
        ParallelBFS::optimized(g, 0, dist);
        baseline_time = timer.elapsed();
        omp_set_num_threads(num_threads);
    }

    // Store results
    result.graph_name = graph_name;
    result.vertex_count = g.vertex_count();
    result.edge_count = g.edge_count();
    result.avg_time_sec = total_time / runs;
    result.throughput_mega_edges_sec = (g.edge_count() / (total_time / runs)) / 1e6;
    result.speedup = (num_threads > 1) ? (baseline_time / (total_time / runs)) : 1.0;
    result.reachable_vertices = reachable;
}

void print_results(const std::vector<BenchmarkResult>& results) {
    // Table header
    std::cout << std::left
              << std::setw(20) << "Graph"
              << std::setw(12) << "|V|"
              << std::setw(12) << "|E|"
              << std::setw(15) << "Time (ms)"
              << std::setw(20) << "Throughput (M/s)"
              << std::setw(12) << "Speedup"
              << std::setw(15) << "Reachable"
              << "\n";
    
    // Table rows
    for (const auto& res : results) {
        std::cout << std::setw(20) << res.graph_name
                  << std::setw(12) << res.vertex_count
                  << std::setw(12) << res.edge_count
                  << std::setw(15) << res.avg_time_sec * 1000
                  << std::setw(20) << res.throughput_mega_edges_sec
                  << std::setw(12) << res.speedup
                  << std::setw(15) << res.reachable_vertices << " ("
                  << std::fixed << std::setprecision(1) 
                  << (100.0 * res.reachable_vertices / res.vertex_count) << "%)"
                  << "\n";
    }
}

void save_results_to_csv(const std::vector<BenchmarkResult>& results, 
                        const std::string& filename) {
    std::ofstream out(filename);
    out << "Graph,Vertices,Edges,Time(ms),Throughput(M/s),Speedup,Reachable,Reachable(%)\n";
    for (const auto& res : results) {
        out << res.graph_name << ","
            << res.vertex_count << ","
            << res.edge_count << ","
            << res.avg_time_sec * 1000 << ","
            << res.throughput_mega_edges_sec << ","
            << res.speedup << ","
            << res.reachable_vertices << ","
            << (100.0 * res.reachable_vertices / res.vertex_count) << "\n";
    }
}

void thread_scaling_benchmark(const Graph& g, const std::string& graph_name) {
    const int max_threads = omp_get_max_threads();
    std::vector<BenchmarkResult> scaling_results(max_threads);

    std::cout << "\nThread scaling for " << graph_name 
              << " (|V|=" << g.vertex_count() 
              << ", |E|=" << g.edge_count() << "):\n";

    for (int t = 1; t <= max_threads; ++t) {
        omp_set_num_threads(t);
        run_benchmark(g, graph_name, t, scaling_results[t-1]);
    }

    print_results(scaling_results);
    save_results_to_csv(scaling_results, "scaling_" + graph_name + ".csv");
}

int main(int argc, char** argv) {
    // Set default thread count to max available
    const int num_threads = (argc > 1) ? std::stoi(argv[1]) : omp_get_max_threads();
    omp_set_num_threads(num_threads);
    std::cout << "Running benchmarks with " << num_threads << " threads\n";

    // Generate test graphs
    std::vector<std::pair<std::string, Graph>> test_graphs;
    test_graphs.push_back(std::make_pair("Small Dense", GraphGenerator::random(1000, 0.1, 42)));
    test_graphs.push_back(std::make_pair("Medium Sparse", GraphGenerator::random(10000, 0.01, 42)));
    // test_graphs.push_back(std::make_pair("Large Scale-Free", GraphGenerator::scale_free(100000, 500000, 42)));
    // test_graphs.push_back(std::make_pair("Huge R-MAT", GraphGenerator::rmat(20, 1000000, 0.57, 0.19, 0.19, 42))); // 2^20 vertices

    // Run benchmarks
    std::vector<BenchmarkResult> results;
    for (const auto& graph_pair : test_graphs) {
        const std::string& name = graph_pair.first;
        const Graph& graph = graph_pair.second;
        
        BenchmarkResult res;
        run_benchmark(graph, name, num_threads, res);
        results.push_back(res);
        
        // Additional thread scaling analysis for the large graph
        if (name == "Huge R-MAT") {
            thread_scaling_benchmark(graph, name);
        }
    }

    // Print and save results
    print_results(results);
    save_results_to_csv(results, "bfs_benchmark_results.csv");

    return 0;
}