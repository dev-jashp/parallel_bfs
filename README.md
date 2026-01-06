# Optimized Parallel Bi-Directional BFS

[cite_start]A high-performance implementation of Breadth-First Search (BFS) for shared-memory platforms, designed to handle million-node graphs with sub-40ms traversal times[cite: 7].

## 🚀 Overview

[cite_start]This project introduces an optimized graph traversal algorithm that outperforms traditional parallel BFS approaches by up to 45%[cite: 7]. It addresses common bottlenecks in large-scale graph analytics—specifically redundant edge scans and synchronization overhead—through a combination of adaptive algorithmic strategies and low-level system optimizations.

### Key Features
* [cite_start]**Adaptive Direction Switching:** Dynamically switches between "top-down" and "bottom-up" traversal based on real-time frontier and remainder sizes, reducing redundant work by up to 70%[cite: 6, 208].
* [cite_start]**Lock-Free Frontier Construction:** Utilizes a two-phase "count-then-fill" merge strategy with prefix sums, completely eliminating critical sections and locking overhead[cite: 6, 28].
* [cite_start]**Memory Optimization:** Features inlined Compressed Sparse Row (CSR) traversal, atomic compare-exchange for enqueueing, and software prefetch hints to hide DRAM latency[cite: 29].
* [cite_start]**Hybrid Multi-Source Loop:** Capable of visiting all disconnected components in a single frontier-driven loop[cite: 30].

---

## 📊 Performance

[cite_start]Tested on an Intel Core i7 (10 cores) with 16GB RAM[cite: 62]:

| Dataset | Nodes | Edges | Traversal Time | Speedup vs Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Synthetic** | 1 M | 3 M | **33 ms** | [cite_start]15% Faster [cite: 198] |
| **Google Web Graph** | ~1 M | ~5 M | **39 ms** | [cite_start]~45% Faster [cite: 202] |

[cite_start]*Achieved throughput exceeding **128 M edges/s** on real-world graphs[cite: 240].*

---

## 🛠️ Build & Run Instructions

### Prerequisites
* C++ Compiler (GCC/MinGW with OpenMP support)
* CMake

### Steps to Build and Run
Follow these steps to compile the project using MinGW Makefiles as configured in the build system.

**1. Prepare the Build Directory**
Navigate to the `build` directory. If you have changed the code or need a clean build, remove existing files first.
```bash
cd build
rm -r * # Optional: Use if you need to clean existing build files
```



### To Run Code

1. Go to 'build' directory
2. if you have chaged code so please remove existing files under build directory using 'rm -r *'
3. cmake -G "MinGW Makefiles" ..
4. cmake --build . --clean-first
5. Run .exe file
    - .\bin\parallel_bfs.exe (Default)
    - .\bin\parallel_bfs.exe 10000 0.01 (Random Synthetic Graphs)
    - .\bin\parallel_bfs.exe ..\graphs\WikiTalk.txt (To Work with Real World Data)
