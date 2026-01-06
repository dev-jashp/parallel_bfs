# Optimized Parallel Bi-Directional BFS

A high-performance implementation of Breadth-First Search (BFS) for shared-memory platforms, designed to handle million-node graphs with sub-40ms traversal times.

## 🚀 Overview

This project introduces an optimized graph traversal algorithm that outperforms traditional parallel BFS approaches by up to 45%. It addresses common bottlenecks in large-scale graph analytics—specifically redundant edge scans and synchronization overhead—through a combination of adaptive algorithmic strategies and low-level system optimizations.

### Key Features
- **Adaptive Direction Switching:** Dynamically switches between "top-down" and "bottom-up" traversal based on real-time frontier and remainder sizes, reducing redundant work by up to 70%.
- **Lock-Free Frontier Construction:** Utilizes a two-phase "count-then-fill" merge strategy with prefix sums, completely eliminating critical sections and locking overhead.
- **Memory Optimization:** Features inlined Compressed Sparse Row (CSR) traversal, atomic compare-exchange for enqueueing, and software prefetch hints to hide DRAM latency.
- **Hybrid Multi-Source Loop:** Capable of visiting all disconnected components in a single frontier-driven loop.

---

## 📊 Performance

Tested on an Intel Core i7 (10 cores) with 16GB RAM:

| Dataset | Nodes | Edges | Traversal Time | Speedup vs Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Synthetic** | 1 M | 3 M | **33 ms** | 15% Faster |
| **Google Web Graph** | ~1 M | ~5 M | **39 ms** | ~45% Faster |

*Achieved throughput exceeding **128 M edges/s** on real-world graphs.*

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

**2. Configure with CMake Generate the Makefiles using the MinGW generator.**
```bash
cmake -G "MinGW Makefiles" ..
```

**3. Build the Project Compile the executable.**
```bash
cmake --build . --clean-first
```

**3. Run the Executable You can run the BFS with default settings, generate a random synthetic graph, or load a real-world dataset.**
- **Default Run**
```bash
.\bin\parallel_bfs.exe
```

- **Run on Random Synthetic Graph:** Usage: `executable <num_nodes> <density>`
```bash
.\bin\parallel_bfs.exe 10000 0.01
```

- **Run on Real-World Data:** Usage: `executable <path_to_graph_file>`
```bash
.\bin\parallel_bfs.exe ..\graphs\WikiTalk.txt
```

## 🛠️ Project Structure
- `src/`: Core implementation of the BFS algorithms.
- `include/`: Header files for Graph and BFS classes.
- `graphs/`: Directory containing real-world datasets (e.g., WikiTalk.txt).
- `bin/`: Output directory for compiled executables.

## 👥 Contributors
1. Jash Patel
2. Hitesh Avula
3. Sai Katari
4. Nirupam Dashika
5. Viswanada Reddy
