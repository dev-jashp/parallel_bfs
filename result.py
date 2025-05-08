import matplotlib.pyplot as plt

# Data
algorithms = ['Parallel BFS', 'Parallel BD BFS', 'DOBFS', 'Optimized Parallel BD BFS']

# Times taken by each algorithm on large synthetic graph 1M nodes, 3M edges
times_4_threads_large_synthetic = [0.062, 0.046, 0.069, 0.044]  # Time in seconds
times_all_threads_large_synthetic = [0.053, 0.038, 0.089, 0.033]  # Time in seconds

# Times taken by each algorithm on web Google Graph
times_4_threads_web_google = [0.079, 0.078, 0.088, 0.072]  # Time in seconds
times_all_threads_web_google = [0.065, 0.046, 0.146, 0.039]  # Time in seconds

times_4_threads_large_synthetic_ms = [62, 46, 69, 44]  # in milliseconds
times_all_threads_large_synthetic_ms = [53, 38, 89, 33]  # in milliseconds

times_4_threads_web_google_ms = [79, 78, 88, 72]  # in milliseconds 
times_all_threads_web_google_ms = [65, 46, 146, 39]  # in milliseconds
# Highlight the optimized algorithm
colors = ['blue', 'blue', 'blue', 'red']

# Create bar chart
plt.figure()
bars = plt.bar(algorithms, times_4_threads_large_synthetic_ms, color=colors)
plt.ylabel('Time (ms)')
plt.title('Algorithm vs Time on Large Synthetic Graph')
plt.xticks(rotation=45, ha='right')

# Add time labels on each bar
for bar, t in zip(bars, times_4_threads_large_synthetic_ms):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        t,
        f'{t:.2f}ms',
        ha='center',
        va='bottom'
    )

plt.tight_layout()
plt.show()
