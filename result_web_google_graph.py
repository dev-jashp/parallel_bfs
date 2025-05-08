import matplotlib.pyplot as plt
import numpy as np

# Algorithm names
algorithms = ['Parallel BFS', 'Parallel BD BFS', 'DOBFS', 'Optimized Parallel BD BFS']

# Time in milliseconds
times_4_threads_ms = [79, 78, 88, 72]
times_all_threads_ms = [65, 46, 146, 39]

# Bar positions
x = np.arange(len(algorithms))
width = 0.35  # Width of the bars

# Plotting
plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - width/2, times_4_threads_ms, width, label='4 Threads')
bars2 = plt.bar(x + width/2, times_all_threads_ms, width, label='All Threads')

# Labels and titles
plt.ylabel('Traversal Time (ms)')
plt.title('Traversal Time vs Algorithms on Web Google Graph (~1M nodes, ~5M edges)')
plt.xticks(x, algorithms)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add text labels on top of bars
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

# Show plot
plt.tight_layout()
plt.show()
