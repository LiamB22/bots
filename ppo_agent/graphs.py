import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# ---------------------------------------------------------
# 1. DATASET DEFINITIONS
# ---------------------------------------------------------

timesteps = [125000, 250000, 500000, 1000000]
labels = ['125K', '250K', '500K', '1M']

# ---------------------------------------------------------
# GRAPH 1 DATA (Only 5 runs) — Original Dense vs Alt Dense
# ---------------------------------------------------------

mixed_dense_5 = {
    125000: [43.30, 41.80, 36.80, 40.10, 26.00],
    250000: [32.40, 46.40, 43.40, 43.60, 43.50],
    500000: [47.70, 51.70, 47.70, 44.60, 33.60],
    1000000: [58.50, 53.80, 50.40, 51.40, 55.90]
}

mixed_alt_dense_5 = {
    125000: [39.70, 36.20, 37.30, 41.70, 38.10],
    250000: [43.90, 42.20, 42.60, 48.10, 46.00],
    500000: [45.50, 46.30, 39.10, 50.70, 49.60],
    1000000: [54.50, 55.30, 56.50, 51.70, 54.30]
}

# ---------------------------------------------------------
# FULL 10-RUN DATA (Improved Dense = Alt Dense)
# ---------------------------------------------------------

# Mixed Agent (10-run Alt Dense)
mixed_dense_10 = {
    125000: [39.70,36.20,37.30,41.70,38.10,37.80,42.60,37.60,41.10,37.50],
    250000: [43.90,42.20,42.60,48.10,46.00,39.20,42.40,50.50,43.50,49.90],
    500000: [45.50,46.30,39.10,50.70,49.60,56.90,53.60,40.10,43.40,46.00],
    1000000: [54.50,55.30,56.50,51.70,54.30,54.00,57.10,53.30,52.40,55.90]
}

# Mixed Agent Sparse (10 runs)
mixed_sparse_10 = {
    125000: [41.30,46.80,32.50,44.00,32.80,40.80,44.20,35.90,42.40,45.00],
    250000: [41.10,38.60,36.00,41.90,40.10,44.90,36.80,47.40,40.70,30.40],
    500000: [54.10,38.40,45.80,50.60,44.50,45.90,42.90,45.10,49.00,42.80],
    1000000: [54.60,54.00,57.70,50.30,51.90,51.10,50.80,56.50,44.30,48.40]
}

# Vector Agent Dense (10-run Alt Dense)
vector_dense_10 = {
    125000: [43.30,50.30,45.60,35.40,49.20,46.00,41.60,36.10,39.70,41.90],
    250000: [42.10,45.80,37.50,49.00,46.00,52.40,44.40,46.80,45.80,49.40],
    500000: [48.20,50.20,51.40,41.40,50.20,43.10,50.40,36.20,52.30,50.90],
    1000000: [54.30,50.70,58.90,57.90,55.60,50.80,53.40,51.90,47.50,50.90]
}

# Vector Agent Sparse (10 runs)
vector_sparse_10 = {
    125000: [37.40,41.00,44.40,32.20,44.80,40.10,44.70,50.20,43.20,48.70],
    250000: [48.80,45.00,49.90,46.50,47.00,42.40,41.70,44.30,51.10,48.70],
    500000: [41.30,52.70,44.30,53.00,52.00,41.60,53.50,42.90,39.90,45.70],
    1000000: [51.50,53.50,47.50,46.80,53.80,56.90,49.20,52.90,49.40,51.10]
}

# ---------------------------------------------------------
# Helper: Compute mean and std
# ---------------------------------------------------------
def avg_std(data):
    avg = [np.mean(data[t]) for t in timesteps]
    std = [np.std(data[t]) for t in timesteps]
    return avg, std

# COLORS
C1 = "#1f77b4"
C2 = "#d62728"
C3 = "#2ca02c"
C4 = "#9467bd"


# ---------------------------------------------------------
# ✅ GRAPH 1 – Mixed Agent: Original Dense vs Alt Dense (5 runs)
# ---------------------------------------------------------
avg_dense5, std_dense5 = avg_std(mixed_dense_5)
avg_alt5, std_alt5 = avg_std(mixed_alt_dense_5)

plt.figure(figsize=(10,6))
plt.plot(labels, avg_dense5, marker='o', color=C1, label="Original Dense (5 runs)", linewidth=2)
plt.plot(labels, avg_alt5, marker='s', color=C2, label="Alt Dense (5 runs)", linewidth=2)

plt.fill_between(range(len(labels)), 
                 np.array(avg_dense5)-np.array(std_dense5),
                 np.array(avg_dense5)+np.array(std_dense5),
                 alpha=0.2, color=C1)

plt.fill_between(range(len(labels)), 
                 np.array(avg_alt5)-np.array(std_alt5),
                 np.array(avg_alt5)+np.array(std_alt5),
                 alpha=0.2, color=C2)

plt.title("Mixed Agent — Original Dense vs Alt Dense (5 Runs)")
plt.xlabel("Training Timesteps")
plt.ylabel("Win Rate (%)")
plt.legend()
plt.ylim(20,65)
plt.tight_layout()
plt.savefig("bots/new_graphs/dense_vs_alt_dense.png", dpi=300)


# ---------------------------------------------------------
# ✅ GRAPH 2 – Mixed Agent: Improved Dense vs Sparse (10 runs)
# ---------------------------------------------------------
mixed_dense_avg, mixed_dense_std = avg_std(mixed_dense_10)
mixed_sparse_avg, mixed_sparse_std = avg_std(mixed_sparse_10)

plt.figure(figsize=(10,6))
plt.plot(labels, mixed_dense_avg, marker='o', color=C1, label="Dense (Improved)", linewidth=2)
plt.plot(labels, mixed_sparse_avg, marker='s', color=C2, label="Sparse", linewidth=2)

plt.fill_between(range(len(labels)),
                 np.array(mixed_dense_avg)-np.array(mixed_dense_std),
                 np.array(mixed_dense_avg)+np.array(mixed_dense_std),
                 alpha=0.2, color=C1)

plt.fill_between(range(len(labels)),
                 np.array(mixed_sparse_avg)-np.array(mixed_sparse_std),
                 np.array(mixed_sparse_avg)+np.array(mixed_sparse_std),
                 alpha=0.2, color=C2)

plt.title("Mixed Agent — Improved Dense vs Sparse (10 Runs)")
plt.xlabel("Training Timesteps")
plt.ylabel("Win Rate (%)")
plt.legend()
plt.ylim(30,65)
plt.tight_layout()
plt.savefig("bots/new_graphs/mixed_dense_vs_sparse.png", dpi=300)


# ---------------------------------------------------------
# ✅ GRAPH 3 – Vector Agent: Improved Dense vs Sparse (10 runs)
# ---------------------------------------------------------
vec_dense_avg, vec_dense_std = avg_std(vector_dense_10)
vec_sparse_avg, vec_sparse_std = avg_std(vector_sparse_10)

plt.figure(figsize=(10,6))
plt.plot(labels, vec_dense_avg, marker='o', color=C1, label="Dense (Improved)", linewidth=2)
plt.plot(labels, vec_sparse_avg, marker='s', color=C2, label="Sparse", linewidth=2)

plt.fill_between(range(len(labels)),
                 np.array(vec_dense_avg)-np.array(vec_dense_std),
                 np.array(vec_dense_avg)+np.array(vec_dense_std),
                 alpha=0.2, color=C1)

plt.fill_between(range(len(labels)),
                 np.array(vec_sparse_avg)-np.array(vec_sparse_std),
                 np.array(vec_sparse_avg)+np.array(vec_sparse_std),
                 alpha=0.2, color=C2)

plt.title("Vector Agent — Improved Dense vs Sparse (10 Runs)")
plt.xlabel("Training Timesteps")
plt.ylabel("Win Rate (%)")
plt.legend()
plt.ylim(35,65)
plt.tight_layout()
plt.savefig("bots/new_graphs/vector_dense_vs_sparse.png", dpi=300)


# ---------------------------------------------------------
# ✅ GRAPH 4 – Mixed vs Vector (Dense only, improved Alt Dense)
# ---------------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(labels, mixed_dense_avg, marker='o', linewidth=2, label='Mixed (Dense)', color=C1)
plt.plot(labels, vec_dense_avg, marker='s', linewidth=2, label='Vector (Dense)', color=C2)

plt.title("Mixed vs Vector Agents — Dense Reward (Improved)")
plt.xlabel("Training Timesteps")
plt.ylabel("Win Rate (%)")
plt.legend()
plt.ylim(35,65)
plt.tight_layout()
plt.savefig("bots/new_graphs/mixed_vs_vector_dense.png", dpi=300)


# ---------------------------------------------------------
# ✅ GRAPH 5 – Mixed vs Vector (Sparse Only)
# ---------------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(labels, mixed_sparse_avg, marker='o', linewidth=2, label='Mixed (Sparse)', color=C1)
plt.plot(labels, vec_sparse_avg, marker='s', linewidth=2, label='Vector (Sparse)', color=C2)

plt.title("Mixed vs Vector Agents — Sparse Reward")
plt.xlabel("Training Timesteps")
plt.ylabel("Win Rate (%)")
plt.legend()
plt.ylim(30,65)
plt.tight_layout()
plt.savefig("bots/new_graphs/mixed_vs_vector_sparse.png", dpi=300)


# ---------------------------------------------------------
# ✅ GRAPH 6 – All Reward Types Overview
# ---------------------------------------------------------
plt.figure(figsize=(12,7))

plt.plot(labels, mixed_dense_avg, marker='o', linewidth=2, label='Mixed Dense', color=C1)
plt.plot(labels, mixed_sparse_avg, marker='o', linewidth=2, linestyle='--', label='Mixed Sparse', color=C1, alpha=0.7)

plt.plot(labels, vec_dense_avg, marker='s', linewidth=2, label='Vector Dense', color=C2)
plt.plot(labels, vec_sparse_avg, marker='s', linewidth=2, linestyle='--', label='Vector Sparse', color=C2, alpha=0.7)

plt.title("Complete Agent Performance Comparison — All Reward Types")
plt.xlabel("Training Timesteps")
plt.ylabel("Win Rate (%)")
plt.legend()
plt.ylim(30,65)
plt.tight_layout()
plt.savefig("bots/new_graphs/all_reward_comparison.png", dpi=300)


print("\n✅ All graphs have been generated and saved successfully!")
