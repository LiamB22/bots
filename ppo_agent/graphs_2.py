import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')

# ---------------------------------------------------------
# 1. DATASET DEFINITIONS
# ---------------------------------------------------------

timesteps = [125000, 250000, 500000, 1000000]
labels = ['125K', '250K', '500K', '1M']

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
# NEW 5-RUN DATASETS (Numeric = table "Sparse/Alt Dense Agent", Board = "Board *" tables)
# ---------------------------------------------------------

numeric_sparse_5 = {
    125000: [38.00, 38.20, 44.50, 40.20, 36.30],
    250000: [41.10, 32.80, 42.50, 47.10, 40.20],
    500000: [47.00, 42.10, 47.00, 43.70, 38.70],
    1000000: [48.00, 46.00, 46.00, 46.30, 40.00],
}

numeric_alt_dense_5 = {
    125000: [21.70, 38.00, 46.80, 20.90, 42.40],
    250000: [45.00, 40.50, 46.10, 42.70, 42.00],
    500000: [51.70, 42.10, 51.10, 39.40, 45.40],
    1000000: [46.90, 41.20, 39.80, 51.10, 57.30],
}

board_sparse_5 = {
    125000: [39.60, 36.70, 36.40, 42.90, 51.70],
    250000: [36.90, 35.60, 41.70, 38.20, 35.20],
    500000: [46.70, 47.60, 43.10, 46.40, 51.10],
    1000000: [51.30, 50.50, 57.00, 53.10, 54.30],
}

board_alt_dense_5 = {
    125000: [43.40, 36.80, 41.10, 29.60, 35.90],
    250000: [45.90, 38.60, 43.10, 39.90, 47.30],
    500000: [47.90, 46.80, 46.90, 48.60, 45.50],
    1000000: [56.60, 51.80, 56.00, 55.30, 56.00],
}

# Precompute means/stds
num_sparse_avg, num_sparse_std = avg_std(numeric_sparse_5)
num_dense_avg,  num_dense_std  = avg_std(numeric_alt_dense_5)
brd_sparse_avg, brd_sparse_std = avg_std(board_sparse_5)
brd_dense_avg,  brd_dense_std  = avg_std(board_alt_dense_5)

# ---------------------------------------------------------
# GRAPH 7 – All four models (5 runs)
# ---------------------------------------------------------
plt.figure(figsize=(12,7))
plt.plot(labels, num_sparse_avg, marker='o', linewidth=2, label='Numeric Sparse', color=C1)
plt.plot(labels, num_dense_avg,  marker='o', linewidth=2, linestyle='--', label='Numeric Dense (Alt)', color=C1, alpha=0.8)
plt.plot(labels, brd_sparse_avg, marker='s', linewidth=2, label='Board Sparse', color=C2)
plt.plot(labels, brd_dense_avg,  marker='s', linewidth=2, linestyle='--', label='Board Dense (Alt)', color=C2, alpha=0.8)

# # (Optional) uncertainty bands
# for avg, std, col in [
#     (num_sparse_avg, num_sparse_std, C1),
#     (num_dense_avg,  num_dense_std,  C1),
#     (brd_sparse_avg, brd_sparse_std, C2),
#     (brd_dense_avg,  brd_dense_std,  C2),
# ]:
#     plt.fill_between(range(len(labels)),
#                      np.array(avg)-np.array(std),
#                      np.array(avg)+np.array(std),
#                      alpha=0.15, color=col)

plt.title("All Models — Board vs Numeric × Sparse vs Dense (5 Runs)")
plt.xlabel("Training Timesteps")
plt.ylabel("Win Rate (%)")
plt.legend()
plt.ylim(20, 65)
plt.tight_layout()
plt.savefig("bots/new_graphs/all_four_models_5run.png", dpi=300)

# ---------------------------------------------------------
# GRAPH 8 – Board Only: Sparse vs Dense (5 runs)
# ---------------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(labels, brd_sparse_avg, marker='o', linewidth=2, label='Board Sparse', color=C3)
plt.plot(labels, brd_dense_avg,  marker='s', linewidth=2, label='Board Dense (Alt)', color=C4)

plt.fill_between(range(len(labels)),
                 np.array(brd_sparse_avg)-np.array(brd_sparse_std),
                 np.array(brd_sparse_avg)+np.array(brd_sparse_std),
                 alpha=0.2, color=C3)
plt.fill_between(range(len(labels)),
                 np.array(brd_dense_avg)-np.array(brd_dense_std),
                 np.array(brd_dense_avg)+np.array(brd_dense_std),
                 alpha=0.2, color=C4)

plt.title("Board-Only — Sparse vs Dense (Alt) (5 Runs)")
plt.xlabel("Training Timesteps")
plt.ylabel("Win Rate (%)")
plt.legend()
plt.ylim(20, 65)
plt.tight_layout()
plt.savefig("bots/new_graphs/board_only_sparse_vs_dense_5run.png", dpi=300)

# ---------------------------------------------------------
# GRAPH 9 – Numeric Only: Sparse vs Dense (5 runs)
# ---------------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(labels, num_sparse_avg, marker='o', linewidth=2, label='Numeric Sparse', color=C3)
plt.plot(labels, num_dense_avg,  marker='s', linewidth=2, label='Numeric Dense (Alt)', color=C4)

plt.fill_between(range(len(labels)),
                 np.array(num_sparse_avg)-np.array(num_sparse_std),
                 np.array(num_sparse_avg)+np.array(num_sparse_std),
                 alpha=0.2, color=C3)
plt.fill_between(range(len(labels)),
                 np.array(num_dense_avg)-np.array(num_dense_std),
                 np.array(num_dense_avg)+np.array(num_dense_std),
                 alpha=0.2, color=C4)

plt.title("Numeric-Only — Sparse vs Dense (Alt) (5 Runs)")
plt.xlabel("Training Timesteps")
plt.ylabel("Win Rate (%)")
plt.legend()
plt.ylim(20, 65)
plt.tight_layout()
plt.savefig("bots/new_graphs/numeric_only_sparse_vs_dense_5run.png", dpi=300)

# ---------------------------------------------------------
# GRAPH 10 – Board vs Numeric (Dense only, Alt Dense) — 5 runs
# ---------------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(labels, brd_dense_avg, marker='o', linewidth=2, label='Board Dense (Alt)', color=C1)
plt.plot(labels, num_dense_avg, marker='s', linewidth=2, label='Numeric Dense (Alt)', color=C2)

plt.fill_between(range(len(labels)),
                 np.array(brd_dense_avg)-np.array(brd_dense_std),
                 np.array(brd_dense_avg)+np.array(brd_dense_std),
                 alpha=0.2, color=C1)
plt.fill_between(range(len(labels)),
                 np.array(num_dense_avg)-np.array(num_dense_std),
                 np.array(num_dense_avg)+np.array(num_dense_std),
                 alpha=0.2, color=C2)

plt.title("Dense Reward (Alt) — Board vs Numeric (5 Runs)")
plt.xlabel("Training Timesteps")
plt.ylabel("Win Rate (%)")
plt.legend()
plt.ylim(20, 65)
plt.tight_layout()
plt.savefig("bots/new_graphs/board_vs_numeric_dense_5run.png", dpi=300)

# ---------------------------------------------------------
# GRAPH 11 – Board vs Numeric (Sparse only) — 5 runs
# ---------------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(labels, brd_sparse_avg, marker='o', linewidth=2, label='Board Sparse', color=C1)
plt.plot(labels, num_sparse_avg, marker='s', linewidth=2, label='Numeric Sparse', color=C2)

plt.fill_between(range(len(labels)),
                 np.array(brd_sparse_avg)-np.array(brd_sparse_std),
                 np.array(brd_sparse_avg)+np.array(brd_sparse_std),
                 alpha=0.2, color=C1)
plt.fill_between(range(len(labels)),
                 np.array(num_sparse_avg)-np.array(num_sparse_std),
                 np.array(num_sparse_avg)+np.array(num_sparse_std),
                 alpha=0.2, color=C2)

plt.title("Sparse Reward — Board vs Numeric (5 Runs)")
plt.xlabel("Training Timesteps")
plt.ylabel("Win Rate (%)")
plt.legend()
plt.ylim(20, 65)
plt.tight_layout()
plt.savefig("bots/new_graphs/board_vs_numeric_sparse_5run.png", dpi=300)

print("✅ New 5-run graphs generated:")
print(" - all_four_models_5run.png")
print(" - board_only_sparse_vs_dense_5run.png")
print(" - numeric_only_sparse_vs_dense_5run.png")
print(" - board_vs_numeric_dense_5run.png")
print(" - board_vs_numeric_sparse_5run.png")
