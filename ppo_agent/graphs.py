import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')


timesteps = [125000, 250000, 500000, 1000000]
labels = ['125K', '250K', '500K', '1M']

#===========================================================
# Mixed vs Vector
#===========================================================

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

def avg_std(data):
    avg = [np.mean(data[t]) for t in timesteps]
    std = [np.std(data[t]) for t in timesteps]
    return avg, std

C1 = "#1f77b4"
C2 = "#d62728"
C3 = "#2ca02c"
C4 = "#9467bd"


# Mixed Agent: Original Dense vs Alt Dense (5 runs)
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
plt.savefig("bots/graphs_1/dense_vs_alt_dense.png", dpi=300)


# Mixed Agent: Improved Dense vs Sparse (10 runs)
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
plt.savefig("bots/graphs_1/mixed_dense_vs_sparse.png", dpi=300)


# Vector Agent: Improved Dense vs Sparse (10 runs)
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
plt.savefig("bots/graphs_1/vector_dense_vs_sparse.png", dpi=300)


# Mixed vs Vector (Dense only, improved Alt Dense)
plt.figure(figsize=(10,6))
plt.plot(labels, mixed_dense_avg, marker='o', linewidth=2, label='Mixed (Dense)', color=C1)
plt.plot(labels, vec_dense_avg, marker='s', linewidth=2, label='Vector (Dense)', color=C2)

plt.title("Mixed vs Vector Agents — Dense Reward (Improved)")
plt.xlabel("Training Timesteps")
plt.ylabel("Win Rate (%)")
plt.legend()
plt.ylim(35,65)
plt.tight_layout()
plt.savefig("bots/graphs_1/mixed_vs_vector_dense.png", dpi=300)

# Mixed vs Vector (Sparse Only)
plt.figure(figsize=(10,6))
plt.plot(labels, mixed_sparse_avg, marker='o', linewidth=2, label='Mixed (Sparse)', color=C1)
plt.plot(labels, vec_sparse_avg, marker='s', linewidth=2, label='Vector (Sparse)', color=C2)

plt.title("Mixed vs Vector Agents — Sparse Reward")
plt.xlabel("Training Timesteps")
plt.ylabel("Win Rate (%)")
plt.legend()
plt.ylim(30,65)
plt.tight_layout()
plt.savefig("bots/graphs_1/mixed_vs_vector_sparse.png", dpi=300)


# All Reward Types Overview
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
plt.savefig("bots/graphs_1/all_reward_comparison_1.png", dpi=300)


print("\nAll graphs have been generated and saved successfully!")

# Create figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Mixed Dense vs Sparse
axs[0, 0].plot(labels, mixed_dense_avg, marker='o', color=C1, label='Dense (Improved)', linewidth=2)
axs[0, 0].plot(labels, mixed_sparse_avg, marker='s', color=C2, label='Sparse', linewidth=2)
axs[0, 0].fill_between(range(len(labels)),
                       np.array(mixed_dense_avg)-np.array(mixed_dense_std),
                       np.array(mixed_dense_avg)+np.array(mixed_dense_std),
                       alpha=0.2, color=C1)
axs[0, 0].fill_between(range(len(labels)),
                       np.array(mixed_sparse_avg)-np.array(mixed_sparse_std),
                       np.array(mixed_sparse_avg)+np.array(mixed_sparse_std),
                       alpha=0.2, color=C2)
axs[0, 0].set_title("Mixed Agent — Dense vs Sparse")
axs[0, 0].set_ylim(30,65)
axs[0, 0].set_xlabel("Timesteps")
axs[0, 0].set_ylabel("Win Rate (%)")
axs[0, 0].legend()

# Top-right: Vector Dense vs Sparse
axs[0, 1].plot(labels, vec_dense_avg, marker='o', color=C1, label='Dense (Improved)', linewidth=2)
axs[0, 1].plot(labels, vec_sparse_avg, marker='s', color=C2, label='Sparse', linewidth=2)
axs[0, 1].fill_between(range(len(labels)),
                       np.array(vec_dense_avg)-np.array(vec_dense_std),
                       np.array(vec_dense_avg)+np.array(vec_dense_std),
                       alpha=0.2, color=C1)
axs[0, 1].fill_between(range(len(labels)),
                       np.array(vec_sparse_avg)-np.array(vec_sparse_std),
                       np.array(vec_sparse_avg)+np.array(vec_sparse_std),
                       alpha=0.2, color=C2)
axs[0, 1].set_title("Vector Agent — Dense vs Sparse")
axs[0, 1].set_ylim(35,65)
axs[0, 1].set_xlabel("Timesteps")
axs[0, 1].set_ylabel("Win Rate (%)")
axs[0, 1].legend()

# Bottom-left: Mixed vs Vector Dense
axs[1, 0].plot(labels, mixed_dense_avg, marker='o', color=C1, label='Mixed Dense', linewidth=2)
axs[1, 0].plot(labels, vec_dense_avg, marker='s', color=C2, label='Vector Dense', linewidth=2)
axs[1, 0].set_title("Mixed vs Vector — Dense")
axs[1, 0].set_ylim(35,65)
axs[1, 0].set_xlabel("Timesteps")
axs[1, 0].set_ylabel("Win Rate (%)")
axs[1, 0].legend()

# Bottom-right: Mixed vs Vector Sparse
axs[1, 1].plot(labels, mixed_sparse_avg, marker='o', color=C1, label='Mixed Sparse', linewidth=2)
axs[1, 1].plot(labels, vec_sparse_avg, marker='s', color=C2, label='Vector Sparse', linewidth=2)
axs[1, 1].set_title("Mixed vs Vector — Sparse")
axs[1, 1].set_ylim(30,65)
axs[1, 1].set_xlabel("Timesteps")
axs[1, 1].set_ylabel("Win Rate (%)")
axs[1, 1].legend()

plt.tight_layout()
plt.savefig("bots/graphs_1/mixed_vs_vector_subplots.png", dpi=300)

# Mixed vs Vector — final timestep (1M)
final_t = 1000000
agents = ['Mixed Dense', 'Mixed Sparse', 'Vector Dense', 'Vector Sparse']
means = [
    np.mean(mixed_dense_10[final_t]),
    np.mean(mixed_sparse_10[final_t]),
    np.mean(vector_dense_10[final_t]),
    np.mean(vector_sparse_10[final_t])
]
stds = [
    np.std(mixed_dense_10[final_t]),
    np.std(mixed_sparse_10[final_t]),
    np.std(vector_dense_10[final_t]),
    np.std(vector_sparse_10[final_t])
]

plt.figure(figsize=(10,6))
plt.bar(agents, means, yerr=stds, capsize=5, color=[C1, C2, C1, C2], alpha=0.8)
plt.ylabel("Win Rate (%)")
plt.title("Mixed vs Vector — Win Rate at 1M Timesteps")
plt.ylim(30,65)
plt.tight_layout()
plt.savefig("bots/graphs_1/mixed_vs_vector_bar.png", dpi=300)

# Mixed vs Vector — all runs at 1M timesteps
data = [
    mixed_dense_10[final_t],
    mixed_sparse_10[final_t],
    vector_dense_10[final_t],
    vector_sparse_10[final_t]
]

plt.figure(figsize=(10,6))
plt.boxplot(data, tick_labels=agents, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'),
            whiskerprops=dict(color='blue'),
            capprops=dict(color='blue'))
plt.ylabel("Win Rate (%)")
plt.title("Mixed vs Vector — Distribution at 1M Timesteps")
plt.ylim(30,65)
plt.tight_layout()
plt.savefig("bots/graphs_1/mixed_vs_vector_box.png", dpi=300)


#===========================================================
# Board vs Numeric
#===========================================================
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
    500000: [46.70, 47.60, 43.10, 46.40, 51.11],
    1000000: [51.30, 50.50, 57.00, 53.10, 54.30],
}

board_alt_dense_5 = {
    125000: [43.40, 36.80, 41.10, 29.60, 35.90],
    250000: [45.90, 38.60, 43.10, 39.90, 47.30],
    500000: [47.90, 46.80, 46.90, 48.60, 45.50],
    1000000: [56.60, 51.80, 56.00, 55.30, 56.00],
}


# Board Agent: Improved Dense vs Sparse (10 runs)
board_dense_avg, board_dense_std = avg_std(board_alt_dense_5)
board_sparse_avg, board_sparse_std = avg_std(board_sparse_5)

plt.figure(figsize=(10,6))
plt.plot(labels, board_dense_avg, marker='o', color=C4, label="Dense (Improved)", linewidth=2)
plt.plot(labels, board_sparse_avg, marker='s', color=C3, label="Sparse", linewidth=2)

plt.fill_between(range(len(labels)),
                 np.array(board_dense_avg)-np.array(board_dense_std),
                 np.array(board_dense_avg)+np.array(board_dense_std),
                 alpha=0.2, color=C4)

plt.fill_between(range(len(labels)),
                 np.array(board_sparse_avg)-np.array(board_sparse_std),
                 np.array(board_sparse_avg)+np.array(board_sparse_std),
                 alpha=0.2, color=C3)

plt.title("Board Agent — Improved Dense vs Sparse (10 Runs)")
plt.xlabel("Training Timesteps")
plt.ylabel("Win Rate (%)")
plt.legend()
plt.ylim(15,65)
plt.tight_layout()
plt.savefig("bots/graphs_2/board_dense_vs_sparse.png", dpi=300)


# Numeric Agent: Improved Dense vs Sparse (10 runs)
numeric_dense_avg, numeric_dense_std = avg_std(numeric_alt_dense_5)
numeric_sparse_avg, numeric_sparse_std = avg_std(numeric_sparse_5)

plt.figure(figsize=(10,6))
plt.plot(labels, numeric_dense_avg, marker='o', color=C4, label="Dense (Improved)", linewidth=2)
plt.plot(labels, numeric_sparse_avg, marker='s', color=C3, label="Sparse", linewidth=2)

plt.fill_between(range(len(labels)),
                 np.array(numeric_dense_avg)-np.array(numeric_dense_std),
                 np.array(numeric_dense_avg)+np.array(numeric_dense_std),
                 alpha=0.2, color=C4)

plt.fill_between(range(len(labels)),
                 np.array(numeric_sparse_avg)-np.array(numeric_sparse_std),
                 np.array(numeric_sparse_avg)+np.array(numeric_sparse_std),
                 alpha=0.2, color=C3)

plt.title("Numeric Agent — Improved Dense vs Sparse (10 Runs)")
plt.xlabel("Training Timesteps")
plt.ylabel("Win Rate (%)")
plt.legend()
plt.ylim(15,65)
plt.tight_layout()
plt.savefig("bots/graphs_2/numeric_dense_vs_sparse.png", dpi=300)


# Board vs Numeric (Dense only, improved Alt Dense)
plt.figure(figsize=(10,6))
plt.plot(labels, board_dense_avg, marker='o', linewidth=2, label='Board (Dense)', color=C4)
plt.plot(labels, numeric_dense_avg, marker='s', linewidth=2, label='Numeric (Dense)', color=C3)

plt.title("Board vs Numeric Agents — Dense Reward (Improved)")
plt.xlabel("Training Timesteps")
plt.ylabel("Win Rate (%)")
plt.legend()
plt.ylim(15,65)
plt.tight_layout()
plt.savefig("bots/graphs_2/board_vs_numeric_dense.png", dpi=300)


# Board vs Numeric (Sparse Only)
plt.figure(figsize=(10,6))
plt.plot(labels, board_sparse_avg, marker='o', linewidth=2, label='Board (Sparse)', color=C4)
plt.plot(labels, numeric_sparse_avg, marker='s', linewidth=2, label='Numeric (Sparse)', color=C3)

plt.title("Board vs Numeric Agents — Sparse Reward")
plt.xlabel("Training Timesteps")
plt.ylabel("Win Rate (%)")
plt.legend()
plt.ylim(15,65)
plt.tight_layout()
plt.savefig("bots/graphs_2/board_vs_numeric_sparse.png", dpi=300)


# All Reward Types Overview
plt.figure(figsize=(12,7))

plt.plot(labels, board_dense_avg, marker='o', linewidth=2, label='Board Dense', color=C4)
plt.plot(labels, board_sparse_avg, marker='o', linewidth=2, linestyle='--', label='Board Sparse', color=C4, alpha=0.7)

plt.plot(labels, numeric_dense_avg, marker='s', linewidth=2, label='Numeric Dense', color=C3)
plt.plot(labels, numeric_sparse_avg, marker='s', linewidth=2, linestyle='--', label='Numeric Sparse', color=C3, alpha=0.7)

plt.title("Complete Agent Performance Comparison — All Reward Types")
plt.xlabel("Training Timesteps")
plt.ylabel("Win Rate (%)")
plt.legend()
plt.ylim(15,65)
plt.tight_layout()
plt.savefig("bots/graphs_2/all_reward_comparison_2.png", dpi=300)


print("\nAll graphs have been generated and saved successfully!")


# Board vs Numeric Data (Dense/Sparse)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Board Dense vs Sparse
axs[0, 0].plot(labels, board_dense_avg, marker='o', color=C4, label='Dense (Improved)', linewidth=2)
axs[0, 0].plot(labels, board_sparse_avg, marker='s', color=C3, label='Sparse', linewidth=2)
axs[0, 0].fill_between(range(len(labels)),
                       np.array(board_dense_avg)-np.array(board_dense_std),
                       np.array(board_dense_avg)+np.array(board_dense_std),
                       alpha=0.2, color=C4)
axs[0, 0].fill_between(range(len(labels)),
                       np.array(board_sparse_avg)-np.array(board_sparse_std),
                       np.array(board_sparse_avg)+np.array(board_sparse_std),
                       alpha=0.2, color=C3)
axs[0, 0].set_title("Board Agent — Dense vs Sparse")
axs[0, 0].set_ylim(15,65)
axs[0, 0].set_xlabel("Timesteps")
axs[0, 0].set_ylabel("Win Rate (%)")
axs[0, 0].legend()

# Top-right: Numeric Dense vs Sparse
axs[0, 1].plot(labels, numeric_dense_avg, marker='o', color=C4, label='Dense (Improved)', linewidth=2)
axs[0, 1].plot(labels, numeric_sparse_avg, marker='s', color=C3, label='Sparse', linewidth=2)
axs[0, 1].fill_between(range(len(labels)),
                       np.array(numeric_dense_avg)-np.array(numeric_dense_std),
                       np.array(numeric_dense_avg)+np.array(numeric_dense_std),
                       alpha=0.2, color=C4)
axs[0, 1].fill_between(range(len(labels)),
                       np.array(numeric_sparse_avg)-np.array(numeric_sparse_std),
                       np.array(numeric_sparse_avg)+np.array(numeric_sparse_std),
                       alpha=0.2, color=C3)
axs[0, 1].set_title("Numeric Agent — Dense vs Sparse")
axs[0, 1].set_ylim(15,65)
axs[0, 1].set_xlabel("Timesteps")
axs[0, 1].set_ylabel("Win Rate (%)")
axs[0, 1].legend()

# Bottom-left: Board vs Numeric Dense
axs[1, 0].plot(labels, board_dense_avg, marker='o', color=C4, label='Board Dense', linewidth=2)
axs[1, 0].plot(labels, numeric_dense_avg, marker='s', color=C3, label='Numeric Dense', linewidth=2)
axs[1, 0].set_title("Board vs Numeric — Dense")
axs[1, 0].set_ylim(15,65)
axs[1, 0].set_xlabel("Timesteps")
axs[1, 0].set_ylabel("Win Rate (%)")
axs[1, 0].legend()

# Bottom-right: Board vs Numeric Sparse
axs[1, 1].plot(labels, board_sparse_avg, marker='o', color=C4, label='Board Sparse', linewidth=2)
axs[1, 1].plot(labels, numeric_sparse_avg, marker='s', color=C3, label='Numeric Sparse', linewidth=2)
axs[1, 1].set_title("Board vs Numeric — Sparse")
axs[1, 1].set_ylim(15,65)
axs[1, 1].set_xlabel("Timesteps")
axs[1, 1].set_ylabel("Win Rate (%)")
axs[1, 1].legend()

plt.tight_layout()
plt.savefig("bots/graphs_2/board_vs_numeric_subplots.png", dpi=300)

# Board vs Numeric — final timestep (1M)
agents_bn = ['Board Dense', 'Board Sparse', 'Numeric Dense', 'Numeric Sparse']
means_bn = [
    np.mean(board_alt_dense_5[final_t]),
    np.mean(board_sparse_5[final_t]),
    np.mean(numeric_alt_dense_5[final_t]),
    np.mean(numeric_sparse_5[final_t])
]
stds_bn = [
    np.std(board_alt_dense_5[final_t]),
    np.std(board_sparse_5[final_t]),
    np.std(numeric_alt_dense_5[final_t]),
    np.std(numeric_sparse_5[final_t])
]

plt.figure(figsize=(10,6))
plt.bar(agents_bn, means_bn, yerr=stds_bn, capsize=5, color=[C4, C3, C4, C3], alpha=0.8)
plt.ylabel("Win Rate (%)")
plt.title("Board vs Numeric — Win Rate at 1M Timesteps")
plt.ylim(15,65)
plt.tight_layout()
plt.savefig("bots/graphs_2/board_vs_numeric_bar.png", dpi=300)

data_bn = [
    board_alt_dense_5[final_t],
    board_sparse_5[final_t],
    numeric_alt_dense_5[final_t],
    numeric_sparse_5[final_t]
]

plt.figure(figsize=(10,6))
plt.boxplot(data_bn, tick_labels=agents_bn, patch_artist=True,
            boxprops=dict(facecolor='plum', color='purple'),
            medianprops=dict(color='green'),
            whiskerprops=dict(color='purple'),
            capprops=dict(color='purple'))
plt.ylabel("Win Rate (%)")
plt.title("Board vs Numeric — Distribution at 1M Timesteps")
plt.ylim(15,65)
plt.tight_layout()
plt.savefig("bots/graphs_2/board_vs_numeric_box.png", dpi=300)


# ==================================================================
# All models
# ==================================================================

final_t = 1000000

all_agents = [
    'Mixed Dense', 'Mixed Sparse',
    'Vector Dense', 'Vector Sparse',
    'Board Dense', 'Board Sparse',
    'Numeric Dense', 'Numeric Sparse'
]

all_means = [
    np.mean(mixed_dense_10[final_t]),
    np.mean(mixed_sparse_10[final_t]),
    np.mean(vector_dense_10[final_t]),
    np.mean(vector_sparse_10[final_t]),
    np.mean(board_alt_dense_5[final_t]),
    np.mean(board_sparse_5[final_t]),
    np.mean(numeric_alt_dense_5[final_t]),
    np.mean(numeric_sparse_5[final_t])
]

all_stds = [
    np.std(mixed_dense_10[final_t]),
    np.std(mixed_sparse_10[final_t]),
    np.std(vector_dense_10[final_t]),
    np.std(vector_sparse_10[final_t]),
    np.std(board_alt_dense_5[final_t]),
    np.std(board_sparse_5[final_t]),
    np.std(numeric_alt_dense_5[final_t]),
    np.std(numeric_sparse_5[final_t])
]

# Color mapping: Dense vs Sparse
colors_all = [C1, C2, C1, C2, C4, C3, C4, C3]

plt.figure(figsize=(12,6))
plt.bar(all_agents, all_means, yerr=all_stds, capsize=5, color=colors_all, alpha=0.8)
plt.ylabel("Win Rate (%)")
plt.title("All 8 Agents — Win Rate at 1M Timesteps")
plt.ylim(15,65)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("bots/graphs/all_agents_bar.png", dpi=300)


all_data = [
    mixed_dense_10[final_t],
    mixed_sparse_10[final_t],
    vector_dense_10[final_t],
    vector_sparse_10[final_t],
    board_alt_dense_5[final_t],
    board_sparse_5[final_t],
    numeric_alt_dense_5[final_t],
    numeric_sparse_5[final_t]
]

plt.figure(figsize=(12,6))
box_colors = ['lightblue', 'lightcoral', 'lightblue', 'lightcoral',
              'plum', 'mediumpurple', 'plum', 'mediumpurple']

bp = plt.boxplot(all_data, tick_labels=all_agents, patch_artist=True)

# Apply colors to boxes
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)

# Median lines
for median in bp['medians']:
    median.set(color='green', linewidth=2)

plt.ylabel("Win Rate (%)")
plt.title("All 8 Agents — Distribution at 1M Timesteps")
plt.ylim(15,65)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("bots/graphs/all_agents_box.png", dpi=300)
