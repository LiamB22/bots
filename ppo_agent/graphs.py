import matplotlib.pyplot as plt
import numpy as np

# Data for Combined Agent - Dense Reward Function
combined_dense = {
    125000: [43.30, 41.80, 36.80, 40.10, 26.00],
    250000: [32.40, 46.40, 43.40, 43.60, 43.50],
    500000: [47.70, 51.70, 47.70, 44.60, 33.60],
    1000000: [58.50, 53.80, 50.40, 51.40, 55.90]
}

combined_alt_dense = {
    125000: [39.70, 36.20, 37.30, 41.70, 38.10],
    250000: [43.90, 42.20, 42.60, 48.10, 46.00],
    500000: [45.50, 46.30, 39.10, 50.70, 49.60],
    1000000: [54.50, 55.30, 56.50, 51.70, 54.30]
}

# Data for Combined Agent - Sparse Reward Function
combined_sparse = {
    125000: [41.30, 46.80, 32.50, 44.00, 32.80],
    250000: [41.10, 38.60, 36.00, 41.90, 40.10],
    500000: [54.10, 38.40, 45.80, 50.60, 44.50],
    1000000: [54.60, 54.00, 57.70, 50.30, 51.90]
}

# Data for PPO Agent - Dense Reward Function
ppo_dense = {
    125000: [43.80, 43.30, 45.00, 48.90, 26.80],
    250000: [49.90, 45.80, 38.30, 43.50, 47.10],
    500000: [45.00, 43.60, 45.30, 46.90, 50.90],
    1000000: [49.70, 50.80, 52.60, 56.60, 46.90]
}

# Data for PPO Agent - Sparse Reward Function
ppo_sparse = {
    125000: [37.40, 41.00, 44.40, 32.20, 44.80],
    250000: [48.80, 45.00, 49.90, 46.50, 47.00],
    500000: [41.30, 52.70, 44.30, 53.00, 52.00],
    1000000: [51.50, 53.50, 47.50, 46.80, 53.80]
}

# Calculate averages
timesteps = [125000, 250000, 500000, 1000000]
timesteps_labels = ['125K', '250K', '500K', '1M']

combined_dense_avg = [np.mean(combined_dense[t]) for t in timesteps]
combined_alt_dense_avg = [np.mean(combined_alt_dense[t]) for t in timesteps]
combined_sparse_avg = [np.mean(combined_sparse[t]) for t in timesteps]
ppo_dense_avg = [np.mean(ppo_dense[t]) for t in timesteps]
ppo_sparse_avg = [np.mean(ppo_sparse[t]) for t in timesteps]

# Calculate standard deviations for error bars
combined_dense_std = [np.std(combined_dense[t]) for t in timesteps]
combined_alt_dense_std = [np.std(combined_alt_dense[t]) for t in timesteps]
combined_sparse_std = [np.std(combined_sparse[t]) for t in timesteps]
ppo_dense_std = [np.std(ppo_dense[t]) for t in timesteps]
ppo_sparse_std = [np.std(ppo_sparse[t]) for t in timesteps]

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#2E86AB', '#A23B72', '#F18F01', "#FF3300","#0DFF00","#1900FF"]

# # Graph 1: Dense Reward Function Comparison
# fig1, ax1 = plt.subplots(figsize=(10, 6))
# ax1.plot(timesteps_labels, combined_dense_avg, marker='o', linewidth=2, 
#          label='Combined Agent', color=colors[0], markersize=8)
# ax1.plot(timesteps_labels, ppo_dense_avg, marker='s', linewidth=2, 
#          label='PPO Agent', color=colors[1], markersize=8)
# ax1.fill_between(range(len(timesteps_labels)), 
#                   np.array(combined_dense_avg) - np.array(combined_dense_std),
#                   np.array(combined_dense_avg) + np.array(combined_dense_std),
#                   alpha=0.2, color=colors[0])
# ax1.fill_between(range(len(timesteps_labels)), 
#                   np.array(ppo_dense_avg) - np.array(ppo_dense_std),
#                   np.array(ppo_dense_avg) + np.array(ppo_dense_std),
#                   alpha=0.2, color=colors[1])
# ax1.set_xlabel('Training Timesteps', fontsize=12, fontweight='bold')
# ax1.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
# ax1.set_title('Agent Performance Comparison - Dense Reward Function', 
#               fontsize=14, fontweight='bold')
# ax1.legend(fontsize=11)
# ax1.grid(True, alpha=0.3)
# ax1.set_ylim([20, 65])
# plt.tight_layout()
# plt.savefig('dense_comparison.png', dpi=300, bbox_inches='tight')

# # Graph 2: Sparse Reward Function Comparison
# fig2, ax2 = plt.subplots(figsize=(10, 6))
# ax2.plot(timesteps_labels, combined_sparse_avg, marker='o', linewidth=2, 
#          label='Combined Agent', color=colors[0], markersize=8)
# ax2.plot(timesteps_labels, ppo_sparse_avg, marker='s', linewidth=2, 
#          label='PPO Agent', color=colors[1], markersize=8)
# ax2.fill_between(range(len(timesteps_labels)), 
#                   np.array(combined_sparse_avg) - np.array(combined_sparse_std),
#                   np.array(combined_sparse_avg) + np.array(combined_sparse_std),
#                   alpha=0.2, color=colors[0])
# ax2.fill_between(range(len(timesteps_labels)), 
#                   np.array(ppo_sparse_avg) - np.array(ppo_sparse_std),
#                   np.array(ppo_sparse_avg) + np.array(ppo_sparse_std),
#                   alpha=0.2, color=colors[1])
# ax2.set_xlabel('Training Timesteps', fontsize=12, fontweight='bold')
# ax2.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
# ax2.set_title('Agent Performance Comparison - Sparse Reward Function', 
#               fontsize=14, fontweight='bold')
# ax2.legend(fontsize=11)
# ax2.grid(True, alpha=0.3)
# ax2.set_ylim([35, 60])
# plt.tight_layout()
# plt.savefig('sparse_comparison.png', dpi=300, bbox_inches='tight')

# # Graph 3: All Results Combined
# fig3, ax3 = plt.subplots(figsize=(12, 7))
# ax3.plot(timesteps_labels, combined_dense_avg, marker='o', linewidth=2, 
#          label='Combined Agent (Dense)', color=colors[0], markersize=8)
# ax3.plot(timesteps_labels, combined_sparse_avg, marker='o', linewidth=2, 
#          label='Combined Agent (Sparse)', color=colors[0], markersize=8, 
#          linestyle='--', alpha=0.7)
# ax3.plot(timesteps_labels, ppo_dense_avg, marker='s', linewidth=2, 
#          label='PPO Agent (Dense)', color=colors[1], markersize=8)
# ax3.plot(timesteps_labels, ppo_sparse_avg, marker='s', linewidth=2, 
#          label='PPO Agent (Sparse)', color=colors[1], markersize=8, 
#          linestyle='--', alpha=0.7)
# ax3.set_xlabel('Training Timesteps', fontsize=12, fontweight='bold')
# ax3.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
# ax3.set_title('Complete Agent Performance Comparison - All Reward Functions', 
#               fontsize=14, fontweight='bold')
# ax3.legend(fontsize=10, loc='best')
# ax3.grid(True, alpha=0.3)
# ax3.set_ylim([20, 65])
# plt.tight_layout()
# plt.savefig('all_comparison.png', dpi=300, bbox_inches='tight')

# # Graph 4: Combined Agent - Dense vs Sparse
# fig4, ax4 = plt.subplots(figsize=(10, 6))
# ax4.plot(timesteps_labels, combined_dense_avg, marker='o', linewidth=2.5, 
#          label='Dense Reward', color=colors[2], markersize=10)
# ax4.plot(timesteps_labels, combined_sparse_avg, marker='s', linewidth=2.5, 
#          label='Sparse Reward', color=colors[3], markersize=10)
# ax4.fill_between(range(len(timesteps_labels)), 
#                   np.array(combined_dense_avg) - np.array(combined_dense_std),
#                   np.array(combined_dense_avg) + np.array(combined_dense_std),
#                   alpha=0.2, color=colors[2])
# ax4.fill_between(range(len(timesteps_labels)), 
#                   np.array(combined_sparse_avg) - np.array(combined_sparse_std),
#                   np.array(combined_sparse_avg) + np.array(combined_sparse_std),
#                   alpha=0.2, color=colors[3])
# ax4.set_xlabel('Training Timesteps', fontsize=12, fontweight='bold')
# ax4.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
# ax4.set_title('Combined Agent: Dense vs Sparse Reward Functions', 
#               fontsize=14, fontweight='bold')
# ax4.legend(fontsize=11)
# ax4.grid(True, alpha=0.3)
# ax4.set_ylim([30, 60])
# plt.tight_layout()
# plt.savefig('combined_dense_vs_sparse.png', dpi=300, bbox_inches='tight')

# # Graph 5: PPO Agent - Dense vs Sparse
# fig5, ax5 = plt.subplots(figsize=(10, 6))
# ax5.plot(timesteps_labels, ppo_dense_avg, marker='o', linewidth=2.5, 
#          label='Dense Reward', color=colors[2], markersize=10)
# ax5.plot(timesteps_labels, ppo_sparse_avg, marker='s', linewidth=2.5, 
#          label='Sparse Reward', color=colors[3], markersize=10)
# ax5.fill_between(range(len(timesteps_labels)), 
#                   np.array(ppo_dense_avg) - np.array(ppo_dense_std),
#                   np.array(ppo_dense_avg) + np.array(ppo_dense_std),
#                   alpha=0.2, color=colors[2])
# ax5.fill_between(range(len(timesteps_labels)), 
#                   np.array(ppo_sparse_avg) - np.array(ppo_sparse_std),
#                   np.array(ppo_sparse_avg) + np.array(ppo_sparse_std),
#                   alpha=0.2, color=colors[3])
# ax5.set_xlabel('Training Timesteps', fontsize=12, fontweight='bold')
# ax5.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
# ax5.set_title('PPO Agent: Dense vs Sparse Reward Functions', 
#               fontsize=14, fontweight='bold')
# ax5.legend(fontsize=11)
# ax5.grid(True, alpha=0.3)
# ax5.set_ylim([35, 55])
# plt.tight_layout()
# plt.savefig('ppo_dense_vs_sparse.png', dpi=300, bbox_inches='tight')

# Graph 6: Combined Agent - Alt Dense vs Dense vs Sparse
fig6, ax6 = plt.subplots(figsize=(10, 6))
ax6.plot(timesteps_labels, combined_alt_dense_avg, marker='^', linewidth=2.5, 
         label='Alt Dense Reward', color=colors[3], markersize=10)
ax6.plot(timesteps_labels, combined_dense_avg, marker='o', linewidth=2.5, 
         label='Dense Reward', color=colors[4], markersize=10)
ax6.plot(timesteps_labels, combined_sparse_avg, marker='s', linewidth=2.5, 
         label='Sparse Reward', color=colors[5], markersize=10)
ax6.fill_between(range(len(timesteps_labels)), 
                  np.array(combined_alt_dense_avg) - np.array(combined_alt_dense_std),
                  np.array(combined_alt_dense_avg) + np.array(combined_alt_dense_std),
                  alpha=0.2, color=colors[3])
ax6.fill_between(range(len(timesteps_labels)), 
                  np.array(combined_dense_avg) - np.array(combined_dense_std),
                  np.array(combined_dense_avg) + np.array(combined_dense_std),
                  alpha=0.2, color=colors[4])
ax6.fill_between(range(len(timesteps_labels)), 
                  np.array(combined_sparse_avg) - np.array(combined_sparse_std),
                  np.array(combined_sparse_avg) + np.array(combined_sparse_std),
                  alpha=0.2, color=colors[5])
ax6.set_xlabel('Training Timesteps', fontsize=12, fontweight='bold')
ax6.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
ax6.set_title('Combined Agent: Alt Dense vs Dense vs Sparse Reward Functions', 
              fontsize=14, fontweight='bold')
ax6.legend(fontsize=11)
ax6.grid(True, alpha=0.3)
ax6.set_ylim([35, 55])
plt.tight_layout()
plt.savefig('combined_alt_dense_vs_dense_vs_sparse.png', dpi=300, bbox_inches='tight')

# print("All graphs have been generated and saved!")
# print("\nAverage Win Rates Summary:")
# print("\nCombined Agent (Dense):", [f"{x:.2f}%" for x in combined_dense_avg])
# print("Combined Agent (Sparse):", [f"{x:.2f}%" for x in combined_sparse_avg])
# print("\nPPO Agent (Dense):", [f"{x:.2f}%" for x in ppo_dense_avg])
# print("PPO Agent (Sparse):", [f"{x:.2f}%" for x in ppo_sparse_avg])