
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import torch
from scipy import stats

HOME = os.path.dirname(os.path.realpath(__file__))

# Define algorithm names
algorithms = ['BC-custom',
        'AWAC-custom',
        'CQL-custom',
        'IQL-custom', 
        'REM-custom', # num_critics 50
        'SACN-custom', # num_critics 50
        'EDAC-custom',# num_critics 10
        'TD3_BC-custom',
        'LB-SAC-custom'] # num_critics 10
# Colors for different algorithms
colors = ['chocolate', 'blue', 'orange', 'green', 'red', 'purple', 'brown', 'black', 'olive']
labels = ['BC', 'AWAC', 'CQL', 'IQL', 'REM', 'SAC-N', 'EDAC', 'TD3+BC', 'LB-SAC']
frames_per_episode = 1000  

rows, cols = 3, 3
fig, axes = plt.subplots(rows, cols, figsize=(15, 8), sharex=True, sharey=True)
axes = axes.flatten()

last_score = []
for idx, alg_name in enumerate(algorithms):
    # file for this algorithm
    file_path = os.path.join(HOME, 'models', alg_name, 'evaluations.pkl')
    with open(file_path, 'rb') as f:
        score = pickle.load(f)
        score = np.array(score) / frames_per_episode
    last_score.append(score[-10:].mean()) 
    episodes = np.arange(len(score))
    ax = axes[idx]
    ax.plot(episodes, score, label=labels[idx], color=colors[idx])
    ax.set_xlabel('Episodes', fontsize=14)
    ax.set_ylabel('Average Reward', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=12)

print('Last scores:', last_score)
# Hide unused subplots if any
for i in range(len(algorithms), len(axes)):
    fig.delaxes(axes[i])

# fig.suptitle('Scores from different models with confidence intervals')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{HOME}/imgs/offline-train.pdf", bbox_inches="tight")
plt.show()