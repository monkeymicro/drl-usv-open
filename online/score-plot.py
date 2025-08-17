import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

HOME = os.path.dirname(os.path.realpath(__file__))
# Define algorithm names
algorithms = ['DDPG.pkl', 'TD3.pkl', 'SAC-2.pkl']

# Colors for different algorithms
colors = ['blue', 'orange', 'green']
frames_per_episode = 1000
file_name = []
for algorithm in algorithms:
    algorithm_path = os.path.join(HOME, 'models', algorithm)
    file_name.append(algorithm_path)
print('file_name is', file_name) 

num_files = len(file_name)
cols = 3
rows = (num_files + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey='row', squeeze=False)
axes = axes.flatten()

last_score = []
for idx, file in enumerate(file_name):
    try:
        data = torch.load(file, map_location=torch.device('cpu'))
        if 'score' in data:
            score = np.array(data['score']) / frames_per_episode
            last_score.append(score[-50:].mean())  # mean of last 50 episodes
            ax = axes[idx]
            ax.plot(score, label=file.split('/')[-1].split('.')[0], color=colors[idx])
            ax.set_xlabel('Episodes', fontsize=14)
            if idx % cols == 0:  # Only label the first subplot of each row
                ax.set_ylabel('Average Reward', fontsize=14)
            ax.legend(loc='center right', fontsize=12)
            ax.grid(True)
    except pickle.UnpicklingError as e:
        print(f"Could not load {file}: {e}")
        continue
print('Last scores:', last_score)
for i in range(num_files, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{HOME}/imgs/online-train.pdf", bbox_inches="tight")
plt.show()