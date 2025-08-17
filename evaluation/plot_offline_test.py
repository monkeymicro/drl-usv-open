import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

HOME = os.path.dirname(os.path.realpath(__file__))
cols = 3
rows = 8

subplot_width_px = 600   # 每个子图宽度 400 px
subplot_height_px = 200  # 每个子图高度 250 px
dpi = 100                # 图像分辨率（像素/英寸）

# 计算整张图的尺寸 (inch)
fig_width = (subplot_width_px * cols) / dpi
fig_height = (subplot_height_px * rows) / dpi

# fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
# axes = axes.flatten()
fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=dpi, squeeze=False)
axes = axes.flatten()

env_names = ['BC-custom', 'AWAC-custom', 'CQL-custom', 'REM-custom', 'SACN-custom', 'EDAC-custom', 'TD3_BC-custom', 'LB-SAC-custom']
label_names = ['BC', 'AWAC', 'CQL', 'REM', 'SAC-N', 'EDAC', 'TD3+BC', 'LB-SAC']
colors = ['blue', 'blue', 'red', 'blue']
data_all = dict()
idx = 0
for env_name in env_names:
    with open(os.path.join(HOME, f'results/{env_name}_test_data.pkl'), 'rb') as f:
        data_all[f'{env_name}'] = pickle.load(f)

    time = data_all[f'{env_name}']['time']

    # yaw
    ax = axes[0 + idx*3]
    ax.plot(time, data_all[f'{env_name}']['psi'], label=f'{label_names[0 + idx]}', color=colors[0]) 
    ax.plot(time, data_all[f'{env_name}']['exp_psi'], label='Reference', color='red')
    ax.set_ylabel('Yaw (deg)', fontsize=14)
    ax.set_ylim(-60, 65)
    if idx == len(env_names)-1:  # 最后一行加 xlabel
        ax.set_xlabel('Time (sec)', fontsize=14)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.5)

    # velocity
    ax = axes[1 + idx*3]
    ax.plot(time, data_all[f'{env_name}']['V'], label=f'{label_names[0 + idx]}', color='red')
    ax.plot(time, data_all[f'{env_name}']['exp_speed'], label='Reference', color=colors[1])
    ax.set_ylabel('Velocity (m/s)', fontsize=14)
    ax.set_ylim(-2.5, 2)
    if idx == len(env_names)-1:
        ax.set_xlabel('Time (sec)', fontsize=14)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.5)

    # rpm
    ax = axes[2 + idx*3]
    ax.plot(time, data_all[f'{env_name}']['rspl'], label=r'$\delta_{port}$', color=colors[2])
    ax.plot(time, data_all[f'{env_name}']['rspr'], label=r'$\delta_{stbd}$', color=colors[3])
    ax.set_ylabel('RPM', fontsize=14)
    ax.set_ylim(-5000, 3000)
    if idx == len(env_names)-1:
        ax.set_xlabel('Time (sec)', fontsize=14)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.5)

    idx += 1
    # 计算MAE
    mae_psi = np.mean(np.abs(np.array(data_all[f'{env_name}']['psi']) - np.array(data_all[f'{env_name}']['exp_psi'])))
    mae_speed = np.mean(np.abs(np.array(data_all[f'{env_name}']['V']) - np.array(data_all[f'{env_name}']['exp_speed'])))
    mae_port = np.mean(np.abs(np.array(data_all[f'{env_name}']['rspl']) - 0))
    mae_stbd = np.mean(np.abs(np.array(data_all[f'{env_name}']['rspr']) - 0))

    print(f"MAE for {env_name}: Yaw: {mae_psi:.2f}, Speed: {mae_speed:.2f}, port: {mae_port:.2f}, stbd: {mae_stbd:.2f}")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{HOME}/imgs/offline-test.pdf", bbox_inches="tight")
plt.show()