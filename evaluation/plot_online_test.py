import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

HOME = os.path.dirname(os.path.realpath(__file__))
cols = 3
rows = 2
subplot_width_px = 600   # 每个子图宽度 400 px
subplot_height_px = 250  # 每个子图高度 250 px
dpi = 100                # 图像分辨率（像素/英寸）

# 计算整张图的尺寸 (inch)
fig_width = (subplot_width_px * cols) / dpi
fig_height = (subplot_height_px * rows) / dpi

# fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
# axes = axes.flatten()
fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=dpi, squeeze=False)
axes = axes.flatten()

env_names = ['TD3', 'SAC']
colors = ['blue', 'blue', 'red', 'blue']
data_all = dict()
idx = 0
for env_name in env_names:
    with open(os.path.join(HOME, f'results/{env_name}_test_data.pkl'), 'rb') as f:
        data_all[f'{env_name}'] = pickle.load(f)

    time = data_all[f'{env_name}']['time']

    # print('time is', len(time))
    # print('data_all is', len(data_all[f'{env_name}']['psi']))
    axes[0 + idx*3].plot(time, data_all[f'{env_name}']['psi'], label=f'{env_name}', color=colors[0]) 
    axes[0 + idx*3].plot(time, data_all[f'{env_name}']['exp_psi'], label='Reference', color='red')
    axes[0 + idx*3].set_ylabel('Yaw (deg)', fontsize=14)
    if idx == len(env_names)-1:  # 最后一行加 xlabel
        axes[0 + idx*3].set_xlabel('Time (sec)', fontsize=14)
    axes[0 + idx*3].set_ylim(-60, 65)
    axes[0 + idx*3].legend(loc='upper right', fontsize=12)
    axes[0 + idx*3].grid(True, alpha=0.5)

    axes[1 + idx*3].plot(time, data_all[f'{env_name}']['V'], label=f'{env_name}', color='red')
    axes[1 + idx*3].plot(time, data_all[f'{env_names[0]}']['exp_speed'], label='Reference', color=colors[1])
    axes[1 + idx*3].set_ylabel('Velocity (m/s)', fontsize=14)
    if idx == len(env_names)-1:  # 最后一行加 xlabel
        axes[1 + idx*3].set_xlabel('Time (sec)', fontsize=14)
    axes[1 + idx*3].set_ylim(-2.5, 2)
    axes[1 + idx*3].legend(loc='upper left', fontsize=12)
    axes[1 + idx*3].grid(True, alpha=0.5)

    axes[2 + idx*3].plot(time, data_all[f'{env_name}']['rspl'], label=r'$\delta_{port}$', color=colors[2])
    axes[2 + idx*3].plot(time, data_all[f'{env_name}']['rspr'], label=r'$\delta_{stbd}$', color=colors[3])
    axes[2 + idx*3].set_ylabel('RPM', fontsize=14)
    if idx == len(env_names)-1:  # 最后一行加 xlabel
        axes[2 + idx*3].set_xlabel('Time (sec)', fontsize=14)
    axes[2 + idx*3].legend(loc='upper left', fontsize=12)
    axes[2 + idx*3].grid(True, alpha=0.5)

    idx += 1
    # 计算MAE
    mae_psi = np.mean(np.abs(np.array(data_all[f'{env_name}']['psi']) - np.array(data_all[f'{env_name}']['exp_psi'])))
    mae_speed = np.mean(np.abs(np.array(data_all[f'{env_name}']['V']) - np.array(data_all[f'{env_name}']['exp_speed'])))
    mae_port = np.mean(np.abs(np.array(data_all[f'{env_name}']['rspl']) - 0))
    mae_stbd = np.mean(np.abs(np.array(data_all[f'{env_name}']['rspr']) - 0))

    print(f"MAE for {env_name}: Yaw: {mae_psi:.2f}, Speed: {mae_speed:.2f}, port: {mae_port:.2f}, stbd: {mae_stbd:.2f}")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{HOME}/imgs/online-test.pdf", bbox_inches="tight")
plt.show()

