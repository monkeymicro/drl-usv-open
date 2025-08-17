import matplotlib.pyplot as plt
import re
import os

HOME = os.path.dirname(os.path.realpath(__file__))
# 训练时间数据
data = {
    "BC": "23m",
    "AWAC": "1h52m",
    "CQL": "4h12m",
    "IQL": "1h10m",
    "REM": "6h18m",
    "SCA-N": "6h5m",
    "EDAC": "2h55m",
    "TD3+BC": "43m",
    "LB-SAC": "15h19m",
}

# 将 "XhYm" 或 "Xm" 转换为分钟
def to_minutes(s: str) -> int:
    h = re.search(r"(\d+)\s*h", s)
    m = re.search(r"(\d+)\s*m", s)
    hours = int(h.group(1)) if h else 0
    mins = int(m.group(1)) if m else 0
    return hours * 60 + mins

algos = list(data.keys())
minutes = [to_minutes(t) for t in data.values()]

# 颜色循环
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = [color_cycle[i % len(color_cycle)] for i in range(len(algos))]

# 绘制柱状图
plt.figure(figsize=(7, 4), dpi=300)  # 稍宽一点
bars = plt.bar(algos, minutes, color=colors)

# 坐标轴标签
plt.ylabel("Training Time (minutes)")
plt.xlabel("Algorithm")
plt.xticks(rotation=30, ha='right')
plt.ylim(0, max(minutes) * 1.15)

# 在柱子顶部标注原始训练时间
for bar, t_str in zip(bars, data.values()):
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height(),
        f"{t_str}",
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.tight_layout()

# 保存高质量图片
# plt.savefig("offline-time.png", bbox_inches="tight")
plt.savefig(f"{HOME}/imgs/offline-time.pdf", bbox_inches="tight")
plt.show()
