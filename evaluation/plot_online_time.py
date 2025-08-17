import matplotlib.pyplot as plt
import re
import os

HOME = os.path.dirname(os.path.realpath(__file__))
# 训练时间数据
data = {
    "DDPG": "1h17m",
    "TD3": "1h4m",
    "SAC": "2h5m",
}

# 将 "XhYm" 转换为分钟
def to_minutes(s: str) -> int:
    h = re.search(r"(\d+)\s*h", s)
    m = re.search(r"(\d+)\s*m", s)
    hours = int(h.group(1)) if h else 0
    mins = int(m.group(1)) if m else 0
    return hours * 60 + mins

algos = list(data.keys())
minutes = [to_minutes(t) for t in data.values()]

# 给每个算法设置不同颜色（Matplotlib 内置颜色）
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝、橙、绿

# 绘制柱状图
plt.figure(figsize=(5, 3.2), dpi=300)  # 高分辨率期刊尺寸
bars = plt.bar(algos, minutes, color=colors)

# 坐标轴标签
plt.ylabel("Training Time (minutes)")
plt.xlabel("Algorithm")
plt.xticks(rotation=0)
plt.ylim(0, max(minutes) * 1.15)

# 在柱子顶部标注数值
for bar, val, t_str in zip(bars, minutes, data.values()):
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height(),
        f"{t_str}",  # 显示原始“1h17m”格式
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

# 保存高质量图片
# plt.savefig("online-time.png", bbox_inches="tight")
plt.savefig(f"{HOME}/imgs/online-time.pdf", bbox_inches="tight")
plt.show()
