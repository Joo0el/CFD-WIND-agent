import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from pathlib import Path

# 输入/输出目录
input_dir  = r"F:\Wind ML\CSV\tile_1_2"
output_dir = r"F:\Wind ML\PNG\tile_1_2"
gif_output_dir = r"F:\Wind ML\PNG\GIF"

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)
os.makedirs(gif_output_dir, exist_ok=True)

# 统一的色标范围
vmin, vmax = 0, 15

# 遍历所有 CSV
for fname in os.listdir(input_dir):
    if not fname.lower().endswith(".csv"):
        continue

    csv_path = os.path.join(input_dir, fname)
    df = pd.read_csv(csv_path)

    # 1 m 网格 binning
    df["x_bin"] = (df["x"] / 1).round() * 1
    df["y_bin"] = (df["y"] / 1).round() * 1
    df2 = df.groupby(["x_bin", "y_bin"], as_index=False).agg(WS=("WS", "mean"))

    xi = np.sort(df2["x_bin"].unique())
    yi = np.sort(df2["y_bin"].unique())

    # 跳过空数据
    if xi.size == 0 or yi.size == 0:
        print(f"[WARN] {fname}: no bins → skip")
        continue

    Xi, Yi = np.meshgrid(xi, yi)

    Z = (
        df2
        .pivot(index="y_bin", columns="x_bin", values="WS")
        .reindex(index=yi, columns=xi)
        .values
    )
    if Z.size == 0 or np.all(np.isnan(Z)):
        print(f"[WARN] {fname}: empty grid → skip")
        continue

    # 从文件名中提取 "t" 后的数值并转换为时间格式（小时:分钟）
    base_name = os.path.splitext(fname)[0]
    if "t" in base_name:
        t_value = base_name.split("t")[-1]
        try:
            seconds = int(t_value)
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            time_label = f"{hours}h{minutes:02d}m"
        except ValueError:
            time_label = t_value
    else:
        time_label = base_name

    # 画图
    fig, ax = plt.subplots(figsize=(6, 6))
    pcm = ax.pcolormesh(
        Xi, Yi, Z,
        shading="auto",
        vmin=vmin, vmax=vmax,
        cmap="viridis"
    )

    cbar = fig.colorbar(pcm, ax=ax, label="WS (m/s)")
    cbar.set_ticks(np.linspace(vmin, vmax, 6))

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Wind Speed\n{time_label}")

    # 保存
    out_png = os.path.join(output_dir, f"{time_label}.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")

# 找到所有 PNG 并按时间顺序排序
def extract_time(fname):
    base = Path(fname).stem
    if "h" in base and "m" in base:
        try:
            h_part = int(base.split("h")[0])
            m_part = int(base.split("h")[-1].replace("m", ""))
            return h_part * 60 + m_part
        except:
            return float('inf')
    return float('inf')

png_files = sorted(
    [f for f in os.listdir(output_dir) if f.lower().endswith(".png")],
    key=extract_time
)

# 合成 GIF
images = []
for png_file in png_files:
    image_path = os.path.join(output_dir, png_file)
    images.append(imageio.imread(image_path))

out_gif = os.path.join(gif_output_dir, "wind_speed.gif")
imageio.mimsave(out_gif, images, duration=0.25)

print(f"GIF saved to: {out_gif}")
