import os
import csv
import numpy as np
import rasterio
from rasterio.transform import Affine
from stl import mesh

# === 配置开始 ===
# 输入目录：GeoTIFF 文件所在目录
tif_dir = r"F:\Wind ML"
# 中间目录：CSV 输出目录
csv_dir = r"F:\Wind ML\建筑矩阵"
# 输出目录：STL 输出目录
stl_dir = r"F:\Wind ML\stl"
# NoData 值，CSV 中空值、STL 中高度为 0
# 实际 nodata 值会从 GeoTIFF 中读取，并转换为 np.nan
# === 配置结束 ===


def read_transform_coords(transform: Affine, width: int, height: int):
    """
    计算每列 (xs) 和每行 (ys) 的中心坐标
    """
    xs = []
    for j in range(width):
        x, y = transform * (j + 0.5, 0.5)
        xs.append(x)
    ys = []
    for i in range(height):
        x, y = transform * (0.5, i + 0.5)
        ys.append(y)
    return xs, ys


def tif_to_grid(tif_path):
    """
    从 GeoTIFF 中读取数据、坐标及生成高度矩阵 (np.float32)，nodata->np.nan
    返回: xs, ys, grid
    """
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
        nodata = src.nodata
        height, width = data.shape
        # nodata 转 np.nan
        if nodata is not None:
            data[data == nodata] = np.nan
        # 计算中心坐标
        xs, ys = read_transform_coords(transform, width, height)
    return xs, ys, data


def write_csv(xs, ys, grid, csv_path):
    """
    写 CSV：空值写空字符串
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Y / X"] + [f"{x:.2f}" for x in xs])
        for i, y in enumerate(ys):
            row = [f"{y:.2f}"]
            for val in grid[i]:
                row.append("" if np.isnan(val) else f"{val:.3f}")
            writer.writerow(row)
    print(f"CSV 已生成: {csv_path}")


def grid_to_stl(xs, ys, grid, stl_path):
    """
    将矩阵转换为立方体网格的 STL 输出
    单元由相邻 xs[j],xs[j+1] 和 ys[i],ys[i+1] 定义
    空值或 nan 当作高度 0
    """
    n_y, n_x = grid.shape
    vertices = []
    faces = []

    def add_cell(i, j, h):
        x0, x1 = xs[j], xs[j+1]
        y0, y1 = ys[i], ys[i+1]
        base_z = 0.0
        top_z = 0.0 if np.isnan(h) else float(h)
        pts = [
            (x0,y0,base_z),(x1,y0,base_z),(x1,y1,base_z),(x0,y1,base_z),
            (x0,y0,top_z),(x1,y0,top_z),(x1,y1,top_z),(x0,y1,top_z)
        ]
        idx0 = len(vertices)
        vertices.extend(pts)
        # 底面
        faces.extend([[idx0,idx0+2,idx0+1],[idx0,idx0+3,idx0+2]])
        # 顶面
        faces.extend([[idx0+4,idx0+5,idx0+6],[idx0+4,idx0+6,idx0+7]])
        # 四侧面
        sides = [(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
        for a,b,c,d in sides:
            faces.extend([[idx0+a,idx0+b,idx0+c],[idx0+a,idx0+c,idx0+d]])

    for i in range(n_y-1):
        for j in range(n_x-1):
            add_cell(i, j, grid[i,j])

    vertices_np = np.array(vertices)
    faces_np = np.array(faces)
    stl_mesh = mesh.Mesh(np.zeros(faces_np.shape[0], dtype=mesh.Mesh.dtype))
    for idx, f in enumerate(faces_np):
        for k in range(3):
            stl_mesh.vectors[idx][k] = vertices_np[f[k]]
    os.makedirs(os.path.dirname(stl_path), exist_ok=True)
    stl_mesh.save(stl_path)
    print(f"STL 已生成: {stl_path}")


def main():
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(stl_dir, exist_ok=True)
    tif_files = [f for f in os.listdir(tif_dir) if f.lower().endswith('.tif')]
    if not tif_files:
        print("未找到 GeoTIFF 文件")
        return

    for tif in tif_files:
        base = os.path.splitext(tif)[0]
        tif_path = os.path.join(tif_dir, tif)
        xs, ys, grid = tif_to_grid(tif_path)
        csv_path = os.path.join(csv_dir, f"{base}.csv")
        stl_path = os.path.join(stl_dir, f"{base}.stl")
        write_csv(xs, ys, grid, csv_path)
        grid_to_stl(xs, ys, grid, stl_path)

if __name__ == '__main__':
    main()
