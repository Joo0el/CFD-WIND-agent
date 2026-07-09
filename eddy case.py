import os
import glob
import shutil
import subprocess
import rhino3dm

# —— 一、路径配置 —— #
TEMPLATE_DIR = r"F:\Wind ML\eddy_case"                    # 模板所在目录（包含 eddy.yaml + region.yaml）
SRC_3DM_DIR  = r"F:\Wind ML\building90"                   # 源 .3dm 文件目录
OUT_ROOT     = r"F:\Wind ML\output\building90"            # 输出所有 Case 的根目录

# —— 二、开始批量处理 —— #
for idx, src3dm in enumerate(glob.glob(os.path.join(SRC_3DM_DIR, "*.3dm")), start=1):
    case_name = f"case_{idx:03d}"
    case_dir  = os.path.join(OUT_ROOT, case_name)

    # 1) 复制模板目录到新 Case
    shutil.copytree(TEMPLATE_DIR, case_dir)               # 复制整棵目录树 :contentReference[oaicite:0]{index=0}

    # 2) 导入几何：清空旧的 STL 并将 .3dm 导出为 geometry.stl
    tri_dir = os.path.join(case_dir, "constant", "triSurface")
    # 清理旧文件
    for f in os.listdir(tri_dir):
        os.remove(os.path.join(tri_dir, f))
    # 读取 .3dm 并写入 STL（binary）
    model = rhino3dm.File3dm.Read(src3dm)
    stl_fp = os.path.join(tri_dir, "geometry.stl")
    with open(stl_fp, "wb") as stl_file:
        for obj in model.Objects:
            brep = obj.Geometry if isinstance(obj.Geometry, rhino3dm.Brep) else None
            if brep:
                meshes = rhino3dm.Mesh.CreateFromBrep(brep)  # Brep→Mesh :contentReference[oaicite:1]{index=1}
                for mesh in meshes:
                    mesh.Write(stl_file, 1)

    # 3) Mesh & Solver：调用 OpenFOAM / Eddy3D 命令
    #    a) blockMesh（若使用 snappyHexMesh，请在模板中已配置 snappyHexMeshDict，并替换下行）
    subprocess.run(["blockMesh", "-case", case_dir], check=True)  # 生成背景网格 :contentReference[oaicite:2]{index=2}
    #    b) snappyHexMesh（可选，若模板中已配置）
    subprocess.run(["snappyHexMesh", "-overwrite", "-case", case_dir], check=True)
    #    c) Eddy3D 求解
    subprocess.run(["mpiexec", "-np", "4", "eddy", "-case", case_dir], check=True)

    print(f"[完成] {case_name} 模拟完成")

print("所有 Case 已全部处理！")
