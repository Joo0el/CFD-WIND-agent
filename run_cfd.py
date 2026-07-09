# run_windAroundBuildings_final.py
# -*- coding: utf-8 -*-
import os
import shutil
import subprocess
import math
import numpy as np
from scipy.interpolate import griddata
import meshio
import re
from textwrap import indent
# Configuration paths
stl_dir      = r"F:\Wind ML\stl\NY500"
template_dir = r"F:\Wind ML\windAroundBuildingsny"
output_root  = r"F:\Wind ML\wind_cases"
CSV_ROOT     = r"F:\Wind ML\CSV"
# OpenFOAM environment locator
def get_of_binaries():
    setvars = r"C:\Progra~1\blueCFD-Core-2020\setvars_OF8.bat"
    bin_dir = r"C:\Progra~1\blueCFD-Core-2020\OpenFOAM-8\platforms\mingw_w64GccDPInt32Opt\bin"
    return setvars, bin_dir

setvars_bat, OF_BIN = get_of_binaries()
BLOCKMESH       = os.path.join(OF_BIN, "blockMesh.exe")
SURFACEFEATURES = os.path.join(OF_BIN, "surfaceFeatures.exe")
SNAPPY          = os.path.join(OF_BIN, "snappyHexMesh.exe")
SIMPLEFOAM      = os.path.join(OF_BIN, "simpleFoam.exe")
POSTPROCESS     = os.path.join(OF_BIN, "postProcess.exe")
FOAM_TO_VTK     = os.path.join(OF_BIN, "foamToVTK.exe")

PLANE_Z    = 2         # 要提取的高度
PLANE_TOL  = 0.1       # 容差(m)
BASE_U = 5

# Simulation parameters
cellSize = 4       # target cell size in meters
ANGLES   = list(range(0,360,45))

# 1) 先列出时序（times, Ux_values）
times = [i * 300.0 for i in range(288)]
Ux_values = [
    1.5000, 1.4773, 1.4545, 1.4318, 1.4091, 1.3864, 1.3636, 1.3409, 1.3182, 1.2955, 1.2727, 1.2500,
    1.2375, 1.2250, 1.2125, 1.2000, 1.1875, 1.1750, 1.1625, 1.1500, 1.1375, 1.1250, 1.1125, 1.1000,
    1.1045, 1.1091, 1.1136, 1.1182, 1.1227, 1.1273, 1.1318, 1.1364, 1.1409, 1.1455, 1.1500, 1.1545,
    1.1591, 1.1636, 1.1682, 1.1727, 1.1773, 1.1818, 1.1864, 1.1909, 1.1955, 1.2000, 1.2091, 1.2182,
    1.2273, 1.2364, 1.2455, 1.2545, 1.2636, 1.2727, 1.2818, 1.2909, 1.3000, 1.3273, 1.3545, 1.3818,
    1.4091, 1.4364, 1.4636, 1.4909, 1.5182, 1.5455, 1.5727, 1.6000, 1.7091, 1.8182, 1.9273, 2.0364,
    2.1455, 2.2545, 2.3636, 2.4727, 2.5818, 2.6909, 2.8000, 3.0364, 3.2727, 3.5091, 3.7455, 3.9818,
    4.2182, 4.4545, 4.6909, 4.9273, 5.1636, 5.4000, 5.5136, 5.6273, 5.7409, 5.8545, 5.9682, 6.0818,
    6.1955, 6.3091, 6.4227, 6.5364, 6.6500, 6.7273, 6.8045, 6.8818, 6.9591, 7.0364, 7.1136, 7.1909,
    7.2682, 7.3455, 7.4227, 7.5000, 7.5045, 7.5091, 7.5136, 7.5182, 7.5227, 7.5273, 7.5318, 7.5364,
    7.5409, 7.5455, 7.5500, 7.5455, 7.5409, 7.5364, 7.5318, 7.5273, 7.5227, 7.5182, 7.5136, 7.5091,
    7.5045, 7.5000, 7.4773, 7.4545, 7.4318, 7.4091, 7.3864, 7.3636, 7.3409, 7.3182, 7.2955, 7.2727,
    7.2500, 7.2045, 7.1591, 7.1136, 7.0682, 7.0227, 6.9773, 6.9318, 6.8864, 6.8409, 6.7955, 6.7500,
    6.6818, 6.6136, 6.5455, 6.4773, 6.4091, 6.3409, 6.2727, 6.2045, 6.1364, 6.0682, 6.0000, 5.9091,
    5.8182, 5.7273, 5.6364, 5.5455, 5.4545, 5.3636, 5.2727, 5.1818, 5.0909, 5.0000, 4.9091, 4.8182,
    4.7273, 4.6364, 4.5455, 4.4545, 4.3636, 4.2727, 4.1818, 4.0909, 4.0000, 3.8636, 3.7273, 3.5909,
    3.4545, 3.3182, 3.1818, 3.0455, 2.9091, 2.7727, 2.6364, 2.5000, 2.4545, 2.4091, 2.3636, 2.3182,
    2.2727, 2.2273, 2.1818, 2.1364, 2.0909, 2.0455, 2.0000, 1.9636, 1.9273, 1.8909, 1.8545, 1.8182,
    1.7818, 1.7455, 1.7091, 1.6727, 1.6364, 1.6000, 1.5727, 1.5455, 1.5182, 1.4909, 1.4636, 1.4364,
    1.4091, 1.3818, 1.3545, 1.3273, 1.3000, 1.2818, 1.2636, 1.2455, 1.2273, 1.2091, 1.1909, 1.1727,
    1.1545, 1.1364, 1.1182, 1.1000, 1.0864, 1.0727, 1.0591, 1.0455, 1.0318, 1.0182, 1.0045, 0.9909,
    0.9773, 0.9636, 0.9500, 0.9364, 0.9227, 0.9091, 0.8955, 0.8818, 0.8682, 0.8545, 0.8409, 0.8273,
    0.8136, 0.8000, 0.7909, 0.7818, 0.7727, 0.7636, 0.7545, 0.7455, 0.7364, 0.7273, 0.7182, 0.7091,
    0.7000, 0.6909, 0.6818, 0.6727, 0.6636, 0.6545, 0.6455, 0.6364, 0.6273, 0.6182, 0.6091, 0.6000,
    0.5909, 0.5818, 0.5727, 0.5636, 0.5545, 0.5455, 0.5364, 0.5273, 0.5182, 0.5091, 0.5000
]
time_series = list(zip(times, Ux_values))


# Templates (unchanged from your last version)
SURFACE_FEATURES_DICT = '''/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
|  \\\\    /   O peration     | Website: https://openfoam.org
|   \\\\  /    A nd           | Version:  8
|    \\/     M anipulation    |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      surfaceFeaturesDict;
}}

// List of STL surfaces to extract feature edges from:
surfaces
(
    "{stl_file}"
);

// Angle threshold (degrees) for feature detection:
includedAngle    150;

subsetFeatures
{{
    nonManifoldEdges yes;
    openEdges        yes;
}}

trimFeatures
{{
    minElem 0;
    minLen  0;
}}

writeObj yes;

// ************************************************************************* //
'''



SNAPPY_DICT_TEMPLATE = '''/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
|  \\\\    /   O peration     | Website: https://openfoam.org
|   \\\\  /    A nd           | Version:  8
|    \\/     M anipulation    |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}

// include standard settings
#includeEtc "caseDicts/mesh/generation/snappyHexMeshDict.cfg"

castellatedMesh    on;
snap               on;
addLayers          off;

geometry
{{
    {case_name}
    {{
        type triSurfaceMesh;
        file "{stl_file}";
    }}
    refinementBox_{case_name}
    {{
        type searchableBox;
        // buffered bounds by {buffer} meters
        min ({xmin_buf:.3f} {ymin_buf:.3f} {zmin_buf:.3f});
        max ({xmax_buf:.3f} {ymax_buf:.3f} {zmax_buf:.3f});
    }}
}};   // <–– 一定要在这里闭合 geometry 并加分号

castellatedMeshControls
{{
    mergePatchPairs ();
    features
    (
        {{ file "{case_name}.eMesh"; level 2; }}
    );

    refinementSurfaces
    {{
        {case_name}
        {{
            level     (2 2);
            patchInfo {{ type wall; }}
        }}
    }}

    refinementRegions
    {{
        refinementBox_{case_name}
        {{
            mode   inside;
            levels ((1E15 1));
        }}
    }}

    // a point inside the boxed region
    locationInMesh ({cx:.3f} {cy:.3f} {cz:.3f});

    // skip merge‐patch‐faces step
    mergePatchPairs ();

    maxLocalCells       300000000;
    maxGlobalCells      900000000;
    minRefinementCells  50;
    maxLoadUnbalance    1.10;
    nCellsBetweenLevels 4;
    resolveFeatureAngle 150;
    allowFreeStandingZoneFaces false;
}}

snapControls
{{
    explicitFeatureSnap  false;
    implicitFeatureSnap  true;
    nSmoothPatch         4;
    tolerance            1.0;
    nSolveIter           20;
}}

addLayersControls
{{
    layers
    {{
        "{case_name}"
        {{
            nSurfaceLayers 2;
        }}
    }}
    relativeSizes       true;
    expansionRatio      1.2;
    finalLayerThickness 0.5;
    minThickness        1e-3;
    nGrow               0;
    featureAngle        60;
}}

meshQualityControls
{{
    maxNonOrtho           75;
    maxBoundarySkewness   20;
    maxInternalSkewness   4;
    maxConcave            80;
    minFlatness           0.5;
    minVol                1e-15;
    minTetQuality         1e-40;
    nSmoothScale          4;
    errorReduction        0.75;
}}

writeFlags
(
    scalarLevels
    layerSets
    layerFields
);

mergeTolerance 1e-4;


// ************************************************************************* //
'''


BLOCKMESH_DICT_TEMPLATE = '''
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website: https://openfoam.org
    \\\\  /    A nd           | Version:  8
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

#inputSyntax slash;

backgroundMesh
{{
    xMin   {xmin};
    xMax   {xmax};
    yMin   {ymin};
    yMax   {ymax};
    zMin   {zmin};
    zMax   {zmax};
    xCells {xCells};
    yCells {yCells};
    zCells {zCells};
}}

convertToMeters 1;

vertices
(
    ({xmin} {ymin} {zmin})
    ({xmax} {ymin} {zmin})
    ({xmax} {ymax} {zmin})
    ({xmin} {ymax} {zmin})

    ({xmin} {ymin} {zmax})
    ({xmax} {ymin} {zmax})
    ({xmax} {ymax} {zmax})
    ({xmin} {ymax} {zmax})
);

blocks
(
    hex (0 1 2 3 4 5 6 7)
    (
        {xCells}
        {yCells}
        {zCells}
    )
    simpleGrading (1 1 1)
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            (0 3 7 4)
        );
    }}

    outlet
    {{
        type patch;
        faces
        (
            (1 5 6 2)
        );
    }}

    ground
    {{
        type wall;
        faces
        (
            (0 1 2 3)
        );
    }}

    frontAndBack
    {{
        type symmetry;
        faces
        (
            (0 4 5 1)
            (3 2 6 7)
            (4 7 6 5)
        );
    }}
);

// ************************************************************************* //
'''
CONTROL_DICT_TEMPLATE ='''
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  8                                     |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}

application     pisoFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         84800;
deltaT          600;

writeControl    timeStep;
writeInterval   1;
purgeWrite      0;

writeFormat     ascii;
writePrecision  3;
writeCompression off;

timeFormat      general;
timePrecision   3;
runTimeModifiable true;

functions
{{

   samplePlane
    {{
        type               sampledPlane;
        libs               ("libsampling.so");
        enabled            true;
        writeControl       timeStep;
        writeInterval      6;
        surfaceFormat      raw;
        interpolationScheme cellPoint;

        plane
        {{
            type           plane;
            planeType      pointAndNormal;
            basePoint      (0 0 2);
            normalVector   (0 0 1);
        }}

        fields             (U);
    }}
}}
'''

# 2) 然后定义完整的模板（注意使用 f-string 引入 coded_block）
U0_template = r"""/*--------------------------------*- C++ -*----------------------------------*\\
| OpenFOAM: uniformFixedValue + Function1（table）示例              |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{{
    inlet
    {{
        type            uniformFixedValue;
        // 直接把 table 作为关键字
        uniformValue     table
        (
{table_entries}
        );
        outOfRange       clamp;
    }}

    outlet
    {{
        type            pressureInletOutletVelocity;
        value           uniform (0 0 0);
    }}

    wall
    {{
        type            noSlip;
    }}

    frontAndBack
    {{
        type            symmetry;
    }}
}}
"""


# 先列出所有时刻和对应的Ux值

def run_cmd(cmd, case_dir):
    exe = ' '.join(f'"{c}"' for c in cmd)
    subprocess.run(f'call "{setvars_bat}" && cd /d "{case_dir}" && {exe}',
                   shell=True, check=True)

def get_bbox_and_center(path):
    mesh = meshio.read(path)
    pts = np.asarray(mesh.points)
    xmin, xmax = pts[:,0].min(), pts[:,0].max()
    ymin, ymax = pts[:,1].min(), pts[:,1].max()
    zmin, zmax = pts[:,2].min(), pts[:,2].max()
    return (xmin,xmax,ymin,ymax,zmin,zmax), \
           ((xmin+xmax)/2,(ymin+ymax)/2,(zmin+zmax)/2)

def write_blockMeshDict(case_dir, bbox):
    sysd = os.path.join(case_dir,'system'); os.makedirs(sysd,exist_ok=True)
    xmin,xmax,ymin,ymax,zmin,zmax = bbox
    dx,dy,dz = xmax-xmin,ymax-ymin,zmax-zmin
    xCells = max(1,int(round(dx/cellSize)))
    yCells = max(1,int(round(dy/cellSize)))
    zCells = max(1,int(round(dz/cellSize)))
    content = BLOCKMESH_DICT_TEMPLATE.format(**locals())
    with open(os.path.join(sysd,'blockMeshDict'),'w') as f:
        f.write(content)

def get_inlet_face_range(bnd_path):
    """返回 (startFace, nFaces)"""
    startFace = nFaces = None
    inside = False
    with open(bnd_path) as f:
        for line in f:
            linestripped = line.strip()
            if not inside:
                if linestripped.startswith("inlet"):
                    inside = True
                continue
            # 进到 inlet 块里
            if linestripped.startswith("startFace"):
                startFace = int(linestripped.split()[1].rstrip(';'))
            elif linestripped.startswith("nFaces"):
                nFaces = int(linestripped.split()[1].rstrip(';'))
            elif linestripped == "}":
                break
    if startFace is None or nFaces is None:
        raise RuntimeError("没能在 boundary 文件里找到 startFace/nFaces")
    return startFace, nFaces


def write_surfaceFeaturesDict(case_dir,name):
    sysd=os.path.join(case_dir,'system');os.makedirs(sysd,exist_ok=True)
    content = SURFACE_FEATURES_DICT.format(stl_file=name+'.stl')
    with open(os.path.join(sysd,'surfaceFeaturesDict'),'w') as f:
        f.write(content)

def generate_snappy_dict(case_name, stl_file, bounds, buffer=100):
    """
    Generate snappyHexMeshDict content for a given case.
    - case_name: name of the case (used for patch and file names)
    - stl_file: STL filename to reference
    - bounds: tuple (xmin, xmax, ymin, ymax, zmin, zmax)
    - buffer: extent by which to expand the bounding box
    """
    # unpack original bounds
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    # apply buffer
    xmin_buf = xmin - buffer
    xmax_buf = xmax + buffer
    ymin_buf = ymin - buffer
    ymax_buf = ymax + buffer
    zmin_buf = zmin - buffer
    zmax_buf = zmax + buffer
    # compute mesh location point (center of buffered box)
    cx = 0.5 * (xmin_buf + xmax_buf)
    cy = 0.5 * (ymin_buf + ymax_buf)
    cz = 0.5 * (zmin_buf + zmax_buf)

    # fill template
    return SNAPPY_DICT_TEMPLATE.format(
        case_name=case_name,
        stl_file=stl_file,
        xmin_buf=xmin_buf, ymin_buf=ymin_buf, zmin_buf=zmin_buf,
        xmax_buf=xmax_buf, ymax_buf=ymax_buf, zmax_buf=zmax_buf,
        cx=cx, cy=cy, cz=cz,
        buffer=buffer
    )
def write_timeVarying_table(case_dir, time_series):
    """在 constant/boundaryData/inlet/ 下生成表 inlet，格式为 (time (Ux Uy Uz))"""
    bd = os.path.join(case_dir, "constant", "boundaryData", "inlet")
    os.makedirs(bd, exist_ok=True)
    fn = os.path.join(bd, "inlet")    # 无扩展名
    with open(fn, "w") as f:
        f.write("(\n")
        for t, ux in time_series:
            f.write(f"    ({t:.-f}    ({ux:.3f} 0.0 0.0))\n")
        f.write(")\n")
    print(f"➡ 写入 timeVaryingUniformFixedValue 表: {fn}")

def write_U0_with_table(case_dir, time_series):
    zero_dir = os.path.join(case_dir, "0"); os.makedirs(zero_dir, exist_ok=True)
    # 构造 table_entries 字符串
    lines = []
    for t, ux in time_series:
        lines.append(f"    ({int(t):<5}    ({ux:.3f} 0 0))")
    table_entries = "\n".join(lines)
    # 渲染模板
    content = U0_template.format(table_entries=table_entries)
    # 写文件
    with open(os.path.join(zero_dir, "U"), "w") as f:
        f.write(content)
    print(f"➡ 写入 U (uniformFixedValue + table) 到 {zero_dir}/U")


def write_snappyDict(case_dir, name, bbox, center, buffer=100):
    # 确保 system 目录存在
    sysd = os.path.join(case_dir, 'system')
    os.makedirs(sysd, exist_ok=True)
    # 调用 generate_snappy_dict 生成带 buffer 的完整字典内容
    content = generate_snappy_dict(
        case_name=name,
        stl_file=name+'.stl',
        bounds=bbox,
        buffer=buffer
    )
    # 写文件
    with open(os.path.join(sysd, 'snappyHexMeshDict'), 'w') as f:
        f.write(content)


# 插入一个写 controlDict 的辅助函数
def write_controlDict(case_dir):
    system_dir = os.path.join(case_dir, 'system')
    os.makedirs(system_dir, exist_ok=True)
    dst = os.path.join(system_dir, 'controlDict')
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(CONTROL_DICT_TEMPLATE.format())
    print(f'>>> Wrote controlDict to {dst}')


def convert_and_extract(case_dir, csv_root, height, tol):
    """
    每个时间步：
      1) 删除旧的 VTK 根目录
      2) 调用 foamToVTK，把当前时间步输出到 VTK/
      3) 读取 VTK/ 下的所有 .vtk
    """
    times = []
    for name in os.listdir(case_dir):
        try:
            float(name)
        except ValueError:
            continue
        p = os.path.join(case_dir, name)
        if os.path.isdir(p):
            times.append(name)
    times.sort(key=lambda s: float(s))

    vtk_root = os.path.join(case_dir, "VTK")
    for t in times:
        print(f"\n=== Processing time {t} ===")
        # 1) 清除旧的 VTK
        if os.path.isdir(vtk_root):
            shutil.rmtree(vtk_root)
        os.makedirs(vtk_root, exist_ok=True)

        # 2) 用 foamToVTK 只输出这个时间步
        subprocess.run(
            f'call "{setvars_bat}" && cd /d "{case_dir}" && '
            f'"{FOAM_TO_VTK}" -case "{case_dir}" -time {t}',
            shell=True, check=True
        )

        # 3) 只遍历新生成的 VTK/*.vtk
        vtk_files = [
            fn for fn in os.listdir(vtk_root)
            if fn.lower().endswith(".vtk")
        ]
        vtk_files.sort()

        for vtk_file in vtk_files:
            mesh_vtk = os.path.join(vtk_root, vtk_file)
            mesh = meshio.read(mesh_vtk)
            pts = mesh.points
            U = mesh.point_data.get("U")
            if U is None:
                continue

            mask = np.abs(pts[:, 2] - height) < tol
            pts2, U2 = pts[mask], U[mask]

            Ux, Uy, Uz = U2[:,0], U2[:,1], U2[:,2]
            WS = np.linalg.norm(U2, axis=1)
            WF = (np.degrees(np.arctan2(-Ux, -Uy)) + 360) % 360

            # 输出目录
            case_name = os.path.basename(case_dir)
            out_dir = os.path.join(csv_root, case_name)
            os.makedirs(out_dir, exist_ok=True)

            time_int = int(float(t))
            h_int = int(round(height))
            base = f"U_plane_z{h_int}_t{time_int}"

            csv_path = os.path.join(out_dir, base + ".csv")
            npy_path = os.path.join(out_dir, base + ".npy")

            arr = np.column_stack((pts2, U2, WS, WF))
            np.save(npy_path, arr)
            header = "x,y,z,Ux,Uy,Uz,WS,WF"
            np.savetxt(csv_path, arr, delimiter=",", header=header, comments="")

            print(f"    ✓ Saved {csv_path}")



def main():
    os.makedirs(output_root, exist_ok=True)
    for fname in os.listdir(stl_dir):
        if not fname.lower().endswith('.stl'):
            continue
        case = os.path.splitext(fname)[0]
        src  = os.path.join(stl_dir, fname)
        for deg in ANGLES:
            case_dir = os.path.join(output_root, f"{case}_{deg}")
            # …（其余步骤同上）…
        case_dir = os.path.join(output_root,case)
        # copy template
        if os.path.isdir(case_dir):
            shutil.rmtree(case_dir)
        shutil.copytree(template_dir,case_dir)

        zero_dir = os.path.join(case_dir, "0")
        os.makedirs(zero_dir, exist_ok=True)
        ux = BASE_U * math.cos(math.radians(deg))
        uy = BASE_U * math.sin(math.radians(deg))

        print(f">>> Wrote initial U at {zero_dir}/U")

        # copy STL
        dst_tri = os.path.join(case_dir,'constant','triSurface')
        os.makedirs(dst_tri,exist_ok=True)
        shutil.copy2(os.path.join(stl_dir,fname),dst_tri)
        # compute bbox & center
        bbox,center = get_bbox_and_center(os.path.join(stl_dir,fname))
        # blockMesh
        write_blockMeshDict(case_dir,bbox)
        run_cmd([BLOCKMESH],case_dir)
        # surfaceFeatures -> generate .eMesh
        write_surfaceFeaturesDict(case_dir,case)
        run_cmd([SURFACEFEATURES],case_dir)
        # snappyHexMesh
        write_snappyDict(case_dir,case,bbox,center)
        run_cmd([SNAPPY,'-overwrite'],case_dir)
        # 在生成 mapping points 之后：
        write_U0_with_table(case_dir, time_series)
        print("➡ 写入 timeVaryingUniformFixedValue 边界条件到 0/U")

        write_controlDict(case_dir)

        # 再调用 simpleFoam
        run_cmd([SIMPLEFOAM], case_dir)
        run_cmd([POSTPROCESS, '-func', 'samplePlane'], case_dir)
        convert_and_extract(case_dir, CSV_ROOT, PLANE_Z, PLANE_TOL)

if __name__ == '__main__':
    main()
