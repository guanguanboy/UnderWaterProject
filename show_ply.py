import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def render_point_cloud(ply_file_path):
    # 读取PLY文件
    points = read_ply(ply_file_path)
    
    # 检查点云数据是否读取成功
    if points is None:
        print("Failed to load the point cloud.")
        return

    # 使用matplotlib绘制3D点云
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 提取XYZ坐标
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # 绘制点云
    ax.scatter(x, y, z, c=z, cmap='viridis', s=1)

    # 设置图像标题
    ax.set_title("3D Point Cloud")

    # 显示点云图
    plt.show()

def read_ply(ply_file_path):
    """读取PLY文件并返回点云数据"""
    try:
        with open(ply_file_path, 'r') as f:
            lines = f.readlines()
        
        # 找到点数据的开始行
        vertex_start_index = None
        for i, line in enumerate(lines):
            if line.strip() == 'end_header':
                vertex_start_index = i + 1
                break
        
        # 提取点云数据
        points = []
        for line in lines[vertex_start_index:]:
            # 跳过空行
            if line.strip() == "":
                continue
            points.append(list(map(float, line.strip().split())))
        
        return np.array(points)
    except Exception as e:
        print(f"Error reading PLY file: {e}")
        return None

# 使用示例
ply_file_path = '/data/gl/Codes/UnderWater3D/curasao.ply'  # 替换为你的PLY文件路径
render_point_cloud(ply_file_path)
